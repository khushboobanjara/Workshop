"""Phase 4: tool guard, DB-failure classification, safety hook, DB audit, startup."""
import asyncio
import inspect
import json
import logging

import mysql.connector.errors as mye
import pytest

from mcp_layer.audit_db import CompositeAuditor, DbAuditor
from mcp_layer.bootstrap import build_gateway, database_available, initialize_mcp
from mcp_layer.gateway import Decision, MCPGateway, SafetyDecision, safe_argument_names
from mcp_layer.guard import classify_exception, guarded
from mcp_layer.schemas import ErrorCode, MCPResult, Ownership, Principal, Role, ToolMeta
from mcp_layer.servers import build_registry

run = asyncio.run
UID = Principal(user_id=7, role=Role.USER)
SRV = "mcp_01_patient_profile"


class Allow:
    def check(self, *a): return Decision(True)


class Capture:
    def __init__(self): self.events = []
    def record(self, e): self.events.append(e)


def gw(safety=None, auditor=None):
    cap = auditor or Capture()
    return MCPGateway(build_registry(), authorizer=Allow(), auditor=cap, safety=safety), cap


def add_tool(g, fn, name="t", perm="profile.read.own"):
    g.registry.get(SRV).tool(ToolMeta(name=name, allowed_roles=frozenset({Role.USER}), permission=perm))(fn)


# ================================================================= classification
@pytest.mark.parametrize("exc,code", [
    (mye.InterfaceError("x", errno=2003), ErrorCode.DATABASE_UNAVAILABLE),
    (mye.OperationalError("x", errno=2006), ErrorCode.DATABASE_UNAVAILABLE),
    (mye.OperationalError("lock wait", errno=1205), ErrorCode.DATABASE_UNAVAILABLE),
    (mye.DatabaseError("x", errno=1045), ErrorCode.DATABASE_UNAVAILABLE),
    (mye.DatabaseError("x", errno=2013), ErrorCode.DATABASE_UNAVAILABLE),
    (mye.ProgrammingError("bad sql", errno=1064), ErrorCode.INTERNAL_ERROR),
    (mye.IntegrityError("dup", errno=1062), ErrorCode.INTERNAL_ERROR),
    (ConnectionRefusedError("x"), ErrorCode.DATABASE_UNAVAILABLE),
    (TimeoutError("x"), ErrorCode.DATABASE_UNAVAILABLE),
    (PermissionError("x"), ErrorCode.ACCESS_DENIED),
    (ValueError("x"), ErrorCode.INVALID_INPUT),
    (KeyError("x"), ErrorCode.INVALID_INPUT),
    (RuntimeError("x"), ErrorCode.INTERNAL_ERROR),
])
def test_classification(exc, code):
    assert classify_exception(exc) == code


# ====================================================================== guard
def test_guard_preserves_signature_and_passes_results_through():
    def tool(a: int, b: str = "x") -> dict: return {"ok": a}
    w = guarded(tool, "s", "t")
    assert inspect.signature(w) == inspect.signature(tool) and w.__name__ == "tool"
    assert w(1) == {"ok": 1}


def test_guard_supports_async_tools():
    async def tool() -> dict: raise mye.OperationalError("down", errno=2003)
    out = run(guarded(tool, "s", "t")())
    assert out["error_code"] == ErrorCode.DATABASE_UNAVAILABLE and out["verified"] is False


def test_database_outage_inside_tool_is_database_unavailable_end_to_end(capsys):
    g, cap = gw()
    def down() -> dict:
        raise mye.InterfaceError("2003: Can't connect to MySQL server on 'prod-db.internal:3306' (hunter2)", errno=2003)
    add_tool(g, down)
    r = run(g.invoke(UID, SRV, "t"))
    assert r.error_code == ErrorCode.DATABASE_UNAVAILABLE
    assert r.success is False and r.verified is False and r.data is None
    assert r.message == "The required system data is currently unavailable."
    assert cap.events[0]["error_code"] == ErrorCode.DATABASE_UNAVAILABLE
    seen = capsys.readouterr()
    for stream in (seen.out, seen.err):                    # no traceback / driver text printed anywhere
        assert "hunter2" not in stream and "prod-db" not in stream and "Traceback" not in stream
    assert "hunter2" not in json.dumps(r.model_dump(mode="json"))


def test_guard_never_logs_exception_message(caplog):
    g, _ = gw()
    def boom() -> dict: raise RuntimeError("password=hunter2")
    add_tool(g, boom)
    with caplog.at_level(logging.DEBUG):
        run(g.invoke(UID, SRV, "t"))
    assert "hunter2" not in caplog.text


def test_sql_bug_is_internal_error_not_unavailable():
    g, _ = gw()
    def bug() -> dict: raise mye.ProgrammingError("syntax", errno=1064)
    add_tool(g, bug)
    assert run(g.invoke(UID, SRV, "t")).error_code == ErrorCode.INTERNAL_ERROR


def test_valid_tool_result_still_verified():
    g, _ = gw()
    def fine() -> dict: return MCPResult.ok({"n": 1}, source="mysql").model_dump(mode="json")
    add_tool(g, fine)
    r = run(g.invoke(UID, SRV, "t"))
    assert r.success and r.verified and r.source == "mysql" and r.data == {"n": 1}


# ================================================================== safety hook
class Block:
    def __init__(self, allow=False): self.allow, self.calls = allow, 0
    def check(self, *a):
        self.calls += 1
        return SafetyDecision(self.allow, "" if self.allow else "suspicious_input")


def test_safety_block_prevents_execution_and_is_audited():
    ran = []
    safety = Block()
    g, cap = gw(safety)
    def t() -> dict: ran.append(1); return MCPResult.ok({}, source="mysql").model_dump(mode="json")
    add_tool(g, t)
    r = run(g.invoke(UID, SRV, "t"))
    assert r.error_code == ErrorCode.SAFETY_BLOCKED and not ran
    e = cap.events[0]
    assert e["security_result"] == "BLOCKED" and e["denial_reason"] == "suspicious_input" and e["permission_result"] == "ALLOWED"


def test_safety_pass_is_audited_and_tool_runs():
    g, cap = gw(Block(allow=True))
    add_tool(g, lambda: MCPResult.ok({}, source="mysql").model_dump(mode="json"))
    assert run(g.invoke(UID, SRV, "t")).success
    assert cap.events[0]["security_result"] == "PASSED"


def test_safety_crash_blocks():
    class Broken:
        def check(self, *a): raise RuntimeError("x")
    g, cap = gw(Broken())
    add_tool(g, lambda: MCPResult.ok({}, source="mysql").model_dump(mode="json"))
    assert run(g.invoke(UID, SRV, "t")).error_code == ErrorCode.SAFETY_BLOCKED
    assert cap.events[0]["denial_reason"] == "safety_error"


def test_safety_not_consulted_when_rbac_denies():
    class Deny:
        def check(self, *a): return Decision(False, "nope")
    safety = Block(allow=True)
    g, cap = gw(safety)
    g.authorizer = Deny()
    r = run(g.invoke(UID, SRV, "server_ping"))
    assert r.error_code == ErrorCode.ACCESS_DENIED and safety.calls == 0
    assert cap.events[0]["security_result"] == "NOT_CHECKED"


# ================================================================ argument names
def test_argument_names_are_sanitized_and_bounded():
    names = safe_argument_names({"ok_name": 1, "bad name; DROP": 2, "x" * 500: 3, 7: 4, ("t",): 5})
    assert all(len(n) <= 40 and n.replace("_", "").isalnum() for n in names)
    assert len(safe_argument_names({f"k{i}": 0 for i in range(100)})) == 20
    assert safe_argument_names("not a dict") == []


def test_mixed_type_keys_do_not_break_auditing():
    g, cap = gw()
    r = run(g.invoke(UID, SRV, "server_ping", {1: "a", "b": 2}))
    assert len(cap.events) == 1 and r.request_id


# ================================================================= DB auditor
class AConn:
    def __init__(self, fail=None): self.fail, self.sql, self.params = fail, None, None
    committed = rolled = closed = False
    def cursor(self): return self
    def execute(self, sql, params):
        if self.fail: raise self.fail
        self.sql, self.params = sql, params
    def commit(self): self.committed = True
    def rollback(self): self.rolled = True
    def close(self): self.closed = True


EVENT = {"request_id": "a" * 32, "user_id": 7, "role": "USER", "server_id": SRV, "tool": "t",
         "argument_names": ["note"], "permission_result": "DENIED", "security_result": "NOT_CHECKED",
         "denial_reason": "permission_not_held", "success": False, "error_code": "ACCESS_DENIED",
         "source": "none", "verified": False, "duration_ms": 1.5}


def test_db_auditor_inserts_with_bound_parameters():
    c = AConn()
    DbAuditor(lambda: c).record(EVENT)
    assert c.committed and c.closed and "%s" in c.sql and "permission_not_held" not in c.sql
    assert c.params[0] == "a" * 32 and c.params[1] == 7 and c.params[5] == "note"
    assert len(c.params) == c.sql.count("%s")


def test_db_auditor_clips_oversized_fields():
    c = AConn()
    DbAuditor(lambda: c).record({**EVENT, "tool": "t" * 500, "argument_names": ["n" * 40] * 20, "role": "R" * 99})
    assert len(c.params[4]) == 64 and len(c.params[5]) <= 255 and len(c.params[2]) == 20


def test_db_down_falls_back_to_log_and_does_not_raise(caplog):
    with caplog.at_level(logging.WARNING):
        DbAuditor(lambda: AConn(fail=mye.OperationalError("down", errno=2003))).record(EVENT)
    assert "mcp_call_unstored" in caplog.text


def test_missing_user_fk_retries_without_user_link():
    calls = []
    class FK(AConn):
        def execute(self, sql, params):
            calls.append(params[1])
            if params[1] is not None: raise mye.IntegrityError("fk", errno=1452)
    DbAuditor(lambda: FK()).record(EVENT)
    assert calls == [7, None]


def test_composite_isolates_failing_sink():
    good = Capture()
    class Bad:
        def record(self, e): raise RuntimeError("x")
    CompositeAuditor([Bad(), good]).record(EVENT)
    assert good.events == [EVENT]


def test_audit_event_end_to_end_into_db_has_no_values():
    c = AConn()
    g, _ = gw(auditor=DbAuditor(lambda: c))
    run(g.invoke(UID, SRV, "server_ping", {"note": "s3cret-value"}))
    assert "s3cret-value" not in json.dumps(c.params, default=str)


# ================================================================= migration 003
def test_migration_003_is_idempotent_and_scoped():
    from pathlib import Path
    sql = (Path(__file__).resolve().parents[1] / "mcp_layer/migrations/003_audit_security_columns.sql").read_text()
    code = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    assert "INFORMATION_SCHEMA.COLUMNS" in code and "security_result" in code and "denial_reason" in code
    assert "mcp_audit_logs" in code and "users" not in code.lower() and "DROP" not in code.upper()


# ===================================================================== startup
class Q:
    def __init__(self, fail=False, rows=()): self.fail, self.rows, self.log = fail, rows, []
    def cursor(self): return self
    def execute(self, sql, p=None):
        if self.fail: raise mye.InterfaceError("down", errno=2003)
        self.log.append(sql)
    def fetchall(self): return list(self.rows)
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass


def test_database_probe():
    assert database_available(lambda: Q()) is True
    assert database_available(lambda: Q(fail=True)) is False


def test_initialize_with_database_down_does_not_raise_and_skips_sync():
    g = build_gateway(use_db_overrides=False, use_db_audit=False)
    rep = run(initialize_mcp(g, connection_factory=lambda: Q(fail=True)))
    assert rep["database"] == "down" and "skipped" in rep["registry_sync"]
    assert set(rep["health"].values()) == {"healthy"}      # servers are in-process; DB outage is separate


def test_initialize_with_database_up_syncs_registry():
    g = build_gateway(use_db_overrides=False, use_db_audit=False)
    rep = run(initialize_mcp(g, connection_factory=lambda: Q()))
    assert rep["database"] == "up" and rep["registry_sync"]["servers"] == 10


def test_initialize_survives_sync_failure():
    class SyncBoom(Q):
        def execute(self, sql, p=None):
            if sql.startswith("INSERT"): raise RuntimeError("x")
            super().execute(sql, p)
    g = build_gateway(use_db_overrides=False, use_db_audit=False)
    rep = run(initialize_mcp(g, connection_factory=lambda: SyncBoom()))
    assert rep["registry_sync"] == {"error": "RuntimeError"} and rep["health"]
