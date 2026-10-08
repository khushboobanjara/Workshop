import asyncio
import json
import pathlib

import pytest

from mcp_layer.config import MCPSettings
from mcp_layer.gateway import Decision, MCPGateway, current_principal
from mcp_layer.roles import resolve_role
from mcp_layer.schemas import ErrorCode, MCPResult, Ownership, Principal, Role, ServerType, ToolKind, ToolMeta
from mcp_layer.servers import build_registry

SQL = pathlib.Path("mcp_layer/migrations/001_mcp_foundation.sql")
UID = Principal(user_id=7, role=Role.USER)


class AllowAll:
    def check(self, *a, **k):
        return Decision(True)


class Capture:
    def __init__(self):
        self.events = []

    def record(self, e):
        self.events.append(e)


def run(coro):
    return asyncio.run(coro)


def gw(**kw):
    kw.setdefault("auditor", Capture())
    return MCPGateway(build_registry(), **kw)


# ---------------------------------------------------------------- registry
def test_ten_servers_5_3_2():
    c = build_registry().counts()
    assert c == {"total": 10, "USER": 5, "ADMIN": 3, "SYSTEM": 2}


def test_server_roles_are_separated():
    for s in build_registry().servers():
        if s.server_type is ServerType.SYSTEM:
            assert s.allowed_roles == {Role.SYSTEM}
        else:
            assert Role.SYSTEM not in s.allowed_roles
        if s.server_type is ServerType.ADMIN:
            assert Role.USER not in s.allowed_roles


def test_user_cannot_discover_admin_or_system_servers():
    seen = {d["server_type"] for d in build_registry().discover(Role.USER)}
    assert seen == {"USER"}


def test_forbidden_tool_parameters_rejected():
    spec = build_registry().get("mcp_01_patient_profile")
    own = ToolMeta(name="x_own", allowed_roles=frozenset({Role.USER}), permission="profile.read.own",
                   ownership=Ownership.CURRENT_USER_ONLY)
    with pytest.raises(ValueError):
        spec.tool(own)(lambda user_id: None)
    role = ToolMeta(name="x_role", allowed_roles=frozenset({Role.USER}), permission="profile.read.own")
    with pytest.raises(ValueError):
        spec.tool(role)(lambda role: None)


def test_tool_metadata_rules():
    with pytest.raises(ValueError):   # destructive tool exposed to USER
        ToolMeta(name="del", allowed_roles=frozenset({Role.USER}), permission="user.delete", kind=ToolKind.DESTRUCTIVE)
    with pytest.raises(ValueError):   # write tool without audit
        ToolMeta(name="w", allowed_roles=frozenset({Role.ADMIN}), permission="a.write", kind=ToolKind.WRITE,
                 audit_required=False)


# ------------------------------------------------------------------ result
def test_failed_result_is_never_verified():
    r = MCPResult(success=False, data={"x": 1}, verified=True, source="mysql")
    assert r.verified is False and r.data is None and r.error_code


def test_verified_needs_source():
    with pytest.raises(ValueError):
        MCPResult(success=True, data=1, verified=True)


# ------------------------------------------------------------------- roles
@pytest.mark.parametrize("db,status,expected", [
    ("PATIENT", None, Role.USER), ("patient", None, Role.USER), ("ADMIN", None, Role.ADMIN),
    ("DOCTOR", "APPROVED", Role.DOCTOR), ("DOCTOR", "PENDING", None), ("DOCTOR", "SUSPENDED", None),
    ("DOCTOR", None, None), (None, None, None), ("", None, None), ("SYSTEM", None, None), ("root", None, None),
])
def test_resolve_role(db, status, expected):
    assert resolve_role(db, status) == expected


# ----------------------------------------------------------------- gateway
def test_default_gateway_denies_everything():
    g = gw()
    r = run(g.invoke(UID, "mcp_01_patient_profile", "server_ping"))
    assert (r.success, r.error_code, r.verified) == (False, ErrorCode.ACCESS_DENIED, False)


def test_unauthenticated():
    r = run(gw(authorizer=AllowAll()).invoke(None, "mcp_01_patient_profile", "server_ping"))
    assert r.error_code == ErrorCode.UNAUTHENTICATED


def test_unknown_server_and_tool():
    g = gw(authorizer=AllowAll())
    assert run(g.invoke(UID, "nope", "server_ping")).error_code == ErrorCode.SERVER_NOT_FOUND
    assert run(g.invoke(UID, "mcp_01_patient_profile", "drop_everything")).error_code == ErrorCode.TOOL_NOT_FOUND
    assert run(g.invoke(UID, ["x"], 5)).error_code == ErrorCode.SERVER_NOT_FOUND


def test_allowed_call_runs_through_fastmcp():
    r = run(gw(authorizer=AllowAll()).invoke(UID, "mcp_03_pharmacy", "server_ping"))
    assert r.success and r.verified and r.source == "mcp_internal" and r.request_id


def test_identity_comes_from_gateway_not_arguments():
    g = gw(authorizer=AllowAll())
    spec = g.registry.get("mcp_01_patient_profile")

    @spec.tool(ToolMeta(name="whoami", allowed_roles=frozenset({Role.USER}), permission="profile.read.own",
                        ownership=Ownership.CURRENT_USER_ONLY))
    def whoami() -> dict:
        return MCPResult.ok({"uid": current_principal().user_id}, source="test").model_dump()

    r = run(g.invoke(UID, "mcp_01_patient_profile", "whoami"))
    assert r.data == {"uid": 7}
    # an attacker-supplied user_id argument is not a parameter, so it is rejected, not honoured
    r2 = run(g.invoke(UID, "mcp_01_patient_profile", "whoami", {"user_id": 1}))
    assert not r2.success and r2.verified is False


def test_tool_exception_is_sanitized():
    g = gw(authorizer=AllowAll())
    spec = g.registry.get("mcp_01_patient_profile")

    @spec.tool(ToolMeta(name="boom", allowed_roles=frozenset({Role.USER}), permission="profile.read.own"))
    def boom() -> dict:
        raise RuntimeError("mysql://root:hunter2@localhost secret")

    r = run(g.invoke(UID, "mcp_01_patient_profile", "boom"))
    assert not r.success and r.verified is False
    assert "hunter2" not in json.dumps(r.model_dump())


def test_raw_tool_output_is_not_trusted():
    g = gw(authorizer=AllowAll())
    spec = g.registry.get("mcp_01_patient_profile")

    @spec.tool(ToolMeta(name="raw", allowed_roles=frozenset({Role.USER}), permission="profile.read.own"))
    def raw() -> dict:
        return {"appointment": "tomorrow 10am"}   # not an MCPResult

    r = run(g.invoke(UID, "mcp_01_patient_profile", "raw"))
    assert not r.success and not r.verified and r.data is None


def test_oversized_arguments_rejected():
    g = gw(authorizer=AllowAll(), settings=MCPSettings(True, 15, 100, 60))
    r = run(g.invoke(UID, "mcp_01_patient_profile", "server_ping", {"a": "x" * 500}))
    assert r.error_code == ErrorCode.INVALID_INPUT


def test_disabled_and_unavailable_server():
    off = gw(authorizer=AllowAll(), settings=MCPSettings(False, 15, 1000, 60))
    assert run(off.invoke(UID, "mcp_01_patient_profile", "server_ping")).error_code == ErrorCode.MCP_DISABLED
    g = gw(authorizer=AllowAll())
    g.registry.get("mcp_03_pharmacy").status = "disabled"
    assert run(g.invoke(UID, "mcp_03_pharmacy", "server_ping")).error_code == ErrorCode.MCP_UNAVAILABLE


def test_health_check_all_ten():
    g = gw()
    report = run(g.health_check())
    assert len(report) == 10 and set(report.values()) == {"healthy"}


def test_audit_records_names_never_values():
    cap = Capture()
    g = MCPGateway(build_registry(), auditor=cap)   # default deny
    run(g.invoke(UID, "mcp_01_patient_profile", "server_ping", {"note": "s3cret-value"}))
    blob = json.dumps(cap.events)
    assert "s3cret-value" not in blob
    ev = cap.events[0]
    assert ev["argument_names"] == ["note"] and ev["permission_result"] == "DENIED" and ev["role"] == "USER"


# --------------------------------------------------------------- migration
def test_migration_parses_as_mysql():
    import sqlglot
    from sqlglot import exp
    stmts = sqlglot.parse(SQL.read_text(), read="mysql")
    tables = [s.this.this.name for s in stmts if isinstance(s, exp.Create)]
    assert sorted(tables) == sorted(["mcp_servers", "mcp_tools", "mcp_permissions",
                                     "mcp_audit_logs", "llm_pricing", "llm_usage"])
    assert "INSERT INTO llm_pricing" not in "\n".join(l for l in SQL.read_text().splitlines()
                                                      if not l.strip().startswith("--"))
