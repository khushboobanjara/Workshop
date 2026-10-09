"""Phase 7: MCP 09 (Security / RBAC / Audit), run end to end through the real gateway + RBAC.
The database is replaced by in-memory fakes. Needs fastmcp, pydantic and mysql-connector installed
(as the earlier phase tests do)."""
import asyncio

import mysql.connector.errors as mye
import pytest

from mcp_layer.gateway import MCPGateway
from mcp_layer.overrides import NoOverrides
from mcp_layer.permissions import validate_registry_permissions
from mcp_layer.rbac import RBACAuthorizer
from mcp_layer.schemas import ErrorCode, Principal, Role, ServerType, ToolKind
from mcp_layer.servers import build_registry
from mcp_layer.servers import mcp_09_security_audit as m09
from src.database import mcp_admin_repository as arepo
from src.database import mcp_security_repository as srepo

run = asyncio.run
P09 = "mcp_09_security_audit"

USER = Principal(user_id=7, role=Role.USER)
DOCTOR = Principal(user_id=20, role=Role.DOCTOR, doctor_id=3)
ADMIN = Principal(user_id=1, role=Role.ADMIN)
SUPER = Principal(user_id=2, role=Role.SUPER_ADMIN)
SYSTEM = Principal.system()

READ_ARGS = {
    "get_audit_logs": {},
    "get_audit_summary": {},
    "get_security_events": {},
    "check_permission": {"role_name": "USER", "server_id": "mcp_08_user_system_admin", "tool_name": "list_users"},
    "check_role": {"role_name": "USER"},
}
WRITE_ARGS = {"event_type": "SUSPICIOUS_ACTIVITY", "reason": "repeated denials"}


class Capture:
    def __init__(self): self.events = []
    def record(self, e): self.events.append(e)


class Revoke:
    """Database revocation stand-in: ADMIN may not use admin_list_medicines."""
    def is_revoked(self, role, server_id, tool_name):
        return role is Role.ADMIN and tool_name == "admin_list_medicines"


def make_gateway(overrides=None):
    cap = Capture()
    gw = MCPGateway(build_registry(with_tools=True), authorizer=RBACAuthorizer(overrides or NoOverrides()), auditor=cap)
    m09.bind_gateway(gw)
    return gw, cap


def call(gw, who, tool, args=None, server=P09):
    return run(gw.invoke(who, server, tool, args or {}))


def db_down(*a, **k):
    raise mye.InterfaceError("2003: Can't connect to MySQL server on 'prod-db.internal:3306'", errno=2003)


class World:
    def __init__(self):
        self.calls = []
        self.users = {7: {"user_id": 7, "role": "PATIENT"}}

    def install(self, mp):
        w = self
        def rec(name, ret):
            def fn(*a, **k):
                w.calls.append((name, a, k))
                return ret
            return fn
        mp.setattr(srepo, "query_audit_logs", rec("query_audit_logs", [{"audit_id": 1, "tool_name": "x"}]))
        mp.setattr(srepo, "audit_totals", rec("audit_totals", {"total_calls": 9, "denied": 2, "blocked": 1, "failed": 3, "distinct_users": 2}))
        mp.setattr(srepo, "audit_by_server", rec("audit_by_server", [{"server_id": "mcp_02_doctor_appointment", "calls": 5}]))
        mp.setattr(srepo, "top_denial_reasons", rec("top_denial_reasons", [{"denial_reason": "role_not_allowed_on_server", "occurrences": 2}]))
        mp.setattr(srepo, "security_events", rec("security_events", [{"audit_id": 4, "permission_result": "DENIED"}]))
        mp.setattr(srepo, "repeat_denials", rec("repeat_denials", [{"user_id": 7, "denials": 6}]))
        mp.setattr(srepo, "insert_security_event", rec("insert_security_event", None))
        mp.setattr(arepo, "get_user", lambda i: w.users.get(i))


@pytest.fixture
def env(monkeypatch):
    world = World()
    world.install(monkeypatch)
    gw, cap = make_gateway()
    yield world, gw, cap
    m09.bind_gateway(None)


# ===================================================================== registry
def test_exactly_ten_servers_five_user_three_admin_two_system():
    reg = build_registry(with_tools=True)
    assert len(reg.servers()) == 10
    assert len(reg.servers(ServerType.USER)) == 5
    assert len(reg.servers(ServerType.ADMIN)) == 3
    assert len(reg.servers(ServerType.SYSTEM)) == 2


def test_permission_matrix_covers_mcp_09():
    assert validate_registry_permissions(build_registry(with_tools=True)) == []


def test_mcp_09_tools_are_system_or_super_admin_only():
    tools = build_registry(with_tools=True).get(P09).tools
    assert set(tools) == {"server_ping", "get_audit_logs", "get_audit_summary", "get_security_events",
                          "check_permission", "check_role", "write_audit_log"}
    for name, meta in tools.items():
        if name == "server_ping":
            continue
        assert not ({Role.USER, Role.DOCTOR, Role.ADMIN} & meta.allowed_roles), name
    assert tools["write_audit_log"].allowed_roles == frozenset({Role.SYSTEM})
    assert tools["write_audit_log"].kind is ToolKind.WRITE
    for name in READ_ARGS:
        assert tools[name].allowed_roles == frozenset({Role.SUPER_ADMIN})
        assert tools[name].kind is ToolKind.READ


def test_only_mcp_10_is_still_ping_only():
    reg = build_registry(with_tools=True)
    assert set(reg.get("mcp_10_llm_monitoring").tools) == {"server_ping"}
    assert len(reg.get(P09).tools) > 1


# ========================================================================== RBAC
@pytest.mark.parametrize("who", [USER, DOCTOR, ADMIN])
@pytest.mark.parametrize("tool", sorted(READ_ARGS))
def test_non_super_admin_roles_are_denied_every_read_tool(env, who, tool):
    world, gw, _ = env
    r = call(gw, who, tool, READ_ARGS[tool])
    assert r.error_code == ErrorCode.ACCESS_DENIED and r.data is None and not world.calls


@pytest.mark.parametrize("who", [USER, DOCTOR, ADMIN, SUPER])
def test_write_audit_log_is_denied_to_every_human_role(env, who):
    world, gw, _ = env
    r = call(gw, who, "write_audit_log", WRITE_ARGS)
    assert r.error_code == ErrorCode.ACCESS_DENIED and not world.calls


@pytest.mark.parametrize("tool", sorted(READ_ARGS))
def test_super_admin_may_read(env, tool):
    _, gw, _ = env
    r = call(gw, SUPER, tool, READ_ARGS[tool])
    assert r.success and r.verified, r.message


@pytest.mark.parametrize("tool", sorted(READ_ARGS))
def test_system_identity_cannot_use_the_super_admin_read_tools(env, tool):
    _, gw, _ = env
    assert call(gw, SYSTEM, tool, READ_ARGS[tool]).error_code == ErrorCode.ACCESS_DENIED


def test_system_identity_may_write_audit_events(env):
    world, gw, _ = env
    r = call(gw, SYSTEM, "write_audit_log", WRITE_ARGS)
    assert r.success and r.data == {"logged": True, "event_type": "SUSPICIOUS_ACTIVITY"}
    name, args, _ = world.calls[-1]
    assert name == "insert_security_event" and args[1] is None and args[2] == "SUSPICIOUS_ACTIVITY"


def test_no_session_is_unauthenticated(env):
    _, gw, _ = env
    assert call(gw, None, "get_audit_logs").error_code == ErrorCode.UNAUTHENTICATED


def test_role_smuggling_is_refused(env):
    world, gw, _ = env
    r = call(gw, SUPER, "get_audit_logs", {"role": "SYSTEM"})
    assert r.error_code == ErrorCode.ACCESS_DENIED and not world.calls


# ==================================================================== audit reads
def test_audit_logs_pass_validated_filters_to_the_repository(env):
    world, gw, _ = env
    r = call(gw, SUPER, "get_audit_logs", {"hours": 48, "only_problems": True, "server_filter": "MCP_02_doctor_appointment",
                                           "tool_filter": "book_appointment", "target_user_id": 7})
    assert r.success and r.metadata["row_count"] == 1
    assert world.calls[-1][1] == (48, True, "mcp_02_doctor_appointment", "book_appointment", 7, 100)


@pytest.mark.parametrize("bad", [{"hours": 0}, {"hours": 721}, {"hours": "0"}, {"hours": "9999"},
                                 {"server_filter": "x'; DROP TABLE users;--"},
                                 {"tool_filter": "a b"}, {"target_user_id": -1}, {"target_user_id": 0}])
def test_audit_log_bad_input_never_reaches_the_database(env, bad):
    world, gw, _ = env
    r = call(gw, SUPER, "get_audit_logs", bad)
    assert not r.success and r.error_code == ErrorCode.INVALID_INPUT and not world.calls


def test_fastmcp_coerces_harmless_type_slips_before_our_checks_run(env):
    """FastMCP validates arguments with pydantic first, so "24" becomes 24, True becomes 1 and "yes"
    becomes True. These are in range and harmless; out-of-range values (above) are still rejected."""
    world, gw, _ = env
    assert call(gw, SUPER, "get_audit_logs", {"hours": "24"}).success
    assert call(gw, SUPER, "get_audit_logs", {"only_problems": "yes"}).success
    assert world.calls[0][1][0] == 24 and world.calls[1][1][1] is True


def test_empty_audit_result_is_a_controlled_answer(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(srepo, "query_audit_logs", lambda *a, **k: [])
    r = call(gw, SUPER, "get_audit_logs")
    assert r.success and r.verified and r.data == [] and r.metadata["row_count"] == 0 and r.message


def test_audit_summary_shape_and_bounds(env):
    world, gw, _ = env
    r = call(gw, SUPER, "get_audit_summary", {"hours": 6})
    assert r.success and set(r.data) == {"window_hours", "totals", "by_server", "top_denial_reasons"}
    assert r.data["window_hours"] == 6 and r.data["totals"]["denied"] == 2
    assert call(gw, SUPER, "get_audit_summary", {"hours": 1000}).error_code == ErrorCode.INVALID_INPUT


def test_security_events_include_repeat_denials_and_validate_threshold(env):
    _, gw, _ = env
    r = call(gw, SUPER, "get_security_events", {"hours": 24, "min_denials": 3})
    assert r.success and r.data["events"] and r.data["repeat_denials"][0]["user_id"] == 7
    assert call(gw, SUPER, "get_security_events", {"min_denials": 1}).error_code == ErrorCode.INVALID_INPUT


@pytest.mark.parametrize("tool", ["get_audit_logs", "get_audit_summary", "get_security_events"])
def test_database_outage_blocks_verified_data(env, monkeypatch, tool):
    _, gw, _ = env
    for fn in ("query_audit_logs", "audit_totals", "security_events"):
        monkeypatch.setattr(srepo, fn, db_down)
    r = call(gw, SUPER, tool)
    assert not r.success and not r.verified and r.data is None
    assert r.error_code == ErrorCode.DATABASE_UNAVAILABLE
    assert "prod-db" not in repr(r.model_dump()) and "3306" not in repr(r.model_dump())


# ========================================================================== RBAC tools
def check(gw, role, server, tool):
    return call(gw, SUPER, "check_permission", {"role_name": role, "server_id": server, "tool_name": tool})


def test_check_permission_matches_the_real_rbac_engine(env):
    _, gw, _ = env
    r = check(gw, "USER", "mcp_08_user_system_admin", "list_users")
    assert r.success and r.data["allowed"] is False and r.data["reason"] == "role_not_allowed_on_server"
    r = check(gw, "ADMIN", "mcp_08_user_system_admin", "set_user_role")
    assert r.data["allowed"] is False and r.data["reason"] == "role_not_allowed_on_tool"
    assert r.data["required_permission"] == "user.role.manage"
    assert check(gw, "SUPER_ADMIN", "mcp_08_user_system_admin", "set_user_role").data["allowed"] is True
    assert check(gw, "DOCTOR", "mcp_06_doctor_admin", "get_my_doctor_profile").data["allowed"] is True
    assert check(gw, "USER", "mcp_02_doctor_appointment", "find_doctors").data["allowed"] is True
    assert check(gw, "DOCTOR", P09, "get_audit_logs").data["allowed"] is False


def test_check_permission_shows_database_revocations():
    gw, _ = make_gateway(Revoke())
    try:
        r = check(gw, "ADMIN", "mcp_07_pharmacy_admin", "admin_list_medicines")
        assert r.data["allowed"] is False and r.data["reason"] == "revoked_by_policy"
    finally:
        m09.bind_gateway(None)


def test_check_permission_does_not_run_the_tool(env):
    world, gw, _ = env
    check(gw, "SUPER_ADMIN", "mcp_08_user_system_admin", "set_user_role")
    assert not world.calls


@pytest.mark.parametrize("args", [
    {"role_name": "root", "server_id": "mcp_02_doctor_appointment", "tool_name": "find_doctors"},
    {"role_name": "", "server_id": "mcp_02_doctor_appointment", "tool_name": "find_doctors"},
    {"role_name": "USER", "server_id": "bad server!", "tool_name": "find_doctors"},
    {"role_name": "USER", "server_id": "mcp_02_doctor_appointment", "tool_name": "../etc"},
])
def test_check_permission_rejects_bad_input(env, args):
    _, gw, _ = env
    assert call(gw, SUPER, "check_permission", args).error_code == ErrorCode.INVALID_INPUT


def test_check_permission_unknown_target_is_not_found(env):
    _, gw, _ = env
    assert check(gw, "USER", "mcp_99_nope", "x_tool").error_code == ErrorCode.NOT_FOUND
    assert check(gw, "USER", "mcp_02_doctor_appointment", "nope_tool").error_code == ErrorCode.NOT_FOUND


def test_rbac_tools_fail_closed_when_no_gateway_is_bound(env):
    _, gw, _ = env
    m09.bind_gateway(None)
    assert check(gw, "USER", "mcp_02_doctor_appointment", "find_doctors").error_code == ErrorCode.MCP_UNAVAILABLE
    assert call(gw, SUPER, "check_role", {"role_name": "USER"}).error_code == ErrorCode.MCP_UNAVAILABLE


def test_check_role_lists_only_what_the_role_can_reach(env):
    _, gw, _ = env
    user = call(gw, SUPER, "check_role", {"role_name": "user"}).data
    names = {(t["server_id"], t["tool_name"]) for t in user["tools"]}
    assert ("mcp_02_doctor_appointment", "find_doctors") in names
    assert not any(s in ("mcp_06_doctor_admin", "mcp_07_pharmacy_admin", P09) for s, _ in names)
    assert "user.manage" not in user["permissions"] and "audit.read" not in user["permissions"]
    assert user["tool_count"] == len(user["tools"])

    admin = {(t["server_id"], t["tool_name"]) for t in call(gw, SUPER, "check_role", {"role_name": "ADMIN"}).data["tools"]}
    assert ("mcp_07_pharmacy_admin", "admin_list_medicines") in admin
    assert ("mcp_08_user_system_admin", "set_user_role") not in admin and (P09, "get_audit_logs") not in admin

    doctor = {t["server_id"] for t in call(gw, SUPER, "check_role", {"role_name": "DOCTOR"}).data["tools"]}
    assert P09 not in doctor and "mcp_08_user_system_admin" not in doctor

    system = {(t["server_id"], t["tool_name"]) for t in call(gw, SUPER, "check_role", {"role_name": "SYSTEM"}).data["tools"]}
    assert (P09, "write_audit_log") in system and (P09, "get_audit_logs") not in system


# ================================================================== write_audit_log
def test_write_audit_log_validates_event_and_reason(env):
    world, gw, _ = env
    for bad in ({"event_type": "DROP_TABLE", "reason": "ok reason"},
                {"event_type": "SUSPICIOUS_ACTIVITY", "reason": "x'; DROP TABLE users;--"},
                {"event_type": "SUSPICIOUS_ACTIVITY", "reason": "a"},
                {"event_type": "SUSPICIOUS_ACTIVITY", "reason": "<script>alert(1)</script>"},
                {"event_type": "", "reason": "ok reason"}):
        assert call(gw, SYSTEM, "write_audit_log", bad).error_code == ErrorCode.INVALID_INPUT
    assert not world.calls


def test_write_audit_log_records_a_blocked_event_for_a_real_user(env):
    world, gw, _ = env
    r = call(gw, SYSTEM, "write_audit_log", {**WRITE_ARGS, "event_type": "prompt_injection_blocked", "subject_user_id": 7})
    assert r.success
    _, args, _ = world.calls[-1]
    assert args[1] == 7 and args[2] == "PROMPT_INJECTION_BLOCKED" and args[3] == "repeated denials"


def test_write_audit_log_unknown_user_writes_nothing(env):
    world, gw, _ = env
    r = call(gw, SYSTEM, "write_audit_log", {**WRITE_ARGS, "subject_user_id": 999})
    assert r.error_code == ErrorCode.NOT_FOUND and not world.calls
    assert call(gw, SYSTEM, "write_audit_log", {**WRITE_ARGS, "subject_user_id": -4}).error_code == ErrorCode.INVALID_INPUT


def test_write_audit_log_database_outage_is_not_reported_as_logged(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(srepo, "insert_security_event", db_down)
    r = call(gw, SYSTEM, "write_audit_log", WRITE_ARGS)
    assert not r.success and not r.verified and r.error_code == ErrorCode.DATABASE_UNAVAILABLE


# ======================================================================== audit trail
def test_mcp_09_calls_are_audited_without_argument_values(env):
    _, gw, cap = env
    call(gw, SUPER, "get_audit_logs", {"server_filter": "secret_server_name_xyz"})
    event = [e for e in cap.events if e["tool"] == "get_audit_logs"][-1]
    assert event["server_id"] == P09 and event["permission_result"] == "ALLOWED"
    assert "server_filter" in event["argument_names"] and "secret_server_name_xyz" not in repr(cap.events)


def test_denied_attempts_on_mcp_09_are_audited(env):
    _, gw, cap = env
    call(gw, ADMIN, "get_audit_logs")
    event = cap.events[-1]
    assert event["permission_result"] == "DENIED" and event["role"] == "ADMIN" and event["success"] is False
