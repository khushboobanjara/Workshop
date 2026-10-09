"""Phase 3: authentication -> role resolver -> RBAC (end to end through the gateway)."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from mcp_layer.auth import AuthOutcome, principal_from_request, resolve_principal
from mcp_layer.bootstrap import build_gateway
from mcp_layer.gateway import MCPGateway
from mcp_layer.overrides import DbPermissionOverrides
from mcp_layer.permissions import (ADMIN_PERMISSIONS, DOCTOR_PERMISSIONS, ROLE_PERMISSIONS,
                                   SUPER_ADMIN_PERMISSIONS, USER_PERMISSIONS, validate_registry_permissions)
from mcp_layer.rbac import RBACAuthorizer
from mcp_layer.schemas import ErrorCode, MCPResult, Ownership, Principal, Role, ToolKind, ToolMeta
from mcp_layer.servers import build_registry

run = asyncio.run
U, D, A, SA, S = Role.USER, Role.DOCTOR, Role.ADMIN, Role.SUPER_ADMIN, Role.SYSTEM


class Capture:
    def __init__(self): self.events = []
    def record(self, e): self.events.append(e)


def _ok():
    return MCPResult.ok({"fine": True}, source="test").model_dump(mode="json")


def make_gateway(overrides=None):
    reg = build_registry()
    s = lambda sid: reg.get(sid)

    def add(sid, name, roles, perm, own=Ownership.NONE, kind=ToolKind.READ):
        meta = ToolMeta(name=name, allowed_roles=frozenset(roles), permission=perm, ownership=own, kind=kind)
        @s(sid).tool(meta)
        def _t() -> dict: return _ok()
    add("mcp_01_patient_profile", "get_my_profile", {U, D, A, SA}, "profile.read.own", Ownership.CURRENT_USER_ONLY)
    add("mcp_02_doctor_appointment", "find_doctors", {U, A, SA}, "doctor.search")
    add("mcp_06_doctor_admin", "approve_doctor", {A, SA}, "doctor.approve", Ownership.ADMINISTRATIVE, ToolKind.WRITE)
    add("mcp_06_doctor_admin", "set_my_availability", {D}, "doctor.availability.update.own", Ownership.DOCTOR_OWN, ToolKind.WRITE)
    add("mcp_08_user_system_admin", "set_user_role", {SA}, "user.role.manage", Ownership.ADMINISTRATIVE, ToolKind.WRITE)
    add("mcp_09_security_audit", "write_audit", {S}, "audit.write", kind=ToolKind.WRITE)
    add("mcp_09_security_audit", "read_audit", {SA}, "audit.read")   # human read access; SYSTEM writes only
    cap = Capture()
    gw = MCPGateway(reg, authorizer=RBACAuthorizer(overrides), auditor=cap)
    return gw, cap


def call(gw, principal, server, tool, args=None):
    return run(gw.invoke(principal, server, tool, args or {}))


USER = Principal(user_id=7, role=U)
DOC = Principal(user_id=20, role=D, doctor_id=3)
DOC_UNLINKED = Principal(user_id=21, role=D, doctor_id=None)
ADMIN = Principal(user_id=1, role=A)
SUPER = Principal(user_id=2, role=SA)


# ============================================================ RBAC matrix
@pytest.mark.parametrize("who,server,tool,allowed", [
    (USER, "mcp_01_patient_profile", "get_my_profile", True),        # USER -> user MCP
    (USER, "mcp_02_doctor_appointment", "find_doctors", True),
    (USER, "mcp_06_doctor_admin", "approve_doctor", False),          # USER -> admin MCP
    (USER, "mcp_08_user_system_admin", "set_user_role", False),
    (USER, "mcp_09_security_audit", "read_audit", False),            # USER -> system MCP
    (USER, "mcp_09_security_audit", "write_audit", False),
    (DOC, "mcp_06_doctor_admin", "set_my_availability", True),       # DOCTOR -> permitted tool
    (DOC, "mcp_01_patient_profile", "get_my_profile", True),
    (DOC, "mcp_08_user_system_admin", "set_user_role", False),       # DOCTOR -> super-admin op
    (DOC, "mcp_06_doctor_admin", "approve_doctor", False),
    (DOC, "mcp_09_security_audit", "read_audit", False),
    (ADMIN, "mcp_06_doctor_admin", "approve_doctor", True),          # ADMIN -> admin MCP
    (ADMIN, "mcp_01_patient_profile", "get_my_profile", True),
    (ADMIN, "mcp_08_user_system_admin", "set_user_role", False),     # role changes: super admin only
    (ADMIN, "mcp_09_security_audit", "read_audit", False),           # ADMIN -> system MCP
    (SUPER, "mcp_06_doctor_admin", "approve_doctor", True),          # SUPER_ADMIN -> admin MCP
    (SUPER, "mcp_08_user_system_admin", "set_user_role", True),
    (SUPER, "mcp_09_security_audit", "read_audit", True),            # read-only system access
    (SUPER, "mcp_09_security_audit", "write_audit", False),          # internal writes stay internal
    (SUPER, "mcp_06_doctor_admin", "set_my_availability", False),    # not a doctor
])
def test_rbac_matrix(who, server, tool, allowed):
    gw, _ = make_gateway()
    r = call(gw, who, server, tool)
    assert r.success is allowed
    if not allowed:
        assert r.error_code == ErrorCode.ACCESS_DENIED and r.verified is False and r.data is None


def test_system_principal_can_use_internal_tools():
    gw, _ = make_gateway()
    assert call(gw, Principal.system(), "mcp_09_security_audit", "write_audit").success


def test_unauthenticated_is_rejected():
    gw, _ = make_gateway()
    r = call(gw, None, "mcp_01_patient_profile", "get_my_profile")
    assert r.error_code == ErrorCode.UNAUTHENTICATED


def test_denial_message_does_not_reveal_why():
    gw, _ = make_gateway()
    a = call(gw, USER, "mcp_06_doctor_admin", "approve_doctor")
    b = call(gw, DOC, "mcp_08_user_system_admin", "set_user_role")
    assert a.message == b.message == "You do not have access to this action."


# ================================================= role / ownership manipulation
@pytest.mark.parametrize("args", [
    {"role": "ADMIN"}, {"is_admin": True}, {"token": "x"}, {"sql": "DROP TABLE users"},
    {"password": "x"}, {"command": "rm"}, {" ROLE ": "SUPER_ADMIN"},
])
def test_role_escalation_arguments_denied_everywhere(args):
    gw, cap = make_gateway()
    for who in (USER, ADMIN, SUPER):
        r = call(gw, who, "mcp_01_patient_profile", "get_my_profile", args)
        assert r.error_code == ErrorCode.ACCESS_DENIED
    assert all(e["denial_reason"] == "forbidden_argument" for e in cap.events)


@pytest.mark.parametrize("key", ["user_id", "patient_id", "owner_id", "doctor_id"])
def test_cannot_name_another_users_record_on_own_data_tool(key):
    gw, cap = make_gateway()
    r = call(gw, USER, "mcp_01_patient_profile", "get_my_profile", {key: 999})
    assert r.error_code == ErrorCode.ACCESS_DENIED
    assert cap.events[-1]["denial_reason"] == "owner_argument_not_allowed"


def test_doctor_cannot_choose_which_doctor_to_act_as():
    gw, _ = make_gateway()
    assert call(gw, DOC, "mcp_06_doctor_admin", "set_my_availability", {"doctor_id": 99}).error_code == ErrorCode.ACCESS_DENIED


def test_doctor_without_linked_doctor_row_is_denied_own_tools():
    gw, cap = make_gateway()
    assert not call(gw, DOC_UNLINKED, "mcp_06_doctor_admin", "set_my_availability").success
    assert cap.events[-1]["denial_reason"] == "no_doctor_identity"


def test_unknown_server_and_tool():
    gw, _ = make_gateway()
    assert call(gw, SUPER, "mcp_99", "x").error_code == ErrorCode.SERVER_NOT_FOUND
    assert call(gw, SUPER, "mcp_01_patient_profile", "drop_everything").error_code == ErrorCode.TOOL_NOT_FOUND


def test_registry_rejects_doctor_own_tool_with_doctor_id_parameter():
    spec = build_registry().get("mcp_06_doctor_admin")
    meta = ToolMeta(name="bad", allowed_roles=frozenset({D}), permission="doctor.availability.update.own",
                    ownership=Ownership.DOCTOR_OWN, kind=ToolKind.WRITE)
    with pytest.raises(ValueError):
        @spec.tool(meta)
        def bad(doctor_id: int) -> dict: return {}


# ===================================================== permission matrix itself
def test_doctor_holds_no_admin_or_super_admin_permissions():
    admin_only = ADMIN_PERMISSIONS - USER_PERMISSIONS
    assert not (DOCTOR_PERMISSIONS & (admin_only | SUPER_ADMIN_PERMISSIONS - USER_PERMISSIONS))


def test_user_holds_no_administrative_permission():
    assert not any(p.startswith(("doctor.manage", "user.", "pharmacy.manage", "rbac", "audit"))
                   for p in USER_PERMISSIONS)


def test_super_admin_is_superset_of_admin_and_only_super_admin_manages_roles():
    assert ADMIN_PERMISSIONS < SUPER_ADMIN_PERMISSIONS
    assert "user.role.manage" in SUPER_ADMIN_PERMISSIONS and "user.role.manage" not in ADMIN_PERMISSIONS


def test_system_permissions_are_not_held_by_human_roles():
    internal = ROLE_PERMISSIONS[S] - {"mcp.ping", "system.monitor"}
    for role in (U, D, A, SA):
        assert not (ROLE_PERMISSIONS[role] & internal)


def test_unlisted_permission_is_held_by_nobody():
    gw, cap = make_gateway()
    spec = gw.registry.get("mcp_02_doctor_appointment")
    @spec.tool(ToolMeta(name="mystery", allowed_roles=frozenset({U}), permission="made.up.perm"))
    def mystery() -> dict: return _ok()
    assert not call(gw, USER, "mcp_02_doctor_appointment", "mystery").success
    assert cap.events[-1]["denial_reason"] == "permission_not_held"


def test_real_registry_and_test_registry_are_consistent():
    assert validate_registry_permissions(build_registry()) == []
    gw, _ = make_gateway()
    assert validate_registry_permissions(gw.registry) == []


def test_validator_flags_a_role_that_lacks_the_permission():
    reg = build_registry()
    @reg.get("mcp_09_security_audit").tool(
        ToolMeta(name="shared_read", allowed_roles=frozenset({S, SA}), permission="audit.read"))
    def shared_read() -> dict: return {}
    assert validate_registry_permissions(reg) == ["mcp_09_security_audit.shared_read: SYSTEM lacks 'audit.read'"]


def test_bootstrap_refuses_inconsistent_registry():
    reg = build_registry()
    @reg.get("mcp_02_doctor_appointment").tool(
        ToolMeta(name="oops", allowed_roles=frozenset({U}), permission="not.in.matrix"))
    def oops() -> dict: return {}
    with pytest.raises(RuntimeError):
        build_gateway(reg, use_db_overrides=False)


# ========================================================= gateway hardening
def test_authorizer_crash_denies_instead_of_allowing():
    class Broken:
        def check(self, *a): raise RuntimeError("boom")
    gw, cap = make_gateway()
    gw.authorizer = Broken()
    r = call(gw, SUPER, "mcp_01_patient_profile", "get_my_profile")
    assert r.error_code == ErrorCode.ACCESS_DENIED and cap.events[-1]["denial_reason"] == "authorizer_error"


def test_audit_has_reason_but_never_argument_values():
    gw, cap = make_gateway()
    call(gw, USER, "mcp_06_doctor_admin", "approve_doctor", {"note": "s3cret-value"})
    e = cap.events[0]
    assert e["denial_reason"] == "role_not_allowed_on_server" and e["permission_result"] == "DENIED"
    assert "s3cret-value" not in json.dumps(e)


# ====================================================== DB permission overrides
class OvConn:
    def __init__(self, rows=(), fail=False): self.rows, self.fail, self.closed = list(rows), fail, False
    def cursor(self): return self
    def execute(self, sql, p=None):
        if self.fail: raise ConnectionError("mysql down")
    def fetchall(self): return self.rows
    def close(self): self.closed = True


def test_db_row_can_revoke_access():
    ov = DbPermissionOverrides(lambda: OvConn([("USER", "mcp_02_doctor_appointment", "find_doctors")]))
    gw, cap = make_gateway(ov)
    assert not call(gw, USER, "mcp_02_doctor_appointment", "find_doctors").success
    assert cap.events[-1]["denial_reason"] == "revoked_by_policy"
    assert call(gw, ADMIN, "mcp_02_doctor_appointment", "find_doctors").success   # other roles unaffected


def test_db_row_can_never_widen_access():
    # a bogus grant row for USER on an admin tool: only granted=0 rows are ever loaded
    ov = DbPermissionOverrides(lambda: OvConn([]))
    gw, _ = make_gateway(ov)
    assert not call(gw, USER, "mcp_06_doctor_admin", "approve_doctor").success


def test_override_store_down_with_no_cache_denies():
    gw, cap = make_gateway(DbPermissionOverrides(lambda: OvConn(fail=True)))
    assert not call(gw, USER, "mcp_01_patient_profile", "get_my_profile").success
    assert cap.events[-1]["denial_reason"] == "policy_store_unavailable"


def test_override_store_down_keeps_last_known_policy():
    now = [0.0]
    conns = [OvConn([("USER", "mcp_02_doctor_appointment", "find_doctors")]), OvConn(fail=True)]
    ov = DbPermissionOverrides(lambda: conns.pop(0) if len(conns) > 1 else conns[0], ttl_seconds=10, clock=lambda: now[0])
    gw, _ = make_gateway(ov)
    assert not call(gw, USER, "mcp_02_doctor_appointment", "find_doctors").success   # loads, revoked
    now[0] = 50.0                                                                     # cache expired, DB now down
    assert not call(gw, USER, "mcp_02_doctor_appointment", "find_doctors").success   # still revoked (stale copy)
    assert call(gw, USER, "mcp_01_patient_profile", "get_my_profile").success         # others still work


# ============================================ authentication / role resolver
class Store:
    def __init__(self, users=None, doctors=None, boom=False):
        self.users, self.doctors, self.boom = users or {}, doctors or {}, boom
    def get_user(self, uid):
        if self.boom: raise ConnectionError("mysql://root:hunter2@db")
        return self.users.get(uid)
    def get_doctor_id_by_email(self, email): return self.doctors.get(email)


def u(role, status=None, email="a@x.com"): return {"user_id": 1, "email": email, "role": role, "doctor_status": status}


@pytest.mark.parametrize("raw", [None, "", "abc", 0, -5, "-1", True, False, 1.5e999, [], {}])
def test_bad_session_user_ids_are_unauthenticated(raw):
    out = resolve_principal(raw, Store({1: u("ADMIN")}))
    assert not out.ok and out.error_code == ErrorCode.UNAUTHENTICATED


def test_unknown_user_is_unauthenticated():
    assert resolve_principal(99, Store({})).error_code == ErrorCode.UNAUTHENTICATED


@pytest.mark.parametrize("db_role,status,expected", [
    ("PATIENT", None, U), ("ADMIN", None, A), ("SUPER_ADMIN", None, SA), ("DOCTOR", "APPROVED", D),
])
def test_role_comes_from_database(db_role, status, expected):
    assert resolve_principal(1, Store({1: u(db_role, status)})).principal.role is expected


def test_session_role_is_never_trusted():
    # cookie claims SUPER_ADMIN, database says PATIENT -> USER
    out = resolve_principal(1, Store({1: u("PATIENT")}), session_role="SUPER_ADMIN")
    assert out.principal.role is U


def test_demoted_account_loses_access_without_relogin():
    store = Store({1: u("ADMIN")})
    assert resolve_principal(1, store, "ADMIN").principal.role is A
    store.users[1]["role"] = "PATIENT"
    assert resolve_principal(1, store, "ADMIN").principal.role is U


@pytest.mark.parametrize("status", ["PENDING", "REJECTED", "SUSPENDED", "BLOCKED", None, ""])
def test_unapproved_doctor_gets_no_principal(status):
    out = resolve_principal(1, Store({1: u("DOCTOR", status)}))
    assert not out.ok and out.error_code == ErrorCode.ACCESS_DENIED


def test_unknown_db_role_gets_no_principal():
    for bad in ("SYSTEM", "ROOT", None, ""):
        assert resolve_principal(1, Store({1: u(bad)})).error_code == ErrorCode.ACCESS_DENIED


def test_doctor_gets_doctor_id_from_email():
    out = resolve_principal(1, Store({1: u("DOCTOR", "APPROVED", "dr@x.com")}, {"dr@x.com": 42}))
    assert out.principal.doctor_id == 42


def test_doctor_without_doctor_row_has_no_doctor_id():
    assert resolve_principal(1, Store({1: u("DOCTOR", "APPROVED")})).principal.doctor_id is None


def test_database_failure_gives_no_principal_and_no_leak():
    out = resolve_principal(1, Store(boom=True))
    assert not out.ok and out.error_code == ErrorCode.DATABASE_UNAVAILABLE
    assert "hunter2" not in repr(out)


def test_system_role_can_never_come_from_authentication():
    assert resolve_principal(1, Store({1: u("SYSTEM")})).principal is None


def test_principal_from_request_uses_session_user_id_and_caches():
    calls = []
    class Counting(Store):
        def get_user(self, uid): calls.append(uid); return super().get_user(uid)
    req = SimpleNamespace(session={"user_id": 1, "role": "SUPER_ADMIN"}, state=SimpleNamespace())
    store = Counting({1: u("PATIENT")})
    first = principal_from_request(req, store)
    second = principal_from_request(req, store)
    assert first.principal.role is U and second is first and calls == [1]


def test_principal_from_request_without_session():
    assert principal_from_request(SimpleNamespace(session={}, state=SimpleNamespace()), Store()).error_code == ErrorCode.UNAUTHENTICATED
    assert principal_from_request(SimpleNamespace(), Store()).error_code == ErrorCode.UNAUTHENTICATED


# ====================================================== full chain, real registry
def test_full_chain_with_production_gateway():
    gw = build_gateway(use_db_overrides=False, auditor=Capture())
    who = resolve_principal(1, Store({1: u("PATIENT")}), "ADMIN").principal
    assert call(gw, who, "mcp_01_patient_profile", "server_ping").success
    assert call(gw, who, "mcp_06_doctor_admin", "server_ping").error_code == ErrorCode.ACCESS_DENIED
    assert call(gw, who, "mcp_09_security_audit", "server_ping").error_code == ErrorCode.ACCESS_DENIED
    boss = resolve_principal(2, Store({2: u("SUPER_ADMIN")})).principal
    assert all(call(gw, boss, s.server_id, "server_ping").success for s in gw.registry.servers())
