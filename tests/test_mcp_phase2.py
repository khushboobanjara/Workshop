"""Phase 1-2: SUPER_ADMIN role, richer MCPResult, registry->DB sync, migration 002."""
from datetime import timezone
from pathlib import Path

import pytest

from mcp_layer.registry_sync import sync_registry_to_db
from mcp_layer.roles import resolve_role
from mcp_layer.schemas import MCPResult, Ownership, Role, ToolKind, ToolMeta
from mcp_layer.servers import build_registry

MIG = Path(__file__).resolve().parents[1] / "mcp_layer" / "migrations"


# ------------------------------------------------------------------ roles
def test_super_admin_resolves_from_db():
    assert resolve_role("SUPER_ADMIN") is Role.SUPER_ADMIN
    assert resolve_role(" super_admin ") is Role.SUPER_ADMIN


@pytest.mark.parametrize("bad", ["SYSTEM", "ROOT", "SUPERADMIN", "SUPER ADMIN", "", None, 5])
def test_unknown_or_internal_roles_get_no_access(bad):
    assert resolve_role(bad) is None


def test_doctor_never_inherits_super_admin():
    reg = build_registry()
    doctor_servers = {d["server_id"] for d in reg.discover(Role.DOCTOR)}
    assert doctor_servers == {"mcp_01_patient_profile", "mcp_06_doctor_admin"}


# -------------------------------------------------------------- visibility
def test_discovery_per_role():
    reg = build_registry()
    n = lambda r: len(reg.discover(r))
    assert n(Role.USER) == 5
    assert n(Role.ADMIN) == 8          # 5 user + 3 admin, no system
    assert n(Role.SUPER_ADMIN) == 10   # everything
    assert n(Role.SYSTEM) == 2


def test_admin_cannot_discover_system_servers():
    types = {d["server_type"] for d in build_registry().discover(Role.ADMIN)}
    assert "SYSTEM" not in types


# ------------------------------------------------------------ tool rules
def _meta(roles, kind=ToolKind.READ, perm="audit.read"):
    return ToolMeta(name="t", allowed_roles=frozenset(roles), permission=perm, kind=kind)


def test_system_read_tool_may_be_shared_with_super_admin():
    _meta({Role.SYSTEM, Role.SUPER_ADMIN})


@pytest.mark.parametrize("roles,kind", [
    ({Role.SYSTEM, Role.SUPER_ADMIN}, ToolKind.WRITE),     # internal writes stay internal
    ({Role.SYSTEM, Role.ADMIN}, ToolKind.READ),
    ({Role.SYSTEM, Role.USER}, ToolKind.READ),
    ({Role.SYSTEM, Role.DOCTOR}, ToolKind.READ),
    ({Role.SYSTEM, Role.SUPER_ADMIN, Role.ADMIN}, ToolKind.READ),
])
def test_system_tool_sharing_is_restricted(roles, kind):
    with pytest.raises(ValueError):
        ToolMeta(name="t", allowed_roles=frozenset(roles), permission="audit.read", kind=kind,
                 audit_required=True)


# ------------------------------------------------------------- MCPResult
def test_result_has_metadata_and_utc_timestamp():
    r = MCPResult.ok({"a": 1}, source="mysql")
    assert r.metadata == {} and r.timestamp.tzinfo is timezone.utc


def test_failure_result_also_has_timestamp_and_stays_unverified():
    r = MCPResult.fail("DATABASE_UNAVAILABLE", metadata={"retryable": True})
    assert r.verified is False and r.data is None and r.timestamp and r.metadata == {"retryable": True}


def test_result_survives_json_roundtrip():
    r = MCPResult.ok({"a": 1}, source="mysql", request_id="x", metadata={"row_count": 1})
    again = MCPResult.model_validate(r.model_dump(mode="json"))
    assert again == r


def test_metadata_default_is_not_shared_between_instances():
    a, b = MCPResult.ok(1, source="mysql"), MCPResult.ok(2, source="mysql")
    a.metadata["x"] = 1
    assert b.metadata == {}


# ------------------------------------------------------------- DB sync
class FakeCursor:
    def __init__(self, log, existing): self.log, self.existing, self._rows = log, existing, []
    def execute(self, sql, params=None):
        self.log.append((sql, params))
        if sql.startswith("SELECT"):
            self._rows = self.existing
    def fetchall(self): return self._rows


class FakeConn:
    def __init__(self, existing=()):
        self.log, self.existing = [], list(existing)
        self.committed = self.rolled_back = self.closed = False
    def cursor(self): return FakeCursor(self.log, self.existing)
    def commit(self): self.committed = True
    def rollback(self): self.rolled_back = True
    def close(self): self.closed = True


def test_sync_upserts_every_server_and_tool():
    conn = FakeConn()
    out = sync_registry_to_db(build_registry(), lambda: conn)
    assert out == {"servers": 10, "tools": 10, "orphaned_tools": []}   # 10 pings for now
    assert conn.committed and conn.closed and not conn.rolled_back


def test_sync_uses_bound_parameters_not_string_building():
    conn = FakeConn()
    sync_registry_to_db(build_registry(), lambda: conn)
    for sql, params in conn.log:
        assert "mcp_01" not in sql and "server_ping" not in sql   # values never inlined


def test_sync_never_overwrites_runtime_state():
    conn = FakeConn()
    sync_registry_to_db(build_registry(), lambda: conn)
    writes = " ".join(sql for sql, _ in conn.log if sql.startswith("INSERT"))
    for protected in ("status", "health_status", "is_enabled", "last_health_check"):
        assert f"{protected} = VALUES" not in writes


def test_sync_reports_orphans_without_deleting():
    conn = FakeConn(existing=[("mcp_01_patient_profile", "server_ping"), ("mcp_01_patient_profile", "old_tool")])
    out = sync_registry_to_db(build_registry(), lambda: conn)
    assert out["orphaned_tools"] == ["mcp_01_patient_profile.old_tool"]
    assert not any(sql.startswith(("DELETE", "UPDATE")) for sql, _ in conn.log)


def test_sync_rolls_back_on_failure():
    class Boom(FakeConn):
        def cursor(self):
            c = super().cursor()
            def bad(sql, params=None): raise RuntimeError("db down")
            c.execute = bad
            return c
    conn = Boom()
    with pytest.raises(RuntimeError):
        sync_registry_to_db(build_registry(), lambda: conn)
    assert conn.rolled_back and conn.closed and not conn.committed


# ------------------------------------------------------------- migration
def _executable(path):
    return "\n".join(l for l in path.read_text().splitlines() if not l.strip().startswith("--"))


def test_migration_002_parses_and_only_touches_mcp_tables():
    sqlglot = pytest.importorskip("sqlglot")
    from sqlglot import exp
    sql = _executable(MIG / "002_super_admin_role.sql")
    stmts = [s for s in sqlglot.parse(sql, read="mysql") if s]
    assert len(stmts) == 1 and isinstance(stmts[0], exp.Alter)
    assert stmts[0].this.name == "mcp_permissions"
    assert "SUPER_ADMIN" in sql
    assert "users" not in sql.lower()


def test_users_table_script_is_fully_manual():
    sql = _executable(MIG / "optional" / "002b_users_super_admin.sql")
    assert "ALTER" not in sql.upper() and "UPDATE" not in sql.upper()   # destructive lines stay commented


def test_down_migration_restores_original_roles():
    assert "SUPER_ADMIN" not in _executable(MIG / "002_super_admin_role_down.sql")
