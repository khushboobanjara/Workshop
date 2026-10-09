"""Phase 8: MCP 10 (LLM / token / cost / monitoring), run end to end through the real gateway + RBAC.
The database is replaced by in-memory fakes. Needs fastmcp, pydantic and mysql-connector installed
(as the earlier phase tests do)."""
import asyncio
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import mysql.connector.errors as mye
import pytest

from mcp_layer.gateway import MCPGateway
from mcp_layer.llm_cost import calculate_cost, money_text
from mcp_layer.llm_tracking import track_llm_usage, usage_from_response
from mcp_layer.overrides import NoOverrides
from mcp_layer.permissions import validate_registry_permissions
from mcp_layer.rbac import RBACAuthorizer
from mcp_layer.schemas import ErrorCode, Principal, Role, ToolKind
from mcp_layer.servers import build_registry
from mcp_layer.servers import mcp_10_llm_monitoring as m10
from src.database import mcp_admin_repository as arepo
from src.database import mcp_llm_repository as lrepo
from src.database import mcp_security_repository as srepo

run = asyncio.run
P10 = "mcp_10_llm_monitoring"
MODEL = "openai/gpt-oss-120b"

USER = Principal(user_id=7, role=Role.USER)
DOCTOR = Principal(user_id=20, role=Role.DOCTOR, doctor_id=3)
ADMIN = Principal(user_id=1, role=Role.ADMIN)
SUPER = Principal(user_id=2, role=Role.SUPER_ADMIN)
SYSTEM = Principal.system()

USAGE_ARGS = {"model": MODEL, "input_tokens": 1000, "output_tokens": 500}
SUPER_READS = {"get_llm_usage_summary": {}, "get_llm_usage_logs": {}, "get_llm_pricing": {}}
SET_PRICE = {"model_name": "demo-model", "input_price_per_million": 0.075, "output_price_per_million": 0.3,
             "effective_from": "2026-10-01"}
ALL_TOOLS = {
    "calculate_cost": {"model": MODEL, "input_tokens": 10, "output_tokens": 10},
    "record_llm_usage": USAGE_ARGS,
    "get_llm_usage_summary": {}, "get_llm_usage_logs": {}, "get_llm_pricing": {},
    "set_llm_pricing": SET_PRICE, "get_mcp_health": {},
}


class Capture:
    def __init__(self): self.events = []
    def record(self, e): self.events.append(e)


def make_gateway():
    cap = Capture()
    gw = MCPGateway(build_registry(with_tools=True), authorizer=RBACAuthorizer(NoOverrides()), auditor=cap)
    m10.bind_gateway(gw)
    return gw, cap


def call(gw, who, tool, args=None):
    return run(gw.invoke(who, P10, tool, args or {}))


def db_down(*a, **k):
    raise mye.InterfaceError("2003: Can't connect to MySQL server on 'prod-db.internal:3306'", errno=2003)


class World:
    def __init__(self):
        self.calls = []
        self.usage = []
        self.prices_added = []
        self.prices = {MODEL: {"model_name": MODEL, "input_price_per_million": Decimal("0.15"),
                               "output_price_per_million": Decimal("0.60"), "currency": "USD",
                               "effective_from": date(2026, 1, 1)}}
        self.users = {7: {"user_id": 7, "role": "PATIENT"}, 1: {"user_id": 1, "role": "ADMIN"}}

    def install(self, mp):
        w = self
        mp.setattr(lrepo, "get_active_price", lambda m: w.prices.get(m))

        def insert_usage(*a):
            w.calls.append(("insert_usage", a))
            if any(u[0] == a[0] for u in w.usage):
                return False
            w.usage.append(a)
            return True
        mp.setattr(lrepo, "insert_usage", insert_usage)

        def insert_pricing(*a):
            w.calls.append(("insert_pricing", a))
            if any(p[0] == a[0] and p[4] == a[4] for p in w.prices_added):
                return False
            w.prices_added.append(a)
            return True
        mp.setattr(lrepo, "insert_pricing", insert_pricing)

        def rec(name, ret):
            def fn(*a, **k):
                w.calls.append((name, a))
                return ret
            return fn
        mp.setattr(lrepo, "list_pricing", rec("list_pricing", [{"model_name": MODEL, "input_price_per_million": 0.15}]))
        mp.setattr(lrepo, "usage_totals", rec("usage_totals", {"requests": 3, "total_tokens": 4500, "unpriced_requests": 1}))
        mp.setattr(lrepo, "cost_by_currency", rec("cost_by_currency", [{"currency": "USD", "total_cost": 0.002}]))
        mp.setattr(lrepo, "usage_by_model", rec("usage_by_model", [{"model": MODEL, "requests": 3}]))
        mp.setattr(lrepo, "usage_by_tool", rec("usage_by_tool", [{"tool_name": "get_verified_context", "requests": 2}]))
        mp.setattr(lrepo, "usage_top_users", rec("usage_top_users", [{"user_id": 7, "requests": 3}]))
        mp.setattr(lrepo, "query_usage", rec("query_usage", [{"usage_id": 1, "model": MODEL}]))
        mp.setattr(arepo, "get_user", lambda i: w.users.get(i))
        mp.setattr(srepo, "audit_by_server", lambda h: [{"server_id": "mcp_02_doctor_appointment", "calls": 5, "denied": 1,
                                                          "failed": 0, "last_activity": datetime(2026, 10, 9, 8, 0)}])


@pytest.fixture
def env(monkeypatch):
    world = World()
    world.install(monkeypatch)
    gw, cap = make_gateway()
    yield world, gw, cap
    m10.bind_gateway(None)


# ===================================================================== pure cost maths
def test_cost_formula_matches_the_specification():
    c = calculate_cost(1000, 500, Decimal("0.15"), Decimal("0.60"))
    assert (c.input_cost, c.output_cost, c.total_cost) == (Decimal("0.00015000"), Decimal("0.00030000"), Decimal("0.00045000"))
    assert calculate_cost(1_000_000, 1_000_000, 2.5, 10).total_cost == Decimal("12.50000000")


def test_cost_is_exact_and_total_is_the_sum_of_the_rounded_parts():
    c = calculate_cost(1000, 1000, Decimal("0.000005"), Decimal("0.000005"))     # 5e-9 each -> rounds HALF_UP
    assert c.input_cost == Decimal("0.00000001") and c.total_cost == c.input_cost + c.output_cost
    assert calculate_cost(0, 0, 1, 1).total_cost == Decimal("0E-8") == 0
    assert money_text(Decimal("0.00000001")) == "0.00000001" and money_text(None) is None


@pytest.mark.parametrize("args", [(-1, 0, 1, 1), (0, -1, 1, 1), (True, 0, 1, 1), (1.5, 0, 1, 1),
                                  (10_000_001, 0, 1, 1), (1, 1, -0.1, 1), (1, 1, 1, Decimal("NaN")), (1, 1, True, 1)])
def test_cost_rejects_nonsense(args):
    with pytest.raises(ValueError):
        calculate_cost(*args)


# ========================================================================== registry
def test_every_server_now_has_real_tools_and_the_matrix_is_consistent():
    reg = build_registry(with_tools=True)
    assert all(len(s.tools) > 1 for s in reg.servers())
    assert validate_registry_permissions(reg) == []


def test_mcp_10_tool_set_and_role_shape():
    tools = build_registry(with_tools=True).get(P10).tools
    assert set(tools) == {"server_ping", *ALL_TOOLS}
    for name, meta in tools.items():
        if name != "server_ping":
            assert not ({Role.USER, Role.DOCTOR, Role.ADMIN} & meta.allowed_roles), name
    assert tools["record_llm_usage"].allowed_roles == frozenset({Role.SYSTEM})
    assert tools["record_llm_usage"].kind is ToolKind.WRITE
    assert tools["calculate_cost"].allowed_roles == frozenset({Role.SYSTEM})
    assert tools["set_llm_pricing"].allowed_roles == frozenset({Role.SUPER_ADMIN})
    assert tools["get_mcp_health"].allowed_roles == frozenset({Role.SYSTEM, Role.SUPER_ADMIN})
    for name in SUPER_READS:
        assert tools[name].allowed_roles == frozenset({Role.SUPER_ADMIN})


# ============================================================================= RBAC
@pytest.mark.parametrize("who", [USER, DOCTOR, ADMIN])
@pytest.mark.parametrize("tool", sorted(ALL_TOOLS))
def test_user_doctor_and_admin_are_denied_every_mcp_10_tool(env, who, tool):
    world, gw, _ = env
    r = call(gw, who, tool, ALL_TOOLS[tool])
    assert r.error_code == ErrorCode.ACCESS_DENIED and r.data is None and not world.calls


@pytest.mark.parametrize("tool", ["record_llm_usage", "calculate_cost"])
def test_super_admin_cannot_use_system_only_tools(env, tool):
    world, gw, _ = env
    assert call(gw, SUPER, tool, ALL_TOOLS[tool]).error_code == ErrorCode.ACCESS_DENIED and not world.calls


@pytest.mark.parametrize("tool", ["get_llm_usage_summary", "get_llm_usage_logs", "get_llm_pricing", "set_llm_pricing"])
def test_system_identity_cannot_use_the_super_admin_tools(env, tool):
    world, gw, _ = env
    assert call(gw, SYSTEM, tool, ALL_TOOLS[tool]).error_code == ErrorCode.ACCESS_DENIED and not world.calls


@pytest.mark.parametrize("tool", [*SUPER_READS, "set_llm_pricing", "get_mcp_health"])
def test_super_admin_may_use_its_tools(env, tool):
    _, gw, _ = env
    r = call(gw, SUPER, tool, ALL_TOOLS[tool])
    assert r.success, r.message


@pytest.mark.parametrize("tool", ["record_llm_usage", "calculate_cost", "get_mcp_health"])
def test_system_identity_may_use_its_tools(env, tool):
    _, gw, _ = env
    r = call(gw, SYSTEM, tool, ALL_TOOLS[tool])
    assert r.success, r.message


def test_no_session_is_unauthenticated_and_role_smuggling_is_refused(env):
    world, gw, _ = env
    assert call(gw, None, "get_llm_pricing").error_code == ErrorCode.UNAUTHENTICATED
    assert call(gw, SUPER, "get_llm_pricing", {"role": "SYSTEM"}).error_code == ErrorCode.ACCESS_DENIED
    assert call(gw, SYSTEM, "record_llm_usage", {**USAGE_ARGS, "role": "SUPER_ADMIN"}).error_code == ErrorCode.ACCESS_DENIED
    assert not world.calls


# ============================================================ record_llm_usage / calculate_cost
def test_usage_is_stored_with_cost_from_the_pricing_table(env):
    world, gw, _ = env
    r = call(gw, SYSTEM, "record_llm_usage", {**USAGE_ARGS, "subject_user_id": 7, "mcp_server_id": "mcp_05_ai_assistant",
                                              "mcp_tool_name": "get_verified_context", "status": "success"})
    assert r.success and r.verified and r.data["priced"] is True
    assert r.data["total_tokens"] == 1500 and r.data["total_cost"] == "0.00045000" and r.data["currency"] == "USD"
    ref, user, role, server, tool, model, tin, tout, cin, cout, ctot, cur, status = world.usage[0]
    assert (user, role, server, tool, model, tin, tout, status) == (
        7, "USER", "mcp_05_ai_assistant", "get_verified_context", MODEL, 1000, 500, "SUCCESS")
    assert (cin, cout, ctot, cur) == (Decimal("0.00015000"), Decimal("0.00030000"), Decimal("0.00045000"), "USD")
    assert r.data["request_id"] == ref and len(ref) == 32


def test_unpriced_model_is_stored_with_null_cost_and_never_guessed(env):
    world, gw, _ = env
    r = call(gw, SYSTEM, "record_llm_usage", {**USAGE_ARGS, "model": "brand-new-model"})
    assert r.success and r.data["priced"] is False and r.data["total_cost"] is None
    assert world.usage[0][8:11] == (None, None, None)
    assert world.usage[0][6] == 1000 and world.usage[0][7] == 500     # tokens are still tracked


def test_the_same_request_is_never_counted_twice(env):
    world, gw, _ = env
    ref = "a" * 32
    assert call(gw, SYSTEM, "record_llm_usage", {**USAGE_ARGS, "request_ref": ref}).success
    r = call(gw, SYSTEM, "record_llm_usage", {**USAGE_ARGS, "request_ref": ref})
    assert not r.success and r.error_code == ErrorCode.CONFLICT and len(world.usage) == 1


@pytest.mark.parametrize("bad", [
    {"input_tokens": -1}, {"output_tokens": -5}, {"input_tokens": 10_000_001},
    {"status": "OK"}, {"status": ""}, {"model": "x'; DROP TABLE llm_usage;--"}, {"model": ""},
    {"request_ref": "ABCDEF" * 5 + "AB"}, {"request_ref": "z" * 32}, {"request_ref": "abc"},
    {"mcp_server_id": "bad server!"}, {"mcp_tool_name": "../x"}, {"subject_user_id": -1}, {"subject_user_id": 0},
])
def test_bad_usage_input_never_reaches_the_database(env, bad):
    world, gw, _ = env
    r = call(gw, SYSTEM, "record_llm_usage", {**USAGE_ARGS, **bad})
    assert not r.success and r.error_code == ErrorCode.INVALID_INPUT and not world.usage and not world.calls


def test_usage_for_an_unknown_user_is_refused(env):
    world, gw, _ = env
    r = call(gw, SYSTEM, "record_llm_usage", {**USAGE_ARGS, "subject_user_id": 999})
    assert r.error_code == ErrorCode.NOT_FOUND and not world.usage


def test_database_outage_is_never_reported_as_recorded(env, monkeypatch):
    world, gw, _ = env
    monkeypatch.setattr(lrepo, "get_active_price", db_down)
    r = call(gw, SYSTEM, "record_llm_usage", USAGE_ARGS)
    assert not r.success and not r.verified and r.error_code == ErrorCode.DATABASE_UNAVAILABLE and not world.usage
    monkeypatch.setattr(lrepo, "get_active_price", lambda m: world.prices.get(m))
    monkeypatch.setattr(lrepo, "insert_usage", db_down)
    r = call(gw, SYSTEM, "record_llm_usage", USAGE_ARGS)
    assert not r.success and r.error_code == ErrorCode.DATABASE_UNAVAILABLE
    assert "prod-db" not in repr(r.model_dump())


def test_calculate_cost_uses_the_database_price(env):
    world, gw, _ = env
    r = call(gw, SYSTEM, "calculate_cost", {"model": MODEL, "input_tokens": 2_000_000, "output_tokens": 1_000_000})
    assert r.success and r.data["total_cost"] == "0.90000000" and r.data["total_tokens"] == 3_000_000
    world.prices[MODEL]["input_price_per_million"] = Decimal("1.00")          # change in the table -> new answer
    assert call(gw, SYSTEM, "calculate_cost", {"model": MODEL, "input_tokens": 2_000_000, "output_tokens": 0}).data["total_cost"] == "2.00000000"


def test_calculate_cost_for_an_unpriced_model_says_so(env):
    _, gw, _ = env
    r = call(gw, SYSTEM, "calculate_cost", {"model": "who-knows", "input_tokens": 1, "output_tokens": 1})
    assert r.success and r.data["priced"] is False and r.data["total_cost"] is None and r.message


def test_calculate_cost_rejects_bad_input_and_outage(env, monkeypatch):
    _, gw, _ = env
    assert call(gw, SYSTEM, "calculate_cost", {"model": MODEL, "input_tokens": -1, "output_tokens": 1}).error_code == ErrorCode.INVALID_INPUT
    monkeypatch.setattr(lrepo, "get_active_price", db_down)
    r = call(gw, SYSTEM, "calculate_cost", ALL_TOOLS["calculate_cost"])
    assert not r.success and not r.verified and r.error_code == ErrorCode.DATABASE_UNAVAILABLE


# ================================================================================ pricing
def test_set_pricing_keeps_exact_prices_and_history(env):
    world, gw, _ = env
    r = call(gw, SUPER, "set_llm_pricing", SET_PRICE)
    assert r.success and r.data["input_price_per_million"] == "0.075" and r.data["effective_from"] == "2026-10-01"
    name, args = world.calls[-1]
    assert args == ("demo-model", Decimal("0.075"), Decimal("0.3"), "USD", date(2026, 10, 1))


def test_a_second_price_for_the_same_date_is_a_conflict(env):
    _, gw, _ = env
    assert call(gw, SUPER, "set_llm_pricing", SET_PRICE).success
    assert call(gw, SUPER, "set_llm_pricing", SET_PRICE).error_code == ErrorCode.CONFLICT


@pytest.mark.parametrize("bad", [
    {"input_price_per_million": -1}, {"output_price_per_million": -0.01}, {"input_price_per_million": 100001},
    {"input_price_per_million": 0.0000001}, {"currency": "usd1"}, {"currency": "US"}, {"currency": ""},
    {"effective_from": "2026-13-40"}, {"effective_from": "2019-12-31"}, {"effective_from": "2100-01-01"},
    {"effective_from": "tomorrow"}, {"model_name": "x'; DROP TABLE llm_pricing;--"},
])
def test_bad_pricing_input_is_rejected(env, bad):
    world, gw, _ = env
    r = call(gw, SUPER, "set_llm_pricing", {**SET_PRICE, **bad})
    assert not r.success and r.error_code == ErrorCode.INVALID_INPUT and not world.calls


def test_pricing_outage_is_not_reported_as_saved(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(lrepo, "insert_pricing", db_down)
    r = call(gw, SUPER, "set_llm_pricing", SET_PRICE)
    assert not r.success and r.error_code == ErrorCode.DATABASE_UNAVAILABLE


def test_empty_pricing_table_is_a_controlled_answer(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(lrepo, "list_pricing", lambda: [])
    r = call(gw, SUPER, "get_llm_pricing")
    assert r.success and r.data == [] and r.metadata["row_count"] == 0 and r.message


# ================================================================================ usage reads
def test_usage_summary_shape_and_bounds(env):
    _, gw, _ = env
    r = call(gw, SUPER, "get_llm_usage_summary", {"hours": 12})
    assert r.success and set(r.data) == {"window_hours", "totals", "cost_by_currency", "by_model", "by_tool", "top_users"}
    assert r.data["totals"]["unpriced_requests"] == 1 and r.data["window_hours"] == 12
    for bad in (0, 721):
        assert call(gw, SUPER, "get_llm_usage_summary", {"hours": bad}).error_code == ErrorCode.INVALID_INPUT


def test_usage_logs_pass_validated_filters_and_reject_bad_ones(env):
    world, gw, _ = env
    r = call(gw, SUPER, "get_llm_usage_logs", {"hours": 48, "model_filter": MODEL, "target_user_id": 7})
    assert r.success and r.metadata["row_count"] == 1 and world.calls[-1] == ("query_usage", (48, MODEL, 7, 100))
    world.calls.clear()
    for bad in ({"hours": 0}, {"model_filter": "a;b"}, {"target_user_id": -3}):
        assert call(gw, SUPER, "get_llm_usage_logs", bad).error_code == ErrorCode.INVALID_INPUT
    assert not world.calls


@pytest.mark.parametrize("tool", ["get_llm_usage_summary", "get_llm_usage_logs", "get_llm_pricing"])
def test_usage_reads_block_verified_data_during_an_outage(env, monkeypatch, tool):
    _, gw, _ = env
    for fn in ("usage_totals", "query_usage", "list_pricing"):
        monkeypatch.setattr(lrepo, fn, db_down)
    r = call(gw, SUPER, tool)
    assert not r.success and not r.verified and r.data is None and r.error_code == ErrorCode.DATABASE_UNAVAILABLE


# ================================================================================ monitoring
@pytest.mark.parametrize("who", [SUPER, SYSTEM])
def test_mcp_health_reports_all_ten_servers_with_activity(env, who):
    _, gw, _ = env
    r = call(gw, who, "get_mcp_health")
    assert r.success and len(r.data["servers"]) == 10 and r.data["activity_available"] is True
    by_id = {s["server_id"]: s for s in r.data["servers"]}
    assert all(s["health"] == "healthy" and s["tool_count"] > 1 for s in by_id.values())
    assert by_id["mcp_02_doctor_appointment"]["calls_24h"] == 5 and by_id["mcp_02_doctor_appointment"]["denied_24h"] == 1
    assert by_id["mcp_03_pharmacy"]["calls_24h"] == 0


def test_mcp_health_survives_an_audit_database_outage_but_says_so(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(srepo, "audit_by_server", db_down)
    r = call(gw, SUPER, "get_mcp_health")
    assert r.success and r.data["activity_available"] is False and r.message
    assert len(r.data["servers"]) == 10 and all("health" in s for s in r.data["servers"])


def test_mcp_health_fails_closed_when_no_gateway_is_bound(env):
    _, gw, _ = env
    m10.bind_gateway(None)
    assert call(gw, SUPER, "get_mcp_health").error_code == ErrorCode.MCP_UNAVAILABLE


# ============================================================================ tracking helper
def test_usage_is_read_from_a_provider_response_without_guessing():
    ok_resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30))
    assert usage_from_response(ok_resp) == (120, 30)
    for bad in (SimpleNamespace(), SimpleNamespace(usage=None), SimpleNamespace(usage=SimpleNamespace(prompt_tokens=None, completion_tokens=5)),
                SimpleNamespace(usage=SimpleNamespace(prompt_tokens=-1, completion_tokens=5)),
                SimpleNamespace(usage=SimpleNamespace(prompt_tokens=True, completion_tokens=5))):
        assert usage_from_response(bad) is None


def test_track_llm_usage_records_through_the_gateway_as_system(env):
    world, gw, cap = env
    r = run(track_llm_usage(gw, model=MODEL, input_tokens=300, output_tokens=100, user_id=7,
                            server_id="mcp_05_ai_assistant", tool_name="get_verified_context"))
    assert r.success and world.usage[0][1] == 7 and world.usage[0][6:8] == (300, 100)
    event = [e for e in cap.events if e["tool"] == "record_llm_usage"][-1]
    assert event["role"] == "SYSTEM" and event["server_id"] == P10 and event["permission_result"] == "ALLOWED"


def test_track_llm_usage_never_raises(env, monkeypatch):
    world, gw, _ = env
    monkeypatch.setattr(lrepo, "insert_usage", db_down)
    r = run(track_llm_usage(gw, model=MODEL, input_tokens=1, output_tokens=1))
    assert r is not None and not r.success and r.error_code == ErrorCode.DATABASE_UNAVAILABLE

    class Broken:
        async def invoke(self, *a, **k): raise RuntimeError("boom")
    assert run(track_llm_usage(Broken(), model=MODEL, input_tokens=1, output_tokens=1)) is None


# ============================================================================ audit trail
def test_mcp_10_calls_are_audited_without_argument_values(env):
    _, gw, cap = env
    call(gw, SYSTEM, "record_llm_usage", {**USAGE_ARGS, "model": "secret-model-xyz"})
    event = [e for e in cap.events if e["tool"] == "record_llm_usage"][-1]
    assert "model" in event["argument_names"] and "secret-model-xyz" not in repr(cap.events)


def test_denied_attempts_on_mcp_10_are_audited(env):
    _, gw, cap = env
    call(gw, ADMIN, "set_llm_pricing", SET_PRICE)
    event = cap.events[-1]
    assert event["permission_result"] == "DENIED" and event["role"] == "ADMIN" and event["success"] is False
