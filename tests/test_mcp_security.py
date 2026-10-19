"""MCP: authentication, role enforcement on every call, input validation, rate limiting, audit logging, fail-closed
behaviour, and proof that the existing REST API is untouched."""
import dataclasses
import json

import pytest
from pydantic import SecretStr

from app import auth
from app.config import settings
from app.db import AuditLog, SessionLocal, User
from app.mcp_server import audit as audit_module, routes as mcp_routes
from app.mcp_server.server import server
from app.mcp_server.contracts import ToolResult
from tests.mcp_support import ALL_TOOLS, SYSTEM_KEY, TOOLS_BY_WHO, mcp_rows, world  # noqa: F401


def _auth_rows():
    with SessionLocal() as s:
        return s.query(AuditLog).filter_by(stage="mcp_auth").all()


# ---------------------------------------------------------------- authentication
def test_no_token_or_a_bad_token_is_401_and_audited_without_leaking_the_token(world):
    r = world.rpc("ping")
    assert r.status_code == 401 and "bearer" in r.headers["www-authenticate"].lower()
    assert world.rpc("ping", token="supersecrettoken123").status_code == 401
    rows = _auth_rows()
    assert [(x.action, x.details["reason"]) for x in rows] == [("auth_failed", "missing_token"), ("auth_failed", "invalid_token")]
    assert "supersecrettoken123" not in json.dumps([x.details for x in rows])


def test_a_session_cookie_alone_is_not_accepted(world):
    assert world.client.post("/auth/login", json={"username": "admin1", "password": "correct-horse-battery"}).status_code == 200
    assert world.client.get("/auth/me").status_code == 200                # the cookie really is valid for the REST API ...
    assert world.rpc("ping").status_code == 401                           # ... and still useless for MCP


def test_the_ingest_api_key_is_not_an_mcp_credential(world, monkeypatch):
    key = "ingest-key-" + "a" * 30
    monkeypatch.setattr(settings, "api_key", SecretStr(key))
    assert world.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"X-API-Key": key}).status_code == 401
    assert world.rpc("ping", token=key).status_code == 401


def test_the_system_key_gives_the_system_role_and_a_wrong_key_does_not(world):
    _, p = world.call("get_my_access", {}, "system")
    assert p["data"]["mcp_role"] == "system" and p["data"]["authenticated_via"] == "mcp_system_key"
    assert p["data"]["username"] == "mcp-system"
    assert world.rpc("ping", token=SYSTEM_KEY[:-1] + "!").status_code == 401


def test_system_role_is_unavailable_when_no_key_is_configured(world, monkeypatch):
    monkeypatch.setattr(settings, "mcp_system_key", SecretStr(""))
    assert world.rpc("ping", token="").status_code == 401
    assert world.rpc("ping", token=SYSTEM_KEY).status_code == 401


def test_disabling_a_user_cuts_off_their_token_on_the_next_call(world):
    assert world.rpc("ping", token=world.tokens["view1"]).status_code == 200
    with SessionLocal() as s:
        uid = s.query(User).filter_by(username="view1").one().id
    auth.update_user(uid, "admin1", is_active=False)
    assert world.rpc("ping", token=world.tokens["view1"]).status_code == 401


def test_repeated_failed_sign_ins_are_throttled(world):
    codes = [world.rpc("ping", token="nope").status_code for _ in range(21)]
    assert codes[:20] == [401] * 20 and codes[20] == 429
    assert len(_auth_rows()) == 20                                        # the throttled attempt is not logged: no log flooding


# ---------------------------------------------------------------- role enforcement on every call
def test_role_matrix_is_enforced_on_every_tool_for_every_role(world):
    wrong = []
    for who, allowed in TOOLS_BY_WHO.items():
        for tool in sorted(ALL_TOOLS):
            args = {"invoice_id": world.ids["i1"]} if tool in ("get_invoice", "get_invoice_audit_trail") else {}
            result, payload = world.call(tool, args, who)
            if tool in allowed:
                ok = result["isError"] is False and payload["source"] == "opspilot_database" and payload["tool"] == tool
            else:
                ok = result["isError"] is True and payload["error"]["code"] == "forbidden"
            if not ok:
                wrong.append((who, tool, payload))
    assert not wrong, wrong


def test_a_denied_call_never_reaches_the_tool_handler(world, monkeypatch):
    calls = []
    real = server.tools["list_users"]
    monkeypatch.setitem(server.tools, "list_users",
                        dataclasses.replace(real, handler=lambda ctx, a: (calls.append(ctx.mcp_role), ToolResult({"x": 1}))[1]))
    _, denied = world.call("list_users", {}, "view1")
    assert denied["error"]["code"] == "forbidden" and calls == []
    _, allowed = world.call("list_users", {}, "admin1")
    assert allowed["data"] == {"x": 1} and calls == ["super_admin"]


def test_listing_filters_but_calling_is_checked_independently(world):
    """A client that skips tools/list and guesses a tool name is refused exactly the same."""
    result, payload = world.call("get_settings", {}, "rev1")
    assert result["isError"] and payload["error"]["code"] == "forbidden"


# ---------------------------------------------------------------- input validation
BAD_ARGUMENTS = [
    ("get_invoice", {}), ("get_invoice", {"invoice_id": 0}), ("get_invoice", {"invoice_id": -5}),
    ("get_invoice", {"invoice_id": "1"}), ("get_invoice", {"invoice_id": 1.5}), ("get_invoice", {"invoice_id": True}),
    ("get_invoice", {"invoice_id": 99_999_999_999}), ("get_invoice", {"invoice_id": 1, "extra": "x"}),
    ("get_invoice", {"invoice_id": None}), ("get_metrics", {"unexpected": 1}),
    ("search_invoices", {"status": "bogus"}), ("search_invoices", {"decision": "maybe"}),
    ("search_invoices", {"limit": 0}), ("search_invoices", {"limit": 101}), ("search_invoices", {"limit": "5"}),
    ("search_invoices", {"vendor": "x" * 101}), ("search_invoices", {"vendor": 5}),
    ("get_audit_log", {"stage": "a b"}), ("get_audit_log", {"actor": "x;DROP"}), ("get_audit_log", {"limit": 201}),
    ("list_users", {"limit": 0}), ("get_review_queue", {"limit": 1000}),
]


def test_invalid_arguments_are_rejected_as_tool_errors_and_never_executed(world):
    wrong = []
    for tool, args in BAD_ARGUMENTS:
        result, payload = world.call(tool, args, "admin1")
        if not (result["isError"] and payload["error"]["code"] == "invalid_arguments" and payload["error"]["details"]):
            wrong.append((tool, args, payload))
    assert not wrong, wrong
    assert [r.action for r in mcp_rows() if not r.action.endswith(":invalid")] == []     # nothing ran, everything logged


def test_rejected_values_are_not_echoed_back_or_logged(world):
    result, _ = world.call("get_invoice", {"invoice_id": "DROP TABLE invoices"}, "view1")
    assert "DROP TABLE" not in json.dumps(result)
    assert "DROP TABLE" not in json.dumps([r.details for r in mcp_rows()])


def test_hostile_search_text_is_harmless_and_wildcards_are_literal(world):
    _, p = world.call("search_invoices", {"vendor": "'; DROP TABLE invoices; --"}, "view1")
    assert p["found"] is False and p["data"]["returned"] == 0
    _, p = world.call("search_invoices", {"vendor": "%"}, "view1")        # only the vendor that really contains a '%'
    assert [i["id"] for i in p["data"]["invoices"]] == [world.ids["i3"]]
    _, p = world.call("search_invoices", {"vendor": "_"}, "view1")
    assert [i["id"] for i in p["data"]["invoices"]] == [world.ids["i3"]]
    _, p = world.call("search_invoices", {}, "view1")                      # the table is still there
    assert p["data"]["total_matching"] == 3


# ---------------------------------------------------------------- limits and failures
def test_rate_limit_is_per_caller_and_audited(world, monkeypatch):
    monkeypatch.setattr(settings, "mcp_rate_limit_per_minute", 3)
    results = [world.call("get_metrics", {}, "view1")[1] for _ in range(4)]
    assert [bool(p.get("ok", True)) and p.get("source") == "opspilot_database" for p in results] == [True, True, True, False]
    assert results[3]["error"]["code"] == "rate_limited"
    assert world.call("get_metrics", {}, "rev1")[1]["source"] == "opspilot_database"       # someone else is unaffected
    assert "get_metrics:rate_limited" in [r.action for r in mcp_rows()]


def test_unexpected_failures_do_not_leak_details_and_are_logged_with_a_reference(world, monkeypatch):
    def boom(ctx, a):
        raise RuntimeError("connection string mysql://root:hunter2@db/prod")
    monkeypatch.setitem(server.tools, "get_metrics", dataclasses.replace(server.tools["get_metrics"], handler=boom))
    result, payload = world.call("get_metrics", {}, "view1")
    assert payload["error"]["code"] == "internal_error" and "hunter2" not in json.dumps(result)
    row = [r for r in mcp_rows() if r.action == "get_metrics:error"][0]
    assert row.details["error_ref"] in payload["error"]["message"] and "hunter2" not in json.dumps(row.details)


def test_oversized_results_are_refused_not_truncated_silently(world, monkeypatch):
    monkeypatch.setattr("app.mcp_server.protocol.MAX_RESULT_CHARS", 50)
    _, payload = world.call("get_metrics", {}, "view1")
    assert payload["error"]["code"] == "result_too_large"


def test_if_the_audit_row_cannot_be_written_no_data_is_returned(world, monkeypatch):
    def broken(**kw):
        raise RuntimeError("audit database down")
    monkeypatch.setattr(audit_module, "record", broken)
    result, payload = world.call("get_invoice", {"invoice_id": world.ids["i1"]}, "view1")
    assert result["isError"] and payload["error"]["code"] == "audit_unavailable"
    assert "Acme" not in json.dumps(result) and "structuredContent" in result and "source" not in result["structuredContent"]


# ---------------------------------------------------------------- audit trail
def test_each_call_leaves_one_complete_append_only_audit_row(world):
    world.call("get_invoice", {"invoice_id": world.ids["i1"]}, "view1", headers={"X-Request-ID": "mcp-trace-12345"})
    (row,) = mcp_rows()
    assert (row.stage, row.action, row.actor, row.request_id) == ("mcp", "get_invoice:ok", "view1", "mcp-trace-12345")
    d = row.details
    assert d["tool"] == "get_invoice" and d["outcome"] == "ok" and d["role"] == "viewer" and d["mcp_role"] == "user"
    assert d["via"] == "bearer" and d["transport"] == "http" and d["entity"] == f"invoice:{world.ids['i1']}"
    assert d["arguments"] == {"invoice_id": world.ids["i1"]} and d["result_count"] == 1 and row.duration_ms is not None


def test_the_audit_row_records_the_request_but_never_the_returned_data_or_the_token(world):
    world.call("get_invoice", {"invoice_id": world.ids["i1"]}, "view1")
    blob = json.dumps([r.details for r in mcp_rows()])
    assert "Acme Corp" not in blob and "INV-1001" not in blob and world.tokens["view1"] not in blob


def test_denials_unknown_tools_and_system_calls_are_logged_too(world):
    world.call("list_users", {}, "view1")
    world.rpc("tools/call", {"name": "bad name!", "arguments": {}}, token=world.tokens["view1"])
    world.call("get_metrics", {}, "system")
    actions = [(r.actor, r.action) for r in mcp_rows()]
    assert actions == [("view1", "list_users:denied"), ("view1", "invalid_tool_name:unknown_tool"), ("mcp-system", "get_metrics:ok")]
    assert mcp_rows()[2].details["mcp_role"] == "system"


def test_admins_can_review_mcp_activity_through_the_existing_audit_endpoint(world):
    world.call("get_metrics", {}, "view1")
    admin = {"Authorization": f"Bearer {world.tokens['admin1']}"}
    rows = world.client.get("/audit", params={"stage": "mcp"}, headers=admin).json()
    assert [(r["actor"], r["action"]) for r in rows] == [("view1", "get_metrics:ok")]
    _, p = world.call("get_audit_log", {"stage": "mcp", "actor": "view1"}, "admin1")
    assert [e["action"] for e in p["data"]["entries"]] == ["get_metrics:ok"]


# ---------------------------------------------------------------- the existing API is untouched
def test_existing_rest_api_behaves_as_before_and_agrees_with_mcp(world):
    c = world.client
    admin = {"Authorization": f"Bearer {world.tokens['admin1']}"}
    viewer = {"Authorization": f"Bearer {world.tokens['view1']}"}
    reviewer = {"Authorization": f"Bearer {world.tokens['rev1']}"}
    assert c.get("/health").json()["auth_enabled"] is True and c.get("/invoices").status_code == 401
    assert c.get("/audit", headers=viewer).status_code == 403 and c.get("/audit", headers=admin).status_code == 200

    rest = c.get("/invoices", headers=admin).json()
    _, p = world.call("search_invoices", {"limit": 100}, "view1")
    assert {i["id"] for i in p["data"]["invoices"]} == {i["id"] for i in rest}

    body = ("Vendor: Brand New Ltd\nInvoice INV-90001\nInvoice Date: 2026-10-01\nDue Date: 2026-10-31\nTotal Amount: USD 5200.00")
    r = c.post("/emails", json={"message_id": "rest-1", "subject": "Invoice INV-90001", "body": body}, headers=reviewer)
    assert r.status_code == 200 and r.json()["invoice"] is not None
    new_id = r.json()["invoice"]["id"]
    rest_inv = c.get(f"/invoices/{new_id}", headers=viewer).json()
    _, p = world.call("get_invoice", {"invoice_id": new_id}, "view1")
    mcp_inv = p["data"]["invoice"]
    for f in ("vendor_name", "invoice_number", "total_amount", "currency", "decision", "status", "risk_score", "decision_rule"):
        assert mcp_inv[f] == rest_inv[f], f
