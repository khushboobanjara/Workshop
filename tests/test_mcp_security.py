<<<<<<< HEAD
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
=======
"""Small unit tests for MCP role policy; these tests do not require a live MySQL DB."""

import os
import unittest
from unittest.mock import patch

from mcp_layer.security import (
    ADMIN_TOOLS,
    ALL_TOOLS,
    ROLE_ADMIN,
    ROLE_SUPER_ADMIN,
    ROLE_SYSTEM,
    ROLE_USER,
    SYSTEM_TOOLS,
    USER_TOOLS,
    can_access_tool,
    role_for_bearer_token,
    normalize_session_role,
)
from mcp_layer.tools import TOOL_CATALOG, invoke_tool, list_tools_for_role
from mcp_layer.usage import _price, pricing_rates


class MCPRoleTests(unittest.TestCase):
    def test_expected_tool_groups_and_total(self):
        self.assertEqual(len(USER_TOOLS), 5)
        self.assertEqual(len(ADMIN_TOOLS), 3)
        self.assertEqual(len(SYSTEM_TOOLS), 2)
        self.assertEqual(len(ALL_TOOLS), 10)
        self.assertEqual(len(TOOL_CATALOG), 10)

    def test_user_cannot_call_admin_or_system_tools(self):
        self.assertTrue(can_access_tool(ROLE_USER, "user_search_doctors"))
        self.assertFalse(can_access_tool(ROLE_USER, "admin_approve_doctor"))
        self.assertFalse(can_access_tool(ROLE_USER, "system_database_health"))

    def test_admin_inherits_user_tools_but_not_system_tools(self):
        self.assertTrue(can_access_tool(ROLE_ADMIN, "user_search_medicines"))
        self.assertTrue(can_access_tool(ROLE_ADMIN, "admin_clinic_metrics"))
        self.assertFalse(can_access_tool(ROLE_ADMIN, "system_database_health"))

    def test_system_is_restricted_to_system_tools(self):
        self.assertTrue(can_access_tool(ROLE_SYSTEM, "system_mcp_status"))
        self.assertFalse(can_access_tool(ROLE_SYSTEM, "user_search_doctors"))
        self.assertFalse(can_access_tool(ROLE_SYSTEM, "admin_approve_doctor"))

    def test_super_admin_can_access_every_tool(self):
        for tool in ALL_TOOLS:
            self.assertTrue(can_access_tool(ROLE_SUPER_ADMIN, tool))

    def test_session_role_mapping(self):
        self.assertEqual(normalize_session_role("PATIENT"), ROLE_USER)
        self.assertEqual(normalize_session_role("DOCTOR"), ROLE_USER)
        self.assertEqual(normalize_session_role("ADMIN"), ROLE_ADMIN)
        self.assertEqual(normalize_session_role("SUPER_ADMIN"), ROLE_SUPER_ADMIN)
        self.assertIsNone(normalize_session_role("GUEST"))

    def test_token_lookup_fails_closed_on_duplicate_tokens(self):
        with patch.dict(os.environ, {
            "MCP_USER_TOKEN": "duplicate",
            "MCP_ADMIN_TOKEN": "duplicate",
            "MCP_SYSTEM_TOKEN": "system-secret",
            "MCP_SUPER_ADMIN_TOKEN": "super-secret",
        }, clear=False):
            self.assertIsNone(role_for_bearer_token("duplicate"))
            self.assertEqual(role_for_bearer_token("system-secret"), ROLE_SYSTEM)
            self.assertIsNone(role_for_bearer_token("unknown"))

    def test_tool_catalog_hides_restricted_tools(self):
        self.assertEqual(len(list_tools_for_role(ROLE_USER)), 5)
        self.assertEqual(len(list_tools_for_role(ROLE_ADMIN)), 8)
        self.assertEqual(len(list_tools_for_role(ROLE_SYSTEM)), 2)
        self.assertEqual(len(list_tools_for_role(ROLE_SUPER_ADMIN)), 10)

    def test_database_outage_blocks_data_tool_before_handler_runs(self):
        with patch("mcp_layer.tools.check_database", return_value={
            "available": False,
            "message": "Clinic database is unavailable.",
        }):
            result = invoke_tool("user_list_specializations", {}, role=ROLE_USER)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "database_unavailable")
        self.assertEqual(result["source"], "database_guard")

    def test_pricing_uses_selected_model_rates_and_cached_tokens(self):
        env = {
            "GROQ_CHAT_MODEL": "openai/gpt-oss-120b",
            "GROQ_VISION_MODEL": "qwen/qwen3.8-27b",
            "LLM_INPUT_PRICE_PER_1M_USD": "0.15",
            "LLM_CACHED_INPUT_PRICE_PER_1M_USD": "0.075",
            "LLM_OUTPUT_PRICE_PER_1M_USD": "0.60",
            "GROQ_VISION_INPUT_PRICE_PER_1M_USD": "",
            "GROQ_VISION_CACHED_INPUT_PRICE_PER_1M_USD": "",
            "GROQ_VISION_OUTPUT_PRICE_PER_1M_USD": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(pricing_rates()["model"], "openai/gpt-oss-120b")
            cost = _price({
                "model": "openai/gpt-oss-120b",
                "prompt_tokens": 100,
                "cached_input_tokens": 10,
                "completion_tokens": 20,
            })
        self.assertAlmostEqual(cost, 0.00002625, places=10)

    def test_unknown_or_unconfigured_model_is_not_silently_priced(self):
        env = {
            "GROQ_CHAT_MODEL": "openai/gpt-oss-120b",
            "GROQ_VISION_MODEL": "qwen/qwen3.8-27b",
            "GROQ_VISION_INPUT_PRICE_PER_1M_USD": "",
            "GROQ_VISION_CACHED_INPUT_PRICE_PER_1M_USD": "",
            "GROQ_VISION_OUTPUT_PRICE_PER_1M_USD": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertIsNone(_price({
                "model": "unconfigured/alternate-model",
                "prompt_tokens": 100,
                "completion_tokens": 20,
            }))


if __name__ == "__main__":
    unittest.main()
>>>>>>> 7054a180b5f6da5609ab045ecaa17bd840c09873
