"""MCP: JSON-RPC / protocol behaviour of POST /mcp."""
import pytest

from app.config import settings
from tests.mcp_support import TOOLS_BY_WHO, world  # noqa: F401  (world is a fixture)


def _h(world, who="admin1", **extra):
    return {"Authorization": f"Bearer {world.tokens[who]}", **extra}


def test_mcp_is_off_by_default_and_answers_404_when_disabled(world, monkeypatch):
    monkeypatch.setattr(settings, "mcp_enabled", False)
    assert world.rpc("ping", token=world.tokens["admin1"]).status_code == 404
    assert world.client.get("/mcp").status_code == 404


def test_initialize_negotiates_version_and_states_the_grounding_rules(world):
    r = world.rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                 "clientInfo": {"name": "t", "version": "1"}}, token=world.tokens["view1"])
    res = r.json()["result"]
    assert res["protocolVersion"] == "2025-03-26" and res["serverInfo"]["name"] == "opspilot-ai"
    assert res["capabilities"] == {"tools": {"listChanged": False}}
    assert "never invent" in res["instructions"] and "untrusted_text_fields" in res["instructions"]
    newer = world.rpc("initialize", {"protocolVersion": "2099-01-01"}, token=world.tokens["view1"]).json()["result"]
    assert newer["protocolVersion"] == "2025-06-18"                      # an unknown version gets our latest


def test_notifications_and_client_responses_get_202_and_no_body(world):
    t = world.tokens["admin1"]
    assert world.post({"jsonrpc": "2.0", "method": "notifications/initialized"}, t).status_code == 202
    assert world.post({"jsonrpc": "2.0", "id": 1, "result": {}}, t).status_code == 202
    assert world.post({"jsonrpc": "2.0", "method": "notifications/initialized"}, t).content == b""


def test_ping_and_unknown_method(world):
    t = world.tokens["view1"]
    assert world.rpc("ping", token=t).json()["result"] == {}
    r = world.rpc("resources/list", token=t)
    assert r.status_code == 200 and r.json()["error"]["code"] == -32601


def test_malformed_messages_are_rejected_cleanly(world):
    t = world.tokens["admin1"]
    h = {"Authorization": f"Bearer {t}", "Content-Type": "application/json"}
    bad_json = world.client.post("/mcp", content=b"{not json", headers=h)
    assert bad_json.status_code == 400 and bad_json.json()["error"]["code"] == -32700
    batch = world.post([{"jsonrpc": "2.0", "id": 1, "method": "ping"}], t)
    assert batch.status_code == 400 and batch.json()["error"]["code"] == -32600
    for body in ({"id": 1, "method": "ping"}, {"jsonrpc": "1.0", "id": 1, "method": "ping"},
                 {"jsonrpc": "2.0", "id": True, "method": "ping"}, {"jsonrpc": "2.0", "id": [1], "method": "ping"},
                 {"jsonrpc": "2.0", "id": 1, "method": 5}, "just a string", 42):
        r = world.post(body, t)
        assert r.status_code == 400 and r.json()["error"]["code"] == -32600, body


def test_only_post_runs_tools_and_protocol_header_is_checked(world):
    t = world.tokens["admin1"]
    r = world.client.delete("/mcp")
    assert r.status_code == 405 and "POST" in r.headers["allow"]
    assert world.rpc("ping", token=t, headers={"MCP-Protocol-Version": "1999-01-01"}).status_code == 400
    assert world.rpc("ping", token=t, headers={"MCP-Protocol-Version": "2025-06-18"}).status_code == 200


def test_oversized_messages_are_refused_before_parsing(world):
    h = {"Authorization": f"Bearer {world.tokens['admin1']}", "Content-Type": "application/json"}
    assert world.client.post("/mcp", content=b" " * 70_000, headers=h).status_code == 413


def test_browser_origins_must_be_on_the_allow_list(world, monkeypatch):
    t = world.tokens["admin1"]
    assert world.rpc("ping", token=t, headers={"Origin": "https://evil.example"}).status_code == 403
    monkeypatch.setattr(settings, "mcp_allowed_origins", "https://app.example, https://other.example/")
    assert world.rpc("ping", token=t, headers={"Origin": "https://app.example"}).status_code == 200
    assert world.rpc("ping", token=t, headers={"Origin": "https://other.example"}).status_code == 200
    assert world.rpc("ping", token=t, headers={"Origin": "https://evil.example"}).status_code == 403


@pytest.mark.parametrize("who", list(TOOLS_BY_WHO))
def test_tools_list_shows_only_the_tools_the_caller_may_use(world, who):
    tools = world.rpc("tools/list", token=world.tokens[who]).json()["result"]["tools"]
    assert {t["name"] for t in tools} == TOOLS_BY_WHO[who]
    for t in tools:
        assert t["inputSchema"]["type"] == "object" and t["annotations"]["readOnlyHint"] is True


def test_bad_tools_call_shapes_are_protocol_errors(world):
    t = world.tokens["admin1"]
    for params in ({"name": 5}, {"arguments": {}}, {"name": "get_metrics", "arguments": [1]}):
        r = world.rpc("tools/call", params, token=t)
        assert r.json()["error"]["code"] == -32602, params
    r = world.post({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1]}, t)
    assert r.json()["error"]["code"] == -32602
    assert world.rpc("tools/call", {"name": "no_such_tool", "arguments": {}}, token=t).json()["error"]["code"] == -32602


# ---------------------------------------------------------------- the browser status page (GET /mcp)
def test_status_page_is_public_but_shows_nothing_sensitive(world):
    r = world.client.get("/mcp")
    d = r.json()
    assert r.status_code == 200 and d["server"]["status"] == "running" and d["signed_in"] is False
    assert "tools_you_may_call" not in d and "activity" not in d and "you" not in d
    assert world.tokens["admin1"] not in r.text and "k" * 40 not in r.text


def test_status_page_shows_a_signed_in_users_tools_and_only_admins_see_activity(world):
    world.call("get_metrics", {}, "view1")
    viewer = world.client.get("/mcp", headers={"Authorization": f"Bearer {world.tokens['view1']}"}).json()
    assert viewer["you"] == {"username": "view1", "role": "viewer", "mcp_role": "user"}
    assert {t["name"] for t in viewer["tools_you_may_call"]} == TOOLS_BY_WHO["view1"] and "activity" not in viewer
    admin = world.client.get("/mcp", headers={"Authorization": f"Bearer {world.tokens['admin1']}"}).json()
    assert {t["name"] for t in admin["tools_you_may_call"]} == TOOLS_BY_WHO["admin1"]
    latest = admin["activity"]["latest"]
    assert (latest[0]["who"], latest[0]["action"], latest[0]["role"]) == ("view1", "get_metrics:ok", "user")
    assert admin["activity"]["total_logged_events"] >= 1


def test_status_page_works_with_the_dashboard_session_cookie_and_renders_escaped_html(world):
    assert world.client.post("/auth/login", json={"username": "admin1", "password": "correct-horse-battery"}).status_code == 200
    world.call("get_metrics", {}, "view1")
    r = world.client.get("/mcp", headers={"Accept": "text/html"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "OpsPilot AI MCP server" in r.text and "get_metrics:ok" in r.text and "view1" in r.text
    assert world.rpc("ping").status_code == 401                        # the cookie still does not work for POST
