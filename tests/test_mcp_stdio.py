"""MCP: the stdio transport (python -m app.mcp_server) shares the same protocol core, RBAC and audit logging."""
import io
import json

from pydantic import SecretStr

from app import auth
from app.config import settings
from app.db import AuditLog, SessionLocal, User
from app.mcp_server import stdio
from app.mcp_server.auth import principal_from_token
from tests.mcp_support import mcp_rows, world  # noqa: F401


def run(lines, resolver):
    out = io.StringIO()
    stdio.serve(iter(lines), out, resolver)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def msg(method, params=None, msg_id=1):
    body = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        body["params"] = params
    return json.dumps(body) + "\n"


def test_a_full_session_over_stdio_uses_the_same_rules_and_is_audited(world):
    token = world.tokens["view1"]
    lines = [msg("initialize", {"protocolVersion": "2025-06-18"}, 1),
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n", "\n",
             msg("tools/list", None, 2),
             msg("tools/call", {"name": "get_metrics", "arguments": {}}, 3),
             msg("tools/call", {"name": "list_users", "arguments": {}}, 4)]
    out = run(lines, lambda: principal_from_token(token))
    assert [o["id"] for o in out] == [1, 2, 3, 4]                         # the notification and the blank line made no output
    assert out[0]["result"]["serverInfo"]["name"] == "opspilot-ai"
    assert "list_users" not in {t["name"] for t in out[1]["result"]["tools"]}
    assert out[2]["result"]["isError"] is False
    assert json.loads(out[3]["result"]["content"][0]["text"])["error"]["code"] == "forbidden"
    rows = mcp_rows()
    assert [(r.action, r.details["transport"]) for r in rows] == [("get_metrics:ok", "stdio"), ("list_users:denied", "stdio")]
    assert all(r.request_id for r in rows)                                # every message gets its own request id


def test_bad_input_and_failed_authentication_get_protocol_errors(world):
    out = run(["{broken\n", "[1, 2]\n", json.dumps("x" * 70_000) + "\n"], lambda: principal_from_token(world.tokens["admin1"]))
    assert [o["error"]["code"] for o in out] == [-32700, -32600, -32600]
    out = run([msg("ping", None, 7)], lambda: None)
    assert out == [{"jsonrpc": "2.0", "id": 7, "error": {"code": stdio.AUTH_FAILED, "message": "Authentication failed"}}]
    assert mcp_rows() == []                                               # nothing ran, so there are no tool-call rows


def test_the_credential_is_rechecked_before_every_message(world):
    token = world.tokens["view1"]
    with SessionLocal() as s:
        uid = s.query(User).filter_by(username="view1").one().id
    lines = iter([msg("tools/call", {"name": "get_metrics", "arguments": {}}, 1),
                  msg("tools/call", {"name": "get_metrics", "arguments": {}}, 2)])
    out = io.StringIO()
    first = True
    for line in lines:
        stdio.serve([line], out, lambda: principal_from_token(token))
        if first:
            auth.update_user(uid, "admin1", is_active=False)               # revoked between the two messages
            first = False
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert "result" in replies[0] and replies[1]["error"]["code"] == stdio.AUTH_FAILED


def test_system_key_over_stdio_is_the_system_role(world):
    out = run([msg("tools/call", {"name": "get_my_access", "arguments": {}})], lambda: principal_from_token(world.tokens["system"]))
    payload = json.loads(out[0]["result"]["content"][0]["text"])
    assert payload["data"]["mcp_role"] == "system"


def test_main_refuses_to_run_when_disabled_unsafe_or_without_a_credential(monkeypatch, capsys):
    monkeypatch.setattr(settings, "mcp_enabled", False)
    assert stdio.main() == 2 and "disabled" in capsys.readouterr().err
    monkeypatch.setattr(settings, "mcp_enabled", True)
    monkeypatch.setattr(settings, "auth_enabled", False)
    assert stdio.main() == 2 and "AUTH_ENABLED" in capsys.readouterr().err
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "mcp_stdio_token", SecretStr(""))
    assert stdio.main() == 2 and "OPSPILOT_MCP_STDIO_TOKEN" in capsys.readouterr().err
