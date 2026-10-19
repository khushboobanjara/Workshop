"""stdio transport, for desktop MCP clients that launch the server as a subprocess:  python -m app.mcp_server

One JSON-RPC message per line on stdin, one per line on stdout. stdout carries protocol messages ONLY (anything else
printed by imported code is redirected to stderr). The credential comes from the environment (OPSPILOT_MCP_STDIO_TOKEN:
a user's JWT or the MCP system key) and is re-verified before EVERY message, so revoking it takes effect immediately.
"""
import json
import sys
import uuid

from app.config import settings
from app.context import request_id_var
from app.mcp_server.auth import principal_from_token
from app.mcp_server.protocol import INVALID_REQUEST, PARSE_ERROR, rpc_error
from app.mcp_server.server import server as default_server

AUTH_FAILED = -32001


def _send(out, obj: dict) -> None:
    out.write(json.dumps(obj) + "\n")
    out.flush()


def serve(lines, out, resolve, srv=default_server) -> None:
    """`resolve()` returns the current Principal or None. Kept separate from main() so tests can drive it."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line.encode("utf-8")) > settings.mcp_max_message_bytes:
            _send(out, rpc_error(None, INVALID_REQUEST, "Message too large"))
            continue
        try:
            message = json.loads(line)
        except ValueError:
            _send(out, rpc_error(None, PARSE_ERROR, "Parse error"))
            continue
        if isinstance(message, list):
            _send(out, rpc_error(None, INVALID_REQUEST, "Batch requests are not supported"))
            continue
        principal = resolve()
        if principal is None:
            msg_id = message.get("id") if isinstance(message, dict) else None
            _send(out, rpc_error(msg_id, AUTH_FAILED, "Authentication failed"))
            continue
        token = request_id_var.set(uuid.uuid4().hex)           # so audit rows of one message share a request id
        try:
            response = srv.handle_message(message, principal, "stdio")
        finally:
            request_id_var.reset(token)
        if response is not None:
            _send(out, response)


def main() -> int:
    from app.mcp_server.routes import startup_checks
    out, real_stdout = sys.stdout, sys.stdout
    sys.stdout = sys.stderr                                    # stray prints must not corrupt the protocol stream
    try:
        if not settings.mcp_enabled:
            print("MCP is disabled. Set OPSPILOT_MCP_ENABLED=true.", file=sys.stderr)
            return 2
        try:
            startup_checks()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        token = settings.mcp_stdio_token.get_secret_value()
        if not token:
            print("Set OPSPILOT_MCP_STDIO_TOKEN (a user JWT or the MCP system key).", file=sys.stderr)
            return 2
        serve(sys.stdin, out, lambda: principal_from_token(token))
        return 0
    finally:
        sys.stdout = real_stdout
