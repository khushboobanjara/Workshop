"""Role-based access control and bearer-token authentication for MCP."""

from __future__ import annotations

import contextvars
import hmac
import json
import os
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send


ROLE_USER = "USER"
ROLE_ADMIN = "ADMIN"
ROLE_SYSTEM = "SYSTEM"
ROLE_SUPER_ADMIN = "SUPER_ADMIN"

# These are ten distinct tools, grouped as requested: five user, three admin,
# and two system tools. ADMIN inherits the read-only USER tool set. SUPER_ADMIN
# can invoke every tool. SYSTEM is a service identity and only gets system tools.
USER_TOOLS = {
    "user_search_doctors",
    "user_list_specializations",
    "user_get_doctor_availability",
    "user_search_medicines",
    "user_get_medicine_details",
}
ADMIN_TOOLS = {
    "admin_list_pending_doctors",
    "admin_approve_doctor",
    "admin_clinic_metrics",
}
SYSTEM_TOOLS = {
    "system_database_health",
    "system_mcp_status",
}
ALL_TOOLS = USER_TOOLS | ADMIN_TOOLS | SYSTEM_TOOLS

ROLE_ALLOWLIST = {
    ROLE_USER: USER_TOOLS,
    ROLE_ADMIN: USER_TOOLS | ADMIN_TOOLS,
    ROLE_SYSTEM: SYSTEM_TOOLS,
    ROLE_SUPER_ADMIN: ALL_TOOLS,
}

# Normal application roles -> MCP roles. Registering an ADMIN, SYSTEM, or
# SUPER_ADMIN account through the public registration form is not permitted.
SESSION_ROLE_MAP = {
    "PATIENT": ROLE_USER,
    "USER": ROLE_USER,
    "DOCTOR": ROLE_USER,
    "ADMIN": ROLE_ADMIN,
    "SYSTEM": ROLE_SYSTEM,
    "SUPER_ADMIN": ROLE_SUPER_ADMIN,
}

TOKEN_ENV_BY_ROLE = {
    ROLE_USER: "MCP_USER_TOKEN",
    ROLE_ADMIN: "MCP_ADMIN_TOKEN",
    ROLE_SYSTEM: "MCP_SYSTEM_TOKEN",
    ROLE_SUPER_ADMIN: "MCP_SUPER_ADMIN_TOKEN",
}

_current_mcp_role: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_mcp_role", default=None
)


def normalize_session_role(value: Any) -> str | None:
    """Map an authenticated application session role to an MCP role."""
    return SESSION_ROLE_MAP.get(str(value or "").strip().upper())


def configured_role_tokens() -> dict[str, str]:
    """Return configured role tokens; never expose these values to clients."""
    return {
        role: os.getenv(env_name, "").strip()
        for role, env_name in TOKEN_ENV_BY_ROLE.items()
        if os.getenv(env_name, "").strip()
    }


def role_for_bearer_token(token: str | None) -> str | None:
    """Resolve a bearer token to a role, failing closed on missing/duplicate tokens."""
    if not token:
        return None

    matches: list[str] = []
    # Empty secrets are never valid. If an operator accidentally configures the
    # same token for multiple roles, reject it rather than taking the first role.
    for role, env_name in TOKEN_ENV_BY_ROLE.items():
        expected = os.getenv(env_name, "").strip()
        if expected and hmac.compare_digest(token, expected):
            matches.append(role)
    return matches[0] if len(matches) == 1 else None


def can_access_tool(role: str | None, tool_name: str) -> bool:
    role = str(role or "").strip().upper()
    return tool_name in ROLE_ALLOWLIST.get(role, set())


def require_tool_access(role: str | None, tool_name: str) -> None:
    if tool_name not in ALL_TOOLS:
        raise ValueError("Unknown MCP tool.")
    if not can_access_tool(role, tool_name):
        raise PermissionError("Your role is not authorized to use this MCP tool.")


def current_mcp_role() -> str | None:
    return _current_mcp_role.get()


class MCPBearerAuthMiddleware:
    """Protect the mounted MCP HTTP transport with role-specific bearer tokens."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_auth = headers.get(b"authorization", b"").decode("latin-1").strip()
        scheme, _, token = raw_auth.partition(" ")
        role = role_for_bearer_token(token.strip() if scheme.lower() == "bearer" else None)

        if role is None:
            if scope.get("type") == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            payload = json.dumps({
                "error": "unauthorized",
                "message": "A valid role-specific MCP bearer token is required.",
            }).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                    (b"www-authenticate", b"Bearer"),
                    (b"cache-control", b"no-store"),
                ],
            })
            await send({"type": "http.response.body", "body": payload})
            return

        context_token = _current_mcp_role.set(role)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_mcp_role.reset(context_token)
