"""Who is calling? Bearer token -> Principal, or None.

Two kinds of token, both sent as `Authorization: Bearer <token>`:
  * a user's JWT from POST /auth/login. Signature, expiry, active flag and token_version are all re-checked on every
    request (it reuses the application's own verifier), so disabling a user or changing a password cuts MCP access at once;
  * OPSPILOT_MCP_SYSTEM_KEY, which becomes the 'system' principal. It is separate from OPSPILOT_API_KEY on purpose.

Deliberately NOT accepted: the session cookie (a browser page must not be able to drive MCP with ambient credentials,
which is the CSRF / DNS-rebinding problem the MCP spec warns about) and the ingest X-API-Key.
"""
import secrets

from app import auth as app_auth
from app.auth import Principal
from app.config import settings
from app.mcp_server.rbac import SYSTEM_VIA

MIN_KEY_LENGTH = 32


def principal_from_token(token: str) -> Principal | None:
    if not token:
        return None
    key = settings.mcp_system_key.get_secret_value()
    if len(key) >= MIN_KEY_LENGTH and secrets.compare_digest(token.encode(), key.encode()):
        return Principal("mcp-system", "service", None, SYSTEM_VIA)
    user = app_auth._principal_from_token(token)             # the application's own JWT check (module-private by name only)
    return Principal(user.username, user.role, user.user_id, "bearer") if user else None


def bearer_token(authorization_header: str) -> str:
    h = authorization_header or ""
    return h[7:].strip() if h.lower().startswith("bearer ") else ""
