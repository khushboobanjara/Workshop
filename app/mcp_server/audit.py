"""Audit logging for MCP. Rows go into the existing append-only audit_log (so the database triggers that refuse
UPDATE/DELETE protect them too) with entity_type='mcp'. Filter them in the dashboard or GET /audit?stage=mcp.

What is recorded: who, which tool, validated arguments (cleaned), the outcome, how many records came back, how long it
took, and the request id. What is NOT recorded: tokens, keys, and the returned data itself (the records are in the
database already; copying them into a log would only create a second place for sensitive data to leak from).
"""
import re

from app.db import AuditLog, SessionLocal
from app.mcp_server.sanitize import clean_text, sanitize_json

OUTCOMES = ("ok", "denied", "invalid", "rate_limited", "unknown_tool", "error", "result_too_large")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,60}$")


def _tool_name(name) -> str:
    return name if isinstance(name, str) and _SAFE_NAME.match(name) else "invalid_tool_name"   # client text never goes in raw


def record(*, principal, mcp_role: str | None, transport: str, tool, outcome: str, args: dict | None = None,
           entity: str | None = None, result_count: int | None = None, duration_ms: float | None = None,
           extra: dict | None = None) -> None:
    """Writes one row and commits. RAISES if it cannot: the caller decides whether that is fatal (it is for a
    successful tool call: no audit row, no data)."""
    name = _tool_name(tool)
    details = {"tool": name, "outcome": outcome, "role": principal.role, "mcp_role": mcp_role,
               "via": principal.via, "transport": transport}
    if args is not None:
        details["arguments"] = sanitize_json(args)
    if entity:
        details["entity"] = entity
    if result_count is not None:
        details["result_count"] = result_count
    if extra:
        details.update(sanitize_json(extra))
    with SessionLocal() as s:
        s.add(AuditLog(entity_type="mcp", entity_id=None, stage="mcp", action=f"{name}:{outcome}"[:100],
                       actor=clean_text(principal.username, 100, redact=False) or "unknown",
                       details=details, duration_ms=duration_ms))
        s.commit()


def record_auth_failure(*, reason: str, ip: str, transport: str) -> None:
    with SessionLocal() as s:
        s.add(AuditLog(entity_type="mcp", entity_id=None, stage="mcp_auth", action="auth_failed", actor="unauthenticated",
                       details={"reason": reason, "ip": clean_text(ip, 64, redact=False), "transport": transport}))
        s.commit()
