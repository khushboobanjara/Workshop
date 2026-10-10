"""Chatbot <-> MCP bridge (Phase 11).

The chatbot never talks to a repository for a migrated intent; it asks the MCP gateway, which runs
authentication-derived identity -> RBAC -> safety -> tool -> audit. This module is the only place the
chatbot touches that machinery.

Switch:  MCP_CHATBOT_ENABLED=true   (default OFF, so deploying this changes nothing until you flip it)
When ON the bridge FAILS CLOSED: if MCP cannot be built or the caller has no verified identity, a
migrated intent returns a controlled message. It never falls back to reading the database directly.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from mcp_layer.schemas import ErrorCode, MCPResult, Principal

log = logging.getLogger("chatbot.mcp")

ENV_FLAG = "MCP_CHATBOT_ENABLED"
RETRY_BUILD_AFTER_SECONDS = 30.0

_DEFAULT_UNAVAILABLE = ("I'm unable to provide verified information right now because the required "
                        "system data is unavailable.")
GENERIC_TEXT = "Sorry, I am unable to process your request right now."
BLOCKED_TEXT = ("I can't help with that request. If you have a health question, "
                "please ask it in your own words.")

def unavailable_text() -> str:
    """The controlled 'no verified data' sentence. Uses the grounding gate's own wording so the chat says
    the same thing whether a tool failed directly or the gate blocked the LLM."""
    try:
        from mcp_layer.grounding import UNAVAILABLE_MESSAGE
        return UNAVAILABLE_MESSAGE
    except Exception:
        return _DEFAULT_UNAVAILABLE


# ------------------------------------------------------------------ switch
def enabled() -> bool:
    return os.getenv(ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------- runtime
@dataclass
class Runtime:
    gateway: Any
    gate: Any        # mcp_layer.grounding.GroundingGate


_lock = threading.Lock()
_runtime: Optional[Runtime] = None
_failed_at: Optional[float] = None


def configure(gateway: Any, gate: Any = None) -> Runtime:
    """Install the gateway explicitly (app start-up in Phase 12, and tests)."""
    global _runtime, _failed_at
    if gate is None:
        from mcp_layer.grounding import GroundingGate
        gate = GroundingGate.from_gateway(gateway)
    with _lock:
        _runtime, _failed_at = Runtime(gateway, gate), None
        return _runtime


def reset() -> None:
    global _runtime, _failed_at
    with _lock:
        _runtime, _failed_at = None, None


def get_runtime() -> Optional[Runtime]:
    """The shared gateway, built on first use. A failed build is retried at most every 30 seconds."""
    global _failed_at
    if _runtime is not None:
        return _runtime
    with _lock:
        if _runtime is not None:
            return _runtime
        if _failed_at is not None and time.monotonic() - _failed_at < RETRY_BUILD_AFTER_SECONDS:
            return None
        try:
            from mcp_layer.bootstrap import build_gateway
            from mcp_layer.grounding import GroundingGate
            gateway = build_gateway()
            return configure_locked(Runtime(gateway, GroundingGate.from_gateway(gateway)))
        except Exception as exc:
            _failed_at = time.monotonic()
            log.error("MCP runtime could not be built", extra={"extra_data": {"exc_type": type(exc).__name__}})
            return None


def configure_locked(runtime: Runtime) -> Runtime:      # caller already holds _lock
    global _runtime, _failed_at
    _runtime, _failed_at = runtime, None
    return runtime


# -------------------------------------------------------------- identity
async def principal_for_request(request: Any) -> Optional[Principal]:
    """The verified caller for this HTTP request, or None. Cheap no-op while the switch is off.
    Re-reads the role from the database (mcp_layer.auth), so the session's cached role is never trusted."""
    if not enabled():
        return None
    from mcp_layer.auth import principal_from_request
    try:
        outcome = await asyncio.to_thread(principal_from_request, request)
    except Exception as exc:
        log.error("identity lookup failed", extra={"extra_data": {"exc_type": type(exc).__name__}})
        return None
    return outcome.principal


# ----------------------------------------------------------------- calls
def _fail(code: str, message: Optional[str] = None) -> MCPResult:
    return MCPResult.fail(code, source="chatbot", message=message)


async def call(principal: Optional[Principal], server_id: str, tool: str,
               arguments: Optional[Dict[str, Any]] = None) -> MCPResult:
    """One tool call through the gateway AS the signed-in user. Never raises."""
    if principal is None:
        return _fail(ErrorCode.UNAUTHENTICATED, "Please log in again.")
    runtime = get_runtime()
    if runtime is None:
        return _fail(ErrorCode.MCP_UNAVAILABLE)
    try:
        return await runtime.gateway.invoke(principal, server_id, tool, arguments or {})
    except Exception as exc:
        log.error("gateway call failed", extra={"extra_data": {"tool": tool, "exc_type": type(exc).__name__}})
        return _fail(ErrorCode.INTERNAL_ERROR)


def usable(result: Any) -> bool:
    """Only a successful AND verified result may be shown or given to the LLM."""
    return bool(getattr(result, "success", False) and getattr(result, "verified", False)
                and getattr(result, "data", None) is not None)


def user_message(result: Any) -> str:
    """Safe text for a failed result. Tool messages are written to be user-presentable; everything
    else gets a fixed sentence, so internal detail can never leak into the chat."""
    code = getattr(result, "error_code", None)
    if code in (ErrorCode.DATABASE_UNAVAILABLE, ErrorCode.MCP_UNAVAILABLE, ErrorCode.MCP_DISABLED,
                ErrorCode.TIMEOUT, ErrorCode.INVALID_TOOL_OUTPUT):
        return unavailable_text()
    if code == ErrorCode.UNAUTHENTICATED:
        return "Please log in again."
    if code == ErrorCode.ACCESS_DENIED:
        return "You do not have access to this action."
    if code == ErrorCode.SAFETY_BLOCKED:
        return "This request cannot be processed."
    if code in (ErrorCode.NOT_FOUND, ErrorCode.CONFLICT, ErrorCode.INVALID_INPUT):
        return getattr(result, "message", None) or GENERIC_TEXT
    return GENERIC_TEXT


# ---------------------------------------------------------- safety screen
def scan_text_fn():
    """The Phase 10 text scanner, or None if it cannot be loaded (logged, and tool-level safety still applies)."""
    try:
        from mcp_layer.safety import scan_text
        return scan_text
    except Exception as exc:
        log.error("safety scanner unavailable", extra={"extra_data": {"exc_type": type(exc).__name__}})
        return None


async def screen_message(message: str, principal: Optional[Principal]) -> Optional[str]:
    """Blocks prompt-injection / SQL / unsafe text BEFORE it can reach intent detection or the LLM.
    Returns the reply to show, or None if the message is fine. The message itself is never logged."""
    scan = scan_text_fn()
    if scan is None:
        return None
    try:
        reason = scan(message)
    except Exception:
        return None
    if not reason:
        return None
    log.warning("chat message blocked", extra={"extra_data": {
        "reason": reason, "user_id": getattr(principal, "user_id", None)}})
    await _audit_block(reason, principal)
    return BLOCKED_TEXT


async def _audit_block(reason: str, principal: Optional[Principal]) -> None:
    """Best-effort record through MCP 09 as SYSTEM. A failure here must never change the reply."""
    runtime = get_runtime()
    if runtime is None:
        return
    args: Dict[str, Any] = {
        "event_type": "PROMPT_INJECTION_BLOCKED" if reason == "prompt_injection" else "SUSPICIOUS_ACTIVITY",
        "reason": "chat message flagged as " + reason.replace("_", " ")}
    if principal is not None and principal.user_id:
        args["subject_user_id"] = principal.user_id
    try:
        await runtime.gateway.invoke(Principal.system(), "mcp_09_security_audit", "write_audit_log", args)
    except Exception as exc:
        log.warning("security event not recorded", extra={"extra_data": {"exc_type": type(exc).__name__}})
