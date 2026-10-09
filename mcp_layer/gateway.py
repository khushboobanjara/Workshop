"""MCP Gateway / orchestrator.

Every tool call from FastAPI goes through MCPGateway.invoke(). The gateway:
  1. needs an authenticated Principal built server-side,
  2. finds the server and tool in the registry,
  3. asks the Authorizer (default: DENY EVERYTHING until Phase 3),
  4. runs the tool through FastMCP's in-memory client with the principal set in
     a contextvar (tools never receive identity from the caller),
  5. validates the output into an MCPResult (anything else = unverified failure),
  6. emits an audit event (argument NAMES only, never values).
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from fastmcp import Client

from .config import MCPSettings, load_settings
from .registry import MCPRegistry, MCPServerSpec
from .schemas import ErrorCode, MCPResult, Principal, Role, ToolMeta
from .servers import PING_TOOL

log = logging.getLogger("mcp.gateway")
audit_log = logging.getLogger("mcp.audit")

_current_principal: contextvars.ContextVar[Optional[Principal]] = contextvars.ContextVar(
    "mcp_current_principal", default=None)


def current_principal() -> Principal:
    """For use inside tools. Raises if called outside a gateway invocation."""
    principal = _current_principal.get()
    if principal is None:
        raise PermissionError("no authenticated principal")
    return principal


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


class Authorizer(Protocol):
    def check(self, principal: Principal, server: MCPServerSpec, tool: ToolMeta,
              arguments: Dict[str, Any]) -> Decision: ...


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str = ""


class SafetyGate(Protocol):
    """Phase 10 plugs the real safety layer in here. Runs AFTER RBAC allows the call and BEFORE
    the tool executes. RBAC asks 'may this role do it?'; safety asks 'is it safe to run?'."""
    def check(self, principal: Principal, server: MCPServerSpec, tool: ToolMeta,
              arguments: Dict[str, Any]) -> SafetyDecision: ...


class NoSafety:
    def check(self, principal, server, tool, arguments) -> SafetyDecision:
        return SafetyDecision(True)


_NAME_CLEAN = re.compile(r"[^A-Za-z0-9_]")


def safe_argument_names(arguments: Any) -> list:
    """Names are caller-controlled, so bound and clean them before they reach logs or the DB."""
    if not isinstance(arguments, dict):
        return []
    return sorted({_NAME_CLEAN.sub("_", str(k))[:40] for k in arguments})[:20]


class DenyAllAuthorizer:
    """Safe default. Phase 3 replaces this with the real RBAC engine."""

    def check(self, principal, server, tool, arguments) -> Decision:
        return Decision(False, "no authorizer configured")


class Auditor(Protocol):
    def record(self, event: dict) -> None: ...


class LoggingAuditor:
    """Phase 2 placeholder. Phase 9 stores events in mcp_audit_logs."""

    def record(self, event: dict) -> None:
        audit_log.info("mcp_call", extra={"extra_data": event})


class MCPGateway:
    def __init__(self, registry: MCPRegistry, authorizer: Optional[Authorizer] = None,
                 auditor: Optional[Auditor] = None, settings: Optional[MCPSettings] = None,
                 safety: Optional[SafetyGate] = None):
        self.registry = registry
        self.safety = safety or NoSafety()
        self.authorizer = authorizer or DenyAllAuthorizer()
        self.auditor = auditor or LoggingAuditor()
        self.settings = settings or load_settings()

    # ---------------------------------------------------------------- public
    async def invoke(self, principal: Optional[Principal], server_id: str, tool_name: str,
                     arguments: Optional[Dict[str, Any]] = None) -> MCPResult:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        arguments = arguments or {}
        ctx: Dict[str, str] = {}
        result, permission = await self._run(principal, server_id, tool_name, arguments, request_id, ctx)
        result.request_id = request_id
        await asyncio.to_thread(self._audit, principal, server_id, tool_name, arguments, permission,
                                result, started, ctx)
        return result

    async def health_check(self) -> Dict[str, str]:
        """Pings every server with the internal SYSTEM identity (bypasses RBAC, ping only)."""
        report = {}
        for spec in self.registry.servers():
            if spec.status != "active":
                spec.health_status = "down"
            else:
                res = await self._execute(Principal.system(), spec, PING_TOOL, {})
                spec.health_status = "healthy" if res.success else "down"
            report[spec.server_id] = spec.health_status
        return report

    # --------------------------------------------------------------- private
    async def _run(self, principal, server_id, tool_name, arguments, request_id, ctx):
        if not self.settings.enabled:
            return MCPResult.fail(ErrorCode.MCP_DISABLED, message="MCP is turned off."), "NOT_CHECKED"
        if principal is None:
            return MCPResult.fail(ErrorCode.UNAUTHENTICATED, message="Please log in."), "DENIED"
        try:
            if len(json.dumps(arguments, default=str).encode()) > self.settings.max_argument_bytes:
                return MCPResult.fail(ErrorCode.INVALID_INPUT, message="Request is too large."), "NOT_CHECKED"
        except (TypeError, ValueError):
            return MCPResult.fail(ErrorCode.INVALID_INPUT, message="Invalid request."), "NOT_CHECKED"

        server = self.registry.get(server_id) if isinstance(server_id, str) else None
        if server is None:
            return MCPResult.fail(ErrorCode.SERVER_NOT_FOUND, message="Unknown service."), "NOT_CHECKED"
        tool = server.tools.get(tool_name) if isinstance(tool_name, str) else None
        if tool is None:
            return MCPResult.fail(ErrorCode.TOOL_NOT_FOUND, message="Unknown action."), "NOT_CHECKED"

        try:
            decision = self.authorizer.check(principal, server, tool, arguments)
        except Exception as exc:   # an authorizer bug must deny, never allow or crash the request
            log.error("authorizer failed", extra={"extra_data": {"exc_type": type(exc).__name__}})
            decision = Decision(False, "authorizer_error")
        if not decision.allowed:
            ctx["denial_reason"] = decision.reason[:60]   # audit only; the caller sees a generic message
            return MCPResult.fail(ErrorCode.ACCESS_DENIED, message="You do not have access to this action."), "DENIED"

        if server.status != "active":
            return MCPResult.fail(ErrorCode.MCP_UNAVAILABLE, message="This service is currently unavailable."), "ALLOWED"
        try:
            verdict = self.safety.check(principal, server, tool, arguments)
        except Exception as exc:   # a broken safety layer blocks; it never waves calls through
            log.error("safety layer failed", extra={"extra_data": {"exc_type": type(exc).__name__}})
            verdict = SafetyDecision(False, "safety_error")
        if not verdict.allowed:
            ctx["security_result"], ctx["denial_reason"] = "BLOCKED", verdict.reason[:60]
            return MCPResult.fail(ErrorCode.SAFETY_BLOCKED, message="This request cannot be processed."), "ALLOWED"
        ctx["security_result"] = "PASSED"
        return await self._execute(principal, server, tool_name, arguments), "ALLOWED"

    async def _execute(self, principal: Principal, server: MCPServerSpec, tool_name: str,
                       arguments: Dict[str, Any]) -> MCPResult:
        token = _current_principal.set(principal)
        try:
            async with Client(server.mcp) as client:
                raw = await asyncio.wait_for(
                    client.call_tool(tool_name, arguments, raise_on_error=False),
                    timeout=self.settings.tool_timeout_seconds)
            if raw.is_error:
                return MCPResult.fail(ErrorCode.INVALID_INPUT, source=server.server_id,
                                      message="The request could not be processed.")
            payload = raw.structured_content if raw.structured_content is not None else raw.data
            try:
                return MCPResult.model_validate(payload)
            except Exception:
                return MCPResult.fail(ErrorCode.INVALID_TOOL_OUTPUT, source=server.server_id,
                                      message="The service returned an unexpected response.")
        except asyncio.TimeoutError:
            return MCPResult.fail(ErrorCode.TIMEOUT, source=server.server_id, message="The service took too long.")
        except Exception as exc:  # sanitized: log type only, never the message or arguments
            log.error("tool execution failed", extra={"extra_data": {
                "server": server.server_id, "tool": tool_name, "exc_type": type(exc).__name__}})
            return MCPResult.fail(ErrorCode.INTERNAL_ERROR, source=server.server_id,
                                  message="Something went wrong. Please try again.")
        finally:
            _current_principal.reset(token)

    def _audit(self, principal, server_id, tool_name, arguments, permission, result, started, ctx=None):
        ctx = ctx or {}
        try:
            self.auditor.record({
                "request_id": result.request_id,
                "user_id": getattr(principal, "user_id", None),
                "role": principal.role.value if principal else None,
                "server_id": server_id if isinstance(server_id, str) else None,
                "tool": tool_name if isinstance(tool_name, str) else None,
                "argument_names": safe_argument_names(arguments),
                "permission_result": permission,
                "security_result": ctx.get("security_result", "NOT_CHECKED"),
                "denial_reason": ctx.get("denial_reason") or None,
                "success": result.success,
                "error_code": result.error_code,
                "source": result.source,
                "verified": result.verified,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            })
        except Exception:
            log.error("audit failure")   # auditing must never break or leak into a request
