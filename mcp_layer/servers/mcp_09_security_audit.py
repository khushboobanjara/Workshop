"""MCP 09 - Security / RBAC / Audit (SYSTEM server).

Who can call what:
  * SUPER_ADMIN only (read): get_audit_logs, get_audit_summary, get_security_events,
    check_permission, check_role.
  * SYSTEM only (write):     write_audit_log - internal code (chatbot, grounding gate, safety layer)
    calls it with Principal.system(); a browser session can never reach it.
Admins, doctors and users are refused at the server level by the gateway's RBAC.

check_permission does not re-implement RBAC: it asks the gateway's own authorizer, so the answer is
exactly what a real call would get. It evaluates a hypothetical caller with that role; ownership of a
specific record is not part of the answer.

NOT here: security_check(). It must wrap the Phase 10 safety layer, so it is built there.
"""
import re
import uuid
from typing import Any, Optional

from src.database import mcp_admin_repository as admin_repo
from src.database import mcp_security_repository as repo

from ..permissions import ROLE_PERMISSIONS, role_has_permission
from ..registry import MCPServerSpec
from ..schemas import ErrorCode, Ownership, Principal, Role, ToolKind, ToolMeta
from ._common import BadInput, clean_text, fail, ok, ok_rows, opt_text, positive_int, rules, whole_number

SUPER_ONLY = frozenset({Role.SUPER_ADMIN})
SYSTEM_ONLY = frozenset({Role.SYSTEM})
LOG_LIMIT = 100
MAX_HOURS = 720                      # 30 days
EVENT_TYPES = {
    "PROMPT_INJECTION_BLOCKED", "UNSAFE_REQUEST_BLOCKED", "SUSPICIOUS_ACTIVITY",
    "RATE_LIMIT_EXCEEDED", "GROUNDING_BLOCKED", "ROLE_ESCALATION_ATTEMPT",
}
_IDENT = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_REASON = re.compile(r"^[A-Za-z0-9_ .:-]{3,60}$")

_gateway: Optional[Any] = None


def bind_gateway(gateway: Any) -> None:
    """bootstrap.build_gateway() calls this so check_permission can use the real authorizer."""
    global _gateway
    _gateway = gateway


def _ident(value: Any, label: str) -> str:
    text = clean_text(value, label, 2, 64).lower()
    if not _IDENT.match(text):
        raise BadInput(f"{label} is invalid.")
    return text


def _parse_role(value: Any) -> Role:
    try:
        return Role(clean_text(value, "Role", 4, 15).upper())
    except ValueError:
        raise BadInput("Role must be USER, DOCTOR, ADMIN, SUPER_ADMIN or SYSTEM.") from None


def _hypothetical(role: Role) -> Principal:
    """A stand-in caller used only to evaluate RBAC rules; it is never used to run a tool."""
    if role is Role.SYSTEM:
        return Principal.system()
    return Principal(user_id=1, role=role, doctor_id=1 if role is Role.DOCTOR else None)


def register(spec: MCPServerSpec) -> None:
    # ------------------------------------------------------------- audit (read)
    @spec.tool(ToolMeta(
        name="get_audit_logs", allowed_roles=SUPER_ONLY, permission="audit.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Recent MCP audit rows (newest first, max 100). Argument names only, never values. "
                    "Filters: hours (1-720), only_problems, server_filter, tool_filter, target_user_id."))
    @rules
    def get_audit_logs(hours: int = 24, only_problems: bool = False, server_filter: Optional[str] = None,
                       tool_filter: Optional[str] = None, target_user_id: Optional[int] = None) -> dict:
        window = whole_number(hours, "Hours", 1, MAX_HOURS)
        if not isinstance(only_problems, bool):
            raise BadInput("only_problems must be true or false.")
        server = _ident(server_filter, "Server") if opt_text(server_filter, "Server", 2, 64) else None
        tool = _ident(tool_filter, "Tool") if opt_text(tool_filter, "Tool", 2, 64) else None
        who = positive_int(target_user_id, "User") if target_user_id is not None else None
        rows = repo.query_audit_logs(window, only_problems, server, tool, who, LOG_LIMIT)
        return ok_rows(rows, "No audit records match.")

    @spec.tool(ToolMeta(
        name="get_audit_summary", allowed_roles=SUPER_ONLY, permission="audit.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Totals, denied / blocked / failed counts, per-server activity and top denial "
                    "reasons for the last N hours (1-720)."))
    @rules
    def get_audit_summary(hours: int = 24) -> dict:
        window = whole_number(hours, "Hours", 1, MAX_HOURS)
        return ok({
            "window_hours": window,
            "totals": repo.audit_totals(window),
            "by_server": repo.audit_by_server(window),
            "top_denial_reasons": repo.top_denial_reasons(window),
        })

    @spec.tool(ToolMeta(
        name="get_security_events", allowed_roles=SUPER_ONLY, permission="security.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Denied and safety-blocked calls in the last N hours, plus users with repeated "
                    "denials (min_denials, default 5)."))
    @rules
    def get_security_events(hours: int = 24, min_denials: int = 5) -> dict:
        window = whole_number(hours, "Hours", 1, MAX_HOURS)
        threshold = whole_number(min_denials, "Minimum denials", 2, 1000)
        return ok({
            "window_hours": window,
            "events": repo.security_events(window, LOG_LIMIT),
            "repeat_denials": repo.repeat_denials(window, threshold),
        })

    # ------------------------------------------------------------------- RBAC
    @spec.tool(ToolMeta(
        name="check_permission", allowed_roles=SUPER_ONLY, permission="rbac.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Would a caller with role_name be allowed to run tool_name on server_id? "
                    "Uses the live RBAC engine. Does not run the tool."))
    @rules
    def check_permission(role_name: str, server_id: str, tool_name: str) -> dict:
        if _gateway is None:
            return fail(ErrorCode.MCP_UNAVAILABLE, "The security service is currently unavailable.", source=spec.server_id)
        role = _parse_role(role_name)
        server = _gateway.registry.get(_ident(server_id, "Server"))
        tool = server.tools.get(_ident(tool_name, "Tool")) if server else None
        if not server or not tool:
            return fail(ErrorCode.NOT_FOUND, "That server or tool does not exist.", source=spec.server_id)
        decision = _gateway.authorizer.check(_hypothetical(role), server, tool, {})
        return ok({
            "role": role.value, "server_id": server.server_id, "tool_name": tool.name,
            "allowed": decision.allowed, "reason": decision.reason,
            "required_permission": tool.permission, "kind": tool.kind.value,
        }, source="mcp_internal")

    @spec.tool(ToolMeta(
        name="check_role", allowed_roles=SUPER_ONLY, permission="rbac.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Everything a role holds: its permissions and the tools it can reach (code-defined "
                    "matrix; database revocations are shown by check_permission)."))
    @rules
    def check_role(role_name: str) -> dict:
        if _gateway is None:
            return fail(ErrorCode.MCP_UNAVAILABLE, "The security service is currently unavailable.", source=spec.server_id)
        role = _parse_role(role_name)
        reachable = [
            {"server_id": s.server_id, "tool_name": name, "kind": meta.kind.value}
            for s in _gateway.registry.servers() if role in s.allowed_roles
            for name, meta in s.tools.items()
            if role in meta.allowed_roles and role_has_permission(role, meta.permission)
        ]
        return ok({
            "role": role.value,
            "permissions": sorted(ROLE_PERMISSIONS.get(role, frozenset())),
            "tool_count": len(reachable),
            "tools": reachable,
        }, source="mcp_internal")

    # ----------------------------------------------------------- audit (write)
    @spec.tool(ToolMeta(
        name="write_audit_log", allowed_roles=SYSTEM_ONLY, permission="audit.write",
        ownership=Ownership.NONE, kind=ToolKind.WRITE,
        description="INTERNAL. Record a security event that did not come from a gateway call. "
                    "event_type must be a known type; reason is a short code, never user text."))
    @rules
    def write_audit_log(event_type: str, reason: str, subject_user_id: Optional[int] = None) -> dict:
        kind = clean_text(event_type, "Event type", 5, 40).upper()
        if kind not in EVENT_TYPES:
            raise BadInput("Unknown event type.")
        code = clean_text(reason, "Reason", 3, 60)
        if not _REASON.match(code):
            raise BadInput("Reason must be a short code (letters, digits, space, . : _ -).")
        who = None
        if subject_user_id is not None:
            who = positive_int(subject_user_id, "User")
            if not admin_repo.get_user(who):
                return fail(ErrorCode.NOT_FOUND, "That user could not be found.", source=spec.server_id)
        repo.insert_security_event(uuid.uuid4().hex, who, kind, code)
        return ok({"logged": True, "event_type": kind}, source="mysql")
