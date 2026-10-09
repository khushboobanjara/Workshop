"""MCP 05 - AI Health Assistant.

The assistant never reads protected tables itself. get_verified_context() collects the caller's
own data by calling the other MCP tools THROUGH THE GATEWAY, with the same principal, so each
sub-call is authorised, safety-checked and audited like any other call.

All-or-nothing: if any requested part fails or is unverified, the whole result fails with that
part's error code (for example DATABASE_UNAVAILABLE). The LLM must never receive half a context
and fill the gap by guessing.

The gateway is injected by bootstrap.build_gateway() through bind_gateway().
"""
from typing import Any, Dict, List, Optional

from ..gateway import current_principal
from ..registry import MCPServerSpec
from ..schemas import ErrorCode, Ownership, Role, ToolKind, ToolMeta
from ._common import BadInput, fail, ok, rules

ROLES = frozenset({Role.USER, Role.ADMIN, Role.SUPER_ADMIN})
MAX_TOPICS = 4

# topic -> (server, tool). Only read-only, own-data tools belong here.
TOPICS = {
    "profile": ("mcp_01_patient_profile", "get_my_profile"),
    "appointments": ("mcp_02_doctor_appointment", "get_my_appointments"),
    "cart": ("mcp_03_pharmacy", "get_my_cart"),
    "orders": ("mcp_03_pharmacy", "get_my_orders"),
}

_gateway: Optional[Any] = None


def bind_gateway(gateway: Any) -> None:
    global _gateway
    _gateway = gateway


def register(spec: MCPServerSpec) -> None:
    @spec.tool(ToolMeta(
        name="get_verified_context", allowed_roles=ROLES, permission="assistant.use",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.READ, requires_verified_data=True,
        description="Verified facts about the signed-in user for the assistant. "
                    "topics: any of profile, appointments, cart, orders."))
    @rules
    async def get_verified_context(topics: List[str]) -> dict:
        principal = current_principal()
        if _gateway is None:
            return fail(ErrorCode.MCP_UNAVAILABLE, "The assistant is currently unavailable.", source=spec.server_id)
        if not isinstance(topics, list) or not topics or len(topics) > MAX_TOPICS:
            raise BadInput("Choose between 1 and 4 topics.")
        wanted: List[str] = []
        for topic in topics:
            key = str(topic).strip().lower()
            if key not in TOPICS:
                raise BadInput("Unknown topic. Use profile, appointments, cart or orders.")
            if key not in wanted:
                wanted.append(key)

        context: Dict[str, Any] = {}
        row_counts: Dict[str, int] = {}
        for key in wanted:
            server_id, tool_name = TOPICS[key]
            result = await _gateway.invoke(principal, server_id, tool_name, {})
            if not (result.success and result.verified):
                return fail(result.error_code or ErrorCode.INTERNAL_ERROR,
                            result.message or "The required information is unavailable.", source=spec.server_id)
            context[key] = result.data
            if "row_count" in result.metadata:
                row_counts[key] = result.metadata["row_count"]
        return ok(context, source="mysql", topics=wanted, row_counts=row_counts)
