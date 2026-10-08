"""Declares the 10 MCP servers. Tools are added phase by phase in their own modules."""
from __future__ import annotations

from fastmcp import FastMCP

from ..registry import MCPRegistry, MCPServerSpec
from ..schemas import MCPResult, Ownership, Role, ServerType, ToolKind, ToolMeta

U, D, A, S = Role.USER, Role.DOCTOR, Role.ADMIN, Role.SYSTEM

# (server_id, name, type, description, allowed_roles)
SERVER_DEFINITIONS = [
    ("mcp_01_patient_profile", "Patient / Profile MCP", ServerType.USER,
     "Own profile and preferences", {U, D, A}),
    ("mcp_02_doctor_appointment", "Doctor / Appointment MCP", ServerType.USER,
     "Doctor search, availability and the patient's own appointments", {U, A}),
    ("mcp_03_pharmacy", "Pharmacy MCP", ServerType.USER,
     "Medicines, cart, own orders, nearby pharmacies", {U, A}),
    ("mcp_04_health_screening", "Health Screening MCP", ServerType.USER,
     "Validated inputs to the existing prediction pipeline", {U, A}),
    ("mcp_05_ai_assistant", "AI Health Assistant MCP", ServerType.USER,
     "Healthcare assistant routed through verified tools", {U, A}),
    ("mcp_06_doctor_admin", "Doctor / Appointment Administration MCP", ServerType.ADMIN,
     "Doctors, availability and appointment administration", {A, D}),
    ("mcp_07_pharmacy_admin", "Pharmacy / Inventory Administration MCP", ServerType.ADMIN,
     "Medicines, stock and order administration", {A}),
    ("mcp_08_user_system_admin", "User / System Administration MCP", ServerType.ADMIN,
     "Users, roles and administrative reporting", {A}),
    ("mcp_09_security_audit", "Security / RBAC / Audit MCP", ServerType.SYSTEM,
     "RBAC evaluation, audit and security events", {S}),
    ("mcp_10_llm_monitoring", "LLM / Token / Cost / Monitoring MCP", ServerType.SYSTEM,
     "Token usage, cost and MCP health monitoring", {S}),
]

PING_TOOL = "server_ping"


def _add_ping(spec: MCPServerSpec) -> None:
    meta = ToolMeta(
        name=PING_TOOL, allowed_roles=frozenset(spec.allowed_roles), permission="mcp.ping",
        ownership=Ownership.NONE, kind=ToolKind.READ, audit_required=False,
        description="Liveness probe. Returns no data.",
    )

    @spec.tool(meta)
    def server_ping() -> dict:
        return MCPResult.ok({"server_id": spec.server_id, "alive": True}, source="mcp_internal").model_dump()


def build_registry() -> MCPRegistry:
    registry = MCPRegistry()
    for server_id, name, stype, desc, roles in SERVER_DEFINITIONS:
        spec = MCPServerSpec(server_id=server_id, server_name=name, server_type=stype,
                             description=desc, allowed_roles=frozenset(roles), mcp=FastMCP(name))
        _add_ping(spec)
        registry.add(spec)
    return registry
