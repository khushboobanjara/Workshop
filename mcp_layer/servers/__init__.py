"""Declares the 10 MCP servers. Tools are added phase by phase in their own modules."""
from __future__ import annotations

from fastmcp import FastMCP

from ..registry import MCPRegistry, MCPServerSpec
from ..schemas import MCPResult, Ownership, Role, ServerType, ToolKind, ToolMeta

U, D, A, SA, S = Role.USER, Role.DOCTOR, Role.ADMIN, Role.SUPER_ADMIN, Role.SYSTEM

# (server_id, name, type, description, allowed_roles)
SERVER_DEFINITIONS = [
    ("mcp_01_patient_profile", "Patient / Profile MCP", ServerType.USER,
     "Own profile and preferences", {U, D, A, SA}),
    ("mcp_02_doctor_appointment", "Doctor / Appointment MCP", ServerType.USER,
     "Doctor search, availability and the patient's own appointments", {U, A, SA}),
    ("mcp_03_pharmacy", "Pharmacy MCP", ServerType.USER,
     "Medicines, cart, own orders, nearby pharmacies", {U, A, SA}),
    ("mcp_04_health_screening", "Health Screening MCP", ServerType.USER,
     "Validated inputs to the existing prediction pipeline", {U, A, SA}),
    ("mcp_05_ai_assistant", "AI Health Assistant MCP", ServerType.USER,
     "Healthcare assistant routed through verified tools", {U, A, SA}),
    ("mcp_06_doctor_admin", "Doctor / Appointment Administration MCP", ServerType.ADMIN,
     "Doctors, availability and appointment administration", {A, D, SA}),
    ("mcp_07_pharmacy_admin", "Pharmacy / Inventory Administration MCP", ServerType.ADMIN,
     "Medicines, stock and order administration", {A, SA}),
    ("mcp_08_user_system_admin", "User / System Administration MCP", ServerType.ADMIN,
     "Users, roles and administrative reporting", {A, SA}),
    ("mcp_09_security_audit", "Security / RBAC / Audit MCP", ServerType.SYSTEM,
     "RBAC evaluation, audit and security events", {S, SA}),
    ("mcp_10_llm_monitoring", "LLM / Token / Cost / Monitoring MCP", ServerType.SYSTEM,
     "Token usage, cost and MCP health monitoring", {S, SA}),
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


def build_registry(with_tools: bool = False) -> MCPRegistry:
    registry = MCPRegistry()
    for server_id, name, stype, desc, roles in SERVER_DEFINITIONS:
        spec = MCPServerSpec(server_id=server_id, server_name=name, server_type=stype,
                             description=desc, allowed_roles=frozenset(roles), mcp=FastMCP(name))
        _add_ping(spec)
        registry.add(spec)
    if with_tools:
        register_tools(registry)
    return registry


def register_tools(registry: MCPRegistry) -> None:
    """Adds the real tools (Phase 5+). Imported here, not at module top, because the tool modules
    import gateway.current_principal and gateway itself imports this package."""
    from . import (mcp_01_patient, mcp_02_appointment, mcp_03_pharmacy, mcp_04_screening, mcp_05_assistant,
                   mcp_06_doctor_admin, mcp_07_pharmacy_admin, mcp_08_user_admin,
                   mcp_09_security_audit)

    mcp_01_patient.register(registry.get("mcp_01_patient_profile"))
    mcp_02_appointment.register(registry.get("mcp_02_doctor_appointment"))
    mcp_03_pharmacy.register(registry.get("mcp_03_pharmacy"))
    mcp_04_screening.register(registry.get("mcp_04_health_screening"))
    mcp_05_assistant.register(registry.get("mcp_05_ai_assistant"))
    mcp_06_doctor_admin.register(registry.get("mcp_06_doctor_admin"))
    mcp_07_pharmacy_admin.register(registry.get("mcp_07_pharmacy_admin"))
    mcp_08_user_admin.register(registry.get("mcp_08_user_system_admin"))
    mcp_09_security_audit.register(registry.get("mcp_09_security_audit"))
