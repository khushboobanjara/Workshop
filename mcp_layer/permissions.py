"""Central permission catalogue: which role holds which permission string.

Tool metadata says which roles MAY call a tool; this matrix says which permissions each role
HOLDS. A call needs both. A permission that is not listed here is held by nobody (fail closed).
Naming: <area>.<action>[.<scope>]  with scope own | assigned | all.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List

from .registry import MCPRegistry
from .schemas import Role

_PING = {"mcp.ping"}

USER_PERMISSIONS: FrozenSet[str] = frozenset(_PING | {
    "profile.read.own", "profile.update.own", "health_profile.read.own",
    "doctor.search", "doctor.availability.read",
    "appointment.create.own", "appointment.read.own", "appointment.cancel.own", "appointment.reschedule.own",
    "pharmacy.search", "pharmacy.availability.read", "pharmacy.cart.own",
    "pharmacy.order.create.own", "pharmacy.order.read.own", "pharmacy.order.cancel.own",
    "screening.create.own", "screening.read.own",
    "assistant.use",
})

# Doctors are scoped: own profile, own availability, appointments assigned to them. Nothing global.
DOCTOR_PERMISSIONS: FrozenSet[str] = frozenset(_PING | {
    "profile.read.own", "profile.update.own",
    "doctor.profile.read.own", "doctor.availability.update.own",
    "appointment.read.assigned", "appointment.manage.assigned",
})

ADMIN_PERMISSIONS: FrozenSet[str] = USER_PERMISSIONS | frozenset({
    "doctor.manage", "doctor.approve",
    "appointment.read.all", "appointment.manage",
    "pharmacy.manage", "pharmacy.order.read.all", "pharmacy.order.manage",
    "user.read", "user.manage",
})

# SUPER_ADMIN = everything ADMIN has + RBAC management, audit, monitoring, role changes.
SUPER_ADMIN_PERMISSIONS: FrozenSet[str] = ADMIN_PERMISSIONS | frozenset({
    "user.role.manage", "rbac.read", "rbac.manage",
    "audit.read", "security.read", "system.monitor",
    "llm.usage.read", "llm.pricing.manage",
})

# SYSTEM = internal MCP infrastructure only.
SYSTEM_PERMISSIONS: FrozenSet[str] = frozenset(_PING | {
    "audit.write", "security.check", "rbac.evaluate",
    "llm.usage.record", "llm.cost.calculate", "system.monitor",
})

ROLE_PERMISSIONS: Dict[Role, FrozenSet[str]] = {
    Role.USER: USER_PERMISSIONS,
    Role.DOCTOR: DOCTOR_PERMISSIONS,
    Role.ADMIN: ADMIN_PERMISSIONS,
    Role.SUPER_ADMIN: SUPER_ADMIN_PERMISSIONS,
    Role.SYSTEM: SYSTEM_PERMISSIONS,
}


def role_has_permission(role: Role, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


def validate_registry_permissions(registry: MCPRegistry) -> List[str]:
    """Startup self-check. Every role a tool names must actually hold the tool's permission,
    otherwise the tool is advertised to a role that can never call it (or is mis-declared)."""
    problems: List[str] = []
    for server in registry.servers():
        for name, meta in server.tools.items():
            for role in meta.allowed_roles:
                if not role_has_permission(role, meta.permission):
                    problems.append(f"{server.server_id}.{name}: {role.value} lacks '{meta.permission}'")
    return sorted(problems)
