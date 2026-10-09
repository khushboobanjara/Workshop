"""RBAC engine. Answers ONE question: is this role allowed to perform this operation?

(Whether the operation is *safe* to run is a separate layer - Phase 10.)
Checks run in this order and the first failure denies. Reasons are internal constants
that go to the audit log only; the caller always sees the same generic message.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .gateway import Decision
from .overrides import NoOverrides, PermissionOverrides
from .permissions import role_has_permission
from .registry import FORBIDDEN_PARAMS, MCPServerSpec
from .schemas import Ownership, Principal, Role, ToolMeta

log = logging.getLogger("mcp.rbac")

OWNER_KEYS = frozenset({"user_id", "patient_id", "owner_id", "doctor_id"})
ADMIN_ROLES = frozenset({Role.ADMIN, Role.SUPER_ADMIN, Role.SYSTEM})


def _deny(reason: str) -> Decision:
    return Decision(False, reason)


class RBACAuthorizer:
    def __init__(self, overrides: Optional[PermissionOverrides] = None):
        self.overrides = overrides or NoOverrides()

    def check(self, principal: Principal, server: MCPServerSpec, tool: ToolMeta,
              arguments: Dict[str, Any]) -> Decision:
        if not isinstance(principal, Principal) or not isinstance(principal.role, Role):
            return _deny("invalid_principal")

        keys = {str(k).strip().lower() for k in (arguments or {})}
        if keys & FORBIDDEN_PARAMS:                                   # role/is_admin/sql/... smuggling
            return _deny("forbidden_argument")

        role = principal.role
        if role not in server.allowed_roles:
            return _deny("role_not_allowed_on_server")
        if role not in tool.allowed_roles:
            return _deny("role_not_allowed_on_tool")
        if not role_has_permission(role, tool.permission):
            return _deny("permission_not_held")

        denial = self._ownership(principal, tool, keys)
        if denial:
            return _deny(denial)

        try:
            if self.overrides.is_revoked(role, server.server_id, tool.name):
                return _deny("revoked_by_policy")
        except Exception as exc:
            log.error("override check failed", extra={"extra_data": {"exc_type": type(exc).__name__}})
            return _deny("policy_store_unavailable")
        return Decision(True, "allowed")

    @staticmethod
    def _ownership(principal: Principal, tool: ToolMeta, keys: set) -> Optional[str]:
        if tool.ownership is Ownership.CURRENT_USER_ONLY:
            if not principal.user_id or principal.user_id <= 0:
                return "no_user_identity"
            if keys & OWNER_KEYS:                       # caller tried to name whose data to read
                return "owner_argument_not_allowed"
        elif tool.ownership is Ownership.DOCTOR_OWN:
            if principal.role is not Role.DOCTOR or not principal.doctor_id:
                return "no_doctor_identity"
            if keys & OWNER_KEYS:
                return "owner_argument_not_allowed"
        elif tool.ownership is Ownership.ADMINISTRATIVE:
            if principal.role not in ADMIN_ROLES:
                return "administrative_role_required"
        return None
