"""MCP 08 - User / System Administration.

Separation of powers:
  * ADMIN: read-only on users, and never sees SUPER_ADMIN accounts.
  * SUPER_ADMIN only: change a user's role (permission user.role.manage, which ADMIN does not hold).
A role change can never produce SUPER_ADMIN, never changes your own role and never edits a
SUPER_ADMIN row - granting that stays a manual database step (see the optional 002b script).

Not built yet: activate / deactivate users. users has no is-active column, and the login code
would have to check it for a deactivation to mean anything. That belongs to Phase 12.
"""
from typing import Optional

from src.database import mcp_admin_repository as admin_repo

from ..gateway import current_principal
from ..registry import MCPServerSpec
from ..schemas import ErrorCode, Ownership, Role, ToolKind, ToolMeta
from ._common import BadInput, clean_text, fail, ok, ok_rows, opt_text, positive_int, rules

ADMINS = frozenset({Role.ADMIN, Role.SUPER_ADMIN})
SUPER_ONLY = frozenset({Role.SUPER_ADMIN})
ASSIGNABLE_ROLES = {"PATIENT", "DOCTOR", "ADMIN"}   # users.role values; SUPER_ADMIN is deliberately absent
READABLE_ROLES = ASSIGNABLE_ROLES | {"SUPER_ADMIN"}


def _hide_super_admins(principal) -> bool:
    return principal.role is not Role.SUPER_ADMIN


def register(spec: MCPServerSpec) -> None:
    @spec.tool(ToolMeta(
        name="list_users", allowed_roles=ADMINS, permission="user.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Search users by name or email, optionally by stored role (PATIENT, DOCTOR, ADMIN). Max 50."))
    @rules
    def list_users(role_filter: Optional[str] = None, search: Optional[str] = None) -> dict:
        wanted = None
        if role_filter and role_filter.strip():
            wanted = role_filter.strip().upper()
            if wanted not in READABLE_ROLES:
                raise BadInput("Role must be PATIENT, DOCTOR or ADMIN.")
        term = opt_text(search, "Search text", 2, 60)
        rows = admin_repo.list_users(wanted, term, _hide_super_admins(current_principal()), 50)
        return ok_rows(rows, "No users match.")

    @spec.tool(ToolMeta(
        name="get_user_details", allowed_roles=ADMINS, permission="user.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="One user's account details (never the password)."))
    @rules
    def get_user_details(target_user_id: int) -> dict:
        row = admin_repo.get_user(positive_int(target_user_id, "User"))
        if not row or (row["role"] == "SUPER_ADMIN" and _hide_super_admins(current_principal())):
            return fail(ErrorCode.NOT_FOUND, "That user could not be found.")
        return ok(row)

    @spec.tool(ToolMeta(
        name="set_user_role", allowed_roles=SUPER_ONLY, permission="user.role.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="SUPER_ADMIN only. Set a user's stored role to PATIENT, DOCTOR or ADMIN. "
                    "A new DOCTOR starts as PENDING until approved."))
    @rules
    def set_user_role(target_user_id: int, new_role: str) -> dict:
        principal = current_principal()
        target = positive_int(target_user_id, "User")
        wanted = clean_text(new_role, "Role", 5, 15).upper()
        if wanted not in ASSIGNABLE_ROLES:
            raise BadInput("Role must be PATIENT, DOCTOR or ADMIN.")
        if target == principal.user_id:
            return fail(ErrorCode.CONFLICT, "You cannot change your own role.")
        row = admin_repo.get_user(target)
        if not row:
            return fail(ErrorCode.NOT_FOUND, "That user could not be found.")
        if row["role"] == "SUPER_ADMIN":
            return fail(ErrorCode.CONFLICT, "A super admin account cannot be changed here.")
        if row["role"] == wanted:
            return ok({"user_id": target, "role": wanted}, message="The user already has that role.")
        admin_repo.set_user_role(target, wanted, pending_doctor=(wanted == "DOCTOR"))
        return ok(admin_repo.get_user(target), message="Role updated.")
