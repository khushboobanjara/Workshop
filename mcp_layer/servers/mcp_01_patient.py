"""MCP 01 - Patient / Profile.

Identity ALWAYS comes from current_principal() (set by the gateway). No tool here accepts a
user_id, so there is no argument an attacker could change to read someone else's profile.
"""
from src.database import mcp_repository as repo

from ..gateway import current_principal
from ..registry import MCPServerSpec
from ..schemas import ErrorCode, Ownership, Role, ToolKind, ToolMeta
from ._common import clean_phone, clean_text, fail, ok, rules

EVERYONE = frozenset({Role.USER, Role.DOCTOR, Role.ADMIN, Role.SUPER_ADMIN})


def register(spec: MCPServerSpec) -> None:
    @spec.tool(ToolMeta(
        name="get_my_profile", allowed_roles=EVERYONE, permission="profile.read.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.READ, requires_verified_data=True,
        description="Return the signed-in user's own profile (name, email, phone, role)."))
    @rules
    def get_my_profile() -> dict:
        profile = repo.get_user_profile(current_principal().user_id)
        if not profile:
            return fail(ErrorCode.NOT_FOUND, "Your profile could not be found.")
        return ok(profile)

    @spec.tool(ToolMeta(
        name="update_my_profile", allowed_roles=EVERYONE, permission="profile.update.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.WRITE,
        description="Update the signed-in user's own name and phone number. Email and role cannot be changed here."))
    @rules
    def update_my_profile(full_name: str, phone: str) -> dict:
        user_id = current_principal().user_id
        name = clean_text(full_name, "Name", 2, 100)
        number = clean_phone(phone)
        if not repo.get_user_profile(user_id):
            return fail(ErrorCode.NOT_FOUND, "Your profile could not be found.")
        repo.update_user_profile(user_id, name, number)
        return ok(repo.get_user_profile(user_id), message="Profile updated.")
