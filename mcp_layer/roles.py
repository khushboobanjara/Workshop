"""Maps the role string stored in users.role to an MCP Role.

Unknown / NULL / empty -> None (no access). SYSTEM can never come from the
database, so a tampered row cannot become an internal identity.
"""
from typing import Optional

from .schemas import Role

_DB_TO_ROLE = {
    "PATIENT": Role.USER,
    "DOCTOR": Role.DOCTOR,
    "ADMIN": Role.ADMIN,
}
BLOCKED_DOCTOR_STATUSES = frozenset({"PENDING", "REJECTED", "SUSPENDED", "BLOCKED"})  # mirrors doctor_dashboard.py


def resolve_role(db_role: Optional[str], doctor_status: Optional[str] = None) -> Optional[Role]:
    if not db_role or not isinstance(db_role, str):
        return None
    role = _DB_TO_ROLE.get(db_role.strip().upper())
    if role is Role.DOCTOR and (doctor_status or "").strip().upper() in BLOCKED_DOCTOR_STATUSES:
        return None
    if role is Role.DOCTOR and not doctor_status:
        return None   # fail closed: an approved doctor must have an explicit status
    return role
