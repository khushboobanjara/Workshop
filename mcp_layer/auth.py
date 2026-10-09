"""Authentication -> role resolver: builds the Principal for one request.

Rules:
  * Only the signed session's user_id is used. The session's cached role is NEVER trusted;
    the role is re-read from the database on every request, so a demoted / unapproved account
    loses access immediately instead of at next login.
  * A database failure produces NO principal (fail closed) and a distinct error code.
  * SYSTEM can never be produced here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .roles import resolve_role
from .schemas import ErrorCode, Principal, Role

log = logging.getLogger("mcp.auth")


@dataclass(frozen=True)
class AuthOutcome:
    principal: Optional[Principal]
    error_code: Optional[str] = None
    reason: str = ""            # internal / audit only, never shown to the user

    @property
    def ok(self) -> bool:
        return self.principal is not None


class IdentityStore(Protocol):
    def get_user(self, user_id: int) -> Optional[dict]: ...
    def get_doctor_id_by_email(self, email: str) -> Optional[int]: ...


class RepositoryIdentityStore:
    """Default store: the project's existing repositories."""

    def get_user(self, user_id: int) -> Optional[dict]:
        from src.database.user_repository import get_user_auth_by_id
        return get_user_auth_by_id(user_id)

    def get_doctor_id_by_email(self, email: str) -> Optional[int]:
        from src.database.doctor_dashboard_repository import get_doctor_by_email
        row = get_doctor_by_email(email)
        return int(row["doctor_id"]) if row and row.get("doctor_id") else None


def normalize_user_id(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):          # True would otherwise become user 1
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def resolve_principal(session_user_id: Any, store: IdentityStore,
                      session_role: Optional[str] = None) -> AuthOutcome:
    user_id = normalize_user_id(session_user_id)
    if user_id is None:
        return AuthOutcome(None, ErrorCode.UNAUTHENTICATED, "no_valid_session_user")
    try:
        row = store.get_user(user_id)
        if not row:
            return AuthOutcome(None, ErrorCode.UNAUTHENTICATED, "user_not_found")

        role = resolve_role(row.get("role"), row.get("doctor_status"))
        if session_role and str(session_role).strip().upper() != str(row.get("role") or "").strip().upper():
            log.warning("session role differs from database role",
                        extra={"extra_data": {"user_id": user_id, "session_role": str(session_role)[:20],
                                              "db_role": str(row.get("role"))[:20]}})
        if role is None:
            return AuthOutcome(None, ErrorCode.ACCESS_DENIED, "role_not_resolved")

        doctor_id = None
        if role is Role.DOCTOR:
            doctor_id = store.get_doctor_id_by_email(row.get("email") or "")
        return AuthOutcome(Principal(user_id=user_id, role=role, doctor_id=doctor_id))
    except Exception as exc:   # fail closed; never leak driver messages
        log.error("identity lookup failed", extra={"extra_data": {"exc_type": type(exc).__name__}})
        return AuthOutcome(None, ErrorCode.DATABASE_UNAVAILABLE, "identity_lookup_failed")


def principal_from_request(request: Any, store: Optional[IdentityStore] = None) -> AuthOutcome:
    """Works with any object exposing .session (and optionally .state), e.g. a Starlette Request.
    The outcome is cached on request.state so one HTTP request costs one identity lookup."""
    state = getattr(request, "state", None)
    cached = getattr(state, "mcp_auth", None) if state is not None else None
    if isinstance(cached, AuthOutcome):
        return cached
    session = getattr(request, "session", None) or {}
    outcome = resolve_principal(session.get("user_id"), store or RepositoryIdentityStore(),
                                session.get("role"))
    if state is not None:
        try:
            state.mcp_auth = outcome
        except Exception:
            pass
    return outcome
