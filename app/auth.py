"""Authentication and role-based access.

- Passwords: argon2id hashes only. Login does the same work for unknown users, so response time does not reveal
  which usernames exist, and the error message is identical for "no such user" and "wrong password".
- Sessions: a short-lived JWT in an HttpOnly, SameSite=Strict cookie (or an Authorization: Bearer header for scripts).
  Each user has a token_version; changing a password/role or disabling the account bumps it, which revokes old tokens.
- Cookie sessions also check the Origin header on state-changing requests (CSRF defence in depth).
- Roles: admin, reviewer, viewer. A separate "service" principal (the ingest API key) may only POST /emails.
- With auth disabled (the default for local development) nothing here applies and behaviour is unchanged."""
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import HTTPException, Request

from app.config import settings
from app.db import AuditLog, SessionLocal, User, utcnow

ROLES = ("admin", "reviewer", "viewer")
COOKIE = "opspilot_session"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,40}$")
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_hasher = PasswordHasher()
_DUMMY_HASH = _hasher.hash("timing-equaliser")


@dataclass(frozen=True)
class Principal:
    username: str
    role: str
    user_id: int | None = None
    via: str = "session"          # session | api_key | auth-disabled


# ---------------------------------------------------------------- passwords and tokens
def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def check_password_policy(username: str, password: str) -> None:
    if len(password) < 10:
        raise ValueError("Password must be at least 10 characters")
    if password.lower() == username.lower():
        raise ValueError("Password must not equal the username")


def _secret() -> str:
    secret = settings.jwt_secret.get_secret_value()
    if len(secret) < 32:
        raise RuntimeError("OPSPILOT_JWT_SECRET must be at least 32 characters when authentication is enabled")
    return secret


def make_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": str(user.id), "tv": user.token_version, "iat": now,
              "exp": now + timedelta(minutes=settings.session_minutes)}
    return jwt.encode(claims, _secret(), algorithm="HS256")


def _principal_from_token(token: str) -> Principal | None:
    try:
        claims = jwt.decode(token, _secret(), algorithms=["HS256"], options={"require": ["exp", "sub", "tv"]})
    except jwt.PyJWTError:
        return None
    with SessionLocal() as s:
        user = s.get(User, int(claims["sub"]))
        if user is None or not user.is_active or user.token_version != claims["tv"]:
            return None
        return Principal(user.username, user.role, user.id)


def resolve_principal(request: Request) -> Principal | None:
    header = request.headers.get("authorization", "")
    bearer = header.lower().startswith("bearer ")
    token = header[7:] if bearer else request.cookies.get(COOKIE)
    if token:
        p = _principal_from_token(token)
        if p:
            # "bearer" skips the Origin/CSRF check, so it must mean the token really came from the Authorization header;
            # a cookie session stays "session" even if some proxy added a different Authorization header (e.g. Basic).
            return Principal(p.username, p.role, p.user_id, "bearer" if bearer else "session")
    expected = settings.api_key.get_secret_value()
    supplied = request.headers.get("x-api-key", "")
    if expected and supplied and secrets.compare_digest(supplied.encode(), expected.encode()):
        return Principal("ingest-service", "service", None, "api_key")
    return None


def require_roles(*roles: str):
    """FastAPI dependency: the caller must be signed in with one of these roles (no-op while auth is disabled)."""
    def dependency(request: Request) -> Principal:
        if not settings.auth_enabled:
            return Principal("anonymous", "admin", None, "auth-disabled")
        principal = resolve_principal(request)
        if principal is None:
            raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
        if principal.role not in roles:
            raise HTTPException(403, "Your role is not allowed to do this")
        origin = request.headers.get("origin")
        if (principal.via == "session" and request.method not in SAFE_METHODS and origin
                and urlparse(origin).netloc != request.headers.get("host")):
            raise HTTPException(403, "Cross-origin request blocked")
        return principal
    return dependency


def actor_name(principal: Principal, supplied: str | None) -> str:
    """Who is acting. With auth on it is always the signed-in user, never a name the client typed."""
    if settings.auth_enabled:
        return principal.username
    if not supplied or not supplied.strip():
        raise HTTPException(422, "reviewer name is required")
    return supplied.strip()


# ---------------------------------------------------------------- users
def _audit(s, action: str, actor: str, details: dict | None = None, entity_id: int | None = None) -> None:
    s.add(AuditLog(entity_type="user", entity_id=entity_id, stage="auth", action=action, actor=actor[:50], details=details))


def create_user(username: str, password: str, role: str, actor: str = "system") -> User:
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-40 characters: letters, digits, dot, dash, underscore")
    if role not in ROLES:
        raise ValueError(f"Role must be one of {', '.join(ROLES)}")
    check_password_policy(username, password)
    with SessionLocal() as s:
        if s.query(User).filter(User.username == username).first():
            raise LookupError("That username is already taken")
        user = User(username=username, password_hash=hash_password(password), role=role)
        s.add(user)
        s.flush()
        _audit(s, "user_created", actor, {"username": username, "role": role}, user.id)
        s.commit()
        return user


def authenticate(username: str, password: str) -> User | None:
    with SessionLocal() as s:
        user = s.query(User).filter(User.username == username).first()
        ok = verify_password(user.password_hash if user else _DUMMY_HASH, password)
        if not (user and ok and user.is_active):
            _audit(s, "login_failed", username[:50] or "?", None, user.id if user else None)
            s.commit()
            return None
        user.last_login_at = utcnow()
        _audit(s, "login_success", user.username, None, user.id)
        s.commit()
        return user


def update_user(user_id: int, actor: str, role: str | None = None, is_active: bool | None = None,
                password: str | None = None) -> User:
    with SessionLocal() as s:
        user = s.get(User, user_id)
        if user is None:
            raise KeyError(user_id)
        losing_admin = user.role == "admin" and user.is_active and (
            (role is not None and role != "admin") or is_active is False)
        if losing_admin and s.query(User).filter(User.role == "admin", User.is_active.is_(True), User.id != user.id).count() == 0:
            raise PermissionError("At least one active admin must remain")
        changes = {}
        if role is not None and role != user.role:
            if role not in ROLES:
                raise ValueError(f"Role must be one of {', '.join(ROLES)}")
            changes["role"] = [user.role, role]
            user.role = role
        if is_active is not None and is_active != user.is_active:
            changes["is_active"] = [user.is_active, is_active]
            user.is_active = is_active
        if password is not None:
            check_password_policy(user.username, password)
            user.password_hash = hash_password(password)
            changes["password"] = "changed"
        if changes:
            user.token_version += 1                       # sign the user out everywhere
            _audit(s, "user_updated", actor, changes, user.id)
        s.commit()
        return user


def user_dict(u: User) -> dict:
    return {"id": u.id, "username": u.username, "role": u.role, "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None}


# ---------------------------------------------------------------- startup
def startup_checks_and_bootstrap_pre() -> None:
    """Fail closed before anything else starts: production must have authentication and HTTPS cookies."""
    if settings.env == "production":
        problems = []
        if not settings.auth_enabled:
            problems.append("OPSPILOT_AUTH_ENABLED must be true")
        if not settings.cookie_secure:
            problems.append("OPSPILOT_COOKIE_SECURE must be true (serve over HTTPS)")
        if problems:
            raise RuntimeError("Refusing to start in production: " + "; ".join(problems))
    if settings.auth_enabled:
        _secret()                                          # raises if too short


def startup_checks_and_bootstrap() -> None:
    """Optionally create the first admin from OPSPILOT_BOOTSTRAP_ADMIN_* when there are no users yet."""
    if not settings.auth_enabled:
        return
    with SessionLocal() as s:
        has_users = s.query(User).first() is not None
    name, pw = settings.bootstrap_admin_user, settings.bootstrap_admin_password.get_secret_value()
    if not has_users and name and pw:
        create_user(name, pw, "admin")


def seed_demo_users() -> None:
    """Public demo only: two low-privilege accounts whose password is shown on the login screen. Never an admin."""
    with SessionLocal() as s:
        existing = {u for (u,) in s.query(User.username).all()}
    for name, role in (("demo_viewer", "viewer"), ("demo_reviewer", "reviewer")):
        if name not in existing:
            create_user(name, settings.demo_password, role)
