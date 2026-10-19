from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app import auth
from app.config import settings
from app.db import SessionLocal, User
from app.security import WriteRateLimiter

router = APIRouter()
login_limiter = WriteRateLimiter()
any_user = auth.require_roles("admin", "reviewer", "viewer")
admin_only = auth.require_roles("admin")


class LoginIn(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


class NewUserIn(BaseModel):
    username: str
    password: str = Field(max_length=200)
    role: str


class UserPatch(BaseModel):
    role: str | None = None
    is_active: bool | None = None
    password: str | None = Field(None, max_length=200)


@router.post("/auth/login")
def login(payload: LoginIn, request: Request, response: Response):
    if not settings.auth_enabled:
        raise HTTPException(400, "Authentication is disabled on this server")
    ip = request.client.host if request.client else "unknown"
    if not login_limiter.allow(ip, 10):
        raise HTTPException(429, "Too many login attempts, wait a minute")
    user = auth.authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(401, "Invalid username or password")
    token = auth.make_token(user)
    response.set_cookie(auth.COOKIE, token, httponly=True, samesite="strict", secure=settings.cookie_secure,
                        max_age=settings.session_minutes * 60, path="/")
    return {"user": auth.user_dict(user), "access_token": token, "token_type": "bearer"}


@router.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE, path="/")
    return {"ok": True}


@router.get("/auth/me")
def me(principal: auth.Principal = Depends(any_user)):
    return {"auth_enabled": settings.auth_enabled, "username": principal.username, "role": principal.role}


@router.get("/users")
def list_users(_: auth.Principal = Depends(admin_only)):
    with SessionLocal() as s:
        return [auth.user_dict(u) for u in s.query(User).order_by(User.id).all()]


@router.post("/users", status_code=201)
def add_user(payload: NewUserIn, p: auth.Principal = Depends(admin_only)):
    try:
        return auth.user_dict(auth.create_user(payload.username, payload.password, payload.role, p.username))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except LookupError as e:
        raise HTTPException(409, str(e))


@router.patch("/users/{user_id}")
def patch_user(user_id: int, payload: UserPatch, p: auth.Principal = Depends(admin_only)):
    try:
        return auth.user_dict(auth.update_user(user_id, p.username, payload.role, payload.is_active, payload.password))
    except KeyError:
        raise HTTPException(404, "User not found")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except PermissionError as e:
        raise HTTPException(409, str(e))
