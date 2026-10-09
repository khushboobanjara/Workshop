"""Shared types: roles, tool metadata, the common MCP result, the caller identity."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class Role(str, Enum):
    USER = "USER"        # DB role PATIENT
    DOCTOR = "DOCTOR"    # DB role DOCTOR (scoped admin: own doctor_id only)
    ADMIN = "ADMIN"
    SUPER_ADMIN = "SUPER_ADMIN"   # DB role SUPER_ADMIN: RBAC management, audit, monitoring
    SYSTEM = "SYSTEM"    # internal service identity, never stored on a user row


class ServerType(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"
    SYSTEM = "SYSTEM"


class ToolKind(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"


class Ownership(str, Enum):
    NONE = "none"                        # no per-user data (e.g. public medicine search)
    CURRENT_USER_ONLY = "current_user_only"
    DOCTOR_OWN = "doctor_own"            # doctor limited to their own doctor_id
    ADMINISTRATIVE = "administrative"    # admin acting on any record


class ErrorCode:
    UNAUTHENTICATED = "UNAUTHENTICATED"
    ACCESS_DENIED = "ACCESS_DENIED"
    SERVER_NOT_FOUND = "SERVER_NOT_FOUND"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    MCP_UNAVAILABLE = "MCP_UNAVAILABLE"
    MCP_DISABLED = "MCP_DISABLED"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_TOOL_OUTPUT = "INVALID_TOOL_OUTPUT"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    SAFETY_BLOCKED = "SAFETY_BLOCKED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"


class MCPResult(BaseModel):
    """Every tool returns this. Only success=True AND verified=True counts as
    verified data; the grounding layer (Phase 7) relies on that."""

    success: bool
    data: Optional[Any] = None
    source: str = "none"
    verified: bool = False
    error_code: Optional[str] = None
    message: Optional[str] = None      # safe, user-presentable text only
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)   # non-sensitive context (e.g. row_count)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _consistent(self):
        if not self.success:
            self.verified = False
            self.data = None
            if not self.error_code:
                self.error_code = ErrorCode.INTERNAL_ERROR
        elif self.verified and (not self.source or self.source == "none"):
            raise ValueError("verified results must name their source")
        return self

    @classmethod
    def ok(cls, data: Any, source: str, verified: bool = True, **kw) -> "MCPResult":
        return cls(success=True, data=data, source=source, verified=verified, **kw)

    @classmethod
    def fail(cls, error_code: str, source: str = "none", message: str | None = None, **kw) -> "MCPResult":
        return cls(success=False, data=None, source=source, verified=False,
                   error_code=error_code, message=message, **kw)


class ToolMeta(BaseModel):
    """Metadata every tool must declare (name, server, role, permission,
    ownership, read/write/destructive, audit)."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    allowed_roles: frozenset[Role]
    permission: str = Field(pattern=r"^[a-z_]+(\.[a-z_]+)+$")   # e.g. appointment.read.own
    ownership: Ownership = Ownership.NONE
    kind: ToolKind = ToolKind.READ
    audit_required: bool = True
    requires_verified_data: bool = False   # grounding must block LLM text if this fails
    description: str = ""

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _rules(self):
        if Role.SYSTEM in self.allowed_roles and len(self.allowed_roles) > 1 and self.permission != "mcp.ping":
            # The only role allowed alongside SYSTEM is SUPER_ADMIN, and only for READ tools
            # (e.g. viewing audit logs). SYSTEM write tools stay internal.
            others = self.allowed_roles - {Role.SYSTEM}
            if others != {Role.SUPER_ADMIN} or self.kind is not ToolKind.READ:
                raise ValueError("SYSTEM tools may only be shared with SUPER_ADMIN, and only for READ tools")
        if self.kind is ToolKind.DESTRUCTIVE:
            if not self.audit_required:
                raise ValueError("destructive tools must be audited")
            if Role.USER in self.allowed_roles:
                raise ValueError("destructive tools must not be exposed to USER")
        if self.kind is not ToolKind.READ and not self.audit_required:
            raise ValueError("write tools must be audited")
        return self


class Principal(BaseModel):
    """The authenticated caller. Built ONLY on the server from the session +
    database (Phase 3). Never constructed from request data."""

    user_id: Optional[int] = None
    role: Role
    doctor_id: Optional[int] = None

    model_config = {"frozen": True}

    @classmethod
    def system(cls) -> "Principal":
        return cls(user_id=None, role=Role.SYSTEM)
