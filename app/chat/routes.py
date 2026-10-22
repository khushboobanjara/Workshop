"""POST /chat and GET /chat/status. Authentication and roles are the application's own (`require_roles`), so a chat
request is exactly as privileged as the signed-in user: the model acts through the MCP core with that user's Principal.

Off unless OPSPILOT_CHAT_ENABLED=true. Refuses to start without authentication (an anonymous chat would hand every record
to a third-party model), in demo mode, or without a Groq key. Questions and the records needed to answer them are sent
to Groq: the MCP layer's redaction applies first (no e-mail bodies, card numbers and IBANs masked), but amounts, vendor
names and invoice numbers do leave this server. See docs/CHAT.md.
"""
import logging
import time
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth import Principal, require_roles
from app.chat import audit, service
from app.chat.errors import GroqError, public_message
from app.chat.groq_client import GroqClient
from app.config import settings
from app.mcp_server import rbac
from app.mcp_server.server import server
from app.rag import service as rag_service
from app.security import WriteRateLimiter

logger = logging.getLogger("opspilot.chat")
router = APIRouter(prefix="/chat", tags=["chat"])
ANY = require_roles("admin", "reviewer", "viewer")
chat_limiter = WriteRateLimiter()
MAX_MESSAGE_CHARS = 2000


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]                 # 'system' and 'tool' turns from a client are rejected outright
    content: str = Field(max_length=MAX_MESSAGE_CHARS)


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v


def make_client() -> GroqClient:
    return GroqClient(api_key=settings.groq_api_key.get_secret_value(), model=settings.groq_model,
                      base_url=settings.groq_base_url, timeout=settings.chat_timeout_seconds,
                      temperature=settings.chat_temperature, rate_limit_retries=settings.chat_rate_limit_retries,
                      max_wait=settings.chat_rate_limit_max_wait_seconds)


def _ready() -> bool:
    return bool(settings.chat_enabled and settings.auth_enabled and settings.groq_api_key.get_secret_value())


def startup_checks() -> None:
    """Fail closed: refuse to start the assistant in an unsafe configuration."""
    if not settings.chat_enabled:
        return
    problems = []
    if not settings.auth_enabled:
        problems.append("OPSPILOT_AUTH_ENABLED must be true (the assistant never runs unauthenticated)")
    if settings.demo_mode:
        problems.append("the assistant cannot be enabled together with OPSPILOT_DEMO_MODE")
    if not settings.groq_api_key.get_secret_value():
        problems.append("OPSPILOT_GROQ_API_KEY is not set")
    if not settings.groq_model.strip():
        problems.append("OPSPILOT_GROQ_MODEL is empty")
    url = urlparse(settings.groq_base_url)
    if url.scheme != "https" and url.hostname not in ("localhost", "127.0.0.1"):
        problems.append("OPSPILOT_GROQ_BASE_URL must use https")
    if problems:
        raise RuntimeError("Refusing to start the AI assistant: " + "; ".join(problems))


def _documents_on() -> bool:
    return bool(settings.rag_enabled)


@router.get("/status")
def chat_status(principal: Principal = Depends(ANY)):
    """Lets the dashboard decide whether to show the chat. Reveals nothing about configuration beyond on/off and the tool names."""
    if not _ready():
        return {"enabled": False, "model": None, "tools": []}
    return {"enabled": True, "model": settings.groq_model,
            "tools": [t.name for t in server.allowed_tools(principal)], "documents": _documents_on()}


@router.post("")
def chat(payload: ChatIn, principal: Principal = Depends(ANY)):
    if not settings.chat_enabled:
        raise HTTPException(404, "The AI assistant is not enabled")
    if not _ready():
        raise HTTPException(503, "The AI assistant is not configured")
    label = rbac.mcp_role(principal.role, principal.via)
    if label is None:
        raise HTTPException(403, "This account cannot use the assistant")
    if not chat_limiter.allow(f"chat:{principal.username}", settings.chat_rate_limit_per_minute):
        raise HTTPException(429, "Too many questions. Wait a minute and try again.", headers={"Retry-After": "60"})

    model = settings.groq_model
    started = time.perf_counter()
    try:
        rag = None
        if _documents_on():
            try:
                rag = rag_service.get_service()
            except rag_service.RagError:
                rag = None                                                # documents unavailable: answer from MCP data only
        result = service.answer(principal, payload.message, [t.model_dump() for t in payload.history],
                                llm=make_client(), server=server, role_label=label, rag=rag,
                                limits=service.Limits(fact_check=settings.chat_fact_check, rag_min_support=settings.rag_min_support,
                                                      history_turns=settings.chat_history_turns, history_chars=settings.chat_history_chars,
                                                      hardcode=settings.chat_hardcode, answer_mode=settings.rag_answer_mode,
                                                      direct_min_overlap=settings.rag_direct_min_overlap))
    except GroqError as exc:
        logger.warning("Groq request failed: code=%s detail=%s", exc.code, exc.detail)       # visible in the server console
        _audit_quietly(principal, "rate_limited" if exc.code == "rate_limited" else "failed", payload.message, model,
                       error_code=exc.code, error_detail=exc.detail, started=started)
        message = public_message(exc.code, exc.retry_after)
        if principal.role == "admin" and exc.detail:                          # admins get the provider's reason; others do not
            message += f" [admin details: {exc.detail}]"
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        raise HTTPException(exc.http_status, message, headers=headers) from None
    try:                                                                  # fail closed: no audit row, no answer
        audit.record_turn(principal=principal, outcome="blocked" if result.blocked else "ok", question=payload.message,
                          model=model, result=result, duration_ms=round((time.perf_counter() - started) * 1000, 2))
    except Exception:
        raise HTTPException(503, "The answer could not be recorded in the audit log, so it was withheld.") from None
    return result.to_dict(model)


def _audit_quietly(principal, outcome, question, model, *, error_code, error_detail=None, started) -> None:
    try:
        audit.record_turn(principal=principal, outcome=outcome, question=question, model=model, error_code=error_code,
                          error_detail=error_detail, duration_ms=round((time.perf_counter() - started) * 1000, 2))
    except Exception:
        pass                                                              # the failure response does not depend on the log
