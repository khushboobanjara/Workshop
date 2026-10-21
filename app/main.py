"""OpsPilot AI HTTP API.   Run:  uvicorn app.main:app --reload
Dashboard: http://127.0.0.1:8000/     API docs: http://127.0.0.1:8000/docs"""
import asyncio
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app import auth, demo
from app.ml import registry
from app.auth import Principal, actor_name, require_roles
from app.auth_routes import router as auth_router
from app.context import request_id_var
from app.chat import routes as chat_routes
from app.mcp_server import routes as mcp_routes
from app.rag import routes as rag_routes
from app.config import settings
from app.db import Email, SessionLocal, init_db
from app.pipeline.factory import build_pipeline
from app.review.rules import ReviewError
from app.security import WriteRateLimiter, limit_body_size, make_guard
from app.services import processing


class EmailIn(BaseModel):
    message_id: str = Field(min_length=1, max_length=200)
    sender: str = Field("", max_length=200)
    subject: str = Field("", max_length=500)
    body: str = Field("", max_length=200_000)
    received_at: datetime | None = None    # when the mail server received it; defaults to "now"


class Corrections(BaseModel):
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    total_amount: float | None = None
    po_number: str | None = None


class ReviewIn(BaseModel):
    action: Literal["approve", "reject"]
    reviewer: str | None = Field(None, max_length=100)    # ignored when authentication is on: the signed-in user is used
    comment: str = Field("", max_length=1000)
    corrections: Corrections | None = None


class TriageIn(BaseModel):
    action: Literal["dismiss", "convert"]
    reviewer: str | None = Field(None, max_length=100)
    comment: str = Field("", max_length=1000)


def create_app(pipeline_factory: Callable = build_pipeline) -> FastAPI:
    """Factory so tests can inject a lighter pipeline (e.g. without the ML classifier)."""

    async def reset_loop(app: FastAPI):
        """Demo mode: wipe and reseed on a schedule so a public instance never accumulates junk."""
        while True:
            await asyncio.sleep(settings.demo_reset_minutes * 60)
            await asyncio.to_thread(demo.reset_and_seed, app.state.pipeline)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        auth.startup_checks_and_bootstrap_pre()
        mcp_routes.startup_checks()                  # no-op unless OPSPILOT_MCP_ENABLED=true
        chat_routes.startup_checks()                 # no-op unless OPSPILOT_CHAT_ENABLED=true
        rag_routes.startup_checks()                  # no-op unless OPSPILOT_RAG_ENABLED=true
        init_db()
        auth.startup_checks_and_bootstrap()
        if settings.demo_mode and settings.auth_enabled:
            auth.seed_demo_users()
        try:
            registry.sync_registry()
        except Exception as exc:                       # a bad metadata file must never stop the service starting
            print(f"model registry sync skipped: {type(exc).__name__}: {exc}")
        app.state.pipeline = pipeline_factory()
        if settings.rag_enabled and settings.rag_preload:             # load the embedding model now instead of on the first question
            try:
                await asyncio.to_thread(rag_routes.get_service().ensure_ready)
            except Exception as exc:                  # the service still starts; document search reports the problem when used
                print(f"document search warm-up skipped: {type(exc).__name__}: {exc}")
        task = None
        if settings.demo_mode:
            with SessionLocal() as s:
                empty = s.query(Email).first() is None
            if empty:
                demo.seed(app.state.pipeline)
            task = asyncio.create_task(reset_loop(app))
        yield
        if task:
            task.cancel()

    guard = make_guard(WriteRateLimiter())
    app = FastAPI(title="OpsPilot AI", version="0.8.0", lifespan=lifespan, dependencies=[Depends(guard)])
    app.middleware("http")(limit_body_size)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", rid):          # never trust or echo arbitrary client text
            rid = uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    app.include_router(auth_router)
    app.include_router(mcp_routes.router)           # POST /mcp: answers 404 unless OPSPILOT_MCP_ENABLED=true
    app.include_router(chat_routes.router)          # POST /chat: answers 404 unless OPSPILOT_CHAT_ENABLED=true
    app.include_router(rag_routes.router)           # /rag/*: answers 404 unless OPSPILOT_RAG_ENABLED=true
    ANY, REVIEWER, ADMIN = (require_roles("admin", "reviewer", "viewer"), require_roles("admin", "reviewer"),
                            require_roles("admin"))
    INGEST = require_roles("admin", "reviewer", "service")

    @app.get("/health", include_in_schema=False)
    def health():
        info = {"status": "ok", "demo_mode": settings.demo_mode, "auth_enabled": settings.auth_enabled,
                "auth_required": bool(settings.api_key.get_secret_value()) and not settings.auth_enabled}
        if settings.demo_mode and settings.auth_enabled:       # the demo accounts are public on purpose and low-privilege
            info["demo_accounts"] = {"password": settings.demo_password, "users": ["demo_viewer", "demo_reviewer"]}
        return info

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(Path(__file__).parent / "static" / "dashboard.html")

    @app.post("/emails")
    def post_email(payload: EmailIn, request: Request, _: Principal = Depends(INGEST)):
        """Run an incoming email through the full pipeline and store the outcome."""
        return processing.process_email(request.app.state.pipeline, payload.model_dump())

    @app.get("/review-queue")
    def get_review_queue(limit: int = Query(50, ge=1, le=500), _: Principal = Depends(REVIEWER)):
        """Invoices waiting for a human, highest risk first."""
        return processing.review_queue(limit)

    @app.get("/invoices")
    def get_invoices(status: str | None = None, decision: str | None = None, limit: int = Query(100, ge=1, le=1000), _: Principal = Depends(ANY)):
        return processing.list_invoices(status, decision, limit)

    @app.get("/invoices/{invoice_id}")
    def get_invoice(invoice_id: int, _: Principal = Depends(ANY)):
        try:
            return processing.get_invoice(invoice_id)
        except processing.NotFound as e:
            raise HTTPException(404, str(e))

    @app.get("/invoices/{invoice_id}/audit")
    def get_invoice_audit(invoice_id: int, _: Principal = Depends(REVIEWER)):
        try:
            return processing.get_audit(invoice_id)
        except processing.NotFound as e:
            raise HTTPException(404, str(e))

    @app.post("/invoices/{invoice_id}/review")
    def review_invoice(invoice_id: int, payload: ReviewIn, principal: Principal = Depends(REVIEWER)):
        try:
            corrections = payload.corrections.model_dump(exclude_none=True) if payload.corrections else None
            return processing.review_invoice(invoice_id, payload.action, actor_name(principal, payload.reviewer), payload.comment, corrections)
        except processing.NotFound as e:
            raise HTTPException(404, str(e))
        except ReviewError as e:
            raise HTTPException(409 if e.kind == "conflict" else 422, str(e))

    @app.get("/triage")
    def get_triage(_: Principal = Depends(REVIEWER)):
        return processing.triage_list()

    @app.post("/triage/{email_id}")
    def resolve_triage(email_id: int, payload: TriageIn, request: Request, principal: Principal = Depends(REVIEWER)):
        try:
            return processing.resolve_triage(request.app.state.pipeline, email_id, payload.action, actor_name(principal, payload.reviewer), payload.comment)
        except processing.NotFound as e:
            raise HTTPException(404, str(e))
        except ReviewError as e:
            raise HTTPException(409 if e.kind == "conflict" else 422, str(e))

    @app.get("/metrics")
    def get_metrics(_: Principal = Depends(ANY)):
        return processing.metrics()

    @app.get("/stats")
    def get_stats(_: Principal = Depends(ANY)):
        """Aggregates for the dashboard home page: counts, chart series, recent invoices."""
        return processing.stats()

    @app.get("/vendors")
    def get_vendors(_: Principal = Depends(ANY)):
        return processing.vendors()

    @app.get("/audit")
    def get_audit_feed(limit: int = Query(200, ge=1, le=1000), stage: str | None = None, actor: str | None = None, _: Principal = Depends(ADMIN)):
        return processing.audit_feed(limit, stage, actor)

    @app.get("/invoices.csv", include_in_schema=False)
    def export_csv(_: Principal = Depends(ANY)):
        return Response(processing.invoices_csv(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="opspilot-invoices.csv"'})

    @app.get("/models")
    def get_models(_: Principal = Depends(ANY)):
        return registry.list_models()

    @app.get("/settings")
    def get_settings(_: Principal = Depends(ADMIN)):
        return processing.public_settings()

    return app


app = create_app()
