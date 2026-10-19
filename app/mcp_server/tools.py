"""The MCP tools. All are READ-ONLY (the only thing MCP ever writes is its own audit row).

Each tool = a strict argument model (unknown fields, wrong types and out-of-range values are rejected) + a handler that
reuses app.services.processing, so MCP and the REST API can never disagree about what the data is + a list of the
application roles allowed to call it. Role lists mirror the REST API one-to-one (see rbac.py).

  tool                    viewer  reviewer  admin   system(MCP key)
  get_my_access             x        x        x        x
  search_invoices           x        x        x
  get_invoice               x        x        x
  list_vendors              x        x        x
  get_metrics               x        x        x        x
  list_models               x        x        x        x
  get_invoice_audit_trail            x        x
  get_review_queue                   x        x
  get_system_status                           x        x
  get_audit_log                               x
  get_settings                                x
  list_users                                  x

Deliberately NOT exposed: approve/reject/correct (an assistant must not move money-adjacent state), e-mail bodies,
the triage queue (it contains body previews) and anything that writes.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, text

from app.auth import user_dict
from app.db import AuditLog, Email, Invoice, ModelVersion, SessionLocal, User
from app.mcp_server import grounding as g
from app.mcp_server.sanitize import clean_text, sanitize_json
from app.mcp_server.contracts import Tool, ToolContext, ToolResult
from app.ml import registry
from app.services import processing

VIEWER_UP = frozenset({"viewer", "reviewer", "admin"})
REVIEWER_UP = frozenset({"reviewer", "admin"})
ADMIN = frozenset({"admin"})
EVERYONE = frozenset({"viewer", "reviewer", "admin", "service"})
ADMIN_AND_SYSTEM = frozenset({"admin", "service"})
MAX_ROWS = 100
INT32_MAX = 2_147_483_647


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArgs(_Args):
    pass


class InvoiceIdArgs(_Args):
    invoice_id: int = Field(ge=1, le=INT32_MAX, description="Numeric id of the invoice.")


class LimitArgs(_Args):
    limit: int = Field(25, ge=1, le=MAX_ROWS, description="Maximum number of rows to return.")


class SearchInvoicesArgs(_Args):
    status: Literal["any", "new", "approved", "pending_review", "rejected"] = Field(
        "any", description="Lifecycle status filter; 'any' means no filter.")
    decision: Literal["any", "auto_approved", "human_review", "rejected"] = Field(
        "any", description="The system's original decision; 'any' means no filter.")
    vendor: str = Field("", max_length=100, description="Case-insensitive part of the vendor name; empty means no filter.")
    limit: int = Field(25, ge=1, le=MAX_ROWS, description="Maximum number of invoices to return (newest first).")


class AuditLogArgs(_Args):
    stage: str = Field("", max_length=50, pattern=r"^[A-Za-z0-9_.-]*$",
                       description="Only this stage, e.g. 'human_review' or 'mcp'; empty means all.")
    actor: str = Field("", max_length=100, pattern=r"^[A-Za-z0-9_.@:-]*$", description="Only this actor; empty means all.")
    limit: int = Field(50, ge=1, le=200, description="Maximum number of entries (newest first).")


class UsersArgs(_Args):
    limit: int = Field(100, ge=1, le=200, description="Maximum number of users to return.")


def _like_escape(s: str) -> str:
    return s.replace("!", "!!").replace("%", "!%").replace("_", "!_")          # '!' is the ESCAPE character below


def _ref(kind: str, i) -> str:
    return f"{kind}:{i}"


# ---------------------------------------------------------------- handlers
def _get_my_access(ctx: ToolContext, a: NoArgs) -> ToolResult:
    p = ctx.principal
    return ToolResult({"username": clean_text(p.username, 100, redact=False), "role": p.role, "mcp_role": ctx.mcp_role,
                       "authenticated_via": p.via, "tools_you_may_call": list(ctx.allowed_tools)})


def _search_invoices(ctx: ToolContext, a: SearchInvoicesArgs) -> ToolResult:
    with SessionLocal() as s:
        q = s.query(Invoice)
        if a.status != "any":
            q = q.filter(Invoice.status == a.status)
        if a.decision != "any":
            q = q.filter(Invoice.decision == a.decision)
        if a.vendor:
            q = q.filter(func.lower(Invoice.vendor_name).like(f"%{_like_escape(a.vendor.lower())}%", escape="!"))
        total = q.count()
        rows = q.order_by(Invoice.id.desc()).limit(a.limit).all()
        items = [g.shape_invoice(processing.invoice_to_dict(r)) for r in rows]
    notes = []
    if not items:
        notes.append("No invoices match these filters.")
    elif total > len(items):
        notes.append(f"Showing the newest {len(items)} of {total} matching invoices; narrow the filters to see others.")
    data = {"filters": {"status": a.status, "decision": a.decision, "vendor": clean_text(a.vendor, 100, redact=False)},
            "total_matching": total, "returned": len(items), "invoices": items}
    return ToolResult(data, found=bool(items), refs=[_ref("invoice", i["id"]) for i in items], tables=("invoices",),
                      untrusted=g.UNTRUSTED_INVOICE_FIELDS, notes=tuple(notes))


def _get_invoice(ctx: ToolContext, a: InvoiceIdArgs) -> ToolResult:
    try:
        d = processing.get_invoice(a.invoice_id)
    except processing.NotFound:
        return ToolResult(None, found=False, tables=("invoices",),
                          notes=(f"No invoice with id {a.invoice_id} exists in the database.",))
    ref = _ref("invoice", a.invoice_id)
    return ToolResult({"invoice": g.shape_invoice(d)}, refs=[ref], tables=("invoices", "emails"),
                      untrusted=g.UNTRUSTED_INVOICE_FIELDS, entity=ref,
                      notes=("The e-mail body is intentionally not available through MCP.",))


def _invoice_audit_trail(ctx: ToolContext, a: InvoiceIdArgs) -> ToolResult:
    try:
        rows = processing.get_audit(a.invoice_id)
    except processing.NotFound:
        return ToolResult(None, found=False, tables=("invoices", "audit_log"),
                          notes=(f"No invoice with id {a.invoice_id} exists in the database.",))
    kept = rows[-200:]
    entries = [{"id": r["id"], "timestamp": r["timestamp"], "stage": clean_text(r["stage"], 50, redact=False),
                "action": clean_text(r["action"], 100, redact=False), "actor": clean_text(r["actor"], 100, redact=False),
                "duration_ms": r["duration_ms"], "details": sanitize_json(r["details"])} for r in kept]
    notes = (f"Only the latest {len(kept)} of {len(rows)} entries are shown.",) if len(rows) > len(kept) else ()
    return ToolResult({"invoice_id": a.invoice_id, "entries": entries}, found=bool(entries),
                      refs=[_ref("audit", e["id"]) for e in entries], tables=("invoices", "audit_log"),
                      untrusted=("entries.details",), entity=_ref("invoice", a.invoice_id), notes=notes)


def _review_queue(ctx: ToolContext, a: LimitArgs) -> ToolResult:
    items = [g.shape_invoice(d) for d in processing.review_queue(a.limit)]
    notes = () if items else ("The review queue is empty.",)
    return ToolResult({"returned": len(items), "ordering": "highest risk first", "invoices": items}, found=bool(items),
                      refs=[_ref("invoice", i["id"]) for i in items], tables=("invoices", "emails"),
                      untrusted=g.UNTRUSTED_INVOICE_FIELDS, notes=notes)


def _list_vendors(ctx: ToolContext, a: LimitArgs) -> ToolResult:
    rows = processing.vendors()
    keep = ("currency", "invoices", "total_spend", "pending", "auto_approved", "duplicates", "avg_risk", "last_invoice")
    out = [{"vendor": clean_text(r["vendor"], 200, redact=False), **{k: r.get(k) for k in keep}} for r in rows[: a.limit]]
    notes = ["total_spend excludes duplicates and rejected invoices."]
    if len(rows) > len(out):
        notes.append(f"Showing the top {len(out)} of {len(rows)} vendors by spend.")
    return ToolResult({"total_vendors": len(rows), "returned": len(out), "vendors": sanitize_json(out)}, found=bool(out),
                      tables=("invoices", "emails"), untrusted=("vendors.vendor",), notes=tuple(notes))


def _get_metrics(ctx: ToolContext, a: NoArgs) -> ToolResult:
    return ToolResult(sanitize_json(processing.metrics()), tables=("invoices", "emails", "audit_log"),
                      notes=("estimated_savings is an estimate from stated assumptions, not a measurement.",))


def _list_models(ctx: ToolContext, a: NoArgs) -> ToolResult:
    data = sanitize_json(registry.list_models(), max_depth=8, max_items=200)
    return ToolResult(data, found=bool(data.get("models")), tables=("model_versions", "audit_log"))


def _system_status(ctx: ToolContext, a: NoArgs) -> ToolResult:
    with SessionLocal() as s:
        s.execute(text("SELECT 1"))                                              # a real round trip: "ok" is measured
        data = {"database": "ok",
                "emails": s.query(func.count(Email.id)).scalar() or 0,
                "invoices": s.query(func.count(Invoice.id)).scalar() or 0,
                "invoices_pending_review": s.query(func.count(Invoice.id)).filter(Invoice.status == "pending_review").scalar() or 0,
                "emails_needing_triage": s.query(func.count(Email.id)).filter(Email.triage_status == "needs_triage").scalar() or 0,
                "audit_entries": s.query(func.count(AuditLog.id)).scalar() or 0,
                "production_models": [{"model": n, "version": v} for n, v in s.query(
                    ModelVersion.model_name, ModelVersion.version).filter(ModelVersion.status == "production")
                    .order_by(ModelVersion.model_name).all()]}
    return ToolResult(data, tables=("emails", "invoices", "audit_log", "model_versions"))


def _audit_log(ctx: ToolContext, a: AuditLogArgs) -> ToolResult:
    rows = processing.audit_feed(a.limit, a.stage or None, a.actor or None)
    entries = [{"id": r["id"], "timestamp": r["timestamp"], "entity": clean_text(r["entity"], 60, redact=False),
                "invoice_id": r["invoice_id"], "stage": clean_text(r["stage"], 50, redact=False),
                "action": clean_text(r["action"], 100, redact=False), "actor": clean_text(r["actor"], 100, redact=False),
                "duration_ms": r["duration_ms"]} for r in rows]
    return ToolResult({"filters": {"stage": a.stage, "actor": a.actor}, "returned": len(entries), "entries": entries},
                      found=bool(entries), refs=[_ref("audit", e["id"]) for e in entries], tables=("audit_log",),
                      notes=("Details are omitted here; use get_invoice_audit_trail for one invoice's full history.",))


def _get_settings(ctx: ToolContext, a: NoArgs) -> ToolResult:
    return ToolResult(sanitize_json(processing.public_settings()), tables=())    # an allow-listed view built in processing.py


def _list_users(ctx: ToolContext, a: UsersArgs) -> ToolResult:
    with SessionLocal() as s:
        rows = s.query(User).order_by(User.id).limit(a.limit).all()
        users = [user_dict(u) for u in rows]                                     # id, username, role, active, timestamps
    for u in users:
        u["username"] = clean_text(u["username"], 40, redact=False)
    return ToolResult({"returned": len(users), "users": users}, found=bool(users),
                      refs=[_ref("user", u["id"]) for u in users], tables=("users",),
                      notes=("Password hashes and session tokens are never exposed.",))


# ---------------------------------------------------------------- registry
TOOLS: tuple[Tool, ...] = (
    Tool("get_my_access", "Who am I and what can I call", "Shows the authenticated identity, its MCP role and the tools it may call.",
         NoArgs, _get_my_access, EVERYONE),
    Tool("search_invoices", "Search invoices",
         "Lists stored invoices, newest first, optionally filtered by status, system decision or vendor name. "
         "Fields that were not extracted are null and named in missing_fields.",
         SearchInvoicesArgs, _search_invoices, VIEWER_UP),
    Tool("get_invoice", "Get one invoice",
         "Returns one stored invoice with its decision, rule, reasons, risk breakdown and review status, or found=false if the id does not exist.",
         InvoiceIdArgs, _get_invoice, VIEWER_UP),
    Tool("get_invoice_audit_trail", "Invoice audit trail",
         "Every recorded pipeline stage, system decision and human action for one invoice, oldest first.",
         InvoiceIdArgs, _invoice_audit_trail, REVIEWER_UP),
    Tool("get_review_queue", "Review queue", "Invoices waiting for a human reviewer, highest risk first.",
         LimitArgs, _review_queue, REVIEWER_UP),
    Tool("list_vendors", "Vendor summary", "Per-vendor invoice counts, spend, pending items and average risk, highest spend first.",
         LimitArgs, _list_vendors, VIEWER_UP),
    Tool("get_metrics", "Processing metrics", "Counts by decision and status, automation rate, processing time and the savings estimate.",
         NoArgs, _get_metrics, EVERYONE),
    Tool("list_models", "Model registry", "Registered ML model versions with their status, recorded metrics and the data they were measured on.",
         NoArgs, _list_models, EVERYONE),
    Tool("get_system_status", "System status", "Database reachability, record counts, queue sizes and the production model versions.",
         NoArgs, _system_status, ADMIN_AND_SYSTEM),
    Tool("get_audit_log", "Global audit log", "Newest-first activity across the whole system, including MCP tool calls (stage 'mcp').",
         AuditLogArgs, _audit_log, ADMIN),
    Tool("get_settings", "Rules and thresholds", "The decision thresholds, risk weights and validation rules currently in force. Never includes secrets.",
         NoArgs, _get_settings, ADMIN),
    Tool("list_users", "List users", "User accounts with role and status. Never includes password hashes or tokens.",
         UsersArgs, _list_users, ADMIN),
)
