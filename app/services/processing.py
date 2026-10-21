"""Application service: runs the pipeline AND persists the outcome, plus the human-review
workflow, audit trail and metrics. The API layer stays a thin wrapper around this."""
import csv
import io
from collections import defaultdict
from datetime import timezone

from sqlalchemy import case, func

from app.db import AuditLog, Email, Invoice, SessionLocal, utcnow
from app.pipeline.base import Pipeline, PipelineContext
from app.review.rules import EDITABLE_FIELDS, ReviewError, plan_review, status_for_decision

# Cost-model assumptions for /metrics (estimates, not measurements)
MANUAL_MINUTES_PER_INVOICE = 6.0
HOURLY_COST = 30.0


class NotFound(Exception):
    pass


def _utc_naive(dt):
    """The DB stores naive UTC; convert aware datetimes instead of silently dropping their offset."""
    if dt is not None and dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _iso(dt):
    return dt.isoformat() if dt else None


def invoice_to_dict(inv: Invoice, email: Email | None = None) -> dict:
    d = {
        "id": inv.id, "email_id": inv.email_id,
        "vendor_name": inv.vendor_name, "invoice_number": inv.invoice_number,
        "invoice_date": inv.invoice_date, "due_date": inv.due_date, "currency": inv.currency,
        "total_amount": inv.total_amount, "po_number": inv.po_number,
        "extraction_confidence": inv.extraction_confidence,
        "risk_score": inv.risk_score, "risk_breakdown": inv.risk_breakdown,
        "decision": inv.decision, "decision_rule": inv.decision_rule, "decision_reasons": inv.decision_reasons,
        "status": inv.status, "duplicate_of": inv.duplicate_of, "flags": inv.flags,
        "reviewed_by": inv.reviewed_by, "review_comment": inv.review_comment, "reviewed_at": _iso(inv.reviewed_at),
        "created_at": _iso(inv.created_at),
    }
    if email is not None:
        d["email"] = {"subject": email.subject, "sender": email.sender, "body": email.body}
    return d


def _max_len(field: str) -> int | None:
    return getattr(Invoice.__table__.c[field].type, "length", None)


def _fit(value, field: str):
    """MySQL (unlike SQLite) rejects text longer than its column. Extracted text comes from untrusted email, so trim it
    to fit rather than fail the whole email with a database error."""
    n = _max_len(field)
    return value[:n] if isinstance(value, str) and n and len(value) > n else value


# ---------------------------------------------------------------- ingestion
def process_email(pipeline, raw: dict) -> dict:
    """Idempotent on message_id: re-sending the same email returns the stored result."""
    with SessionLocal() as s:
        email = s.query(Email).filter_by(message_id=raw["message_id"]).one_or_none()
        if email is not None:
            inv = s.query(Invoice).filter_by(email_id=email.id).one_or_none()
            if inv is not None or email.category is not None:
                return {"email_id": email.id, "message_id": email.message_id, "category": email.category,
                        "category_confidence": email.category_confidence,
                        "decision": inv.decision if inv else ("needs_triage" if email.triage_status else "not_invoice"),
                        "invoice": invoice_to_dict(inv) if inv else None,
                        "flags": inv.flags if inv else [], "already_processed": True}
            email_id = email.id            # stored earlier but never finished: process it now
            received_at = email.received_at
        else:
            email = Email(message_id=raw["message_id"], sender=raw.get("sender", ""),
                          subject=raw.get("subject", ""), body=raw.get("body", ""),
                          received_at=_utc_naive(raw.get("received_at")) or utcnow())
            s.add(email)
            s.commit()
            email_id = email.id
            received_at = email.received_at

    # Pipeline runs outside the session; it opens its own short sessions for audit rows.
    ctx = pipeline.run(PipelineContext(
        email={"subject": raw.get("subject", ""), "body": raw.get("body", ""), "sender": raw.get("sender", ""),
               "received_at": received_at},
        email_id=email_id))

    with SessionLocal() as s:
        email = s.get(Email, email_id)
        email.category = ctx.email.get("category")
        email.category_confidence = ctx.email.get("category_confidence")

        invoice = None
        decision = ctx.decision
        if not ctx.invoice and ctx.decision == "human_review":
            # Nothing was extracted because the system could not even tell what this email is (unsure classifier,
            # or a stage failed). That is a question about the EMAIL, not an invoice with missing fields.
            email.triage_status, decision = "needs_triage", "needs_triage"
        elif ctx.decision != "not_invoice":
            invoice = _store_invoice(s, email_id, ctx)
        s.commit()
        return {"email_id": email_id, "message_id": email.message_id, "category": email.category,
                "category_confidence": email.category_confidence,
                "decision": invoice.decision if invoice else decision,
                "invoice": invoice_to_dict(invoice) if invoice else None,
                "flags": list(ctx.flags), "already_processed": False}


def _store_invoice(s, email_id: int, ctx: PipelineContext) -> Invoice:
    inv = ctx.invoice
    # Empty invoice dict = classifier was unsure, so nothing was extracted. We still create a row
    # so the email lands in the review queue instead of silently disappearing.
    outcome = ctx.decision or "human_review"                      # no decision made -> fail safe
    rule = inv.get("decision_rule") or ("low_classification_confidence" if not inv else "no_decision_made")
    reasons = inv.get("decision_reasons") or list(ctx.flags)
    dup = inv.get("duplicate_of")

    row = Invoice(
        email_id=email_id,
        vendor_name=_fit(inv.get("vendor_name"), "vendor_name"),
        invoice_number=_fit(inv.get("invoice_number"), "invoice_number"),
        invoice_date=_fit(inv.get("invoice_date"), "invoice_date"), due_date=_fit(inv.get("due_date"), "due_date"),
        currency=_fit(inv.get("currency"), "currency"),
        total_amount=inv.get("total_amount"), po_number=_fit(inv.get("po_number"), "po_number"),
        extraction_confidence=inv.get("extraction_confidence"),
        risk_score=inv.get("risk_score"), risk_breakdown=inv.get("risk_breakdown"),
        decision=outcome, status=status_for_decision(outcome), decision_rule=_fit(rule, "decision_rule"),
        decision_reasons=reasons, flags=list(ctx.flags),
        duplicate_of=dup if isinstance(dup, int) else None,
    )
    s.add(row)
    s.flush()                                                     # assigns row.id
    s.add(AuditLog(entity_type="invoice", entity_id=row.id, stage="decision", action=outcome, actor="system",
                   details={"rule": rule, "risk_score": row.risk_score, "reasons": reasons,
                            "risk_breakdown": row.risk_breakdown, "status": row.status}))
    return row


# ---------------------------------------------------------------- queries
def review_queue(limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        # Unknown risk (NULL) sorts as highest priority.
        rows = (s.query(Invoice).filter(Invoice.status == "pending_review")
                .order_by(func.coalesce(Invoice.risk_score, 1.0).desc(), Invoice.created_at).limit(limit).all())
        return [invoice_to_dict(r, s.get(Email, r.email_id)) for r in rows]


def list_invoices(status: str | None = None, decision: str | None = None, limit: int = 100) -> list[dict]:
    with SessionLocal() as s:
        q = s.query(Invoice)
        if status:
            q = q.filter(Invoice.status == status)
        if decision:
            q = q.filter(Invoice.decision == decision)
        return [invoice_to_dict(r) for r in q.order_by(Invoice.id.desc()).limit(limit).all()]


def get_invoice(invoice_id: int) -> dict:
    with SessionLocal() as s:
        inv = s.get(Invoice, invoice_id)
        if inv is None:
            raise NotFound(f"Invoice {invoice_id} not found")
        return invoice_to_dict(inv, s.get(Email, inv.email_id))


def get_audit(invoice_id: int) -> list[dict]:
    """Full story of one invoice: every pipeline stage, the system decision, and human actions."""
    with SessionLocal() as s:
        inv = s.get(Invoice, invoice_id)
        if inv is None:
            raise NotFound(f"Invoice {invoice_id} not found")
        rows = (s.query(AuditLog)
                .filter(((AuditLog.entity_type == "invoice") & (AuditLog.entity_id == invoice_id)) |
                        ((AuditLog.entity_type == "email") & (AuditLog.entity_id == inv.email_id)))
                .order_by(AuditLog.timestamp, AuditLog.id).all())
        return [{"id": r.id, "timestamp": _iso(r.timestamp), "entity": f"{r.entity_type}:{r.entity_id}",
                 "stage": r.stage, "action": r.action, "actor": r.actor,
                 "duration_ms": r.duration_ms, "details": r.details} for r in rows]


# ---------------------------------------------------------------- human review
def review_invoice(invoice_id: int, action: str, reviewer: str, comment: str = "",
                   corrections: dict | None = None) -> dict:
    with SessionLocal() as s:
        # FOR UPDATE: a second reviewer (or a double click) waits here and then sees the new status, so an invoice can
        # never be approved and rejected at once. SQLite ignores the lock; MySQL/InnoDB holds it until commit.
        inv = s.get(Invoice, invoice_id, with_for_update=True)
        if inv is None:
            raise NotFound(f"Invoice {invoice_id} not found")

        candidate = {f: getattr(inv, f) for f in EDITABLE_FIELDS}
        changes = {}
        for k, v in (corrections or {}).items():
            if k not in EDITABLE_FIELDS:
                raise ReviewError(f"Field {k!r} cannot be corrected", "invalid")
            if isinstance(v, str):
                v = v.strip().upper() if k == "currency" else v.strip()
            limit = _max_len(k)
            if isinstance(v, str) and limit and len(v) > limit:
                raise ReviewError(f"Field {k!r} is too long (maximum {limit} characters)", "invalid")
            if v != candidate[k]:
                changes[k] = {"from": candidate[k], "to": v}
                candidate[k] = v

        new_status = plan_review(inv.status, action, candidate)   # may raise; nothing is modified yet

        previous_status = inv.status
        for k, ch in changes.items():
            setattr(inv, k, ch["to"])
        inv.status, inv.reviewed_by, inv.review_comment, inv.reviewed_at = new_status, reviewer, comment, utcnow()
        s.add(AuditLog(entity_type="invoice", entity_id=inv.id, stage="human_review", action=new_status,
                       actor=reviewer,
                       details={"comment": comment, "corrections": changes, "previous_status": previous_status,
                                "system_decision": inv.decision, "risk_score": inv.risk_score}))
        s.commit()
        return invoice_to_dict(inv, s.get(Email, inv.email_id))


# ---------------------------------------------------------------- metrics
def metrics() -> dict:
    with SessionLocal() as s:
        total = s.query(func.count(Invoice.id)).scalar() or 0
        by_decision = dict(s.query(Invoice.decision, func.count(Invoice.id)).group_by(Invoice.decision).all())
        by_status = dict(s.query(Invoice.status, func.count(Invoice.id)).group_by(Invoice.status).all())
        emails = s.query(func.count(Email.id)).scalar() or 0

        per_email = defaultdict(float)
        for entity_id, ms in s.query(AuditLog.entity_id, AuditLog.duration_ms).filter(
                AuditLog.entity_type == "email", AuditLog.duration_ms.isnot(None)).all():
            per_email[entity_id] += ms

    auto = by_decision.get("auto_approved", 0)
    hours = auto * MANUAL_MINUTES_PER_INVOICE / 60
    return {
        "emails_received": emails, "invoices": total,
        "by_decision": by_decision, "by_status": by_status,
        "automation_rate": round(auto / total, 4) if total else None,
        "pending_review": by_status.get("pending_review", 0),
        "avg_pipeline_ms_per_email": round(sum(per_email.values()) / len(per_email), 2) if per_email else None,
        "estimated_savings": {
            "hours": round(hours, 2), "cost": round(hours * HOURLY_COST, 2),
            "assumptions": f"{MANUAL_MINUTES_PER_INVOICE:g} min manual effort per invoice at {HOURLY_COST:g}/hour",
            "note": "estimate, not a measurement"},
    }


HIGH_RISK = 0.60


def stats() -> dict:
    """Everything the dashboard's home page needs in one call (so the page makes one request, not six)."""
    m = metrics()
    with SessionLocal() as s:
        rows = s.query(Invoice.decision, Invoice.status, Invoice.duplicate_of, Invoice.risk_score).all()
        categories = dict(s.query(Email.category, func.count(Email.id)).group_by(Email.category).all())
        day = func.date(Email.received_at)
        timeline = (s.query(day, func.count(Invoice.id), func.sum(case((Invoice.decision == "auto_approved", 1), else_=0)))
                    .join(Email, Email.id == Invoice.email_id).group_by(day).order_by(day).all())
        recent = [invoice_to_dict(r) for r in s.query(Invoice).order_by(Invoice.id.desc()).limit(6).all()]

    # Mutually exclusive slices that add up to the total (a duplicate is counted only as a duplicate)
    donut = {"auto_processed": 0, "approved_by_reviewer": 0, "pending_review": 0, "rejected": 0, "duplicates": 0}
    for decision, status, dup, _ in rows:
        if dup is not None:
            donut["duplicates"] += 1
        elif decision == "auto_approved":
            donut["auto_processed"] += 1
        elif status == "approved":
            donut["approved_by_reviewer"] += 1
        elif status == "pending_review":
            donut["pending_review"] += 1
        else:
            donut["rejected"] += 1
    with SessionLocal() as s:
        needs_triage = s.query(func.count(Email.id)).filter(Email.triage_status == "needs_triage").scalar()
    return {**m, "needs_triage": needs_triage, "donut": donut, "duplicates": donut["duplicates"],
            "high_risk": sum(1 for r in rows if (r[3] or 0) >= HIGH_RISK), "high_risk_threshold": HIGH_RISK,
            "categories": {(k or "unclassified"): v for k, v in categories.items()},
            "timeline": [{"date": str(d), "invoices": n, "auto_approved": int(a or 0)} for d, n, a in timeline],
            "recent": recent}


# ---------------------------------------------------------------- vendors, audit feed, export, settings
def vendors() -> list[dict]:
    """One row per vendor. Spend excludes duplicates and rejected invoices, so it is money actually owed."""
    agg: dict[str, dict] = {}
    with SessionLocal() as s:
        rows = (s.query(Invoice.vendor_name, Invoice.currency, Invoice.total_amount, Invoice.risk_score, Invoice.decision,
                        Invoice.status, Invoice.duplicate_of, Email.received_at)
                .join(Email, Email.id == Invoice.email_id).all())
    for name, cur, amt, risk, decision, status, dup, received in rows:
        a = agg.setdefault(name or "Unknown vendor", {"vendor": name or "Unknown vendor", "currency": cur, "invoices": 0,
                           "total_spend": 0.0, "pending": 0, "auto_approved": 0, "duplicates": 0, "_risk": [], "last_invoice": None})
        if cur and a["currency"] and cur != a["currency"]:
            a["currency"] = "MIXED"            # spend below adds amounts of different currencies: do not label it as one of them
        elif cur and not a["currency"]:
            a["currency"] = cur
        a["invoices"] += 1
        a["_risk"].append(risk or 0.0)
        if dup is not None:
            a["duplicates"] += 1
        elif status != "rejected":
            a["total_spend"] += amt or 0.0
        a["pending"] += status == "pending_review"
        a["auto_approved"] += decision == "auto_approved"
        if received and (a["last_invoice"] is None or received > a["last_invoice"]):
            a["last_invoice"] = received
    out = []
    for a in agg.values():
        risks = a.pop("_risk")
        a.update(avg_risk=round(sum(risks) / len(risks), 3), total_spend=round(a["total_spend"], 2),
                 last_invoice=_iso(a["last_invoice"]))
        out.append(a)
    return sorted(out, key=lambda a: -a["total_spend"])


def audit_feed(limit: int = 200, stage: str | None = None, actor: str | None = None) -> list[dict]:
    """Newest-first activity across all invoices (pipeline stages and human actions)."""
    with SessionLocal() as s:
        q = s.query(AuditLog)
        if stage:
            q = q.filter(AuditLog.stage == stage)
        if actor:
            q = q.filter(AuditLog.actor == actor)
        rows = q.order_by(AuditLog.id.desc()).limit(limit).all()
        inv_by_email = {e: i for e, i in s.query(Invoice.email_id, Invoice.id).filter(
            Invoice.email_id.in_({r.entity_id for r in rows if r.entity_type == "email"})).all()}
    return [{"id": r.id, "timestamp": _iso(r.timestamp), "entity": f"{r.entity_type}:{r.entity_id}",
             "invoice_id": r.entity_id if r.entity_type == "invoice" else inv_by_email.get(r.entity_id),
             "stage": r.stage, "action": r.action, "actor": r.actor, "duration_ms": r.duration_ms} for r in rows]


def _csv_safe(v):
    """Spreadsheet apps execute cells starting with = + - @ as formulas, and vendor names come from
    untrusted email, so defuse them with a leading apostrophe (OWASP CSV-injection guidance)."""
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v


CSV_COLUMNS = ["id", "invoice_number", "vendor_name", "invoice_date", "due_date", "currency", "total_amount", "po_number",
               "risk_score", "decision", "status", "duplicate_of", "reviewed_by", "created_at"]


def invoices_csv() -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for inv in list_invoices(limit=100_000):
        w.writerow([_csv_safe(inv.get(c)) for c in CSV_COLUMNS])
    return buf.getvalue()


def public_settings() -> dict:
    """Read-only view of the rules. A whitelist on purpose: never dump the whole settings object (it holds secrets)."""
    from app.config import settings as cfg
    from app.risk.engine import WEIGHTS
    from app.validation import rules
    keys = ["auto_approve_max_risk", "auto_reject_min_risk", "auto_approve_max_amount", "new_vendor_review_amount",
            "classifier_min_confidence"]
    return {"thresholds": {k: getattr(cfg, k) for k in keys}, "risk_weights": WEIGHTS, "high_risk_at": HIGH_RISK,
            "validation_rules": {"max_payment_terms_days": rules.MAX_PAYMENT_TERMS_DAYS,
                                 "max_reasonable_amount": rules.MAX_REASONABLE_AMOUNT,
                                 "allowed_currencies": ", ".join(sorted(rules.ALLOWED_CURRENCIES))},
            "savings_assumptions": {"minutes_per_invoice": MANUAL_MINUTES_PER_INVOICE, "hourly_cost": HOURLY_COST},
            "demo_mode": cfg.demo_mode, "auth_required": bool(cfg.api_key.get_secret_value())}


# ---------------------------------------------------------------- triage: emails the system could not classify
def triage_list(limit: int = 200) -> list[dict]:
    with SessionLocal() as s:
        rows = (s.query(Email).filter(Email.triage_status == "needs_triage")
                .order_by(Email.id.desc()).limit(limit).all())
        return [{"email_id": e.id, "sender": e.sender, "subject": e.subject, "preview": (e.body or "")[:1500],
                 "received_at": _iso(e.received_at), "predicted_category": e.category,
                 "confidence": e.category_confidence} for e in rows]


def resolve_triage(pipeline, email_id: int, action: str, reviewer: str, comment: str = "") -> dict:
    """dismiss = 'not an invoice'. convert = 'it is an invoice': run everything after the classifier and store the
    invoice, which then goes through the normal rules and review like any other."""
    with SessionLocal() as s:
        email = s.get(Email, email_id, with_for_update=True)        # lock: two people cannot convert the same email twice
        if email is None:
            raise NotFound(f"Email {email_id} not found")
        if email.triage_status != "needs_triage":
            raise ReviewError("This email is not waiting for triage", "conflict")
        email.triage_status = "dismissed" if action == "dismiss" else "converted"   # claim it before any slow work
        s.add(AuditLog(entity_type="email", entity_id=email_id, stage="triage",
                       action="dismissed_not_invoice" if action == "dismiss" else "confirmed_invoice",
                       actor=reviewer, details={"comment": comment, "classifier_confidence": email.category_confidence}))
        s.commit()
        if action == "dismiss":
            return {"email_id": email_id, "triage_status": "dismissed", "invoice": None}
        raw = {"subject": email.subject, "body": email.body, "sender": email.sender, "received_at": email.received_at,
               "category": "invoice", "category_confidence": email.category_confidence}

    try:
        rest = Pipeline([st for st in pipeline.stages if st.name != "classify"])
        ctx = rest.run(PipelineContext(email=raw, email_id=email_id))
        with SessionLocal() as s:
            email = s.get(Email, email_id)
            email.category = "invoice"
            invoice = _store_invoice(s, email_id, ctx)
            s.commit()
            return {"email_id": email_id, "triage_status": "converted", "invoice": invoice_to_dict(invoice)}
    except Exception:
        with SessionLocal() as s:                                   # give the email back so a person can retry
            s.get(Email, email_id).triage_status = "needs_triage"
            s.commit()
        raise
