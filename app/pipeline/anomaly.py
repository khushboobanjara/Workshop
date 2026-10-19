from datetime import date, datetime
from pathlib import Path
from typing import Callable

from app.ml.versioning import file_version
from app.anomaly.burst import Activity, burst_signal
from app.anomaly.detector import AmountAnomalyDetector
from app.anomaly.scoring import score_amount
from app.config import settings
from app.pipeline.base import PipelineContext, Stage

AmountHistoryProvider = Callable[[str | None], list[float]]
ActivityProvider = Callable[[str | None], Activity]
COLD_START_SIGNAL = 0.15   # "unknown" is mildly risky, not safe


def db_amount_history(vendor: str | None) -> list[float]:
    from app.db import Invoice, SessionLocal
    with SessionLocal() as s:
        # Only approved, non-duplicate invoices define what is "normal" for a vendor:
        # pending or rejected anomalies must not teach the model that anomalies are normal.
        rows = (s.query(Invoice)
                .filter(Invoice.vendor_name == vendor, Invoice.status == "approved", Invoice.duplicate_of.is_(None))
                .all())
        return [r.total_amount for r in rows if r.total_amount is not None]


def db_activity(vendor: str | None) -> Activity:
    """Every earlier non-duplicate invoice of this vendor (any status: a burst counts even while
    its members are still pending) with its arrival date and whether it was approved."""
    from app.db import Email, Invoice, SessionLocal
    with SessionLocal() as s:
        rows = (s.query(Email.received_at, Invoice.status)
                .join(Invoice, Invoice.email_id == Email.id)
                .filter(Invoice.vendor_name == vendor, Invoice.duplicate_of.is_(None)).all())
        return [(r.received_at.date(), r.status == "approved") for r in rows]


def arrival_date(email: dict) -> date:
    """The email's arrival date: 'received' (ISO string) or 'received_at' (ISO string / datetime), else today."""
    v = email.get("received") or email.get("received_at")
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str) and v:
        return date.fromisoformat(v[:10])
    return date.today()


class AnomalyStage(Stage):
    name = "anomaly"

    def __init__(self, history_provider: AmountHistoryProvider = db_amount_history, model_path: str | None = None,
                 activity_provider: ActivityProvider = db_activity):
        self.history_provider = history_provider
        self.activity_provider = activity_provider
        path = model_path or settings.anomaly_model_path
        self.detector = AmountAnomalyDetector.load(path) if Path(path).exists() else None
        self.model_version = file_version(path)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        amount, vendor = ctx.invoice.get("total_amount"), ctx.invoice.get("vendor_name")
        if amount is None:
            ctx.risk_signals["anomaly"] = 0.0      # missing amount is already flagged upstream
            return ctx
        signal, reasons = score_amount(amount, self.history_provider(vendor), self.detector)
        # Cold start: too little history to judge the amount. The decision engine uses this to
        # send LARGE first invoices from unknown vendors to a human.
        ctx.risk_signals["new_vendor"] = 1.0 if signal is None else 0.0
        if signal is None:
            signal = COLD_START_SIGNAL
        ctx.risk_signals["anomaly"] = signal
        ctx.flags.extend(f"Anomaly: {r}" for r in reasons)

        if vendor:
            burst, burst_reasons = burst_signal(arrival_date(ctx.email), self.activity_provider(vendor))
            ctx.risk_signals["burst"] = burst or 0.0
            ctx.flags.extend(f"Anomaly: {r}" for r in burst_reasons)
        return ctx
