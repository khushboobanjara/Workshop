from typing import Callable

from app.duplicates.matcher import find_duplicate
from app.pipeline.base import PipelineContext, Stage

HistoryProvider = Callable[[dict], list[dict]]


def db_history(inv: dict) -> list[dict]:
    """Default provider: previously stored invoices (candidates narrowed in SQL later for scale)."""
    from app.db import Invoice, SessionLocal
    with SessionLocal() as s:
        return [dict(id=r.id, vendor_name=r.vendor_name, invoice_number=r.invoice_number,
                     invoice_date=r.invoice_date, total_amount=r.total_amount)
                for r in s.query(Invoice).all()]


class DuplicateStage(Stage):
    name = "duplicate_check"

    def __init__(self, history_provider: HistoryProvider = db_history):
        self.history_provider = history_provider

    def run(self, ctx: PipelineContext) -> PipelineContext:
        match = find_duplicate(ctx.invoice, self.history_provider(ctx.invoice))
        ctx.risk_signals["duplicate"] = match.score if match else 0.0
        if match:
            ctx.invoice["duplicate_of"] = match.matched_id
            ctx.flags.append(f"Possible duplicate of invoice {match.matched_id} ({match.kind}: {match.reason})")
        return ctx
