from app.config import settings
from app.pipeline.base import PipelineContext, Stage
from app.risk.decision import Thresholds, decide


def thresholds_from_settings() -> Thresholds:
    return Thresholds(
        auto_approve_max_risk=settings.auto_approve_max_risk,
        auto_reject_min_risk=settings.auto_reject_min_risk,
        auto_approve_max_amount=settings.auto_approve_max_amount,
        new_vendor_review_amount=settings.new_vendor_review_amount,
    )


class DecisionStage(Stage):
    name = "decision"

    def __init__(self, thresholds: Thresholds | None = None):
        self.t = thresholds or thresholds_from_settings()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        d = decide(ctx.invoice, ctx.risk_signals, ctx.invoice.get("risk_score", 0.0),
                   ctx.invoice.get("risk_breakdown", {}), self.t)
        ctx.decision = d.outcome
        ctx.invoice["decision"] = d.outcome
        ctx.invoice["decision_rule"] = d.rule
        ctx.invoice["decision_reasons"] = d.reasons
        ctx.flags.extend(f"Decision: {r}" for r in d.reasons)
        return ctx
