from app.risk.engine import risk_score
from app.pipeline.base import PipelineContext, Stage


class RiskStage(Stage):
    name = "risk"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        score, contributions = risk_score(ctx.risk_signals)
        ctx.invoice["risk_score"] = score
        ctx.invoice["risk_breakdown"] = contributions
        return ctx
