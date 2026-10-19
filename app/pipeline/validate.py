from app.pipeline.base import PipelineContext, Stage
from app.validation.rules import validate_invoice, validation_risk


class ValidateStage(Stage):
    name = "validate"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        issues = validate_invoice(ctx.invoice)
        for i in issues:
            ctx.flags.append(f"Validation: {i.message}")
        ctx.risk_signals["validation_issues"] = validation_risk(issues)
        return ctx
