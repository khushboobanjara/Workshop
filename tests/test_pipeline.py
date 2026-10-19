from app.db import AuditLog, SessionLocal, init_db
from app.pipeline.base import Pipeline, PipelineContext, Stage


class DummyStage(Stage):
    name = "dummy"

    def run(self, ctx):
        ctx.flags.append("hello")
        return ctx


def test_pipeline_writes_audit_log():
    init_db()
    Pipeline([DummyStage()]).run(PipelineContext(email={"subject": "x"}))
    with SessionLocal() as s:
        assert s.query(AuditLog).filter_by(stage="dummy").count() >= 1
