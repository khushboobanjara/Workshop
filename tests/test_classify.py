import pytest

from app.config import settings
from app.db import init_db
from app.pipeline.base import Pipeline, PipelineContext
from app.pipeline.classify import ClassifyStage

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(settings.classifier_model_path).exists(),
    reason="train the model first: python -m app.ml.train_classifier",
)


def run(subject, body):
    init_db()
    ctx = PipelineContext(email={"subject": subject, "body": body})
    return Pipeline([ClassifyStage()]).run(ctx)


def test_invoice_continues():
    ctx = run("Invoice INV-12345", "Please find invoice INV-12345 attached.\nTotal Amount: USD 950.00\nDue Date: 2026-05-01")
    assert ctx.email["category"] == "invoice" and not ctx.stop


def test_newsletter_stops():
    ctx = run("Our monthly newsletter", "Product updates you'll love. Read our latest news.")
    assert ctx.stop and ctx.decision in ("not_invoice", "human_review")
