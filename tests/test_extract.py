import json
from pathlib import Path

import pytest

from app.config import settings
from app.pipeline.base import Pipeline, PipelineContext
from app.pipeline.extract import ExtractStage

STANDARD = """Hello,

Please find invoice INV-12345 attached.
Vendor: Acme Supplies Ltd
Invoice Date: 2026-03-01
Due Date: 2026-03-31
PO Number: PO-4821
Total Amount: USD 1,250.50

Regards,
Acme Supplies Ltd"""

TERSE = """Hi,

Attached is the bill for this month.
Ref INV-12345, amount due USD 1,250.50 by 2026-03-31.

Thanks,
Acme Supplies Ltd"""


def run(subject, body):
    return ExtractStage().run(PipelineContext(email={"subject": subject, "body": body}))


def test_standard_email_fully_extracted():
    inv = run("Invoice INV-12345", STANDARD).invoice
    assert inv["invoice_number"] == "INV-12345"
    assert inv["total_amount"] == 1250.50 and inv["currency"] == "USD"
    assert inv["due_date"] == "2026-03-31" and inv["po_number"] == "PO-4821"
    assert inv["vendor_name"] == "Acme Supplies Ltd"


def test_terse_email_flags_missing_invoice_date():
    ctx = run("Document attached", TERSE)
    assert ctx.invoice["total_amount"] == 1250.50
    assert ctx.invoice["invoice_date"] is None
    assert any("invoice_date" in f for f in ctx.flags)
    assert ctx.risk_signals["extraction_uncertainty"] > 0


def test_conflicting_invoice_numbers_lower_confidence():
    ctx = run("Invoice INV-99999", STANDARD)
    assert ctx.invoice["field_confidence"]["invoice_number"] == 0.50
    assert any("invoice_number" in f for f in ctx.flags)


@pytest.mark.skipif(not Path(settings.classifier_model_path).exists(), reason="train classifier first")
def test_classify_then_extract_end_to_end():
    from app.db import init_db
    from app.pipeline.classify import ClassifyStage
    init_db()
    row = next(json.loads(l) for l in Path("data/synthetic_emails.jsonl").read_text().splitlines()
               if json.loads(l)["category"] == "invoice")
    ctx = Pipeline([ClassifyStage(), ExtractStage()]).run(
        PipelineContext(email={"subject": row["subject"], "body": row["body"]}))
    assert ctx.invoice["invoice_number"] == row["truth"]["invoice_number"]
