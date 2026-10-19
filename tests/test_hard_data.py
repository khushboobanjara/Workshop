import json
from pathlib import Path

import pytest

from app.extraction.oracle import fields_for, oracle_extractor, truth_fields
from app.pipeline.base import PipelineContext
from app.pipeline.extract import ExtractStage

HARD = Path("data/synthetic_emails_hard.jsonl")
needs_data = pytest.mark.skipif(not HARD.exists(), reason="run: python -m scripts.generate_hard_data")


def load():
    return [json.loads(l) for l in HARD.read_text().splitlines()]


@needs_data
def test_hard_data_schema_and_chronological_order():
    rows = load()
    assert len(rows) > 1000
    assert all({"message_id", "subject", "body", "category", "is_duplicate", "is_anomaly", "truth", "received"} <= set(r) for r in rows)
    assert all(rows[i]["received"] <= rows[i + 1]["received"] for i in range(len(rows) - 1))
    assert len({r["message_id"] for r in rows}) == len(rows)


@needs_data
def test_every_duplicate_and_anomaly_kind_is_present():
    rows = load()
    assert {r["dup_kind"] for r in rows if r["dup_kind"]} >= {"exact", "number_fmt", "vendor_variant", "amount_rounded", "resubmitted"}
    assert {r["anomaly_kind"] for r in rows if r["anomaly_kind"]} >= {"spike_mild", "spike_big", "burst", "new_vendor"}


@needs_data
def test_twins_are_legitimate_not_duplicates():
    twins = [r for r in load() if r["twin"]]
    assert twins and all(not r["is_duplicate"] and not r["is_anomaly"] for r in twins)


@needs_data
def test_invoice_rows_carry_full_truth_and_others_do_not():
    for r in load():
        assert (r["truth"] is not None) == (r["category"] == "invoice")
        if r["truth"]:
            assert r["truth"]["vendor_name"] and r["truth"]["total_amount"] > 0 and r["truth"]["due_date"]


def test_truth_fields_wraps_values_and_skips_none():
    f = truth_fields({"vendor_name": "Acme", "invoice_number": "INV-12345", "invoice_date": "2026-01-01",
                      "due_date": "2026-01-31", "currency": "USD", "total_amount": 10.0, "po_number": None})
    assert f["vendor_name"].value == "Acme" and f["po_number"] is None
    assert fields_for({"truth": {"vendor_name": "X"}}, oracle=True)["vendor_name"].value == "X"


def test_extract_stage_accepts_a_custom_extractor():
    truth = {"vendor_name": "Acme", "invoice_number": "INV-12345", "invoice_date": "2026-01-01",
             "due_date": "2026-01-31", "currency": "USD", "total_amount": 10.0, "po_number": None}
    ctx = ExtractStage(oracle_extractor).run(PipelineContext(email={"subject": "x", "body": "unparseable", "_truth": truth}))
    assert ctx.invoice["total_amount"] == 10.0 and not any("Missing" in f for f in ctx.flags)


def test_default_extract_stage_is_unchanged():
    ctx = ExtractStage().run(PipelineContext(email={"subject": "Invoice INV-12345",
                             "body": "Vendor: Acme\nDue Date: 2026-01-31\nTotal Amount: USD 10.00"}))
    assert ctx.invoice["vendor_name"] == "Acme" and ctx.invoice["total_amount"] == 10.0
