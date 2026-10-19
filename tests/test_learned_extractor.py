import json
from pathlib import Path

import pytest

from app.extraction.learned import (LearnedExtractor, build_candidates, normalize_number, parse_amount,
                                    parse_date)
from app.pipeline.base import PipelineContext
from app.pipeline.extract import ExtractStage

HARD = Path("data/synthetic_emails_hard.jsonl")


def test_parse_date_formats():
    assert parse_date("2026-03-01") == "2026-03-01"
    assert parse_date("01/03/2026") == "2026-03-01"            # DD/MM/YYYY
    assert parse_date("March 1, 2026") == "2026-03-01"
    assert parse_date("1 Mar 2026") == "2026-03-01"
    assert parse_date("31/02/2026") is None                     # impossible date is rejected


def test_parse_amount_us_and_european():
    assert parse_amount("1,234.56") == 1234.56
    assert parse_amount("1.234,56") == 1234.56
    assert parse_amount("1234.56") == 1234.56
    assert parse_amount("2,800.00") == 2800.0


def test_normalize_number():
    assert normalize_number("INV 12345") == "INV-12345"
    assert normalize_number("inv-12345") == "INV-12345"
    assert normalize_number("Inv #12345") == "INV-12345"
    assert normalize_number("#29711") == "#29711"
    assert normalize_number("SL-2026-1234") == "SL-2026-1234"


def test_candidates_find_every_amount_and_date_format():
    body = "Subtotal: $1,000.00\nTax: EUR 1.080,00\nTotal: 1,080.00 USD\nIssued: 1 Mar 2026\nDue: 15/03/2026"
    c = build_candidates("Invoice INV-12345", body)
    assert {x.value for x in c["amount"]} == {1000.0, 1080.0}
    assert {x.value for x in c["date"]} == {"2026-03-01", "2026-03-15"}
    assert {x.currency for x in c["amount"]} >= {"USD", "EUR"}


@pytest.fixture(scope="module")
def trained():
    if not HARD.exists():
        pytest.skip("run: python -m scripts.generate_hard_data")
    rows = [r for r in (json.loads(l) for l in HARD.read_text().splitlines()) if r["category"] == "invoice"]
    return LearnedExtractor().fit(rows[:700]), rows[700:900]


def test_learned_extractor_beats_chance_and_never_guesses_wrong(trained):
    ex, test = trained
    right = wrong = 0
    for r in test:
        got = ex({"subject": r["subject"], "body": r["body"]})["total_amount"]
        if got is not None:
            ok = abs(got.value - r["truth"]["total_amount"]) < 0.01
            right += ok; wrong += not ok
    assert right > 0.9 * len(test) and wrong <= 2


def test_abstains_when_information_is_absent(trained):
    ex, _ = trained
    got = ex({"subject": "Document attached", "body": "Hello,\n\nPlease find our invoice attached.\n\nRegards,\nJohn"})
    assert got["total_amount"] is None and got["invoice_number"] is None and got["due_date"] is None


def test_due_date_derived_from_payment_terms(trained):
    ex, _ = trained
    body = "Hello,\n\nInvoice No: INV-55555\nVendor: Acme Supplies Ltd\nInvoice Date: 2026-03-01\nPayment terms: Net 30\nTotal Amount: USD 100.00\n\nRegards,\nAcme Supplies Ltd"
    got = ex({"subject": "Invoice INV-55555", "body": body})
    assert got["invoice_date"].value == "2026-03-01" and got["due_date"].value == "2026-03-31"
    assert got["due_date"].source.startswith("derived")


def test_save_load_roundtrip_and_pipeline_integration(trained, tmp_path):
    ex, test = trained
    ex.save(str(tmp_path / "m.joblib"))
    loaded = LearnedExtractor.load(str(tmp_path / "m.joblib"))
    r = test[0]
    email = {"subject": r["subject"], "body": r["body"]}
    assert {k: (v.value if v else None) for k, v in loaded(email).items()} == {k: (v.value if v else None) for k, v in ex(email).items()}
    ctx = ExtractStage(loaded).run(PipelineContext(email=email))
    assert 0 <= ctx.invoice["extraction_confidence"] <= 1 and "extraction_uncertainty" in ctx.risk_signals
