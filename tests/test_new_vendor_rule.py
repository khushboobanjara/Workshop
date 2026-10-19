from app.risk.decision import Thresholds, decide
from app.risk.engine import risk_score

INV = {"vendor_name": "Brand New Ltd", "invoice_number": "INV-1", "due_date": "2026-04-01"}


def run(amount, new_vendor):
    signals = {"new_vendor": 1.0 if new_vendor else 0.0, "anomaly": 0.15 if new_vendor else 0.0}
    score, contrib = risk_score(signals)
    return decide({**INV, "total_amount": amount}, signals, score, contrib, Thresholds())


def test_large_first_invoice_from_unknown_vendor_goes_to_a_human():
    d = run(5000.0, new_vendor=True)
    assert d.outcome == "human_review" and d.rule == "new_vendor_large_amount"


def test_small_first_invoice_from_unknown_vendor_can_pass():
    assert run(400.0, new_vendor=True).outcome == "auto_approved"


def test_large_invoice_from_known_vendor_is_unaffected():
    assert run(5000.0, new_vendor=False).outcome == "auto_approved"


def test_threshold_is_configurable():
    signals = {"new_vendor": 1.0}
    score, contrib = risk_score(signals)
    d = decide({**INV, "total_amount": 500.0}, signals, score, contrib, Thresholds(new_vendor_review_amount=100.0))
    assert d.rule == "new_vendor_large_amount"
