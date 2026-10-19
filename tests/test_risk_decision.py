from app.pipeline.base import PipelineContext
from app.pipeline.decision import DecisionStage
from app.pipeline.risk import RiskStage
from app.risk.decision import Thresholds, decide
from app.risk.engine import risk_score

CLEAN_INV = {"vendor_name": "Acme", "invoice_number": "INV-12345", "total_amount": 1200.0, "due_date": "2026-04-01"}
ZERO = {"duplicate": 0.0, "anomaly": 0.0, "validation_issues": 0.0, "extraction_uncertainty": 0.0}


def d(inv=None, **sig):
    signals = {**ZERO, **sig}
    score, contrib = risk_score(signals)
    return decide({**CLEAN_INV, **(inv or {})}, signals, score, contrib, Thresholds())


def test_no_signals_means_zero_risk():
    assert risk_score(ZERO)[0] == 0.0


def test_single_strong_signal_is_not_diluted():
    assert risk_score({"anomaly": 1.0})[0] >= 0.8


def test_weak_signals_accumulate_but_stay_moderate():
    one = risk_score({"extraction_uncertainty": 0.2})[0]
    two = risk_score({"extraction_uncertainty": 0.2, "anomaly": 0.15})[0]
    assert 0 < one < two < 0.3


def test_signal_values_are_clamped_and_unknown_ignored():
    assert risk_score({"anomaly": 5.0, "mystery": 1.0})[0] == risk_score({"anomaly": 1.0})[0]


def test_clean_invoice_auto_approved():
    assert d(extraction_uncertainty=0.05).outcome == "auto_approved"


def test_exact_duplicate_rejected():
    r = d({"duplicate_of": 7}, duplicate=1.0)
    assert r.outcome == "rejected" and r.rule == "exact_duplicate"


def test_anomaly_goes_to_review_not_reject():
    assert d(anomaly=1.0).outcome == "human_review"


def test_fuzzy_duplicate_goes_to_review():
    assert d(duplicate=0.70).outcome == "human_review"


def test_missing_required_field_blocks_auto_approval_even_at_zero_risk():
    r = d({"due_date": None})
    assert r.outcome == "human_review" and r.rule == "missing_required_fields"


def test_severe_validation_blocks_auto_approval():
    assert d(validation_issues=0.7).rule == "severe_validation_issue"


def test_big_amount_needs_human_even_if_low_risk():
    r = d({"total_amount": 25_000.0})
    assert r.outcome == "human_review" and r.rule == "amount_above_auto_limit"


def test_stages_set_decision_and_reasons():
    ctx = PipelineContext(email={})
    ctx.invoice = dict(CLEAN_INV)
    ctx.risk_signals = {**ZERO, "anomaly": 1.0}
    ctx = DecisionStage(Thresholds()).run(RiskStage().run(ctx))
    assert ctx.decision == "human_review" and ctx.invoice["risk_score"] > 0.3
    assert any(f.startswith("Decision:") for f in ctx.flags)
