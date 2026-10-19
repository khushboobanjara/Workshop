import pytest

from app.review.rules import ReviewError, plan_review, status_for_decision

GOOD = {"vendor_name": "Acme", "invoice_number": "INV-12345", "total_amount": 100.0, "due_date": "2026-04-01"}


def test_status_for_decision():
    assert status_for_decision("auto_approved") == "approved"
    assert status_for_decision("human_review") == "pending_review"
    assert status_for_decision("rejected") == "rejected"
    assert status_for_decision("something_unexpected") == "pending_review"   # fails safe


def test_approve_and_reject_pending():
    assert plan_review("pending_review", "approve", GOOD) == "approved"
    assert plan_review("pending_review", "reject", {}) == "rejected"


@pytest.mark.parametrize("status", ["approved", "rejected", "new"])
def test_only_pending_can_be_reviewed(status):
    with pytest.raises(ReviewError) as e:
        plan_review(status, "approve", GOOD)
    assert e.value.kind == "conflict"


def test_cannot_approve_with_missing_or_blank_required_field():
    for bad in ({**GOOD, "due_date": None}, {**GOOD, "vendor_name": "   "}):
        with pytest.raises(ReviewError) as e:
            plan_review("pending_review", "approve", bad)
        assert e.value.kind == "invalid"


def test_cannot_approve_non_positive_amount():
    with pytest.raises(ReviewError):
        plan_review("pending_review", "approve", {**GOOD, "total_amount": 0})


def test_unknown_action():
    with pytest.raises(ReviewError):
        plan_review("pending_review", "maybe", GOOD)
