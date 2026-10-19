from datetime import date, datetime, timedelta

from app.anomaly.burst import MIN_APPROVED, burst_signal
from app.pipeline.anomaly import AnomalyStage, arrival_date
from app.pipeline.base import PipelineContext

TODAY = date(2026, 3, 10)


def history(n: int, span_days: int, approved=True):
    """n invoices spread evenly over span_days, all before TODAY."""
    return [(TODAY - timedelta(days=1 + (i * span_days) // n), approved) for i in range(n)]


def test_too_little_history_cannot_judge():
    sig, reasons = burst_signal(TODAY, history(MIN_APPROVED - 1, 60))
    assert sig is None and reasons == []


def test_first_invoice_of_the_day_is_not_a_burst():
    sig, _ = burst_signal(TODAY, history(20, 100))
    assert sig == 0.0


def test_second_invoice_from_a_quiet_vendor_is_suspicious():
    prior = history(12, 120) + [(TODAY, True)]            # ~0.1/day vendor, already one today
    sig, reasons = burst_signal(TODAY, prior)
    assert sig == 1.0 and "#2" in reasons[0]


def test_same_count_from_a_busy_vendor_is_normal():
    prior = history(240, 120) + [(TODAY, True)]           # ~2/day vendor, second one today is routine
    sig, _ = burst_signal(TODAY, prior)
    assert sig == 0.0


def test_pending_and_rejected_invoices_still_count_toward_the_burst():
    approved = history(12, 120)
    sig_pending, _ = burst_signal(TODAY, approved + [(TODAY, False)])
    sig_none, _ = burst_signal(TODAY, approved)
    assert sig_pending > sig_none == 0.0


def test_unapproved_history_does_not_define_the_normal_rate():
    # 200 pending invoices must not make a burst look normal
    prior = history(12, 120) + history(200, 120, approved=False) + [(TODAY, False)]
    sig, _ = burst_signal(TODAY, prior)
    assert sig > 0


def test_arrival_date_accepts_strings_and_datetimes():
    assert arrival_date({"received": "2026-03-10"}) == TODAY
    assert arrival_date({"received_at": datetime(2026, 3, 10, 14, 30)}) == TODAY
    assert arrival_date({"received_at": "2026-03-10T14:30:00+00:00"}) == TODAY
    assert arrival_date({}) == date.today()


def make_stage(prior):
    return AnomalyStage(history_provider=lambda v: [1200.0, 1250.0, 1180.0, 1220.0],
                        model_path="does-not-exist.joblib", activity_provider=lambda v: prior)


def ctx(amount=1210.0, vendor="Acme"):
    return PipelineContext(email={"received": TODAY.isoformat()},
                           invoice={"total_amount": amount, "vendor_name": vendor})


def test_stage_sets_burst_signal_and_flag():
    out = make_stage(history(12, 120) + [(TODAY, True)]).run(ctx())
    assert out.risk_signals["burst"] == 1.0
    assert any("from this vendor today" in f for f in out.flags)


def test_stage_marks_cold_start_vendor_as_new():
    stage = AnomalyStage(history_provider=lambda v: [900.0], model_path="does-not-exist.joblib",
                         activity_provider=lambda v: [])
    out = stage.run(ctx())
    assert out.risk_signals["new_vendor"] == 1.0 and out.risk_signals["burst"] == 0.0


def test_stage_known_vendor_is_not_new():
    out = make_stage([]).run(ctx())
    assert out.risk_signals["new_vendor"] == 0.0
