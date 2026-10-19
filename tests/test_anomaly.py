import numpy as np

from app.anomaly.detector import AmountAnomalyDetector
from app.anomaly.features import featurize
from app.anomaly.scoring import score_amount

HIST = [1180, 1250, 1300, 1210, 1275, 1190, 1230, 1260]


def make_detector():
    rng = np.random.default_rng(0)
    X = []
    for _ in range(300):
        h = list(rng.normal(1200, 150, 8))
        X.append(featurize(float(rng.normal(1200, 150)), h))
    return AmountAnomalyDetector().fit(np.array(X))


def test_cold_start_returns_none():
    sig, reasons = score_amount(500.0, [100.0, 120.0], None)
    assert sig is None and reasons


def test_normal_amount_low_signal():
    sig, _ = score_amount(1240.0, HIST, make_detector())
    assert sig < 0.3


def test_ten_x_amount_high_signal_with_reason():
    sig, reasons = score_amount(12400.0, HIST, make_detector())
    assert sig >= 0.9 and any("typical invoice" in r for r in reasons)


def test_works_without_trained_model():
    sig, _ = score_amount(9000.0, HIST, None)
    assert sig >= 0.9


def test_past_anomaly_in_history_does_not_hide_new_one():
    polluted = HIST + [12000.0]          # median/MAD should shrug this off
    sig, _ = score_amount(11000.0, polluted, None)
    assert sig >= 0.9
