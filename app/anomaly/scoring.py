"""Combine the statistical rule and the Isolation Forest into one 0..1 risk signal.
Pure function: easy to unit test, no DB or pipeline imports."""
from app.anomaly.detector import AmountAnomalyDetector
from app.anomaly.features import featurize


def _z_signal(z: float) -> float:
    return min(1.0, max(0.0, (abs(z) - 2.0) / 4.0))        # 0 at |z|<=2, 1 at |z|>=6


def _if_signal(pct: float) -> float:
    return min(1.0, max(0.0, (pct - 0.90) / 0.10))         # only the top 10% tail counts


def score_amount(amount, vendor_history: list[float], detector: AmountAnomalyDetector | None):
    """Returns (signal | None, reasons). None means 'cannot judge' (cold start)."""
    feats = featurize(amount, vendor_history)
    if feats is None:
        return None, ["Not enough vendor history to judge amount"]
    z = float(feats[1])
    sig, reasons = _z_signal(z), []
    if sig > 0:
        ratio = float(2.718281828 ** feats[0])
        reasons.append(f"Amount is {ratio:.1f}x the vendor's typical invoice (robust z={z:+.1f})")
    if detector is not None:
        isig = _if_signal(detector.signal(feats))
        if isig > 0:
            reasons.append(f"Isolation Forest rates this amount pattern unusual ({isig:.2f})")
        sig = max(sig, isig)
    return round(sig, 4), reasons
