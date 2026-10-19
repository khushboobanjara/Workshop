"""Feature engineering for amount anomalies. Pure numpy, no DB or pipeline imports.

Features compare an invoice to what the SAME vendor normally charges. History must
only contain invoices received BEFORE this one (no peeking at the future)."""
import numpy as np

MIN_HISTORY = 3          # below this we do not trust vendor stats
FEATURE_NAMES = ["log_ratio_to_median", "robust_z", "log_amount"]


def vendor_stats(amounts: list[float]) -> tuple[float, float]:
    """Median and MAD-based spread. Median/MAD (not mean/std) so past anomalies
    sitting in the history do not distort what 'normal' means."""
    a = np.asarray(amounts, dtype=float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    return med, max(mad, 0.05 * med, 1e-6)   # floor stops tiny spreads exploding the z-score


def featurize(amount: float, vendor_history: list[float]) -> np.ndarray | None:
    """Returns None when there is not enough vendor history (cold start)."""
    if amount is None or amount <= 0 or len(vendor_history) < MIN_HISTORY:
        return None
    med, spread = vendor_stats(vendor_history)
    return np.array([np.log(amount / med), (amount - med) / spread, np.log(amount)])


def robust_z(amount: float, vendor_history: list[float]) -> float | None:
    f = featurize(amount, vendor_history)
    return None if f is None else float(f[1])
