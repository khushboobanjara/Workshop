"""Risk Engine: combines the per-stage signals (each 0..1) into one 0..1 risk score.
Pure functions only (no DB, no pipeline imports) so they are easy to test and tune.

Method: weighted noisy-OR.   risk = 1 - prod(1 - weight_i * signal_i)
 - a single strong signal can push risk high (a plain weighted average would dilute it)
 - several weak signals add up gradually
 - weight_i is the most that signal can contribute on its own"""

WEIGHTS = {
    "duplicate": 0.95,
    "anomaly": 0.80,
    "burst": 0.60,
    "validation_issues": 0.80,
    "extraction_uncertainty": 0.60,
}


def risk_score(signals: dict[str, float], weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
    """Returns (score, contributions). Unknown signal names are ignored; missing ones count as 0."""
    weights = weights or WEIGHTS
    contributions = {k: round(w * min(1.0, max(0.0, signals.get(k, 0.0))), 4) for k, w in weights.items()}
    safe = 1.0
    for c in contributions.values():
        safe *= 1.0 - c
    return round(1.0 - safe, 4), contributions
