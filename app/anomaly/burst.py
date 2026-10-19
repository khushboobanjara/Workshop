"""Burst detection: is the vendor sending many invoices on one day compared with its own normal rate?
Pure functions (no DB, no pipeline imports). Uses arrival dates, because bursts are about when mail
arrives, not what the invoice says.

Honest limitation: busy vendors legitimately send several invoices a day, so count alone can never
separate a burst from natural clustering perfectly. This is a SOFT signal that trades some extra
reviews for catching the most surprising days."""
import math
from datetime import date

from scipy.stats import poisson

MIN_APPROVED = 10      # below this we do not trust the vendor's rate
MIN_SPAN_DAYS = 30     # rate is estimated over at least this many days
SIGNAL_START = 1.2     # -log10(p) at which the signal begins
SIGNAL_FULL = 2.2      # -log10(p) at which the signal reaches 1.0

Activity = list[tuple[date, bool]]   # (arrival date, approved?) of this vendor's earlier invoices


def burst_signal(arrival: date, prior: Activity) -> tuple[float | None, list[str]]:
    """Returns (signal 0..1 | None when the vendor has too little history, reasons)."""
    approved = [d for d, ok in prior if ok]
    if len(approved) < MIN_APPROVED:
        return None, []
    span = max(MIN_SPAN_DAYS, (max(approved) - min(approved)).days + 1)
    rate = len(approved) / span                          # normal invoices per day
    count_today = 1 + sum(1 for d, _ in prior if d == arrival)   # includes pending/rejected ones: they did arrive
    if count_today < 2:
        return 0.0, []
    p = float(poisson.sf(count_today - 1, rate))         # P(a normal day has at least this many)
    surprise = -math.log10(max(p, 1e-12))
    sig = min(1.0, max(0.0, (surprise - SIGNAL_START) / (SIGNAL_FULL - SIGNAL_START)))
    if sig == 0.0:
        return 0.0, []
    return round(sig, 4), [f"Invoice #{count_today} from this vendor today (it normally sends {rate:.2f}/day)"]
