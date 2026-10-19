"""Pure duplicate matching. `history` is a list of dicts with keys:
id, vendor_name, invoice_number, invoice_date, total_amount."""
import re
from dataclasses import dataclass
from datetime import timedelta

from rapidfuzz import fuzz

from app.validation.rules import parse_date

VENDOR_SIM_MIN = 90      # 0..100, token-sort ratio
DATE_WINDOW_DAYS = 3
AMOUNT_REL_TOL = 0.005   # 0.5%


@dataclass
class DuplicateMatch:
    matched_id: object
    kind: str      # exact | fuzzy | resubmitted
    score: float   # 0..1 -> becomes a risk signal
    reason: str


def norm_vendor(name: str | None) -> str:
    name = re.sub(r"[^\w\s]", "", (name or "").casefold())
    return re.sub(r"\b(ltd|llc|llp|inc|co|corp|limited)\b", "", name).strip()


def norm_number(num: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (num or "").upper())


def _same_amount(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= max(0.01, AMOUNT_REL_TOL * max(abs(a), abs(b)))


def _exact_amount(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= 0.01


def _close_dates(a, b) -> bool:
    da, db = parse_date(a), parse_date(b)
    return da is not None and db is not None and abs(da - db) <= timedelta(days=DATE_WINDOW_DAYS)


def find_duplicate(inv: dict, history: list[dict]) -> DuplicateMatch | None:
    best: DuplicateMatch | None = None
    v, n = norm_vendor(inv.get("vendor_name")), norm_number(inv.get("invoice_number"))
    for h in history:
        vsim = fuzz.token_sort_ratio(v, norm_vendor(h.get("vendor_name")))
        same_vendor = bool(v) and vsim >= VENDOR_SIM_MIN
        same_num = bool(n) and n == norm_number(h.get("invoice_number"))
        same_amt = _same_amount(inv.get("total_amount"), h.get("total_amount"))

        if same_vendor and same_num and same_amt:
            m = DuplicateMatch(h["id"], "exact", 1.0, "same vendor, invoice number and amount")
        elif same_vendor and same_num:
            m = DuplicateMatch(h["id"], "fuzzy", 0.85, "same vendor and invoice number, different amount")
        elif same_vendor and _exact_amount(inv.get("total_amount"), h.get("total_amount")) and _close_dates(inv.get("invoice_date"), h.get("invoice_date")):
            m = DuplicateMatch(h["id"], "resubmitted", 0.70, "same vendor, amount and date, new invoice number")
        else:
            continue
        if best is None or m.score > best.score:
            best = m
    return best
