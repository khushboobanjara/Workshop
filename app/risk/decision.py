"""Decision Engine: turns risk score + business rules into one of
auto_approved | human_review | rejected, with human-readable reasons.

Hard rules run FIRST (they cannot be outvoted by a low score); thresholds decide the rest."""
from dataclasses import dataclass, field

# Keep in sync with REQUIRED in app/pipeline/extract.py
REQUIRED_FIELDS = ["vendor_name", "invoice_number", "total_amount", "due_date"]


@dataclass
class Thresholds:
    auto_approve_max_risk: float = 0.30
    auto_reject_min_risk: float = 0.85
    auto_approve_max_amount: float = 10_000.0   # business policy: bigger invoices always get human eyes
    severe_validation: float = 0.70             # validation signal at/above this blocks auto-approval
    new_vendor_review_amount: float = 2_000.0   # first invoices from an unknown vendor above this get human eyes


@dataclass
class Decision:
    outcome: str                 # auto_approved | human_review | rejected
    risk_score: float
    rule: str                    # which rule decided (goes to the audit log)
    reasons: list[str] = field(default_factory=list)


def _drivers(contributions: dict[str, float], n: int = 2) -> str:
    top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return ", ".join(f"{k} ({v:.2f})" for k, v in top if v > 0) or "none"


def decide(invoice: dict, signals: dict[str, float], score: float,
           contributions: dict[str, float], t: Thresholds | None = None) -> Decision:
    t = t or Thresholds()

    if signals.get("duplicate", 0.0) >= 1.0:
        return Decision("rejected", score, "exact_duplicate",
                        [f"Exact duplicate of invoice {invoice.get('duplicate_of')} - not paid twice"])

    missing = [k for k in REQUIRED_FIELDS if invoice.get(k) is None]
    if missing:
        return Decision("human_review", score, "missing_required_fields",
                        [f"Missing required fields: {', '.join(missing)}"])

    if signals.get("validation_issues", 0.0) >= t.severe_validation:
        return Decision("human_review", score, "severe_validation_issue",
                        ["Serious validation problem (e.g. due date before invoice date, non-positive amount)"])

    if invoice["total_amount"] > t.auto_approve_max_amount:
        return Decision("human_review", score, "amount_above_auto_limit",
                        [f"Amount {invoice['total_amount']:,.2f} is above the auto-approval limit of {t.auto_approve_max_amount:,.0f}"])

    if signals.get("new_vendor", 0.0) >= 1.0 and invoice["total_amount"] > t.new_vendor_review_amount:
        return Decision("human_review", score, "new_vendor_large_amount",
                        [f"Unknown vendor with little history and amount {invoice['total_amount']:,.2f} "
                         f"above {t.new_vendor_review_amount:,.0f} - first invoices are checked by a person"])

    if score >= t.auto_reject_min_risk:
        return Decision("rejected", score, "risk_above_reject_threshold",
                        [f"Risk {score:.2f} >= {t.auto_reject_min_risk}. Main drivers: {_drivers(contributions)}"])

    if score > t.auto_approve_max_risk:
        return Decision("human_review", score, "risk_above_approve_threshold",
                        [f"Risk {score:.2f} > {t.auto_approve_max_risk}. Main drivers: {_drivers(contributions)}"])

    return Decision("auto_approved", score, "low_risk", [f"All checks passed, risk {score:.2f}"])
