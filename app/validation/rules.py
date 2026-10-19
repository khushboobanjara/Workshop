"""Pure validation rules for extracted invoice data (no DB, no pipeline imports).
Each rule returns an Issue with a severity in 0..1 so the risk engine can weigh it."""
from dataclasses import dataclass
from datetime import date

ALLOWED_CURRENCIES = {"USD", "EUR", "GBP", "INR", "CAD", "AUD"}
MAX_PAYMENT_TERMS_DAYS = 120
MAX_REASONABLE_AMOUNT = 1_000_000


@dataclass
class Issue:
    field: str
    message: str
    severity: float  # 0..1


def parse_date(value) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def validate_invoice(inv: dict, today: date | None = None) -> list[Issue]:
    today = today or date.today()
    issues: list[Issue] = []

    inv_date, due_date = parse_date(inv.get("invoice_date")), parse_date(inv.get("due_date"))
    for key in ("invoice_date", "due_date"):
        if inv.get(key) and parse_date(inv[key]) is None:
            issues.append(Issue(key, f"{key} is not a valid date: {inv[key]!r}", 0.6))

    if inv_date and inv_date > today:
        issues.append(Issue("invoice_date", f"Invoice date {inv_date} is in the future", 0.5))
    if inv_date and due_date:
        if due_date < inv_date:
            issues.append(Issue("due_date", "Due date is before invoice date", 0.7))
        elif (due_date - inv_date).days > MAX_PAYMENT_TERMS_DAYS:
            issues.append(Issue("due_date", f"Unusually long payment terms ({(due_date - inv_date).days} days)", 0.3))

    amount = inv.get("total_amount")
    if amount is not None:
        if amount <= 0:
            issues.append(Issue("total_amount", "Total amount must be positive", 1.0))
        elif amount > MAX_REASONABLE_AMOUNT:
            issues.append(Issue("total_amount", f"Amount {amount:,.2f} exceeds sanity limit", 0.4))

    cur = inv.get("currency")
    if cur and cur not in ALLOWED_CURRENCIES:
        issues.append(Issue("currency", f"Unsupported currency {cur}", 0.4))

    num = inv.get("invoice_number")
    if num and not (4 <= len(num) <= 30):
        issues.append(Issue("invoice_number", f"Suspicious invoice number format: {num!r}", 0.4))

    return issues


def validation_risk(issues: list[Issue]) -> float:
    """Combine issues into one 0..1 signal (worst issue dominates, extras add a little)."""
    if not issues:
        return 0.0
    sev = sorted((i.severity for i in issues), reverse=True)
    return round(min(1.0, sev[0] + 0.1 * sum(sev[1:])), 4)
