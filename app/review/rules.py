"""Pure human-review rules: which status changes are legal. No DB, no web framework."""

REQUIRED_FIELDS = ["vendor_name", "invoice_number", "total_amount", "due_date"]
EDITABLE_FIELDS = ["vendor_name", "invoice_number", "invoice_date", "due_date",
                   "currency", "total_amount", "po_number"]
ACTIONS = ("approve", "reject")

_STATUS_FOR_DECISION = {"auto_approved": "approved", "human_review": "pending_review", "rejected": "rejected"}


class ReviewError(Exception):
    """kind='conflict' -> wrong state (HTTP 409); kind='invalid' -> bad input (HTTP 422)."""

    def __init__(self, message: str, kind: str = "invalid"):
        super().__init__(message)
        self.kind = kind


def status_for_decision(outcome: str) -> str:
    # Unknown outcomes fail SAFE: a human looks at them.
    return _STATUS_FOR_DECISION.get(outcome, "pending_review")


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def plan_review(current_status: str, action: str, invoice_after_corrections: dict) -> str:
    """Returns the new status, or raises ReviewError. Called BEFORE anything is changed."""
    if action not in ACTIONS:
        raise ReviewError(f"Unknown action {action!r}; use one of {ACTIONS}", "invalid")
    if current_status != "pending_review":
        raise ReviewError(f"Invoice is '{current_status}'; only 'pending_review' invoices can be reviewed", "conflict")
    if action == "reject":
        return "rejected"
    missing = [k for k in REQUIRED_FIELDS if _blank(invoice_after_corrections.get(k))]
    if missing:
        raise ReviewError(f"Cannot approve: missing required fields {missing}. Supply them in 'corrections'.", "invalid")
    amount = invoice_after_corrections["total_amount"]
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise ReviewError("Cannot approve: total_amount must be a positive number", "invalid")
    return "approved"
