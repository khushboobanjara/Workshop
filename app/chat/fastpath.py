"""Zero-token answers for common, simple questions.

A short question such as "show the review queue" or "system status" does not need a language model: the right tool is
known, and the answer is a fixed template filled with values copied from the tool result. That costs 0 Groq calls and 0
tokens, and it cannot hallucinate (every figure is the tool's own value, so the figures check and the fact check are
not needed).

Conservative on purpose: only short messages that match a known pattern AND contain no "reasoning" words (why, compare,
explain, trend, ...) are handled here. Everything else returns None and goes to the normal model path in service.py.
Tools are still called through the MCP bridge as the signed-in user, so RBAC, validation, redaction and audit are unchanged.
"""
import re

MAX_FAST_CHARS = 80
FAST_LIST_LIMIT = 10
_NEEDS_REASONING = re.compile(
    r"\b(why|how come|compare|comparison|explain|trend|versus|vs|difference|should|recommend|best|worst|average|sum|"
    r"total of|between|and then|predict|forecast)\b", re.IGNORECASE)

_INVOICE_ID = re.compile(r"^\W*(?:show|get|open|details? (?:of|for)|what(?:'s| is) (?:the )?status of)?\W*invoice\s*(?:id\s*)?#?\s*(\d{1,9})\W*$", re.IGNORECASE)
_ROUTES = (  # (tool, pattern)
    ("get_review_queue", re.compile(r"^\W*(?:show |list |open )?(?:me )?(?:the |my )?(?:review queue|pending review|invoices (?:pending|waiting for|needing) review|items? to review)\W*$", re.I)),
    ("get_metrics", re.compile(r"^\W*(?:show |get )?(?:me )?(?:the )?(?:metrics|stats|statistics|automation rate|savings|processing metrics|dashboard summary)\W*$", re.I)),
    ("list_vendors", re.compile(r"^\W*(?:show |list )?(?:me )?(?:the |all )?(?:top )?vendors?(?: summary| list)?\W*$", re.I)),
    ("list_models", re.compile(r"^\W*(?:show |list )?(?:me )?(?:the )?(?:ml )?models?(?: versions?| registry)?\W*$", re.I)),
    ("get_system_status", re.compile(r"^\W*(?:show |get )?(?:me )?(?:the )?(?:system )?status\W*$", re.I)),
    ("get_my_access", re.compile(r"^\W*(?:what can i (?:do|see|access)|my (?:access|role|permissions))\W*$", re.I)),
)
_FORBIDDEN_TEXT = "Your role is not allowed to see that. Please ask an administrator if you need access."
_NOTHING_TEXT = "The database has no matching records for that."


def route(message: str):
    """-> (tool_name, arguments) or None."""
    msg = (message or "").strip()
    if not msg or len(msg) > MAX_FAST_CHARS or _NEEDS_REASONING.search(msg):
        return None
    m = _INVOICE_ID.match(msg)
    if m:
        return "get_invoice", {"invoice_id": int(m.group(1))}
    for tool, pattern in _ROUTES:
        if pattern.match(msg):
            return tool, ({"limit": FAST_LIST_LIMIT} if tool in ("get_review_queue", "list_vendors") else {})
    return None


def _v(x):
    return "not in the records" if x is None or x == "" else str(x)


def _money(inv):
    return f"{_v(inv.get('total_amount'))} {inv.get('currency') or ''}".strip() if inv.get("total_amount") is not None else "not in the records"


def _invoice_line(i):
    return (f"- id {i.get('id')}, invoice number {_v(i.get('invoice_number'))}, vendor {_v(i.get('vendor_name'))}, "
            f"amount {_money(i)}, risk score {_v(i.get('risk_score'))}, status {_v(i.get('status'))}")


def render(tool: str, payload: dict) -> str | None:
    """Fixed template over the tool's own values. None = no template for this shape -> use the model."""
    data = payload.get("data")
    if not payload.get("found", True) or data is None:
        return _NOTHING_TEXT
    if tool == "get_review_queue":
        n = data.get("returned") or 0
        head = (f"Showing the {n} highest-risk invoices in the review queue; there may be more (ask for system status for the pending count):"
                if n >= FAST_LIST_LIMIT else f"{n} invoice(s) in the review queue (highest risk first):")
        lines = [head]
        return "\n".join(lines + [_invoice_line(i) for i in data.get("invoices", [])])
    if tool == "get_invoice":
        i = data.get("invoice") or {}
        out = [f"Invoice id {i.get('id')}:", _invoice_line(i),
               f"- decision {_v(i.get('decision'))} (rule {_v(i.get('decision_rule'))}), due date {_v(i.get('due_date'))}, invoice date {_v(i.get('invoice_date'))}"]
        if i.get("missing_fields"):
            out.append("- not in the records: " + ", ".join(i["missing_fields"]))
        return "\n".join(out)
    if tool == "get_metrics":
        s = data.get("estimated_savings") or {}
        return "\n".join([
            f"Emails received: {_v(data.get('emails_received'))}; invoices: {_v(data.get('invoices'))}; pending review: {_v(data.get('pending_review'))}.",
            f"By decision: {_v(data.get('by_decision'))}.", f"By status: {_v(data.get('by_status'))}.",
            f"Automation rate: {_v(data.get('automation_rate'))}. Average pipeline time per email: {_v(data.get('avg_pipeline_ms_per_email'))} ms.",
            f"Estimated savings: {_v(s.get('hours'))} hours, cost {_v(s.get('cost'))} ({_v(s.get('assumptions'))}); {_v(s.get('note'))}."])
    if tool == "list_vendors":
        rows = [f"- {_v(v.get('vendor'))}: {_v(v.get('invoices'))} invoices, spend {_v(v.get('total_spend'))} {v.get('currency') or ''}, "
                f"pending {_v(v.get('pending'))}, avg risk {_v(v.get('avg_risk'))}, last invoice {_v(v.get('last_invoice'))}" for v in data.get("vendors", [])]
        return "\n".join([f"{data.get('total_vendors')} vendors; showing {data.get('returned')} by spend (excludes duplicates and rejected):"] + rows)
    if tool == "list_models":
        rows = []
        for name, versions in (data.get("models") or {}).items():
            for v in versions:
                rows.append(f"- {name} {_v(v.get('version'))}: status {_v(v.get('status'))}, algorithm {_v(v.get('algorithm'))}, trained {_v(v.get('trained_at'))}")
        return "\n".join(["Registered models:"] + rows) if rows else _NOTHING_TEXT
    if tool == "get_system_status":
        models = ", ".join(f"{m['model']} {m['version']}" for m in data.get("production_models", [])) or "none recorded"
        return (f"Database: {_v(data.get('database'))}. Emails: {_v(data.get('emails'))}; invoices: {_v(data.get('invoices'))}; "
                f"pending review: {_v(data.get('invoices_pending_review'))}; emails needing triage: {_v(data.get('emails_needing_triage'))}; "
                f"audit entries: {_v(data.get('audit_entries'))}. Production models: {models}.")
    if tool == "get_my_access":
        return (f"You are {_v(data.get('username'))} with role {_v(data.get('role'))}. "
                f"Tools you may call: {', '.join(data.get('tools_you_may_call', [])) or 'none'}.")
    return None


def forbidden_text() -> str:
    return _FORBIDDEN_TEXT