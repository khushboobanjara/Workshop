"""Keeps every MCP answer tied to database records.

* Output is built from ALLOW-LISTS of columns. A column added to a model later is not exposed until someone lists it
  here, and the email body, password hashes, token versions and secrets are never selected at all.
* A missing value is returned as null and named in `missing_fields`. It is never replaced by a default or a guess.
* "Nothing found" is an explicit result (`found: false`), not an empty string the model might fill in.
* Every answer carries an envelope that names its source tables and record ids, flags the text that came from
  outside the system (so a model treats it as data, not as instructions), and repeats the grounding rule.

What the server cannot do is stop a client model from inventing things in its OWN reply. It can only make the real
data complete, explicit about absence, and easy to check against `record_refs`.
"""
from datetime import datetime, timezone

from app.mcp_server.sanitize import clean_text, sanitize_json
from app.mcp_server.contracts import ToolResult

INVOICE_FIELDS = (
    "id", "email_id", "vendor_name", "invoice_number", "invoice_date", "due_date", "currency", "total_amount",
    "po_number", "extraction_confidence", "risk_score", "risk_breakdown", "decision", "decision_rule",
    "decision_reasons", "status", "duplicate_of", "flags", "reviewed_by", "review_comment", "reviewed_at", "created_at")
CORE_FIELDS = ("vendor_name", "invoice_number", "invoice_date", "due_date", "currency", "total_amount", "po_number")
UNTRUSTED_INVOICE_FIELDS = ("vendor_name", "invoice_number", "invoice_date", "due_date", "po_number", "review_comment",
                            "decision_reasons", "flags", "email.subject", "email.sender")

GROUNDING_NOTE = ("Only the values in `data` are verified database records. null or a name listed in `missing_fields` "
                  "means the information is not stored: say it is unknown, do not guess or fill it in.")
NOT_FOUND_NOTE = "The database has no matching record. Do not estimate, infer or invent one."


def shape_invoice(d: dict) -> dict:
    """Allow-listed, cleaned view of one invoice (as produced by processing.invoice_to_dict). The email BODY is never
    copied: it is attacker-controlled text and may hold bank details. Only the subject and a masked sender are shown."""
    out = {k: d.get(k) for k in INVOICE_FIELDS}
    out["vendor_name"] = clean_text(out["vendor_name"], 200, redact=False)
    out["invoice_number"] = clean_text(out["invoice_number"], 100, redact=False)
    out["po_number"] = clean_text(out["po_number"], 100, redact=False)
    out["invoice_date"] = clean_text(out["invoice_date"], 20, redact=False)
    out["due_date"] = clean_text(out["due_date"], 20, redact=False)
    out["reviewed_by"] = clean_text(out["reviewed_by"], 100, redact=False)
    out["review_comment"] = clean_text(out["review_comment"], 500)
    for k in ("risk_breakdown", "decision_reasons", "flags"):
        out[k] = sanitize_json(out[k])
    out["missing_fields"] = [f for f in CORE_FIELDS if out[f] is None or out[f] == ""]
    email = d.get("email")
    if email is not None:
        out["email"] = {"subject": clean_text(email.get("subject"), 200),
                        "sender": clean_text(email.get("sender"), 200, mask_emails=True)}
    return out


def build_envelope(tool: str, r: ToolResult) -> dict:
    notes = list(r.notes)
    if not r.found:
        notes.append(NOT_FOUND_NOTE)
    notes.append(GROUNDING_NOTE)
    return {"source": "opspilot_database", "tool": tool, "as_of": datetime.now(timezone.utc).isoformat(),
            "found": r.found, "record_refs": list(r.refs), "tables": list(r.tables),
            "untrusted_text_fields": list(r.untrusted), "notes": notes, "data": r.data}
