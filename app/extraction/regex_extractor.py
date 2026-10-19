"""Rule-based invoice field extractor. Every field comes back with a confidence
score and the rule that produced it, so later stages know which values to trust."""
import re
from dataclasses import dataclass

DATE = r"(\d{4}-\d{2}-\d{2})"


@dataclass
class Field:
    value: object
    confidence: float
    source: str  # which rule matched (useful for debugging and the audit trail)


def _norm_inv(s: str) -> str:
    return re.sub(r"[-\s]+", "-", s.upper())


def _invoice_number(subject: str, body: str) -> Field | None:
    pat = re.compile(r"\b(INV[-\s]?\d{4,})\b", re.I)
    s, b = pat.search(subject), pat.search(body)
    if s and b:
        sv, bv = _norm_inv(s.group(1)), _norm_inv(b.group(1))
        if sv == bv:
            return Field(bv, 0.98, "subject+body agree")
        return Field(bv, 0.50, "subject and body disagree")
    if b:
        return Field(_norm_inv(b.group(1)), 0.90, "body")
    if s:
        return Field(_norm_inv(s.group(1)), 0.80, "subject only")
    return None


def _amount(body: str) -> tuple[Field | None, Field | None]:
    """Returns (total_amount, currency)."""
    labeled = re.findall(r"(?i:total amount)\s*:?\s*([A-Z]{3})\s+([\d,]+\.\d{2})", body)
    fallback = re.findall(r"(?i:amount due)\s*:?\s*([A-Z]{3})\s+([\d,]+\.\d{2})", body)
    hits, conf, src = (labeled, 0.95, "Total Amount label") if labeled else (fallback, 0.85, "amount due phrase")
    if not hits:
        return None, None
    values = {float(a.replace(",", "")) for _, a in hits}
    if len(values) > 1:
        conf, src = 0.50, "multiple different totals"
    cur, amt = hits[0]
    return Field(float(amt.replace(",", "")), conf, src), Field(cur, conf, src)


def _date(body: str, label: str) -> Field | None:
    m = re.search(rf"(?i:{label})\s*:\s*{DATE}", body)
    return Field(m.group(1), 0.95, f"{label} label") if m else None


def _due_date(body: str) -> Field | None:
    labeled = _date(body, "due date")
    if labeled:
        return labeled
    m = re.search(rf"\bby\s+{DATE}", body)
    return Field(m.group(1), 0.75, "'by <date>' phrase") if m else None


def _po(body: str) -> Field | None:
    m = re.search(r"\b(PO-\d{3,})\b", body)
    return Field(m.group(1), 0.90, "PO pattern") if m else None


def _vendor(body: str) -> Field | None:
    m = re.search(r"(?im)^vendor\s*:\s*(.+)$", body)
    if m:
        return Field(m.group(1).strip(), 0.95, "Vendor label")
    lines = [l.strip() for l in body.strip().splitlines() if l.strip()]
    if lines and len(lines[-1]) < 60 and not re.search(r"[\d@:]", lines[-1]):
        return Field(lines[-1], 0.60, "signature line")
    return None


def extract_fields(subject: str, body: str) -> dict[str, Field | None]:
    total, currency = _amount(body)
    return {
        "vendor_name": _vendor(body),
        "invoice_number": _invoice_number(subject, body),
        "invoice_date": _date(body, "invoice date"),
        "due_date": _due_date(body),
        "currency": currency,
        "total_amount": total,
        "po_number": _po(body),
    }
