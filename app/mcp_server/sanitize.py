"""Cleaning of text before it is returned to an AI assistant or written to the audit log. Pure, standard library only.

Two different risks are handled here, and neither can be solved completely, so this is defence in depth and the
primary control is elsewhere (an allow-list of fields: email bodies and password hashes are never selected at all):

  * sensitive data in free text (card numbers, IBANs, SSNs, e-mail addresses) -> redacted or masked;
  * hidden or control characters that make text look different to a person and to a model (zero-width and bidi
    characters are a known prompt-injection trick) -> removed.

Redaction is best-effort pattern matching. It errs on the side of redacting (a long number that happens to pass the
Luhn check is masked too), and it is only applied to free-text fields, never to identifiers such as invoice numbers.
"""
import math
import re
import unicodedata

MAX_TEXT = 500
TRUNCATED = "…[truncated]"

_DIGIT_RUN = re.compile(r"\d(?:[ -]?\d)*")                        # digits, optionally separated by single spaces/dashes
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_KEEP_CONTROL = {"\n", "\t"}
_MAX_CARD_PASSES = 10


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    s = candidate.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    number = "".join(str(int(ch, 36)) for ch in rearranged)        # A=10 ... Z=35
    return int(number) % 97 == 1


def _redact_one_card(text: str) -> tuple[str, bool]:
    """Masks the first Luhn-valid 13-19 digit number found inside any run of digits; returns (text, changed)."""
    for m in _DIGIT_RUN.finditer(text):
        positions = [m.start() + i for i, ch in enumerate(m.group()) if ch.isdigit()]
        if len(positions) < 13:
            continue
        digits = "".join(text[p] for p in positions)
        for length in range(min(19, len(digits)), 12, -1):
            for start in range(len(digits) - length + 1):
                window = digits[start:start + length]
                if _luhn(window):
                    first, last = positions[start], positions[start + length - 1]
                    return text[:first] + f"[REDACTED-CARD ****{window[-4:]}]" + text[last + 1:], True
    return text, False


def redact_sensitive_text(text: str) -> str:
    # Order matters: IBANs contain long digit runs that would otherwise be mistaken for card numbers.
    text = _IBAN.sub(lambda m: "[REDACTED-IBAN]" if _iban_ok(m.group()) else m.group(), text)
    text = _SSN.sub("[REDACTED-SSN]", text)
    for _ in range(_MAX_CARD_PASSES):
        text, changed = _redact_one_card(text)
        if not changed:
            break
    return text


def mask_email_addresses(text: str) -> str:
    """jane.doe@acme.com -> j***@acme.com: the domain says who it is, the person's name is not needed."""
    return _EMAIL.sub(lambda m: m.group()[0] + "***@" + m.group().split("@", 1)[1], text)


def strip_invisible(text: str) -> str:
    """Removes control, format (zero-width, bidi), surrogate, private-use and unassigned characters. Keeps \\n and \\t."""
    kept = "".join(ch for ch in text
                   if ch in _KEEP_CONTROL or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co", "Cn"))
    return re.sub(r"\n{3,}", "\n\n", kept)


def clean_text(value, max_len: int = MAX_TEXT, *, redact: bool = True, mask_emails: bool = False) -> str | None:
    """The one function every free-text value passes through. None stays None (a missing value is never replaced)."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    pre_cut = len(text) > max_len * 4                                # bound the regex work on huge inputs
    text = strip_invisible(text[: max_len * 4])
    if redact:
        text = redact_sensitive_text(text)
    if mask_emails:
        text = mask_email_addresses(text)
    if pre_cut or len(text) > max_len:
        text = text[:max_len].rstrip() + TRUNCATED
    return text


def sanitize_json(value, *, max_depth: int = 6, max_items: int = 100, max_str: int = 300, _depth: int = 0):
    """Makes an arbitrary JSON-like value safe to return: strings cleaned, NaN/Infinity dropped, size and depth bounded."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return clean_text(value, max_str)
    if _depth >= max_depth:
        return "[truncated]"
    kw = {"max_depth": max_depth, "max_items": max_items, "max_str": max_str, "_depth": _depth + 1}
    if isinstance(value, dict):
        return {("unknown" if k is None else clean_text(str(k), 80, redact=False)): sanitize_json(v, **kw)
                for k, v in list(value.items())[:max_items]}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v, **kw) for v in list(value)[:max_items]]
    return clean_text(str(value), max_str)
