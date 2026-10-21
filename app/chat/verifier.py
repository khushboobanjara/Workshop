"""Answer verification: every number, amount, percentage, date and identifier in the model's answer must be supported by
what the tools returned in THIS turn. Pure Python, no I/O.

What is checked
  * ISO dates (2026-10-31)                       must appear in the tool results (or the user's question / today's date)
  * identifiers like INV-1001, PO-7              must appear in the tool results or the user's question
  * #12 / ids                                    must be a number in the tool results or one the user typed
  * money, decimals, percentages, whole numbers   must equal a number in the tool results (percentages may also be
                                                   a 0-1 value x 100, e.g. risk 0.12 -> 12%)
What is NOT checked here: years, list markers ("1."), and ordinary words. A number is only matched to *some* number in the
results, not to the field it describes, which is why service.py adds a second pass (a fact checker) that compares every
statement with the records. A passing answer is "no unsupported figures", not a proof that every word is true.

The user's own text only vouches for identifiers, dates and whole numbers (so the model may repeat "INV-1001" or "#12"
from the question), never for amounts: a user cannot make "$5,000" true by typing it. Text from earlier ASSISTANT
turns never counts as evidence, so a forged history cannot launder a fact.

One exception, switched on per turn by the caller (`allow_premises`): in a turn that consulted ONLY company documents and no
database record, a what-if question ("an invoice worth 12,000 with risk 0.10: may it be approved?") has to restate the
user's own values to answer. They are accepted as the user's premise, never as a fact about the records: no record was read
in that turn, so there is nothing for them to be mistaken for, and numbers the user did not type are still checked.
"""
import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?")
_IDENT = re.compile(r"\b[A-Za-z]{2,6}-[A-Za-z0-9-]*\d[A-Za-z0-9-]*\b")
_HASHID = re.compile(r"(?<![\w&])#(\d+)\b")
_NUM = re.compile(r"(?P<cur>[$€£₹¥])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?P<pct>\s?%)?")
_CODES = ("USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY")
_NO_NUMBERS_FROM = {"as_of", "timestamp", "created_at", "reviewed_at", "time", "last_invoice"}
MAX_REPORTED = 8


def _dec(x) -> Decimal | None:
    try:
        d = Decimal(repr(x)) if isinstance(x, float) else Decimal(str(x))
    except Exception:
        return None
    return d if d.is_finite() else None


@dataclass
class Evidence:
    text: str = ""                                                 # lower-cased haystack for identifiers and dates
    numbers: set = field(default_factory=set)                      # Decimals that appear in tool results
    percents: set = field(default_factory=set)                     # 0-1 values expressed as percentages
    user_ints: set = field(default_factory=set)                    # whole numbers the user typed
    premises: set = field(default_factory=set)                     # every number the user typed (Decimals)
    premise_percents: set = field(default_factory=set)             # typed 0-1 decimals expressed as percentages
    allow_premises: bool = False                                   # see the module docstring: off unless the caller turns it on

    def add_tool_payload(self, payload) -> None:
        self._walk(payload, None)

    def add_user_text(self, text: str) -> None:
        self.text += "\n" + text.lower()
        for m in re.finditer(r"\d+", text):
            self.user_ints.add(int(m.group()))
        for m in _NUM.finditer(_IDENT.sub(" ", _ISO.sub(" ", text))):
            raw = m.group("num").replace(",", "")
            d = _dec(raw)
            if d is None:
                continue
            self.premises.add(d)
            if "." in raw and 0 < abs(d) <= 1:
                for places in (0, 1, 2):
                    self.premise_percents.add((d * 100).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))

    def add_allowed_text(self, text: str) -> None:
        self.text += "\n" + text.lower()

    def _add_number(self, x) -> None:
        d = _dec(x)
        if d is None:
            return
        self.numbers.add(d)
        if isinstance(x, float) and 0 < abs(d) <= 1:
            for places in (0, 1, 2):
                self.percents.add((d * 100).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))

    def _walk(self, node, key) -> None:
        if node is None or isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            self._add_number(node)
        elif isinstance(node, str):
            self.text += "\n" + node.lower()
            if key not in _NO_NUMBERS_FROM:
                stripped = _IDENT.sub(" ", _ISO.sub(" ", node))
                for m in _NUM.finditer(stripped):
                    self._add_number(m.group("num").replace(",", ""))
        elif isinstance(node, dict):
            for k, v in node.items():
                self._walk(v, k)
        elif isinstance(node, (list, tuple)):
            for v in node:
                self._walk(v, key)


def find_unsupported(answer: str, ev: Evidence) -> list[str]:
    """Returns the figures/identifiers in `answer` that the evidence does not support (empty list = nothing unsupported)."""
    bad: list[str] = []
    text = answer
    for m in _ISO.finditer(text):
        if m.group()[:10] not in ev.text:
            bad.append(m.group())
    text = _ISO.sub(" ", text)
    for m in _IDENT.finditer(text):
        if m.group().lower() not in ev.text:
            bad.append(m.group())
    text = _IDENT.sub(" ", text)
    for m in _HASHID.finditer(text):
        n = int(m.group(1))
        if Decimal(n) not in ev.numbers and n not in ev.user_ints:
            bad.append("#" + m.group(1))
    text = _HASHID.sub(" ", text)
    text = re.sub(r"(?m)^\s*\d{1,2}[.)]\s+", " ", text)                  # list markers such as "1." are not figures
    for m in _NUM.finditer(text):
        num_s, cur, pct = m.group("num"), m.group("cur"), m.group("pct")
        value = _dec(num_s.replace(",", ""))
        if value is None:
            continue
        around = (text[max(0, m.start() - 4):m.start()] + " " + text[m.end():m.end() + 5]).upper()
        if pct:
            kind = "percent"
        elif cur or any(c in around for c in _CODES):
            kind = "money"
        elif "." in num_s or "," in num_s:
            kind = "decimal"
        elif 1900 <= value <= 2100:
            continue                                               # years are not checked
        else:
            kind = "int"                                           # whole numbers, including small counts such as 0 or 3
        ok = value in ev.numbers or (kind == "percent" and value in ev.percents) or (kind == "int" and int(value) in ev.user_ints)
        if not ok and ev.allow_premises:
            ok = value in ev.premises or (kind == "percent" and value in ev.premise_percents)
        if not ok:
            bad.append(m.group().strip())
    seen: list[str] = []
    for b in bad:
        if b not in seen:
            seen.append(b)
    return seen[:MAX_REPORTED]
