"""Learned invoice field extractor: candidate generation + per-field classifiers.

1. Candidate generation (permissive regexes): every date in any common format, every money amount in
   US or European notation, every ID-like token, every vendor-like line. No guessing which one matters yet.
2. A logistic-regression classifier per field scores each candidate from its CONTEXT: the words just before
   and after it, its line position, whether it is the largest amount, the earliest/latest date, and so on.
3. The best candidate wins if its probability clears a threshold; that probability is the confidence.
   Below the threshold the field is reported missing (and the invoice goes to a human) instead of guessed.

Labels come for free: a candidate is positive when its normalised value equals the known truth."""
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from app.extraction.regex_extractor import Field

DEFAULT_MODEL_PATH = "models/field_extractor.joblib"
CURRENCIES = {"USD", "EUR", "GBP", "INR", "CAD", "AUD"}
SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
FIELD_FAMILY = {"vendor_name": "vendor", "invoice_number": "number", "invoice_date": "date",
                "due_date": "date", "total_amount": "amount", "po_number": "po"}

_M = "|".join(m.capitalize() for m in MONTHS)
_S = "|".join(m[:3].capitalize() for m in MONTHS)
DATE_RE = re.compile(rf"\b(\d{{4}}-\d{{2}}-\d{{2}})\b|\b(\d{{2}}/\d{{2}}/\d{{4}})\b|\b((?:{_M}) \d{{1,2}}, \d{{4}})\b|\b(\d{{1,2}} (?:{_S}) \d{{4}})\b")
_CUR = "(?:USD|EUR|GBP|INR|CAD|AUD)"
AMT_RE = re.compile(rf"(?P<pre>{_CUR}\s*|[$£€]\s?)?(?P<num>\d{{1,3}}(?:[.,]\d{{3}})+[.,]\d{{2}}|\d+[.,]\d{{2}})(?P<suf>\s*{_CUR})?(?!\d)")
NUM_RE = re.compile(r"(?i)\binv(?:oice)?\s*#\s?\d{4,}|#\s?\d{4,}|\b[A-Z]{2,4}-\d{4}-\d{3,}\b|\b[A-Z]{2,4}[-\s]?\d{4,}\b")
PO_RE = re.compile(r"(?i)\bPO[-\s#]*(\d{3,})\b")
KV_RE = re.compile(r"^([A-Za-z][A-Za-z .&']{1,24}):\s*(.+)$")
TERMS_RE = re.compile(r"(?i)\bnet\s*(\d{1,3})\b")


# ------------------------------------------------------------------ parsing / normalisation
def parse_date(raw: str) -> str | None:
    try:
        if "-" in raw:
            return date.fromisoformat(raw).isoformat()
        if "/" in raw:
            d, m, y = raw.split("/")                      # assumes DD/MM/YYYY (documented limitation)
            return date(int(y), int(m), int(d)).isoformat()
        parts = raw.replace(",", "").split()
        if parts[0].isalpha():                            # "March 1 2026"
            return date(int(parts[2]), MONTHS.index(parts[0].lower()) + 1, int(parts[1])).isoformat()
        return date(int(parts[2]), [m[:3] for m in MONTHS].index(parts[1].lower()) + 1, int(parts[0])).isoformat()
    except (ValueError, IndexError):
        return None


def parse_amount(num: str) -> float | None:
    s = num
    if "." in s and "," in s:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif re.search(r"[.,]\d{2}$", s) and len(re.findall(r"[.,]", s)) == 1:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "").replace(".", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def normalize_number(raw: str) -> str:
    s = raw.strip()
    m = re.match(r"(?i)^inv(?:oice)?\s*[-#\s]*(\d{4,})$", s)
    if m:
        return f"INV-{m.group(1)}"
    if s.startswith("#"):
        return "#" + re.sub(r"\D", "", s)
    return re.sub(r"[-\s]+", "-", s.upper())


# ------------------------------------------------------------------ candidates
@dataclass
class Occ:
    key: object
    value: object
    raw: str
    line_idx: int
    line: str
    start: int
    end: int
    in_subject: bool
    currency: str | None = None
    label: str | None = None


@dataclass
class Candidate:
    key: object
    value: object
    feats: dict
    currency: str | None = None


def collect(subject: str, body: str):
    lines = [l.rstrip() for l in body.splitlines()]
    docs = [(subject, -1, True)] + [(l, i, False) for i, l in enumerate(lines)]
    out = {k: [] for k in ("date", "amount", "number", "po", "vendor")}
    for text, idx, subj in docs:
        if not text.strip():
            continue
        for m in DATE_RE.finditer(text):
            iso = parse_date(m.group(0))
            if iso:
                out["date"].append(Occ(iso, iso, m.group(0), idx, text, m.start(), m.end(), subj))
        for m in AMT_RE.finditer(text):
            val = parse_amount(m.group("num"))
            if val is None:
                continue
            code = (m.group("pre") or m.group("suf") or "").strip()
            cur = SYMBOLS.get(code, code) if code else None
            out["amount"].append(Occ(val, val, m.group(0), idx, text, m.start(), m.end(), subj, currency=cur or None))
        for m in NUM_RE.finditer(text):
            raw = m.group(0)
            if re.match(rf"(?i){_CUR}\b", raw) or re.match(r"(?i)po\b", raw):
                continue
            n = normalize_number(raw)
            out["number"].append(Occ(n, n, raw, idx, text, m.start(), m.end(), subj))
        for m in PO_RE.finditer(text):
            n = f"PO-{m.group(1)}"
            out["po"].append(Occ(n, n, m.group(0), idx, text, m.start(), m.end(), subj))
        # vendor-like lines
        if subj:
            m = re.match(r"(?i)^(.*?)\s+invoice\b", text)
            cand = m.group(1).strip() if m else ""
            if len(cand) >= 3 and not re.search(r"\d|^(re|fwd):", cand, re.I):
                out["vendor"].append(Occ(cand, cand, cand, idx, text, 0, len(cand), True))
        else:
            s = text.strip()
            kv = KV_RE.match(s)
            if kv and not re.search(r"\d", kv.group(2)) and len(kv.group(2)) <= 60:
                # any "Label: Name" line: the model decides from the label words (and generic shape) whether it is the vendor
                v = kv.group(2).strip()
                start = text.index(v)
                out["vendor"].append(Occ(v, v, v, idx, text, start, start + len(v), False, label=kv.group(1).strip().lower()))
            elif len(s) <= 60 and not s.endswith(",") and ":" not in s:
                start = text.index(s)
                out["vendor"].append(Occ(s, s, s, idx, text, start, start + len(s), False))
    return out, lines


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z]+", s.lower())


def _shape(raw: str) -> str:
    return re.sub(r"[a-z]+", "a", re.sub(r"[A-Z][a-z]+", "Aa", re.sub(r"\d+", "9", raw)))


def _occ_feats(o: Occ, lines: list[str], family: str) -> dict:
    f: dict = {}
    left, right = o.line[:o.start], o.line[o.end:]
    lt = _tok(left)
    for k, t in enumerate(reversed(lt[-3:]), 1):
        f[f"L{k}:{t}"] = 1
    for t in lt[-6:]:
        f["lb:" + t] = 1
    for k, t in enumerate(_tok(right)[:2], 1):
        f[f"R{k}:{t}"] = 1
    if o.line_idx > 0:
        for t in _tok(lines[o.line_idx - 1])[-2:]:
            f["pl:" + t] = 1
    if left.rstrip().endswith(":"):
        f["left_colon"] = 1
    if o.in_subject:
        f["subj"] = 1
    else:
        f[f"pos:{min(3, int(o.line_idx / max(1, len(lines) - 1) * 4))}"] = 1
        if o.line_idx == len(lines) - 1:
            f["last_line"] = 1
    f["shape:" + _shape(o.raw)] = 1
    if family == "amount":
        f["has_cur"] = int(o.currency is not None)
        f["has_symbol"] = int(bool(re.match(r"[$£€]", o.raw)))
    if family == "vendor":
        words = o.raw.split()
        f["nwords:%d" % min(len(words), 6)] = 1
        f["has_digit"] = int(bool(re.search(r"\d", o.raw)))
        f["allcaps"] = int(o.raw.isupper())
        f["endp"] = int(o.raw.endswith("."))
        if o.label:
            f["kv"] = 1
            for t in _tok(o.label):
                f["klab:" + t] = 1
        for t in _tok(o.raw):
            f["w:" + t] = 1
        if words:
            f["lastw:" + words[-1].lower().strip(".,")] = 1
    return f


def build_candidates(subject: str, body: str) -> dict[str, list[Candidate]]:
    occs, lines = collect(subject, body)
    result: dict[str, list[Candidate]] = {}
    for family, items in occs.items():
        groups: dict = {}
        for o in items:
            groups.setdefault(o.key, []).append(o)
        cands = []
        for key, group in groups.items():
            feats: dict = {}
            for o in group:
                for k, v in _occ_feats(o, lines, family).items():
                    feats[k] = max(feats.get(k, 0), v)
            feats[f"n_occ:{min(len(group), 3)}"] = 1
            if any(o.in_subject for o in group) and any(not o.in_subject for o in group):
                feats["subj_and_body"] = 1
            cur = next((o.currency for o in group if o.currency), None)
            cands.append(Candidate(key, group[0].value, feats, cur))
        n = len(cands)
        vals = sorted((c.key for c in cands), reverse=family in ("amount",))
        for i, c in enumerate(cands):
            c.feats[f"rank:{min(i, 5)}"] = 1
            c.feats[f"rank_end:{min(n - 1 - i, 5)}"] = 1
            c.feats[f"n_cand:{min(n, 6)}"] = 1
            if family in ("amount", "date") and n > 1:
                r = vals.index(c.key)
                c.feats[f"vrank:{min(r, 4)}"] = 1
                if r == n - 1:
                    c.feats["vlast"] = 1
        result[family] = cands
    return result


def _is_match(field_name: str, c: Candidate, truth) -> bool:
    if truth is None:
        return False
    if field_name == "total_amount":
        return abs(c.value - truth) < 0.01
    if field_name == "invoice_number":
        return c.value == normalize_number(str(truth))
    if field_name == "po_number":
        return c.value == truth
    return c.value == truth


class _Constant:
    """Stand-in model for a field whose training candidates were all one class (e.g. every PO-like
    string in the data is a real PO number, so there is nothing to discriminate)."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        import numpy as np
        return np.tile([1 - self.p, self.p], (len(X), 1))


# ------------------------------------------------------------------ the extractor
@dataclass
class LearnedExtractor:
    threshold: float = 0.5
    C: float = 3.0
    models: dict = field(default_factory=dict)

    def fit(self, rows: list[dict]) -> "LearnedExtractor":
        data = {f: ([], []) for f in FIELD_FAMILY}
        for r in rows:
            cands = build_candidates(r["subject"], r["body"])
            for f, fam in FIELD_FAMILY.items():
                for c in cands[fam]:
                    data[f][0].append(c.feats)
                    data[f][1].append(int(_is_match(f, c, r["truth"].get(f))))
        for f, (X, y) in data.items():
            if not y:
                raise ValueError(f"no candidates found to train {f}")
            if len(set(y)) < 2:
                self.models[f] = _Constant(float(y[0]))
                continue
            self.models[f] = make_pipeline(DictVectorizer(), LogisticRegression(C=self.C, max_iter=3000)).fit(X, y)
        return self

    def _scored(self, email: dict):
        cands = build_candidates(email.get("subject", ""), email.get("body", ""))
        scores = {}
        for f, fam in FIELD_FAMILY.items():
            cs = cands[fam]
            if cs:
                p = self.models[f].predict_proba([c.feats for c in cs])[:, 1]
                scores[f] = list(zip(map(float, p), cs))
        return scores, cands

    def __call__(self, email: dict) -> dict:
        scores, cands = self._scored(email)
        out: dict = {f: None for f in ("vendor_name", "invoice_number", "invoice_date", "due_date",
                                       "currency", "total_amount", "po_number")}
        thr = self.threshold
        # the two date fields compete for the same candidates: highest probability wins each date
        taken, assigned = set(), set()
        triples = sorted(((p, f, i) for f in ("invoice_date", "due_date") for i, (p, _) in enumerate(scores.get(f, []))), reverse=True)
        for p, f, i in triples:
            c = scores[f][i][1]
            if p >= thr and f not in assigned and c.key not in taken:
                out[f] = Field(c.value, round(p, 4), "learned"); assigned.add(f); taken.add(c.key)
        for f in ("vendor_name", "invoice_number", "total_amount", "po_number"):
            if f in scores:
                p, c = max(scores[f], key=lambda t: t[0])
                if p >= thr:
                    out[f] = Field(c.value, round(p, 4), "learned")
                    if f == "total_amount":
                        out["_total_cand"] = c
        total = out.pop("_total_cand", None)
        # due date missing but the email states payment terms ("Net 30"): derive it from the invoice date
        if out["due_date"] is None and out["invoice_date"] is not None:
            m = TERMS_RE.search(email.get("body", ""))
            if m:
                due = date.fromisoformat(out["invoice_date"].value) + timedelta(days=int(m.group(1)))
                out["due_date"] = Field(due.isoformat(), round(min(0.85, out["invoice_date"].confidence), 4), "derived: payment terms")
        # currency: the chosen total's own currency, else the most common one mentioned in the email
        if total is not None and total.currency:
            out["currency"] = Field(total.currency, out["total_amount"].confidence, "learned: currency of total")
        elif out["total_amount"] is not None:
            seen = Counter(c.currency for _, c in scores.get("total_amount", []) if c.currency)
            text_codes = Counter(re.findall(rf"\b({_CUR})\b", email.get("subject", "") + " " + email.get("body", "")))
            pool = seen or text_codes
            if pool:
                out["currency"] = Field(pool.most_common(1)[0][0], 0.6, "learned: currency mentioned in email")
        return out

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = DEFAULT_MODEL_PATH) -> "LearnedExtractor":
        return joblib.load(path)
