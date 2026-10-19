"""Score the extractor against ground truth.   Run: python -m scripts.evaluate_extraction"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.extraction.regex_extractor import extract_fields

FIELDS = ["vendor_name", "invoice_number", "invoice_date", "due_date", "currency", "total_amount", "po_number"]


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 0.01
    return a == b


def main(data: Path = Path("data/synthetic_emails.jsonl")):
    rows = [json.loads(l) for l in data.read_text().splitlines()]
    rows = [r for r in rows if r["category"] == "invoice"]
    stat = defaultdict(lambda: {"correct": 0, "wrong": 0, "missing": 0})
    hi = {"n": 0, "ok": 0}
    lo = {"n": 0, "ok": 0}

    for r in rows:
        got = extract_fields(r["subject"], r["body"])
        for k in FIELDS:
            truth, f = r["truth"][k], got[k]
            if f is None:
                stat[k]["correct" if truth is None else "missing"] += 1
                continue
            ok = same(f.value, truth)
            stat[k]["correct" if ok else "wrong"] += 1
            if k != "po_number":
                bucket = hi if f.confidence >= 0.9 else lo
                bucket["n"] += 1
                bucket["ok"] += ok

    print(f"{len(rows)} invoice emails\n")
    print(f"{'field':<16}{'correct':>9}{'wrong':>8}{'missing':>9}{'accuracy':>10}")
    for k in FIELDS:
        s = stat[k]
        n = sum(s.values())
        print(f"{k:<16}{s['correct']:>9}{s['wrong']:>8}{s['missing']:>9}{s['correct']/n:>10.1%}")
    print(f"\nConfidence check: values with conf >= 0.9 are right {hi['ok']/max(hi['n'],1):.1%} of the time "
          f"({hi['n']} values); below 0.9: {lo['ok']/max(lo['n'],1):.1%} ({lo['n']} values)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synthetic_emails.jsonl"))
    main(ap.parse_args().data)
