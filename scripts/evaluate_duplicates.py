"""Replay invoice emails in arrival order and score duplicate detection against ground truth.
Run:  python -m scripts.evaluate_duplicates [--data data/synthetic_emails_hard.jsonl]

'twin' rows (hard dataset) are legitimate look-alikes: two real, identical orders. They are NOT duplicates,
so flagging them counts as a false positive - that is the price of the 'resubmitted' rule."""
import argparse
import json
from collections import Counter
from pathlib import Path

from app.duplicates.matcher import find_duplicate
from app.extraction.oracle import fields_for


def main(data: Path, oracle: bool = False) -> None:
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    history = []
    tp = fp = fn = tn = 0
    caught, missed, fp_cause, tp_type = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        if r["category"] != "invoice":
            continue
        f = fields_for(r, oracle)
        inv = {k: (v.value if v else None) for k, v in f.items()}
        m = find_duplicate(inv, history)
        kind = r.get("dup_kind") or ("exact" if r["is_duplicate"] else None)
        if m and r["is_duplicate"]:
            tp += 1; caught[kind] += 1; tp_type[m.kind] += 1
        elif m:
            fp += 1; fp_cause["legit look-alike (twin)" if r.get("twin") else "other"] += 1
        elif r["is_duplicate"]:
            fn += 1; missed[kind] += 1
        else:
            tn += 1
        history.append({"id": r["message_id"], **inv})

    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  precision={prec:.1%} recall={rec:.1%}")
    if tp + fn:
        print("\nBy duplicate kind (caught / total):")
        for k in sorted(set(caught) | set(missed)):
            print(f"  {k:<16}{caught[k]:>4} / {caught[k] + missed[k]}")
    if fp:
        print("False positives by cause:", dict(fp_cause))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synthetic_emails.jsonl"))
    ap.add_argument("--oracle-extraction", action="store_true", help="use true field values instead of the regex extractor")
    a = ap.parse_args()
    main(a.data, a.oracle_extraction)
