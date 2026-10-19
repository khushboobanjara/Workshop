"""Regex vs learned extractor, honestly split.   Run: python -m scripts.compare_extractors

Training: the FIRST 70% of the hard dataset's invoices (chronological). Testing:
  A) the last 30% of the hard dataset            - same wording as training, unseen emails
  B) shifted (development set) and C) shifted_test (clean) - wording and formats never seen in training, i.e. what
     happens when a new vendor's mail arrives. B was used to find weaknesses and fix them, so C is the honest number.
Each test also runs the pipeline end to end (regex vs learned vs true fields), scored on the last 30%.
Anomaly detection uses the plain z-score rule here (no Isolation Forest) so the three runs differ ONLY in extraction."""
import json
from pathlib import Path

from app.extraction.learned import LearnedExtractor
from app.extraction.oracle import oracle_extractor
from app.extraction.regex_extractor import extract_fields
from app.risk.decision import Thresholds
from scripts.evaluate_decisions import evaluate, replay

HARD = Path("data/synthetic_emails_hard.jsonl")
SHIFTED = Path("data/synthetic_emails_shifted.jsonl")            # DEVELOPMENT set: used to find the extractor's weaknesses
SHIFTED_TEST = Path("data/synthetic_emails_shifted_test.jsonl")  # clean test set: other unseen wording, never used for development
FIELDS = ["vendor_name", "invoice_number", "invoice_date", "due_date", "currency", "total_amount", "po_number"]
REQUIRED = ["vendor_name", "invoice_number", "total_amount", "due_date"]
NO_FOREST = "models/none.joblib"      # missing file -> AnomalyStage falls back to the z-score rule only


def invoice_rows(path):
    return [r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip()) if r["category"] == "invoice"]


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 0.01
    return a == b


def field_table(rows, extractors):
    print(f"{'field':<16}" + "".join(f"{n:>34}" for n in extractors))
    print(f"{'':<16}" + "".join(f"{'found right / WRONG / missing':>34}" for _ in extractors))
    for k in FIELDS:
        cells = []
        for fn in extractors.values():
            c = w = m = 0
            for r in rows:
                got, truth = fn(r)[k], r["truth"][k]
                if got is None:
                    c, m = (c + 1, m) if truth is None else (c, m + 1)
                elif same(got.value, truth):
                    c += 1
                else:
                    w += 1
            n = c + w + m
            cells.append(f"{c / n:>8.1%} / {w:>4} / {m:>4}")
        print(f"{k:<16}" + "".join(f"{c:>34}" for c in cells))


def wrong_field(res):
    inv, truth = res["invoice"], res["truth"]
    return any(inv.get(k) is not None and not same(inv[k], truth[k]) for k in REQUIRED)


def end_to_end(path, extractors, frac=0.7):
    print(f"{'':<22}{'auto-approved':>15}{'bad auto-approved':>20}{'good sent to human':>20}{'approved w/ WRONG field':>26}")
    for name, ex in extractors.items():
        res = replay(path, NO_FOREST, extractor=ex)
        res = res[int(len(res) * frac):]
        decisions, auto, leak, good, false_alarm, n = evaluate(res, Thresholds())
        bad_total = sum(x["bad"] for x in res)
        wrong = sum(wrong_field(x) for x in auto)
        print(f"{name:<22}{len(auto) / n:>15.1%}{f'{len(leak)} of {bad_total}':>20}{len(false_alarm) / len(good):>20.1%}{f'{wrong} of {len(auto)}':>26}")


def main() -> None:
    hard = invoice_rows(HARD)
    cut = int(len(hard) * 0.7)
    learned = LearnedExtractor().fit(hard[:cut])
    regex = lambda r: extract_fields(r["subject"], r["body"])
    ml = lambda r: learned({"subject": r["subject"], "body": r["body"]})
    fields = {"regex": regex, "learned": ml}

    print(f"Trained on the first {cut} hard-dataset invoices.\n")
    print(f"=== A) Hard dataset, held-out last {len(hard) - cut} invoices (same wording as training)")
    field_table(hard[cut:], fields)
    for tag, path, note in (("B", SHIFTED, "development set - used to find and fix weaknesses"),
                            ("C", SHIFTED_TEST, "clean test set - different unseen wording, never used for development")):
        if path.exists():
            rows = invoice_rows(path)
            print(f"\n=== {tag}) {path.stem}: {len(rows)} invoices ({note})")
            field_table(rows, fields)

    pipeline_ex = {"regex": None, "learned": learned.__call__, "true fields (ceiling)": oracle_extractor}
    print("\n=== End to end, hard dataset, last 30% scored")
    end_to_end(HARD, pipeline_ex)
    if SHIFTED_TEST.exists():
        print("\n=== End to end, shifted_test (clean), last 30% scored (extractor trained on the hard dataset only)")
        end_to_end(SHIFTED_TEST, pipeline_ex)


if __name__ == "__main__":
    main()
