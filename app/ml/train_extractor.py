"""Train the learned field extractor used by the app.
Run:  python -m app.ml.train_extractor [--data a.jsonl b.jsonl ...] [--model-out models/field_extractor.joblib]

Fits on EVERY invoice in the given files (more wording variety = more robust). For the honest held-out numbers
(chronological split, plus unseen-wording test sets) run:  python -m scripts.compare_extractors"""
import argparse
import json
from pathlib import Path

from app.ml import registry
from app.extraction.learned import DEFAULT_MODEL_PATH, LearnedExtractor

DEFAULT_DATA = ["data/synthetic_emails.jsonl", "data/synthetic_emails_hard.jsonl", "data/synthetic_emails_shifted.jsonl"]


def main(data: list[str], model_out: str) -> None:
    rows = []
    for path in data:
        p = Path(path)
        if not p.exists():
            print(f"skipping {path} (not found)")
            continue
        rows += [r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip()) if r["category"] == "invoice"]
    model = LearnedExtractor().fit(rows)
    model.save(model_out)
    print(f"Trained on {len(rows)} invoice emails from {len(data)} files. Saved to {model_out}")
    registry.write_meta(model_out, "extractor", "LearnedExtractor (see app/extraction/learned.py)", {"training_emails": len(rows)},
                        None, {"files": [str(f) for f in data if Path(f).exists()], "rows": len(rows), "source": "synthetic"},
                        notes="Fit on every row, so there is no held-out score for this file. Held-out accuracy comes from "
                              "python -m scripts.compare_extractors.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    ap.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    a = ap.parse_args()
    main(a.data, a.model_out)
