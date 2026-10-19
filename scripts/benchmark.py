"""Run every evaluation on BOTH datasets and save the raw reports to docs/results/.
Run:  python -m scripts.benchmark          (about a minute)

Models are trained to separate files (models/bench_*.joblib), so your app's models are not touched."""
import subprocess
import sys
from pathlib import Path

DATASETS = {"easy": "data/synthetic_emails.jsonl", "hard": "data/synthetic_emails_hard.jsonl"}
OUT = Path("docs/results")


def run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(args)}\n{r.stdout}\n{r.stderr}")
    return r.stdout.rstrip()


def main() -> None:
    for path, flag in ((DATASETS["hard"], None), ("data/synthetic_emails_shifted.jsonl", "--shifted"),
                       ("data/synthetic_emails_shifted_test.jsonl", "--shifted-test")):
        if not Path(path).exists():
            run("-m", "scripts.generate_hard_data", *([flag] if flag else []))
    OUT.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    for name, data in DATASETS.items():
        clf, ano = f"models/bench_{name}_classifier.joblib", f"models/bench_{name}_anomaly.joblib"
        sections = [
            ("EMAIL CLASSIFIER", run("-m", "app.ml.train_classifier", "--data", data, "--model-out", clf)),
            ("FIELD EXTRACTION", run("-m", "scripts.evaluate_extraction", "--data", data)),
            ("DUPLICATE DETECTION", run("-m", "scripts.evaluate_duplicates", "--data", data)),
            ("ANOMALY DETECTION", run("-m", "app.ml.train_anomaly", "--data", data, "--model-out", ano)),
            ("END TO END DECISIONS", run("-m", "scripts.evaluate_decisions", "--data", data, "--anomaly-model", ano)),
        ]
        text = f"OpsPilot AI benchmark - dataset: {name} ({data})\n" + "".join(f"\n{'=' * 70}\n{t}\n{'=' * 70}\n{body}\n" for t, body in sections)
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"wrote {OUT / (name + '.txt')}")

    # Same hard data, but later stages receive the TRUE field values: separates detection quality from extraction quality.
    data = DATASETS["hard"]
    ano = "models/bench_hard_oracle_anomaly.joblib"
    sections = [
        ("DUPLICATE DETECTION", run("-m", "scripts.evaluate_duplicates", "--data", data, "--oracle-extraction")),
        ("ANOMALY DETECTION", run("-m", "app.ml.train_anomaly", "--data", data, "--model-out", ano, "--oracle-extraction")),
        ("END TO END DECISIONS", run("-m", "scripts.evaluate_decisions", "--data", data, "--anomaly-model", ano, "--oracle-extraction")),
    ]
    text = "OpsPilot AI benchmark - dataset: hard, with ORACLE extraction (true field values; isolates detection quality)\n" + \
        "".join(f"\n{'=' * 70}\n{t}\n{'=' * 70}\n{body}\n" for t, body in sections)
    (OUT / "hard_oracle.txt").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'hard_oracle.txt'}")

    # Regex vs learned extractor, with a chronological split and unseen-wording test sets
    (OUT / "extractors.txt").write_text("OpsPilot AI benchmark - regex vs learned field extractor\n\n" +
                                        run("-m", "scripts.compare_extractors") + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'extractors.txt'}")


if __name__ == "__main__":
    main()
