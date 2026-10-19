"""Train + evaluate the anomaly detector.
Run:  python -m app.ml.train_anomaly [--data data/synthetic_emails_hard.jsonl] [--model-out models/x.joblib]

Replays invoices in arrival order. Each invoice is featurized against its vendor's PRIOR invoices only,
then we split chronologically (train on the first 70%, test on the last 30%), which mimics deploying the
model on future data."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.anomaly.detector import AmountAnomalyDetector
from app.anomaly.features import MIN_HISTORY, featurize
from app.config import settings
from app.ml import registry
from app.extraction.oracle import fields_for

DATA = Path("data/synthetic_emails.jsonl")
Z_THRESHOLD = 4.0   # baseline rule: |robust z| above this = anomaly
IF_CUTOFF = 0.95


def build_dataset(data: Path = DATA, oracle: bool = False):
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    history = defaultdict(list)
    X, y, kinds = [], [], []
    stats = Counter()
    for r in rows:
        if r["category"] != "invoice" or r["is_duplicate"]:
            continue                                 # duplicates are the duplicate detector's job
        f = fields_for(r, oracle)
        vendor = f["vendor_name"].value if f["vendor_name"] else None
        amount = f["total_amount"].value if f["total_amount"] else None
        if vendor is None or amount is None:
            stats["unextractable"] += 1
            stats["unextractable_anomalies"] += int(r["is_anomaly"])
            continue
        feats = featurize(amount, history[vendor])
        history[vendor].append(amount)
        if feats is None:
            stats["cold_start"] += 1
            stats["cold_start_anomalies"] += int(r["is_anomaly"])
            continue
        X.append(feats); y.append(int(r["is_anomaly"])); kinds.append(r.get("anomaly_kind") or "normal")
    return np.array(X), np.array(y), np.array(kinds), stats


def prf(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, p, r


def rates(y_true, y_pred) -> dict:
    tp, fp, fn, p, r = prf(y_true, y_pred)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0}


def main(data: Path = DATA, model_out: str | None = None, oracle: bool = False) -> None:
    model_out = model_out or settings.anomaly_model_path
    X, y, kinds, stats = build_dataset(data, oracle)
    cut = int(len(X) * 0.7)
    X_tr, X_te, y_tr, y_te, k_te = X[:cut], X[cut:], y[:cut], y[cut:], kinds[cut:]
    print(f"{len(X)} scored invoices ({stats['cold_start']} skipped: vendor cold start < {MIN_HISTORY} prior invoices, "
          f"{stats['unextractable']} skipped: vendor/amount not extracted)")
    if stats["cold_start_anomalies"] or stats["unextractable_anomalies"]:
        print(f"NOT SCORED: {stats['cold_start_anomalies']} anomalies hidden by cold start, "
              f"{stats['unextractable_anomalies']} by failed extraction (counted as misses in the end-to-end report)")
    print(f"train={len(X_tr)} test={len(X_te)} | anomalies in test: {int(y_te.sum())}\n")

    base_pred = (np.abs(X_te[:, 1]) > Z_THRESHOLD).astype(int)
    tp, fp, fn, p, r = prf(y_te, base_pred)
    print(f"Baseline |robust z| > {Z_THRESHOLD}:  TP={tp} FP={fp} FN={fn}  precision={p:.1%} recall={r:.1%}")

    det = AmountAnomalyDetector().fit(X_tr)
    sig = np.array([det.signal(x) for x in X_te])
    for thr in (0.90, 0.95, 0.97):
        tp, fp, fn, p, r = prf(y_te, (sig >= thr).astype(int))
        print(f"IsolationForest signal >= {thr}:  TP={tp} FP={fp} FN={fn}  precision={p:.1%} recall={r:.1%}")
    if_pred = (sig >= IF_CUTOFF).astype(int)
    by_threshold = {str(thr): rates(y_te, (sig >= thr).astype(int)) for thr in (0.90, 0.95, 0.97)}

    anomaly_kinds = sorted(set(k_te[y_te == 1]))
    if len(anomaly_kinds) > 1:
        print(f"\nRecall by anomaly kind in the test split (caught / total):  baseline | IsolationForest@{IF_CUTOFF}")
        for k in anomaly_kinds:
            m = k_te == k
            print(f"  {k:<12}{int(base_pred[m].sum()):>3} / {int(m.sum()):<3}   |   {int(if_pred[m].sum()):>3} / {int(m.sum())}")

    # Final model uses all data
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    AmountAnomalyDetector().fit(X).save(model_out)
    print(f"\nSaved model -> {model_out}")

    registry.write_meta(model_out, "anomaly", "Isolation Forest on per-vendor amount features (robust z, ratio to median)",
        {"if_cutoff": IF_CUTOFF, "z_threshold": Z_THRESHOLD, "min_vendor_history": MIN_HISTORY},
        {"evaluation": {"data_source": "synthetic" + (" (true field values, no extraction errors)" if oracle else ""),
                        "split": "chronological: first 70% train, last 30% test", "n_train": len(X_tr), "n_test": len(X_te),
                        "anomalies_in_test": int(y_te.sum()),
                        "note": "Measured on a model trained on the first 70%; the saved model is then refit on all data, "
                                "so these numbers describe the method, not that exact file."},
         "baseline_robust_z": rates(y_te, base_pred), "isolation_forest": {**rates(y_te, if_pred), "cutoff": IF_CUTOFF},
         "isolation_forest_by_cutoff": by_threshold,
         "skipped": {"cold_start": stats["cold_start"], "unextractable": stats["unextractable"],
                     "cold_start_anomalies": stats["cold_start_anomalies"]}},
        {"name": data.name, "rows": len(X), "source": "synthetic"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--model-out", default=None, help="default: the model path from settings")
    ap.add_argument("--oracle-extraction", action="store_true", help="use true field values instead of the regex extractor")
    a = ap.parse_args()
    main(a.data, a.model_out, a.oracle_extraction)
