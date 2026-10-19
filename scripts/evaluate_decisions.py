"""End-to-end replay of invoice emails through every stage up to the decision, scored against ground truth.
Also sweeps the auto-approve threshold so you can tune it.
Run:  python -m scripts.evaluate_decisions [--data data/synthetic_emails_hard.jsonl] [--anomaly-model models/x.joblib]

Ground truth: an invoice is 'bad' if it is a duplicate or an anomaly.
  automation rate - share of invoices approved with no human touch (want HIGH)
  leakage         - bad invoices that got auto-approved (want ZERO)
  false alarms    - good invoices that still went to a human or were rejected (want LOW)

Replay assumptions (they mirror the real app): the email classifier is NOT in the loop (ground-truth category is
used, so its mistakes are measured separately by train_classifier); vendor amount history only learns from
invoices that end up approved, assuming reviewers approve good invoices and reject bad ones."""
import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from app.pipeline.anomaly import AnomalyStage
from app.pipeline.base import PipelineContext
from app.pipeline.duplicates import DuplicateStage
from app.extraction.oracle import oracle_extractor
from app.pipeline.extract import ExtractStage
from app.pipeline.risk import RiskStage
from app.pipeline.validate import ValidateStage
from app.risk.decision import Thresholds, decide

# Cost model assumptions (edit to match your story). Printed with the results.
MINUTES_PER_MANUAL_INVOICE = 6
HOURLY_COST = 30.0


def replay(data: Path, anomaly_model: str | None, oracle: bool = False, extractor=None):
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    inv_history: list[dict] = []
    amounts: dict[str, list[float]] = defaultdict(list)
    activity: dict[str, list] = defaultdict(list)     # vendor -> [(arrival date, approved?)] for burst detection
    has_dates = all("received" in r for r in rows if r["category"] == "invoice")   # old datasets carry no dates: no bursts
    extract_stage = ExtractStage(extractor) if extractor else (ExtractStage(oracle_extractor) if oracle else ExtractStage())
    stages = [extract_stage, ValidateStage(),
              DuplicateStage(history_provider=lambda inv: inv_history),
              AnomalyStage(history_provider=lambda v: amounts[v], model_path=anomaly_model,
                           activity_provider=lambda v: activity[v] if has_dates else []),
              RiskStage()]
    out = []
    for r in rows:
        if r["category"] != "invoice":
            continue
        ctx = PipelineContext(email={"subject": r["subject"], "body": r["body"], "_truth": r["truth"],
                                      **({"received": r["received"]} if has_dates else {})})
        for s in stages:
            ctx = s.run(ctx)
        inv = ctx.invoice
        bad = bool(r["is_duplicate"] or r["is_anomaly"])
        d = decide(inv, ctx.risk_signals, inv["risk_score"], inv["risk_breakdown"], Thresholds())
        inv_history.append({"id": r["message_id"], **{k: inv.get(k) for k in
                            ("vendor_name", "invoice_number", "invoice_date", "total_amount")}})
        approved = d.outcome == "auto_approved" or (d.outcome == "human_review" and not bad)
        if approved and inv.get("total_amount") is not None and "duplicate_of" not in inv:
            amounts[inv.get("vendor_name")].append(inv["total_amount"])
        if has_dates and "duplicate_of" not in inv:
            activity[inv.get("vendor_name")].append((date.fromisoformat(r["received"]), approved))
        kind = r.get("dup_kind") or r.get("anomaly_kind") or ("twin" if r.get("twin") else None)
        out.append(dict(truth=r["truth"], bad=bad, dup=r["is_duplicate"], anom=r["is_anomaly"], kind=kind,
                        invoice=inv, signals=dict(ctx.risk_signals), score=inv["risk_score"], contrib=inv["risk_breakdown"]))
    return out


def evaluate(results, t: Thresholds):
    decisions = [(x, decide(x["invoice"], x["signals"], x["score"], x["contrib"], t)) for x in results]
    n = len(decisions)
    auto = [x for x, d in decisions if d.outcome == "auto_approved"]
    leak = [x for x in auto if x["bad"]]
    good = [(x, d) for x, d in decisions if not x["bad"]]
    false_alarm = [(x, d) for x, d in good if d.outcome != "auto_approved"]
    return decisions, auto, leak, good, false_alarm, n


def main(data: Path, anomaly_model: str | None, oracle: bool = False) -> None:
    results = replay(data, anomaly_model, oracle)
    t = Thresholds()
    decisions, auto, leak, good, false_alarm, n = evaluate(results, t)

    print(f"{n} invoices replayed | thresholds: approve<={t.auto_approve_max_risk} reject>={t.auto_reject_min_risk} "
          f"amount cap {t.auto_approve_max_amount:,.0f}\n")
    print("Decisions:", dict(Counter(d.outcome for _, d in decisions)))
    print(f"Automation rate:      {len(auto)/n:.1%}  ({len(auto)} of {n} approved with no human touch)")
    bad_total = sum(x['bad'] for x in results)
    print(f"Leakage:              {len(leak)} of {bad_total} bad invoices were auto-approved")
    if leak:
        print("   leaked by kind:", dict(Counter(x["kind"] for x in leak)))
    print(f"False alarms:         {len(false_alarm)} of {len(good)} good invoices went to a human or were rejected "
          f"({len(false_alarm)/len(good):.1%})")
    print("Rules behind false alarms:", dict(Counter(d.rule for _, d in false_alarm)))
    dups = [(x, d) for x, d in decisions if x["dup"]]
    anoms = [(x, d) for x, d in decisions if x["anom"]]
    print(f"Duplicates:           {sum(d.outcome == 'rejected' for _, d in dups)} rejected, "
          f"{sum(d.outcome == 'human_review' for _, d in dups)} sent to review, "
          f"{sum(d.outcome == 'auto_approved' for _, d in dups)} leaked, of {len(dups)}")
    print(f"Anomalies:            {sum(d.outcome != 'auto_approved' for _, d in anoms)} stopped "
          f"(review or reject), {sum(d.outcome == 'auto_approved' for _, d in anoms)} leaked, of {len(anoms)}")

    hours = len(auto) * MINUTES_PER_MANUAL_INVOICE / 60
    print(f"\nEstimated saving: {len(auto)} auto-approved x {MINUTES_PER_MANUAL_INVOICE} min = {hours:.1f} h "
          f"= ${hours*HOURLY_COST:,.0f} at ${HOURLY_COST:.0f}/h  (assumptions, not measured)")

    print("\nThreshold sweep (auto_approve_max_risk):")
    print("  threshold  automation  leakage  false_alarm_rate")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
        _, a, l, g, fa, nn = evaluate(results, Thresholds(auto_approve_max_risk=thr))
        print(f"  {thr:<9}  {len(a)/nn:>9.1%}  {len(l):>7}  {len(fa)/len(g):>15.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synthetic_emails.jsonl"))
    ap.add_argument("--anomaly-model", default=None, help="default: the model path from settings")
    ap.add_argument("--oracle-extraction", action="store_true", help="use true field values instead of the regex extractor")
    a = ap.parse_args()
    main(a.data, a.anomaly_model, a.oracle_extraction)
