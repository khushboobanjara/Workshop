"""Run a few synthetic emails through the FULL pipeline and print each decision.
Run:  python -m scripts.try_pipeline"""
import json
from pathlib import Path

from app.db import init_db
from app.pipeline.anomaly import AnomalyStage
from app.pipeline.base import Pipeline, PipelineContext
from app.pipeline.classify import ClassifyStage
from app.pipeline.decision import DecisionStage
from app.pipeline.duplicates import DuplicateStage
from app.pipeline.extract import ExtractStage
from app.pipeline.risk import RiskStage
from app.pipeline.validate import ValidateStage

init_db()
rows = [json.loads(l) for l in Path("data/synthetic_emails.jsonl").read_text().splitlines()]
pipeline = Pipeline([ClassifyStage(), ExtractStage(), ValidateStage(), DuplicateStage(),
                     AnomalyStage(), RiskStage(), DecisionStage()])

# one normal invoice, one anomaly, one non-invoice
picks = [next(r for r in rows if r["category"] == "invoice" and not r["is_anomaly"] and not r["is_duplicate"]),
         next(r for r in rows if r["is_anomaly"]),
         next(r for r in rows if r["category"] == "newsletter")]

for row in picks:
    ctx = pipeline.run(PipelineContext(email={"subject": row["subject"], "body": row["body"]}))
    inv = ctx.invoice
    print(f"\n[{row['category']}{' / anomaly' if row['is_anomaly'] else ''}] {row['subject']}")
    print("  decision:", ctx.decision, "| risk:", inv.get("risk_score"), "| rule:", inv.get("decision_rule"))
    print("  signals :", ctx.risk_signals)
    for f in ctx.flags:
        print("  -", f)
