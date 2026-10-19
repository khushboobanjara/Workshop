"""Model registry. Training writes <model file>.meta.json next to each model (no database needed, so it also works
inside a Docker build); at app start the registry records every model file it can see and marks the ones that are
actually being served as "production". Anything else is a "candidate" until a person promotes it.

Honesty rule: every metric carries the data it was measured on. Nothing here is real-world performance."""
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func

from app.config import settings
from app.db import AuditLog, ModelVersion, SessionLocal
from app.ml.versioning import file_version

META_SUFFIX = ".meta.json"
CANDIDATE_DIR = Path("models/candidates")
DATA_NOTICE = ("All metrics on this page were measured on SYNTHETIC data (held-out splits of generated emails). "
               "They show how the models behave on that data, not how they would perform on real invoices.")
NO_META_NOTE = "No evaluation metadata recorded for this file. Retrain it to record how it was measured."


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def meta_path(model_path) -> Path:
    return Path(str(model_path) + META_SUFFIX)


def write_meta(model_path, model_name: str, algorithm: str, params: dict, metrics: dict | None,
               dataset: dict, notes: str | None = None) -> dict:
    meta = {"model_name": model_name, "version": file_version(model_path), "algorithm": algorithm, "params": params,
            "metrics": metrics, "dataset": dataset, "notes": notes,
            "trained_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")}
    meta_path(model_path).write_text(json.dumps(meta, indent=2, default=_jsonable))
    return meta


def read_meta(model_path) -> dict | None:
    """None if missing, unreadable, or written for a DIFFERENT file (the model was replaced without retraining)."""
    try:
        meta = json.loads(meta_path(model_path).read_text())
    except (OSError, ValueError):
        return None
    return meta if meta.get("version") == file_version(model_path) else None


def serving_paths() -> dict[str, str]:
    from app.extraction.learned import DEFAULT_MODEL_PATH
    return {"classifier": settings.classifier_model_path, "anomaly": settings.anomaly_model_path,
            "extractor": DEFAULT_MODEL_PATH}


def _upsert(s, name: str, path, serving: bool) -> None:
    version = file_version(path)
    if version is None:
        return
    meta = read_meta(path)
    row = s.query(ModelVersion).filter_by(model_name=name, version=version).first()
    if row is None:
        m = meta or {}
        row = ModelVersion(model_name=name, version=version, status="candidate", algorithm=m.get("algorithm"),
                           params=m.get("params"), metrics=m.get("metrics"), dataset=m.get("dataset"),
                           notes=m.get("notes") if meta else NO_META_NOTE, file_path=str(path),
                           trained_at=datetime.fromisoformat(m["trained_at"]) if m.get("trained_at") else None)
        s.add(row)
    if serving:
        row.status = "production"
        for other in s.query(ModelVersion).filter(ModelVersion.model_name == name, ModelVersion.version != version,
                                                  ModelVersion.status == "production"):
            other.status = "archived"


def sync_registry(serving: dict[str, str] | None = None, candidate_dir: Path | None = None) -> None:
    serving = serving or serving_paths()
    with SessionLocal() as s:
        for name, path in serving.items():
            _upsert(s, name, path, serving=True)
        for meta_file in sorted((candidate_dir or CANDIDATE_DIR).glob("*" + META_SUFFIX)):
            model_file = Path(str(meta_file)[: -len(META_SUFFIX)])
            try:
                name = json.loads(meta_file.read_text())["model_name"]
            except (OSError, ValueError, KeyError):
                continue
            _upsert(s, name, model_file, serving=False)
        s.commit()


def list_models() -> dict:
    with SessionLocal() as s:
        runs = dict(s.query(AuditLog.model_version, func.count()).filter(AuditLog.model_version.is_not(None))
                    .group_by(AuditLog.model_version).all())
        rows = s.query(ModelVersion).order_by(ModelVersion.model_name, ModelVersion.registered_at.desc(), ModelVersion.id.desc()).all()
        models: dict[str, list] = {}
        for r in rows:
            models.setdefault(r.model_name, []).append({
                "version": r.version, "status": r.status, "algorithm": r.algorithm, "params": r.params,
                "metrics": r.metrics, "dataset": r.dataset, "notes": r.notes,
                "trained_at": r.trained_at.isoformat() if r.trained_at else None,
                "stage_runs_audited": runs.get(r.version, 0)})
    return {"data_notice": DATA_NOTICE, "models": models}


def headline(meta: dict | None) -> dict:
    """The few numbers shown when comparing a candidate with the production model."""
    m = (meta or {}).get("metrics") or {}
    if "accuracy" in m:
        return {"accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)}
    if "isolation_forest" in m:
        i = m["isolation_forest"]
        return {"precision": round(i["precision"], 4), "recall": round(i["recall"], 4), "false_positive_rate": round(i["fpr"], 4)}
    return {}
