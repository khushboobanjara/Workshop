import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import auth
from app.config import settings
from app.db import AuditLog, Base, ModelVersion, SessionLocal, engine
from app.main import create_app
from app.ml import registry, train_anomaly, train_classifier
from app.ml.versioning import file_version
from scripts.promote_model import promote
from tests.test_mailbox import light_pipeline


def fresh():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def model_file(tmp_path, name, content, with_meta=True, metrics=None):
    f = tmp_path / name
    f.write_bytes(content)
    if with_meta:
        registry.write_meta(f, "classifier", "test-algo", {"k": 1}, metrics or {"accuracy": 0.9, "macro_f1": 0.8},
                            {"name": "d.jsonl", "rows": 10, "source": "synthetic"})
    return f


def statuses():
    with SessionLocal() as s:
        return {(m.model_name, m.version): m.status for m in s.query(ModelVersion).all()}


def test_metadata_roundtrip_and_stale_detection(tmp_path):
    f = model_file(tmp_path, "m.joblib", b"v1")
    assert registry.read_meta(f)["version"] == file_version(f) and registry.read_meta(f)["metrics"]["accuracy"] == 0.9
    f.write_bytes(b"replaced without retraining")
    assert registry.read_meta(f) is None                                        # meta no longer describes this file
    assert registry.read_meta(tmp_path / "missing") is None


def test_sync_marks_the_served_file_production_and_demotes_the_old_one(tmp_path):
    fresh()
    v1 = model_file(tmp_path, "c.joblib", b"one")
    registry.sync_registry({"classifier": str(v1)}, tmp_path / "none")
    assert list(statuses().values()) == ["production"]
    v1.write_bytes(b"two")                                                      # the served file changes (e.g. after promotion)
    registry.write_meta(v1, "classifier", "test-algo", {}, {"accuracy": 0.95, "macro_f1": 0.9}, {"name": "d", "rows": 1, "source": "synthetic"})
    registry.sync_registry({"classifier": str(v1)}, tmp_path / "none")
    assert sorted(statuses().values()) == ["archived", "production"]


def test_candidates_are_listed_but_never_production(tmp_path):
    fresh()
    prod = model_file(tmp_path, "prod.joblib", b"prod")
    cand_dir = tmp_path / "cands"
    cand_dir.mkdir()
    model_file(cand_dir, "cand.joblib", b"cand")
    registry.sync_registry({"classifier": str(prod)}, cand_dir)
    assert sorted(statuses().values()) == ["candidate", "production"]


def test_a_model_without_metadata_is_flagged_not_invented(tmp_path):
    fresh()
    f = model_file(tmp_path, "old.joblib", b"legacy", with_meta=False)
    registry.sync_registry({"classifier": str(f)}, tmp_path / "none")
    with SessionLocal() as s:
        row = s.query(ModelVersion).one()
    assert row.metrics is None and "No evaluation metadata" in row.notes and row.status == "production"


def test_models_endpoint_labels_data_and_counts_audited_runs(tmp_path, monkeypatch):
    fresh()
    f = model_file(tmp_path, "c.joblib", b"served")
    version = file_version(f)
    monkeypatch.setattr(registry, "serving_paths", lambda: {"classifier": str(f)})
    monkeypatch.setattr(registry, "CANDIDATE_DIR", tmp_path / "no-candidates")        # ignore any real models/candidates/
    with SessionLocal() as s:
        s.add_all([AuditLog(entity_type="email", stage="classify", action="completed", model_version=version) for _ in range(3)])
        s.commit()
    with TestClient(create_app(light_pipeline)) as c:
        body = c.get("/models").json()
    assert "SYNTHETIC" in body["data_notice"]
    (entry,) = body["models"]["classifier"]
    assert entry["version"] == version and entry["status"] == "production" and entry["stage_runs_audited"] == 3
    assert entry["dataset"]["source"] == "synthetic"


def test_models_endpoint_is_open_to_every_role_but_not_to_anonymous(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", SecretStr("m" * 48))
    fresh()
    auth.create_user("viewer1", "long-enough-pass", "viewer")
    with TestClient(create_app(light_pipeline)) as c:
        assert c.get("/models").status_code == 401
        c.post("/auth/login", json={"username": "viewer1", "password": "long-enough-pass"})
        assert c.get("/models").status_code == 200


def test_classifier_training_records_honest_metrics(tmp_path):
    out = tmp_path / "clf.joblib"
    train_classifier.main(train_classifier.DATA, str(out))
    m = registry.read_meta(out)
    assert m and m["model_name"] == "classifier" and m["dataset"]["source"] == "synthetic"
    ev, metrics = m["metrics"]["evaluation"], m["metrics"]
    assert ev["data_source"] == "synthetic" and ev["n_test"] > 0 and "hold-out" in ev["split"]
    labels = metrics["confusion_matrix"]["labels"]
    assert len(metrics["confusion_matrix"]["matrix"]) == len(labels) and 0 <= metrics["accuracy"] <= 1
    assert sum(sum(r) for r in metrics["confusion_matrix"]["matrix"]) == ev["n_test"]


def test_anomaly_training_records_baseline_and_forest_with_error_rates(tmp_path):
    out = tmp_path / "an.joblib"
    train_anomaly.main(train_anomaly.DATA, str(out))
    m = registry.read_meta(out)["metrics"]
    for key in ("baseline_robust_z", "isolation_forest"):
        r = m[key]
        assert {"tp", "fp", "fn", "tn", "precision", "recall", "fpr", "fnr"} <= set(r)
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == m["evaluation"]["n_test"]
    assert "chronological" in m["evaluation"]["split"] and "refit on all data" in m["evaluation"]["note"]


def test_promotion_needs_metadata_and_explicit_confirmation(tmp_path, monkeypatch, capsys):
    serving = model_file(tmp_path, "serving.joblib", b"current", metrics={"accuracy": 0.80, "macro_f1": 0.7})
    cand = model_file(tmp_path, "cand.joblib", b"better", metrics={"accuracy": 0.90, "macro_f1": 0.85})
    bare = model_file(tmp_path, "bare.joblib", b"unmeasured", with_meta=False)
    monkeypatch.setattr(registry, "serving_paths", lambda: {"classifier": str(serving)})
    arch = str(tmp_path / "archive")
    assert promote("classifier", str(bare), yes=True, archive_dir=arch) == 1             # unmeasured: refused
    assert serving.read_bytes() == b"current"
    assert promote("classifier", str(cand), yes=False, archive_dir=arch) == 0            # dry run changes nothing
    assert serving.read_bytes() == b"current" and "Dry run" in capsys.readouterr().out
    assert promote("classifier", str(cand), yes=True, archive_dir=arch) == 0
    assert serving.read_bytes() == b"better" and registry.read_meta(serving)["metrics"]["accuracy"] == 0.90
    assert len(list((tmp_path / "archive").glob("classifier-*.joblib"))) == 1            # the old model is kept
    assert promote("nonsense", str(cand), yes=True) == 2
