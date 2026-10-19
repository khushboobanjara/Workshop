import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError

from app import db
from app.db import AuditImmutableError, AuditLog, Base, SessionLocal, engine
from app.main import create_app
from app.ml.versioning import file_version
from app.pipeline.base import Pipeline, PipelineContext, Stage
from tests.test_mailbox import light_pipeline


def fresh():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def test_every_audit_row_of_a_request_shares_its_request_id_and_it_is_returned():
    fresh()
    with TestClient(create_app(light_pipeline)) as c:
        r = c.post("/emails", json={"message_id": "r1", "subject": "Invoice INV-1", "body": "Total Amount: USD 5.00"},
                   headers={"X-Request-ID": "trace-abc-12345"})
        assert r.headers["x-request-id"] == "trace-abc-12345"
        bad = c.get("/health", headers={"X-Request-ID": "<script>alert(1)</script>"})
        assert bad.headers["x-request-id"] != "<script>alert(1)</script>" and len(bad.headers["x-request-id"]) == 32
    with SessionLocal() as s:
        rows = s.query(AuditLog).all()
    assert rows and {a.request_id for a in rows} == {"trace-abc-12345"}


def test_audit_rows_cannot_be_updated_or_deleted():
    fresh()
    with SessionLocal() as s:
        s.add(AuditLog(entity_type="x", stage="t", action="a"))
        s.commit()
        row = s.query(AuditLog).one()
        row.actor = "tampered"
        with pytest.raises(AuditImmutableError):
            s.commit()
        s.rollback()
        s.delete(s.query(AuditLog).one())
        with pytest.raises(AuditImmutableError):
            s.commit()
    with engine.begin() as conn:                                  # raw SQL is stopped by the database itself
        with pytest.raises(DBAPIError, match="append-only"):
            conn.execute(text("UPDATE audit_log SET actor='x'"))
    with engine.begin() as conn:
        with pytest.raises(DBAPIError, match="append-only"):
            conn.execute(text("DELETE FROM audit_log"))


def test_model_version_is_recorded_per_stage(tmp_path):
    class Versioned(Stage):
        name, model_version = "versioned", "sha-abc123"

        def run(self, ctx):
            return ctx

    class Plain(Stage):
        name = "plain"

        def run(self, ctx):
            return ctx

    fresh()
    Pipeline([Versioned(), Plain()]).run(PipelineContext(email={"subject": "s", "body": "b"}))
    with SessionLocal() as s:
        got = {a.stage: a.model_version for a in s.query(AuditLog).all()}
    assert got == {"versioned": "sha-abc123", "plain": None}

    f = tmp_path / "m.bin"
    f.write_bytes(b"model-v1")
    v1 = file_version(f)
    assert v1 == file_version(f) and v1.startswith("sha-") and len(v1) == 16
    f.write_bytes(b"model-v2")
    assert file_version(f) != v1 and file_version(tmp_path / "missing") is None


def test_old_audit_table_gets_the_new_columns_and_triggers(tmp_path, monkeypatch):
    old = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with old.begin() as conn:
        conn.execute(text("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, entity_type VARCHAR(30), entity_id INTEGER, "
                          "stage VARCHAR(50), action VARCHAR(100), actor VARCHAR(50), details JSON, duration_ms FLOAT, timestamp DATETIME)"))
        conn.execute(text("INSERT INTO audit_log (entity_type, stage, action, actor) VALUES ('email','s','a','system')"))
    monkeypatch.setattr(db, "engine", old)
    db._add_missing_columns()
    db._ensure_audit_triggers()
    cols = {c["name"] for c in inspect(old).get_columns("audit_log")}
    assert {"request_id", "model_version"} <= cols
    with old.begin() as conn:
        assert conn.execute(text("SELECT count(*) FROM audit_log")).scalar() == 1          # history kept
        with pytest.raises(DBAPIError, match="append-only"):
            conn.execute(text("DELETE FROM audit_log"))
