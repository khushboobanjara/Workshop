import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, inspect, text

from app import db
from app.config import settings
from app.db import AuditLog, Base, Email, Invoice, SessionLocal, engine
from app.main import create_app
from app.pipeline.base import Pipeline, Stage
from app.pipeline.extract import ExtractStage
from tests.test_api import StubClassifier

INVOICE_TEXT = ("Vendor: Acme Supplies Ltd\nInvoice INV-55555\nInvoice Date: 2026-04-01\n"
                "Due Date: 2026-04-30\nTotal Amount: USD 980.00")


class Boom(Stage):
    name = "boom"

    def run(self, ctx):
        raise RuntimeError("extractor crashed")


def make(factory=lambda: Pipeline([StubClassifier(), ExtractStage()])):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return TestClient(create_app(factory), raise_server_exceptions=False)


def post_unsure(c, mid="u1", body=INVOICE_TEXT):
    return c.post("/emails", json={"message_id": mid, "subject": "unsure what this is", "body": body, "sender": "x@y.example"}).json()


def test_unsure_email_creates_no_invoice_and_shows_in_triage_counts():
    with make() as c:
        out = post_unsure(c)
        assert out["decision"] == "needs_triage" and out["invoice"] is None
        assert c.get("/metrics").json()["invoices"] == 0 and c.get("/review-queue").json() == []
        t = c.get("/triage").json()
        assert len(t) == 1 and t[0]["sender"] == "x@y.example" and "INV-55555" in t[0]["preview"]
        assert c.get("/stats").json()["needs_triage"] == 1
        again = post_unsure(c)                                          # re-delivery of the same email
        assert again["already_processed"] and again["decision"] == "needs_triage" and len(c.get("/triage").json()) == 1


def test_dismiss_removes_it_and_is_audited():
    with make() as c:
        eid = post_unsure(c)["email_id"]
        r = c.post(f"/triage/{eid}", json={"action": "dismiss", "reviewer": "asha", "comment": "newsletter"})
        assert r.status_code == 200 and r.json()["invoice"] is None
        assert c.get("/triage").json() == [] and c.get("/metrics").json()["invoices"] == 0
        with SessionLocal() as s:
            row = s.query(AuditLog).filter_by(stage="triage").one()
        assert row.actor == "asha" and row.action == "dismissed_not_invoice" and row.entity_id == eid
        assert c.post(f"/triage/{eid}", json={"action": "dismiss", "reviewer": "asha"}).status_code == 409


def test_convert_runs_the_rest_of_the_pipeline_once():
    with make() as c:
        eid = post_unsure(c)["email_id"]
        r = c.post(f"/triage/{eid}", json={"action": "convert", "reviewer": "asha"})
        assert r.status_code == 200
        inv = r.json()["invoice"]
        assert inv["total_amount"] == 980.0 and inv["invoice_number"] == "INV-55555"
        assert c.get("/triage").json() == [] and c.get("/metrics").json()["invoices"] == 1
        with SessionLocal() as s:
            assert s.get(Email, eid).category == "invoice" and s.query(Invoice).count() == 1
        assert c.post(f"/triage/{eid}", json={"action": "convert", "reviewer": "asha"}).status_code == 409
        with SessionLocal() as s:
            assert s.query(Invoice).count() == 1                        # the second click did not create another


def test_failed_convert_gives_the_email_back():
    class Gate(Boom):                                                   # only blows up once the email is forced past the classifier
        def run(self, ctx):
            if ctx.email.get("category") == "invoice":
                raise RuntimeError("extractor crashed")
            return ctx

    with make(lambda: Pipeline([StubClassifier(), Gate()])) as c:
        eid = post_unsure(c)["email_id"]
        assert c.post(f"/triage/{eid}", json={"action": "convert", "reviewer": "asha"}).status_code == 500
        assert [t["email_id"] for t in c.get("/triage").json()] == [eid]   # still waiting, nothing lost
        with SessionLocal() as s:
            assert s.query(Invoice).count() == 0


def test_input_validation_and_unknown_ids():
    with make() as c:
        eid = post_unsure(c)["email_id"]
        assert c.post(f"/triage/{eid}", json={"action": "convert", "reviewer": ""}).status_code == 422
        assert c.post(f"/triage/{eid}", json={"action": "delete", "reviewer": "a"}).status_code == 422
        assert c.post("/triage/9999", json={"action": "dismiss", "reviewer": "a"}).status_code == 404


def test_triage_requires_the_api_key_when_one_is_set(monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr("k"))
    with make() as c:
        assert c.get("/triage").status_code == 401 and c.post("/triage/1", json={}).status_code == 401
        assert c.get("/triage", headers={"X-API-Key": "k"}).status_code == 200


def test_old_database_gets_the_new_column_without_losing_rows(tmp_path, monkeypatch):
    old = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with old.begin() as conn:                                           # schema as it was before triage existed
        conn.execute(text("CREATE TABLE emails (id INTEGER PRIMARY KEY, message_id VARCHAR(200), sender VARCHAR(200), "
                          "subject VARCHAR(500), body TEXT, received_at DATETIME, category VARCHAR(50), category_confidence FLOAT)"))
        conn.execute(text("INSERT INTO emails (id, message_id, sender, subject, body) VALUES (1, 'm', 's', 'sub', 'b')"))
    monkeypatch.setattr(db, "engine", old)
    db._add_missing_columns()
    db._add_missing_columns()                                           # idempotent
    assert "triage_status" in {c["name"] for c in inspect(old).get_columns("emails")}
    with old.connect() as conn:
        assert conn.execute(text("SELECT message_id, triage_status FROM emails")).one() == ("m", None)
