"""End-to-end API tests against a temporary test database (MySQL, see conftest.py).
Uses a pipeline WITHOUT the ML classifier/anomaly model so they run on a fresh checkout."""
import pytest
from fastapi.testclient import TestClient

from app.db import Base, engine
from app.main import create_app
from app.pipeline.anomaly import AnomalyStage
from app.pipeline.base import Pipeline, Stage
from app.pipeline.decision import DecisionStage
from app.pipeline.duplicates import DuplicateStage
from app.pipeline.extract import ExtractStage
from app.pipeline.risk import RiskStage
from app.pipeline.validate import ValidateStage


def light_pipeline():
    return Pipeline([ExtractStage(), ValidateStage(), DuplicateStage(),
                     AnomalyStage(model_path="no_such_model.joblib"), RiskStage(), DecisionStage()])


class StubClassifier(Stage):
    """Mimics ClassifyStage outcomes without needing the ML model."""
    name = "classify"

    def run(self, ctx):
        subject = ctx.email.get("subject", "")
        if subject.startswith("newsletter"):
            ctx.email["category"], ctx.decision, ctx.stop = "newsletter", "not_invoice", True
        elif subject.startswith("unsure"):
            ctx.email["category"], ctx.email["category_confidence"] = "invoice", 0.4
            ctx.flags.append("Low classification confidence (0.40) - needs human check")
            ctx.decision, ctx.stop = "human_review", True
        return ctx


def make_email(msg_id, number="INV-10001", amount=1000.0, vendor="Acme Supplies Ltd", with_due=True):
    due = "Due Date: 2026-03-31\n" if with_due else ""
    body = (f"Hello,\n\nPlease find invoice {number} attached.\nVendor: {vendor}\n"
            f"Invoice Date: 2026-03-01\n{due}Total Amount: USD {amount:.2f}\n\nRegards,\n{vendor}")
    return {"message_id": msg_id, "sender": "billing@example.com", "subject": f"{vendor} invoice {number}", "body": body}


@pytest.fixture
def make_client():
    opened = []

    def _make(factory=light_pipeline):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        c = TestClient(create_app(factory))
        c.__enter__()
        opened.append(c)
        return c

    yield _make
    for c in opened:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def post(client, **kw):
    r = client.post("/emails", json=make_email(**kw))
    assert r.status_code == 200, r.text
    return r.json()


def build_history(client, n=4):
    for i, amt in enumerate([1000.0, 1040.0, 980.0, 1020.0][:n]):
        post(client, msg_id=f"hist-{i}", number=f"INV-3000{i}", amount=amt)


def test_clean_invoice_is_auto_approved_and_persisted(client):
    out = post(client, msg_id="m1")
    assert out["decision"] == "auto_approved" and out["invoice"]["status"] == "approved"
    inv = client.get(f"/invoices/{out['invoice']['id']}").json()
    assert inv["vendor_name"] == "Acme Supplies Ltd" and inv["total_amount"] == 1000.0
    assert inv["email"]["subject"].startswith("Acme")

    audit = client.get(f"/invoices/{inv['id']}/audit").json()
    stages = [a["stage"] for a in audit]
    assert "extract" in stages and "risk" in stages and stages[-1] == "decision"


def test_same_message_twice_is_idempotent(client):
    first = post(client, msg_id="dup-msg")
    second = post(client, msg_id="dup-msg")
    assert second["already_processed"] is True and second["invoice"]["id"] == first["invoice"]["id"]
    assert client.get("/metrics").json()["invoices"] == 1


def test_exact_duplicate_is_rejected_and_linked(client):
    first = post(client, msg_id="a", number="INV-20001", amount=1500.0)
    second = post(client, msg_id="b", number="INV-20001", amount=1500.0)
    assert second["decision"] == "rejected" and second["invoice"]["status"] == "rejected"
    assert second["invoice"]["duplicate_of"] == first["invoice"]["id"]


def test_anomaly_goes_to_review_queue_then_human_approves(client):
    build_history(client)
    out = post(client, msg_id="big", number="INV-30009", amount=6000.0)
    assert out["decision"] == "human_review" and out["invoice"]["status"] == "pending_review"
    assert out["invoice"]["risk_breakdown"]["anomaly"] >= 0.7

    queue = client.get("/review-queue").json()
    assert [q["id"] for q in queue] == [out["invoice"]["id"]]
    assert queue[0]["email"]["body"]

    inv_id = out["invoice"]["id"]
    r = client.post(f"/invoices/{inv_id}/review", json={"action": "approve", "reviewer": "alice", "comment": "confirmed with vendor"})
    assert r.status_code == 200 and r.json()["status"] == "approved" and r.json()["reviewed_by"] == "alice"
    assert r.json()["decision"] == "human_review"            # the system's original decision is preserved
    assert client.get("/review-queue").json() == []

    human = [a for a in client.get(f"/invoices/{inv_id}/audit").json() if a["stage"] == "human_review"]
    assert human and human[0]["actor"] == "alice" and human[0]["details"]["comment"] == "confirmed with vendor"

    again = client.post(f"/invoices/{inv_id}/review", json={"action": "reject", "reviewer": "bob"})
    assert again.status_code == 409


def test_pending_anomaly_does_not_pollute_vendor_history(client):
    build_history(client)
    post(client, msg_id="big1", number="INV-30008", amount=6000.0)       # waits in the queue
    second = post(client, msg_id="big2", number="INV-30009", amount=6100.0)
    assert second["decision"] == "human_review"                           # still unusual: 6000 was not learned as normal


def test_cannot_approve_missing_fields_until_corrected(client):
    out = post(client, msg_id="nodue", number="INV-40001", with_due=False)
    inv = out["invoice"]
    assert out["decision"] == "human_review" and inv["decision_rule"] == "missing_required_fields"

    bad = client.post(f"/invoices/{inv['id']}/review", json={"action": "approve", "reviewer": "alice"})
    assert bad.status_code == 422

    ok = client.post(f"/invoices/{inv['id']}/review", json={
        "action": "approve", "reviewer": "alice", "corrections": {"due_date": "2026-03-31"}})
    assert ok.status_code == 200 and ok.json()["due_date"] == "2026-03-31"
    human = [a for a in client.get(f"/invoices/{inv['id']}/audit").json() if a["stage"] == "human_review"][0]
    assert human["details"]["corrections"]["due_date"] == {"from": None, "to": "2026-03-31"}


def test_reject_in_review(client):
    build_history(client)
    out = post(client, msg_id="big", number="INV-30009", amount=6000.0)
    r = client.post(f"/invoices/{out['invoice']['id']}/review", json={"action": "reject", "reviewer": "alice", "comment": "wrong amount"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"


def test_not_found_and_validation_errors(client):
    assert client.get("/invoices/9999").status_code == 404
    assert client.get("/invoices/9999/audit").status_code == 404
    assert client.post("/invoices/9999/review", json={"action": "approve", "reviewer": "a"}).status_code == 404
    assert client.post("/invoices/1/review", json={"action": "maybe", "reviewer": "a"}).status_code == 422
    assert client.post("/emails", json={"subject": "no id"}).status_code == 422


def test_non_invoice_email_is_stored_but_creates_no_invoice(make_client):
    c = make_client(lambda: Pipeline([StubClassifier(), ExtractStage()]))
    out = c.post("/emails", json={"message_id": "n1", "subject": "newsletter: webinar", "body": "join us"}).json()
    assert out["decision"] == "not_invoice" and out["invoice"] is None and out["category"] == "newsletter"
    assert c.get("/metrics").json()["invoices"] == 0 and c.get("/metrics").json()["emails_received"] == 1


def test_unsure_classification_goes_to_triage_not_the_invoice_queue(make_client):
    c = make_client(lambda: Pipeline([StubClassifier(), ExtractStage()]))
    out = c.post("/emails", json={"message_id": "u1", "subject": "unsure what this is", "body": "hmm"}).json()
    assert out["decision"] == "needs_triage" and out["invoice"] is None
    assert c.get("/review-queue").json() == [] and c.get("/metrics").json()["invoices"] == 0
    assert [t["email_id"] for t in c.get("/triage").json()] == [out["email_id"]]


def test_metrics(client):
    build_history(client)
    post(client, msg_id="big", number="INV-30009", amount=6000.0)
    m = client.get("/metrics").json()
    assert m["invoices"] == 5 and m["pending_review"] == 1
    assert m["by_decision"]["auto_approved"] == 4 and m["automation_rate"] == 0.8
    assert m["avg_pipeline_ms_per_email"] > 0 and "estimate" in m["estimated_savings"]["note"]
