import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import demo
from app.config import settings
from app.db import Base, engine
from app.main import create_app
from app.security import WriteRateLimiter
from tests.test_mailbox import light_pipeline

EMAIL = {"message_id": "sec-1", "subject": "hi", "body": "hello"}


@pytest.fixture
def make_client():
    opened = []

    def _make():
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        c = TestClient(create_app(light_pipeline))
        c.__enter__()
        opened.append(c)
        return c

    yield _make
    for c in opened:
        c.__exit__(None, None, None)


def test_no_key_configured_means_open_access(make_client):
    assert make_client().get("/metrics").status_code == 200


def test_api_key_protects_every_api_call_but_not_the_page_or_health(make_client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr("s3cret"))
    c = make_client()
    assert c.get("/").status_code == 200 and c.get("/health").json()["auth_required"] is True
    for method, path in [("get", "/metrics"), ("get", "/review-queue"), ("get", "/invoices"), ("post", "/emails")]:
        assert getattr(c, method)(path).status_code == 401, path
    assert c.get("/metrics", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.post("/emails", json=EMAIL, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_oversized_requests_are_rejected(make_client, monkeypatch):
    c = make_client()
    assert c.post("/emails", json={**EMAIL, "body": "x" * 200_001}).status_code == 422
    assert c.post("/emails", json={**EMAIL, "subject": "x" * 501}).status_code == 422
    monkeypatch.setattr(settings, "max_request_bytes", 500)
    assert c.post("/emails", json={**EMAIL, "body": "x" * 2000}).status_code == 413


def test_demo_mode_rate_limits_writes_only(make_client, monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "demo_writes_per_minute", 3)
    monkeypatch.setattr(demo, "seed", lambda pipeline: 0)           # keep this test about the limiter
    c = make_client()
    codes = [c.post("/emails", json={**EMAIL, "message_id": f"rl-{i}"}).status_code for i in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert c.get("/metrics").status_code == 200                      # reads are never limited
    assert c.get("/health").json()["demo_mode"] is True


def test_limiter_window_slides():
    now = [0.0]
    lim = WriteRateLimiter(clock=lambda: now[0])
    assert [lim.allow("ip", 2) for _ in range(3)] == [True, True, False]
    assert lim.allow("other-ip", 2)
    now[0] = 61
    assert lim.allow("ip", 2)


def test_demo_seed_produces_a_useful_dashboard(make_client):
    c = make_client()
    assert demo.seed(c.app.state.pipeline) > 15
    m = c.get("/metrics").json()
    assert m["pending_review"] >= 2 and m["by_decision"].get("auto_approved", 0) >= 5
    decisions = {i["vendor_name"]: i["decision"] for i in c.get("/invoices?limit=200").json() if i["total_amount"] in (5200.0, 6400.0)}
    assert set(decisions.values()) == {"human_review"}               # unknown-vendor and spike both reach a person
    assert len([i for i in c.get("/invoices?limit=200").json() if i.get("duplicate_of")]) >= 1


def test_demo_reset_wipes_user_activity_and_reseeds(make_client):
    c = make_client()
    n = demo.reset_and_seed(c.app.state.pipeline)
    c.post("/emails", json={**EMAIL, "message_id": "visitor-1"})
    assert c.get("/metrics").json()["emails_received"] == n + 1
    assert demo.reset_and_seed(c.app.state.pipeline) == n
    assert c.get("/metrics").json()["emails_received"] == n
