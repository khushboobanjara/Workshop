import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text

from app import auth
from app.auth_routes import login_limiter
from app.config import settings
from app.db import AuditLog, Base, Invoice, SessionLocal, User, engine
from app.main import create_app
from tests.test_mailbox import light_pipeline

PW = "correct-horse-battery"
UNKNOWN_VENDOR_BIG = ("Vendor: Brand New Ltd\nInvoice INV-90001\nInvoice Date: 2026-10-01\nDue Date: 2026-10-31\n"
                      "Total Amount: USD 5200.00")


@pytest.fixture
def secured(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", SecretStr("x" * 48))
    login_limiter.hits.clear()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    for name, role in [("admin1", "admin"), ("rev1", "reviewer"), ("view1", "viewer")]:
        auth.create_user(name, PW, role)
    opened = []

    def client(as_user=None, **kw):
        c = TestClient(create_app(light_pipeline), **kw)
        c.__enter__()
        opened.append(c)
        if as_user:
            assert c.post("/auth/login", json={"username": as_user, "password": PW}).status_code == 200
        return c

    yield client
    for c in opened:
        c.__exit__(None, None, None)


def test_anonymous_is_rejected_but_page_and_health_are_open(secured):
    c = secured()
    assert c.get("/").status_code == 200 and c.get("/health").json()["auth_enabled"] is True
    for path in ["/metrics", "/stats", "/invoices", "/review-queue", "/vendors", "/audit", "/users", "/auth/me"]:
        assert c.get(path).status_code == 401, path


def test_login_failures_are_uniform_and_audited_and_cookie_is_locked_down(secured):
    c = secured()
    wrong = c.post("/auth/login", json={"username": "rev1", "password": "nope-nope-nope"})
    ghost = c.post("/auth/login", json={"username": "ghost", "password": "nope-nope-nope"})
    assert wrong.status_code == ghost.status_code == 401 and wrong.json() == ghost.json()
    with SessionLocal() as s:
        assert s.query(AuditLog).filter_by(action="login_failed").count() == 2
    ok = c.post("/auth/login", json={"username": "rev1", "password": PW})
    cookie = ok.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert c.get("/auth/me").json() == {"auth_enabled": True, "username": "rev1", "role": "reviewer"}


def test_passwords_are_hashed_not_stored(secured):
    with SessionLocal() as s:
        h = s.query(User).filter_by(username="rev1").one().password_hash
    assert h.startswith("$argon2") and PW not in h


@pytest.mark.parametrize("user,method,path,allowed", [
    ("view1", "get", "/metrics", True), ("view1", "get", "/invoices.csv", True), ("view1", "get", "/vendors", True),
    ("view1", "get", "/review-queue", False), ("view1", "get", "/triage", False), ("view1", "get", "/audit", False),
    ("view1", "get", "/settings", False), ("view1", "get", "/users", False), ("view1", "post", "/emails", False),
    ("rev1", "get", "/review-queue", True), ("rev1", "get", "/triage", True), ("rev1", "post", "/emails", True),
    ("rev1", "get", "/audit", False), ("rev1", "get", "/settings", False), ("rev1", "get", "/users", False),
    ("admin1", "get", "/audit", True), ("admin1", "get", "/settings", True), ("admin1", "get", "/users", True),
])
def test_role_matrix(secured, user, method, path, allowed):
    c = secured(user)
    kwargs = {"json": {"message_id": f"m-{user}", "subject": "s", "body": "b"}} if method == "post" else {}
    status = getattr(c, method)(path, **kwargs).status_code
    assert (status == 403) is (not allowed) and status != 401


def test_reviewer_identity_cannot_be_spoofed(secured):
    c = secured("rev1")
    inv = c.post("/emails", json={"message_id": "big1", "subject": "Invoice INV-90001", "body": UNKNOWN_VENDOR_BIG}).json()["invoice"]
    r = c.post(f"/invoices/{inv['id']}/review", json={"action": "approve", "reviewer": "someone-else", "comment": "ok"})
    assert r.status_code == 200
    with SessionLocal() as s:
        assert s.get(Invoice, inv["id"]).reviewed_by == "rev1"
        assert s.query(AuditLog).filter(AuditLog.actor == "someone-else").count() == 0


def test_ingest_api_key_can_only_post_emails(secured, monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr("service-key"))
    c, h = secured(), {"X-API-Key": "service-key"}
    assert c.post("/emails", json={"message_id": "svc1", "subject": "s", "body": "b"}, headers=h).status_code == 200
    assert c.get("/metrics", headers=h).status_code == 403                     # recognised, but not allowed beyond ingestion
    assert c.post("/invoices/1/review", json={"action": "approve"}, headers=h).status_code == 403
    assert c.post("/emails", json={"message_id": "svc2", "subject": "s", "body": "b"}, headers={"X-API-Key": "wrong"}).status_code == 401


def test_changing_password_or_disabling_signs_the_user_out(secured):
    admin, rev = secured("admin1"), secured("rev1")
    uid = next(u["id"] for u in admin.get("/users").json() if u["username"] == "rev1")
    assert rev.get("/review-queue").status_code == 200
    assert admin.patch(f"/users/{uid}", json={"password": "a-brand-new-password"}).status_code == 200
    assert rev.get("/review-queue").status_code == 401                       # old session revoked
    assert admin.patch(f"/users/{uid}", json={"is_active": False}).status_code == 200
    assert TestClient(create_app(light_pipeline)).post("/auth/login", json={"username": "rev1", "password": "a-brand-new-password"}).status_code == 401


def test_the_last_admin_cannot_be_removed_and_user_input_is_validated(secured):
    c = secured("admin1")
    me = next(u["id"] for u in c.get("/users").json() if u["username"] == "admin1")
    assert c.patch(f"/users/{me}", json={"role": "viewer"}).status_code == 409
    assert c.patch(f"/users/{me}", json={"is_active": False}).status_code == 409
    assert c.post("/users", json={"username": "second", "password": PW, "role": "admin"}).status_code == 201
    assert c.patch(f"/users/{me}", json={"role": "viewer"}).status_code == 200      # now allowed
    admin2 = secured("second")
    assert admin2.post("/users", json={"username": "x", "password": PW, "role": "viewer"}).status_code == 422   # bad username
    assert admin2.post("/users", json={"username": "okname", "password": "short", "role": "viewer"}).status_code == 422
    assert admin2.post("/users", json={"username": "okname", "password": PW, "role": "root"}).status_code == 422
    assert admin2.post("/users", json={"username": "rev1", "password": PW, "role": "viewer"}).status_code == 409
    assert admin2.patch("/users/9999", json={"role": "viewer"}).status_code == 404


def test_cross_origin_writes_with_a_session_cookie_are_blocked(secured):
    c = secured("rev1")
    body = {"message_id": "csrf1", "subject": "s", "body": "b"}
    assert c.post("/emails", json=body, headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.post("/emails", json=body, headers={"Origin": "http://testserver"}).status_code == 200


def test_login_is_rate_limited(secured):
    c = secured()
    codes = [c.post("/auth/login", json={"username": "rev1", "password": "wrong-wrong-wrong"}).status_code for _ in range(12)]
    assert codes[:10] == [401] * 10 and codes[10:] == [429, 429]


def test_bearer_token_works_for_scripts(secured):
    c = secured()
    token = c.post("/auth/login", json={"username": "view1", "password": PW}).json()["access_token"]
    fresh = TestClient(create_app(light_pipeline))
    assert fresh.get("/metrics", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert fresh.get("/metrics", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_bootstrap_admin_is_created_once_and_never_overwritten(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", SecretStr("y" * 48))
    monkeypatch.setattr(settings, "bootstrap_admin_user", "boss")
    monkeypatch.setattr(settings, "bootstrap_admin_password", SecretStr(PW))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    auth.startup_checks_and_bootstrap()
    monkeypatch.setattr(settings, "bootstrap_admin_password", SecretStr("a-different-password"))
    auth.startup_checks_and_bootstrap()
    with SessionLocal() as s:
        users = s.query(User).all()
    assert [(u.username, u.role) for u in users] == [("boss", "admin")]
    assert auth.verify_password(users[0].password_hash, PW)


def test_production_refuses_to_start_unsafe(monkeypatch):
    monkeypatch.setattr(settings, "env", "production")
    with pytest.raises(RuntimeError, match="AUTH_ENABLED"):
        auth.startup_checks_and_bootstrap_pre()
    monkeypatch.setattr(settings, "auth_enabled", True)
    with pytest.raises(RuntimeError, match="COOKIE_SECURE"):
        auth.startup_checks_and_bootstrap_pre()
    monkeypatch.setattr(settings, "cookie_secure", True)
    with pytest.raises(RuntimeError, match="at least 32"):
        auth.startup_checks_and_bootstrap_pre()
    monkeypatch.setattr(settings, "jwt_secret", SecretStr("z" * 40))
    auth.startup_checks_and_bootstrap_pre()                                 # now fine


def test_demo_mode_with_auth_offers_only_low_privilege_accounts(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", SecretStr("d" * 48))
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr("app.demo.seed", lambda pipeline: 0)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(create_app(light_pipeline)) as c:
        info = c.get("/health").json()
        assert info["demo_accounts"]["users"] == ["demo_viewer", "demo_reviewer"]
        assert c.post("/auth/login", json={"username": "demo_reviewer", "password": info["demo_accounts"]["password"]}).status_code == 200
    with SessionLocal() as s:
        assert {u.role for u in s.query(User).all()} == {"viewer", "reviewer"}
