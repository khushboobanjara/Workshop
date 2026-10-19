"""Shared fixtures and helpers for the MCP tests (imported by tests/test_mcp_*.py).

`world` builds a secured app with MCP switched on, three users (admin / reviewer / viewer), and three invoices whose
contents are chosen to exercise the grounding and redaction rules:
  i1  complete, approved; e-mail subject holds a card number, body holds a secret + IBAN, sender is a real address
  i2  pending review with NO invoice number, amount or PO number (must come back as null + missing_fields)
  i3  rejected duplicate of i2 whose vendor name contains LIKE wildcards ("100%_Real Co")
"""
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import auth
from app.auth_routes import login_limiter
from app.config import settings
from app.db import AuditLog, Base, Email, Invoice, SessionLocal, engine
from app.main import create_app
from app.mcp_server import routes as mcp_routes
from app.mcp_server.server import server
from tests.test_mailbox import light_pipeline

PW = "correct-horse-battery"
SYSTEM_KEY = "k" * 40
JWT_SECRET = "x" * 48
USERS = [("admin1", "admin"), ("rev1", "reviewer"), ("view1", "viewer")]
CARD = "4111 1111 1111 1111"                    # passes the Luhn check
IBAN = "DE89 3704 0044 0532 0130 00"            # a valid IBAN
BODY_SECRET = "TOP-SECRET-BODY-TEXT"

ALL_TOOLS = {"get_my_access", "search_invoices", "get_invoice", "get_invoice_audit_trail", "get_review_queue",
             "list_vendors", "get_metrics", "list_models", "get_system_status", "get_audit_log", "get_settings", "list_users"}
USER_TOOLS = {"get_my_access", "search_invoices", "get_invoice", "list_vendors", "get_metrics", "list_models"}
REVIEWER_TOOLS = USER_TOOLS | {"get_invoice_audit_trail", "get_review_queue"}
SYSTEM_TOOLS = {"get_my_access", "get_metrics", "list_models", "get_system_status"}
TOOLS_BY_WHO = {"view1": USER_TOOLS, "rev1": REVIEWER_TOOLS, "admin1": ALL_TOOLS, "system": SYSTEM_TOOLS}


def seed() -> dict:
    with SessionLocal() as s:
        e1 = Email(message_id="m1", sender="Billing Dept <billing@acme.example>", category="invoice",
                   subject=f"Invoice INV-1001 paid by card {CARD}", body=f"{BODY_SECRET} please wire to {IBAN}")
        e2 = Email(message_id="m2", sender="ap@globex.example", subject="Invoice", body="see attachment", category="invoice")
        e3 = Email(message_id="m3", sender="x@real.example", subject="Invoice again", body="dup", category="invoice")
        s.add_all([e1, e2, e3])
        s.flush()
        i1 = Invoice(email_id=e1.id, vendor_name="Acme Corp", invoice_number="INV-1001", invoice_date="2026-10-01",
                     due_date="2026-10-31", currency="USD", total_amount=1250.5, po_number="PO-7",
                     extraction_confidence=0.9, risk_score=0.12, decision="auto_approved", status="approved",
                     decision_rule="low_risk", decision_reasons=["low risk"], flags=[], risk_breakdown={"amount": 0.1},
                     reviewed_by="rev1", review_comment=f"checked, IBAN {IBAN}")
        i2 = Invoice(email_id=e2.id, vendor_name="Globex", invoice_number=None, invoice_date="2026-10-02",
                     due_date="2026-11-01", currency="USD", total_amount=None, po_number=None, risk_score=0.9,
                     decision="human_review", status="pending_review", decision_rule="missing_fields",
                     decision_reasons=["missing fields"], flags=["missing_total"], risk_breakdown={"validation": 0.9})
        s.add_all([i1, i2])
        s.flush()
        i3 = Invoice(email_id=e3.id, vendor_name="100%_Real Co", invoice_number="INV-3", invoice_date="2026-10-03",
                     due_date="2026-11-02", currency="USD", total_amount=10.0, po_number="PO-3", risk_score=0.95,
                     decision="rejected", status="rejected", duplicate_of=i2.id, decision_rule="duplicate",
                     decision_reasons=["duplicate"], flags=["duplicate"], risk_breakdown={"duplicate": 0.95})
        s.add(i3)
        s.flush()
        s.add(AuditLog(entity_type="invoice", entity_id=i1.id, stage="human_review", action="approved", actor="rev1",
                       details={"comment": f"paid by card {CARD}", "previous_status": "pending_review"}))
        s.commit()
        return {"i1": i1.id, "i2": i2.id, "i3": i3.id}


class World:
    def __init__(self, client: TestClient, tokens: dict, ids: dict):
        self.client, self.tokens, self.ids = client, tokens, ids

    def post(self, body, token=None, headers=None, **kw):
        h = dict(headers or {})
        if token:
            h["Authorization"] = f"Bearer {token}"
        return self.client.post("/mcp", json=body, headers=h, **kw)

    def rpc(self, method, params=None, token=None, msg_id=1, **kw):
        body = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            body["params"] = params
        return self.post(body, token, **kw)

    def call(self, tool, args=None, who="admin1", headers=None):
        """-> (tool result dict, parsed payload). `who` is a user name, 'system', or a raw token."""
        token = self.tokens.get(who, who)
        r = self.rpc("tools/call", {"name": tool, "arguments": args or {}}, token=token, headers=headers)
        assert r.status_code == 200, r.text
        result = r.json()["result"]
        return result, json.loads(result["content"][0]["text"])


def mcp_rows():
    with SessionLocal() as s:
        return s.query(AuditLog).filter(AuditLog.entity_type == "mcp").order_by(AuditLog.id).all()


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", SecretStr(JWT_SECRET))
    monkeypatch.setattr(settings, "mcp_enabled", True)
    monkeypatch.setattr(settings, "mcp_system_key", SecretStr(SYSTEM_KEY))
    monkeypatch.setattr(settings, "mcp_rate_limit_per_minute", 1000)
    login_limiter.hits.clear()
    server.limiter.hits.clear()
    mcp_routes.auth_failures.hits.clear()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    for name, role in USERS:
        auth.create_user(name, PW, role)
    ids = seed()
    client = TestClient(create_app(light_pipeline))
    client.__enter__()
    tokens = {}
    for name, _ in USERS:
        r = client.post("/auth/login", json={"username": name, "password": PW})
        assert r.status_code == 200
        tokens[name] = r.json()["access_token"]
    tokens["system"] = SYSTEM_KEY
    client.cookies.clear()                      # tests start without an ambient session cookie
    yield World(client, tokens, ids)
    client.__exit__(None, None, None)
