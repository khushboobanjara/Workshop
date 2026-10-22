"""Fixtures for the AI-assistant API tests. Groq is replaced by a scripted fake behind httpx.MockTransport, so the whole
chat path runs for real (auth, RBAC, MCP core, audit) except the network call. Builds on tests/mcp_support.py."""
import json

import httpx
import pytest
from pydantic import SecretStr

from app.chat import routes as chat_routes
from app.chat.groq_client import GroqClient
from app.config import settings
from app.db import AuditLog, SessionLocal
from tests.mcp_support import world  # noqa: F401  (fixture)

GROQ_KEY = "gsk_" + "t" * 40


def tc(name, args, call_id="call_1"):
    return {"content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}]}


def say(text):
    return {"content": text}


class FakeGroq:
    """Scripted model. Each script item is a dict (assistant message), an httpx.Response, or an exception to raise."""

    def __init__(self):
        self.script, self.requests = [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        step = self.script.pop(0) if self.script else {"content": "(script exhausted)"}
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": step.get("content"), "tool_calls": step.get("tool_calls")},
                                                      "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}})


class ChatWorld:
    def __init__(self, world, fake):
        self.world, self.fake, self.ids = world, fake, world.ids

    def ask(self, message, who="view1", history=None, script=None, **extra):
        if script is not None:
            self.fake.script = list(script)
        body = {"message": message, **({"history": history} if history is not None else {}), **extra}
        token = self.world.tokens.get(who, who)
        return self.world.client.post("/chat", json=body, headers={"Authorization": f"Bearer {token}"})

    def tool_names_offered(self, request_index=0):
        return [t["function"]["name"] for t in self.fake.requests[request_index]["body"].get("tools", [])]


def audit_rows(entity_type):
    with SessionLocal() as s:
        return s.query(AuditLog).filter(AuditLog.entity_type == entity_type).order_by(AuditLog.id).all()


@pytest.fixture
def chat(world, monkeypatch):
    fake = FakeGroq()
    monkeypatch.setattr(settings, "chat_enabled", True)
    monkeypatch.setattr(settings, "groq_api_key", SecretStr(GROQ_KEY))
    monkeypatch.setattr(settings, "groq_model", "test-model")
    monkeypatch.setattr(settings, "chat_rate_limit_per_minute", 1000)
    monkeypatch.setattr(settings, "chat_fact_check", False)       # tests that need the fact checker switch it on and script its verdicts
    monkeypatch.setattr(settings, "chat_hardcode", False)         # the fixed-rules layer has its own tests (test_chat_rules.py); older tests script the model
    chat_routes.chat_limiter.hits.clear()
    monkeypatch.setattr(chat_routes, "make_client", lambda: GroqClient(GROQ_KEY, "test-model", transport=httpx.MockTransport(fake)))
    return ChatWorld(world, fake)
