"""Fixtures for the document-search (RAG) tests. Builds on tests/chat_support.py (secured app, three users, scripted Groq).

Embeddings use the deterministic HashingEmbedder (no model download, no network); the vector index is a real FAISS index in a
temporary folder; documents are stored in the real test database. Only the semantic quality of the real Sentence Transformers
model is not exercised here (see scripts/evaluate_retrieval.py for that).
"""
import pytest

from app.config import settings
from app.rag import service as rag_service
from app.rag.embeddings import HashingEmbedder
from app.rag.service import RagService
from tests.chat_support import chat, tc, say, world  # noqa: F401  (fixtures / helpers)

POLICY = b"""# Accounts Payable Policy

## 1. Approval limits

Invoices up to 2,000 USD may be approved by a single reviewer. Invoices above 10,000 USD need two approvers.
Any invoice from a new vendor above 2,000 USD must go to human review.

## 2. Payment terms

Standard payment terms are net 30 days from the invoice date. Early payment discounts must be approved by finance.
"""
ONBOARDING = b"""# Vendor Onboarding Procedure

New vendors must submit a W-9 form and bank details before the first payment. The procurement team verifies the bank account
by calling the vendor on a known phone number. Vendors are reviewed again every 12 months.
"""
SECRET = b"""# Executive Compensation Plan

The executive bonus pool for the year is 4,500,000 USD and is paid out in March by the board.
"""


class RagWorld:
    def __init__(self, chat_world, svc):
        self.chat, self.svc, self.world, self.fake = chat_world, svc, chat_world.world, chat_world.fake
        self.client = self.world.client

    def headers(self, who):
        return {"Authorization": f"Bearer {self.world.tokens.get(who, who)}"}

    def upload(self, who="admin1", name="policy.md", data=POLICY, min_role="viewer", title=None):
        form = {"min_role": min_role, **({"title": title} if title else {})}
        return self.client.post("/rag/documents", files={"file": (name, data, "application/octet-stream")}, data=form, headers=self.headers(who))

    def get(self, path, who="admin1"):
        return self.client.get(path, headers=self.headers(who))

    def search(self, query, who="view1", limit=5):
        return self.client.post("/rag/search", json={"query": query, "limit": limit}, headers=self.headers(who))


@pytest.fixture
def rag(chat, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "rag_embedder", "hashing")
    monkeypatch.setattr(settings, "rag_data_dir", str(tmp_path / "rag"))
    monkeypatch.setattr(settings, "rag_min_score", 0.12)              # the hashing embedder's scores are lower than a real model's
    monkeypatch.setattr(settings, "rag_search_rate_limit_per_minute", 1000)
    rag_service.search_limiter.hits.clear()
    svc = RagService(HashingEmbedder(), tmp_path / "rag")
    rag_service.set_service(svc)
    yield RagWorld(chat, svc)
    rag_service.set_service(None)
