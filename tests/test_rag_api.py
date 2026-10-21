"""The /rag HTTP API: authentication, role gating, upload validation, visibility by role, deletion and the audit trail."""
import io
import zipfile

import pytest

from app.config import settings
from app.db import AuditLog, RagChunk, RagDocument, SessionLocal
from tests.rag_support import ONBOARDING, POLICY, SECRET, chat, rag, world  # noqa: F401  (fixtures)


def rag_audit():
    with SessionLocal() as s:
        return [(r.action, r.actor, r.entity_id, r.details) for r in s.query(AuditLog).filter(AuditLog.entity_type == "rag").order_by(AuditLog.id)]


def test_everything_is_off_unless_enabled(rag, monkeypatch):
    monkeypatch.setattr(settings, "rag_enabled", False)
    assert rag.get("/rag/status").json() == {"enabled": False}
    assert rag.get("/rag/documents").status_code == 404
    assert rag.upload().status_code == 404


def test_every_endpoint_needs_a_signed_in_user(rag):
    c = rag.client
    assert c.get("/rag/status").status_code == 401
    assert c.get("/rag/documents").status_code == 401
    assert c.post("/rag/search", json={"query": "approval limit"}).status_code == 401
    assert c.post("/rag/documents", files={"file": ("a.md", POLICY)}, data={"min_role": "viewer"}).status_code == 401
    assert c.post("/rag/reindex").status_code == 401
    assert c.delete("/rag/documents/1").status_code == 401
    assert rag.get("/rag/status", who="x" * 40).status_code == 401              # the MCP system key is not a person


def test_only_admins_upload_delete_and_reindex(rag):
    for who in ("rev1", "view1"):
        assert rag.upload(who=who).status_code == 403
        assert rag.client.post("/rag/reindex", headers=rag.headers(who)).status_code == 403
        assert rag.client.delete("/rag/documents/1", headers=rag.headers(who)).status_code == 403
    assert rag.get("/rag/documents").json() == []


def test_upload_stores_cleaned_chunks_and_audits(rag):
    r = rag.upload(title="AP Policy")
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["title"] == "AP Policy" and doc["file_type"] == "md" and doc["chunks"] >= 1 and doc["min_role"] == "viewer"
    assert doc["uploaded_by"] == "admin1" and len(doc["sha256"]) == 64
    with SessionLocal() as s:
        assert s.query(RagChunk).filter(RagChunk.document_id == doc["id"]).count() == doc["chunks"]
    assert rag.svc._store.ntotal == doc["chunks"]
    action, actor, entity_id, details = rag_audit()[-1]
    assert action == "upload:ok" and actor == "admin1" and entity_id == doc["id"] and details["min_role"] == "viewer"
    assert "Invoices up to" not in str(details)                                 # the audit row never holds document text


def test_visibility_follows_the_role_of_the_user(rag):
    rag.upload(name="policy.md", data=POLICY, min_role="viewer")
    rag.upload(name="onboarding.md", data=ONBOARDING, min_role="reviewer")
    rag.upload(name="exec.md", data=SECRET, min_role="admin")
    titles = lambda who: sorted(d["filename"] for d in rag.get("/rag/documents", who=who).json())
    assert titles("view1") == ["policy.md"]
    assert titles("rev1") == ["onboarding.md", "policy.md"]
    assert titles("admin1") == ["exec.md", "onboarding.md", "policy.md"]
    viewer_doc = rag.get("/rag/documents", who="view1").json()[0]
    assert "sha256" not in viewer_doc and "uploaded_by" not in viewer_doc        # internals are admin-only
    assert rag.get("/rag/status", who="view1").json() == {"enabled": True, "documents": 1}
    assert rag.get("/rag/status", who="admin1").json()["documents"] == 3


def test_search_only_returns_passages_the_role_may_see(rag):
    rag.upload(name="policy.md", data=POLICY, min_role="viewer")
    rag.upload(name="exec.md", data=SECRET, min_role="admin")
    q = "executive bonus pool paid out in March"
    admin = rag.search(q, who="admin1").json()
    assert admin["returned"] >= 1 and "4,500,000" in admin["passages"][0]["text"] and admin["passages"][0]["ref"].startswith("DOC-")
    viewer = rag.search(q, who="view1").json()
    assert "4,500,000" not in str(viewer) and "Executive" not in str(viewer)
    top = rag.search("approval limit for a new vendor", who="view1").json()["passages"][0]
    assert top["title"] == "policy" and "human review" in top["text"] and top["score"] > 0.12


def test_search_is_audited_and_fails_closed(rag, monkeypatch):
    rag.upload()
    r = rag.search("payment terms net 30", who="rev1")
    assert r.status_code == 200 and r.json()["returned"] >= 1
    action, actor, _, details = rag_audit()[-1]
    assert action == "search:ok" and actor == "rev1" and details["role"] == "reviewer" and details["refs"] and "text" not in details
    from app.rag import routes
    monkeypatch.setattr(routes.audit, "record", lambda **kw: (_ for _ in ()).throw(RuntimeError("db down")))
    blocked = rag.search("payment terms net 30", who="rev1")
    assert blocked.status_code == 503 and "net 30" not in blocked.text


def test_search_validation_and_rate_limit(rag, monkeypatch):
    assert rag.client.post("/rag/search", json={"query": "ab"}, headers=rag.headers("view1")).status_code == 422
    assert rag.client.post("/rag/search", json={"query": "approval", "limit": 99}, headers=rag.headers("view1")).status_code == 422
    assert rag.client.post("/rag/search", json={"query": "approval", "extra": 1}, headers=rag.headers("view1")).status_code == 422
    monkeypatch.setattr(settings, "rag_search_rate_limit_per_minute", 2)
    codes = [rag.search("approval limits", who="view1").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_duplicate_file_is_refused(rag):
    assert rag.upload().status_code == 201
    r = rag.upload(name="renamed.md")
    assert r.status_code == 409 and "already" in r.json()["detail"]
    with SessionLocal() as s:
        assert s.query(RagDocument).count() == 1


@pytest.mark.parametrize("name,data,status", [
    ("notes.exe", b"MZ....", 415), ("old.doc", b"\xd0\xcf\x11\xe0" + b"x" * 50, 415), ("empty.txt", b"", 422),
    ("fake.pdf", b"this is plain text", 415), ("fake.docx", b"plain text", 415), ("binary.txt", b"abc\x00\x01\x02def", 415),
    ("bad.docx", b"PK\x03\x04 not really a zip", 415)])
def test_unsupported_or_mismatched_files_are_refused(rag, name, data, status):
    r = rag.upload(name=name, data=data)
    assert r.status_code == status, r.text
    with SessionLocal() as s:
        assert s.query(RagDocument).count() == 0
    assert rag_audit()[-1][0] == "upload:refused"


def test_oversized_upload_is_refused(rag, monkeypatch):
    monkeypatch.setattr(settings, "rag_max_upload_bytes", 2000)
    assert rag.upload(data=b"word " * 1000).status_code == 413
    with SessionLocal() as s:
        assert s.query(RagDocument).count() == 0


def test_the_global_body_limit_still_applies_everywhere_else(rag, monkeypatch):
    monkeypatch.setattr(settings, "max_request_bytes", 500)
    r = rag.client.post("/rag/search", json={"query": "x" * 2000}, headers=rag.headers("view1"))
    assert r.status_code == 413
    assert rag.upload(data=POLICY + b"filler " * 300).status_code == 201         # the upload route has its own, larger limit


def test_a_zip_bomb_docx_is_refused(rag):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("word/big.bin", b"\0" * (60 * 1024 * 1024))                 # compresses to a few KB, unpacks to 60 MB
    assert len(buf.getvalue()) < 200_000
    r = rag.upload(name="bomb.docx", data=buf.getvalue())
    assert r.status_code == 413


def test_delete_removes_text_vectors_and_access(rag):
    doc = rag.upload().json()
    assert rag.search("approval limit new vendor").json()["returned"] >= 1
    r = rag.client.delete(f"/rag/documents/{doc['id']}", headers=rag.headers("admin1"))
    assert r.status_code == 200 and r.json()["chunks"] == doc["chunks"]
    assert rag.search("approval limit new vendor").json()["returned"] == 0
    assert rag.svc._store.ntotal == 0
    with SessionLocal() as s:
        assert s.query(RagChunk).count() == 0 and s.query(RagDocument).count() == 0
    assert rag.client.delete(f"/rag/documents/{doc['id']}", headers=rag.headers("admin1")).status_code == 404
    assert rag_audit()[-1][0] == "delete:ok" or rag_audit()[-2][0] == "delete:ok"


def test_a_stale_index_cannot_leak_a_deleted_or_restricted_document(rag):
    """Even if FAISS still holds vectors of a document, MySQL decides what may be returned."""
    doc = rag.upload(name="exec.md", data=SECRET, min_role="admin").json()
    rag.svc._store.save()
    with SessionLocal() as s:                                                   # delete the rows behind the index's back
        s.query(RagChunk).filter(RagChunk.document_id == doc["id"]).delete()
        s.query(RagDocument).filter(RagDocument.id == doc["id"]).delete()
        s.commit()
    assert rag.svc._store.ntotal > 0
    assert rag.search("executive bonus pool", who="admin1").json()["returned"] == 0


def test_reindex_rebuilds_from_the_stored_text(rag):
    rag.upload()
    rag.upload(name="o.md", data=ONBOARDING)
    before = rag.svc._store.ntotal
    rag.svc._store.reset()
    assert rag.svc._store.ntotal == 0
    r = rag.client.post("/rag/reindex", headers=rag.headers("admin1"))
    assert r.status_code == 200 and r.json()["vectors"] == before and rag.svc._store.ntotal == before
    assert rag.search("W-9 form bank details", who="view1").json()["returned"] >= 1
    assert rag.get("/rag/status").json()["index_in_sync"] is True


def test_index_is_reconciled_with_the_database_on_first_use(rag, tmp_path):
    from app.rag.embeddings import HashingEmbedder
    from app.rag.service import RagService
    rag.upload()
    with SessionLocal() as s:                                                   # a row the index has never seen
        s.add(RagChunk(document_id=s.query(RagDocument.id).first()[0], chunk_index=99, page=None, section=None, text="Wire transfers need dual control."))
        s.commit()
    fresh = RagService(HashingEmbedder(), tmp_path / "rag")                     # a new process: loads the saved index, then reconciles
    fresh.ensure_ready()
    with SessionLocal() as s:
        assert fresh._store.ids() == {i for (i,) in s.query(RagChunk.id).all()}
    assert [p.chunk_index for p in fresh.search("admin", "wire transfers dual control").passages][:1] == [99]


def test_an_index_built_with_another_model_is_rebuilt(rag, tmp_path):
    from app.rag.embeddings import HashingEmbedder
    from app.rag.service import RagService
    rag.upload()
    other = RagService(HashingEmbedder(dim=128), tmp_path / "rag")
    other.ensure_ready()                                                        # dimension differs: mismatch -> rebuild from MySQL
    assert other._store.ntotal == rag.svc._store.ntotal and other._store.dim == 128
    assert other.search("viewer", "approval limit new vendor").passages


# ---------------------------------------------------------------- start-up safety
def test_start_up_checks_refuse_unsafe_configurations(rag, monkeypatch):
    from app.rag import routes
    routes.startup_checks()                                                    # the test configuration is fine
    for attr, value, fragment in [("auth_enabled", False, "AUTH_ENABLED"), ("demo_mode", True, "DEMO_MODE"),
                                  ("rag_embedder", "magic", "RAG_EMBEDDER"), ("rag_min_score", 1.5, "MIN_SCORE")]:
        with monkeypatch.context() as m:
            m.setattr(settings, attr, value)
            with pytest.raises(RuntimeError) as e:
                routes.startup_checks()
            assert fragment in str(e.value)
    with monkeypatch.context() as m:
        m.setattr(settings, "env", "production")                               # the word-matching stand-in is not allowed in production
        with pytest.raises(RuntimeError) as e:
            routes.startup_checks()
        assert "hashing" in str(e.value)
    with monkeypatch.context() as m:
        m.setattr(settings, "rag_enabled", False)
        m.setattr(settings, "demo_mode", True)
        routes.startup_checks()                                                # off means nothing to check


def test_the_dashboard_offers_the_documents_page(rag):
    html = rag.client.get("/").text
    assert 'data-tab="documents"' in html and "renderDocuments" in html


def test_two_worker_processes_sharing_one_index_folder_see_each_others_uploads(rag, tmp_path):
    from app.rag.embeddings import HashingEmbedder
    from app.rag.service import RagService
    worker_b = RagService(HashingEmbedder(), tmp_path / "rag")
    worker_b.ensure_ready()                                                    # B loaded the (empty) index before A uploaded anything
    rag.upload(name="policy.md", data=POLICY)                                  # handled by worker A (the app's service)
    assert [p.title for p in worker_b.search("viewer", "approval limit new vendor").passages][:1] == ["policy"]
    worker_b.ingest(data=ONBOARDING, filename="onboarding.md", title="Onboarding", min_role="viewer", actor="b")
    assert rag.search("W-9 form bank details", who="view1").json()["returned"] >= 1      # and A sees what B added
    with SessionLocal() as s:
        assert rag.svc._store.ids() == worker_b._store.ids() == {i for (i,) in s.query(RagChunk.id).all()}
    policy_id = next(d["id"] for d in rag.svc.list_documents("admin") if d["title"] == "policy")
    worker_b.delete(policy_id, "b")
    assert rag.search("standard payment terms net 30 days", who="view1").json()["returned"] == 0     # gone for A as well
    with SessionLocal() as s:
        assert {i for (i,) in s.query(RagChunk.id).all()} == worker_b._store.ids()


def test_real_pdf_and_docx_uploads_end_to_end(rag):
    from tests.test_rag_units import make_docx, make_pdf
    pdf = rag.upload(name="travel.pdf", data=make_pdf(["Travel Policy", "Hotels are reimbursed up to 200 USD per night.", "", "Meals are reimbursed up to 75 USD per day."]), title="Travel PDF")
    assert pdf.status_code == 201 and pdf.json()["file_type"] == "pdf" and pdf.json()["pages"] == 4
    word = rag.upload(name="travel.docx", data=make_docx(), title="Travel DOCX")
    assert word.status_code == 201 and word.json()["file_type"] == "docx"
    hits = rag.search("hotel reimbursement per night", who="view1").json()["passages"]
    assert hits and hits[0]["title"] in ("Travel PDF", "Travel DOCX") and "night" in hits[0]["text"].lower()
    assert rag.search("flights booked days ahead", who="view1").json()["passages"][0]["title"] == "Travel DOCX"
    scanned = rag.upload(name="scan.pdf", data=make_pdf([""]))
    assert scanned.status_code == 422 and "scanned" in scanned.json()["detail"].lower()
