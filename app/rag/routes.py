"""/rag/*: manage and search the uploaded documents. Off (404) unless OPSPILOT_RAG_ENABLED=true.

  GET    /rag/status               any signed-in role   on/off and how many documents you may see (admins: index health too)
  GET    /rag/documents            any signed-in role   the documents YOUR role may retrieve
  POST   /rag/documents            admin                upload (multipart: file, min_role, optional title)
  DELETE /rag/documents/{id}       admin
  POST   /rag/reindex              admin                rebuild the vector index from the stored text
  POST   /rag/search               any signed-in role   the passages the assistant would see for a query (audited, rate limited)

Roles are the application's own (`require_roles`); document visibility is `min_role` (see access.py). Every upload, delete,
re-index and search writes an audit row, and a search with no audit row returns nothing.
"""
import logging
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from app.auth import Principal, require_roles
from app.config import settings
from app.rag import audit
from app.rag.service import MAX_TOP_K, RagError, get_service, search_limiter

logger = logging.getLogger("opspilot.rag")
router = APIRouter(prefix="/rag", tags=["documents"])
ANY = require_roles("admin", "reviewer", "viewer")
ADMIN = require_roles("admin")


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=300)
    limit: int = Field(5, ge=1, le=MAX_TOP_K)


def _service():
    if not settings.rag_enabled:
        raise HTTPException(404, "Document search is not enabled")
    try:
        return get_service()
    except RagError as exc:
        raise HTTPException(exc.http_status, exc.message) from None


def startup_checks() -> None:
    """Fail closed: refuse to start document search in an unsafe or unusable configuration."""
    if not settings.rag_enabled:
        return
    problems = []
    if not settings.auth_enabled:
        problems.append("OPSPILOT_AUTH_ENABLED must be true (document visibility depends on roles)")
    if settings.demo_mode:
        problems.append("document search cannot be enabled together with OPSPILOT_DEMO_MODE")
    kind = settings.rag_embedder.strip().lower()
    if kind not in ("sentence-transformers", "sentence_transformers", "st", "hashing"):
        problems.append("OPSPILOT_RAG_EMBEDDER must be 'sentence-transformers' (or 'hashing' for tests)")
    if kind == "hashing" and settings.env == "production":
        problems.append("OPSPILOT_RAG_EMBEDDER=hashing matches words, not meaning, and is not allowed in production")
    if not 0.0 <= settings.rag_min_score <= 1.0:
        problems.append("OPSPILOT_RAG_MIN_SCORE must be between 0 and 1")
    if settings.rag_chunk_chars < 200 or settings.rag_chunk_overlap_chars < 0:
        problems.append("OPSPILOT_RAG_CHUNK_CHARS must be at least 200 and the overlap not negative")
    try:
        import faiss  # noqa: F401
    except ImportError:
        problems.append("faiss-cpu is not installed (pip install -r requirements.txt)")
    try:
        folder = Path(settings.rag_data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write-test"
        probe.write_text("ok")
        probe.unlink()
    except OSError:
        problems.append(f"OPSPILOT_RAG_DATA_DIR ({settings.rag_data_dir}) is not writable")
    if problems:
        raise RuntimeError("Refusing to start document search: " + "; ".join(problems))


@router.get("/status")
def status(principal: Principal = Depends(ANY)):
    if not settings.rag_enabled:
        return {"enabled": False}
    return _service().status(principal.role)


@router.get("/documents")
def list_documents(principal: Principal = Depends(ANY)):
    return _service().list_documents(principal.role)


@router.post("/documents", status_code=201)
async def upload(file: UploadFile = File(...), min_role: Literal["viewer", "reviewer", "admin"] = Form(...),
                 title: str = Form(""), principal: Principal = Depends(ADMIN)):
    svc = _service()
    data = await file.read(settings.rag_max_upload_bytes + 1)
    if len(data) > settings.rag_max_upload_bytes:
        raise HTTPException(413, f"The file is larger than the {settings.rag_max_upload_bytes // 1_000_000} MB limit.")
    try:
        return await run_in_threadpool(svc.ingest, data=data, filename=file.filename or "document", title=title or None,
                                       min_role=min_role, actor=principal.username)
    except RagError as exc:
        raise HTTPException(exc.http_status, exc.message) from None


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: int, principal: Principal = Depends(ADMIN)):
    try:
        return _service().delete(doc_id, principal.username)
    except RagError as exc:
        raise HTTPException(exc.http_status, exc.message) from None


@router.post("/reindex")
def reindex(principal: Principal = Depends(ADMIN)):
    try:
        return _service().reindex(principal.username)
    except RagError as exc:
        raise HTTPException(exc.http_status, exc.message) from None


@router.post("/search")
def search(payload: SearchIn, principal: Principal = Depends(ANY)):
    svc = _service()
    if not search_limiter.allow(f"rag:{principal.username}", settings.rag_search_rate_limit_per_minute):
        raise HTTPException(429, "Too many searches. Wait a minute and try again.", headers={"Retry-After": "60"})
    started = time.perf_counter()
    try:
        result = svc.search(principal.role, payload.query, payload.limit)
    except RagError as exc:
        raise HTTPException(exc.http_status, exc.message) from None
    try:                                                              # fail closed: no audit row, no passages
        audit.record(actor=principal.username, action="search", outcome="ok" if result.passages else "no_results",
                     details={"role": principal.role, "via": principal.via, "transport": "http", "query": payload.query[:300],
                              "returned": len(result.passages), "documents": list(dict.fromkeys(p.document_id for p in result.passages)),
                              "refs": [p.ref for p in result.passages], "top_score": result.passages[0].score if result.passages else None},
                     duration_ms=round((time.perf_counter() - started) * 1000, 2))
    except Exception:
        logger.exception("could not write the RAG audit row")
        raise HTTPException(503, "The search could not be recorded in the audit log, so it was withheld.") from None
    return {"query": payload.query, "returned": len(result.passages),
            "passages": [{**p.for_model(), "score": p.score} for p in result.passages]}
