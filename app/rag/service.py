"""The document store: ingest, delete, search, re-index. One `RagService` per process (see get_service()).

Source of truth and derived data
  MySQL (rag_documents, rag_chunks) holds the cleaned text and who may see it. FAISS holds only vectors keyed by chunk id.
  A search takes ids from FAISS and then RESOLVES THEM THROUGH MYSQL, applying the document's status and min_role, so a stale
  or tampered index can at worst miss passages: it can never reveal a deleted document or one the user's role may not see.
  On first use the index is checked against the table (reconcile): vectors with no row are dropped, rows with no vector are
  embedded and added, and an index built with another embedding model is rebuilt from the stored text.

Order of writes
  upload: extract -> clean -> chunk -> quarantine -> embed -> [MySQL rows + audit row in ONE transaction] -> FAISS add + save
  delete: [MySQL delete + audit row in ONE transaction] -> FAISS remove + save
  If the FAISS step of an upload fails, the rows are removed again. If it fails after a delete, the leftover vectors are
  harmless (unresolvable) and are removed by the next reconcile.
"""
import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.db import RagChunk, RagDocument, SessionLocal
from app.mcp_server.sanitize import clean_text
from app.rag import access, audit, extract as extractor
from app.rag.chunking import Chunk, chunk_segments, embedding_text
from app.rag.embeddings import _STOP, EmbedderUnavailable, _stem, build_embedder
from app.rag.extract import ExtractionError, Segment
from app.rag.index import FaissStore, IndexMismatch
from app.rag.sanitize import clean_extracted, looks_like_injection
from app.security import WriteRateLimiter

logger = logging.getLogger("opspilot.rag")
MAX_TOP_K = 8
MAX_QUERY_CHARS = 300
PASSAGE_CHARS = 1_000
LEXICAL_WEIGHT = 0.10                     # ranking bonus for query words found verbatim; the relevance FLOOR uses the cosine only
search_limiter = WriteRateLimiter()


class RagError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code, self.message, self.http_status = code, message, http_status


@dataclass
class Passage:
    ref: str                                # DOC-<document id>-C<chunk number>: the citation the model must use
    document_id: int
    chunk_id: int
    chunk_index: int
    title: str
    page: int | None
    section: str | None
    text: str
    score: float                            # cosine similarity; kept server-side (audit, API), never shown to the model

    def for_model(self) -> dict:
        return {"ref": self.ref, "document_id": self.document_id, "title": self.title, "section": self.section,
                "page": self.page, "text": self.text[:PASSAGE_CHARS]}

    def citation(self) -> dict:
        return {"ref": self.ref, "document_id": self.document_id, "title": self.title, "page": self.page,
                "section": self.section, "snippet": self.text[:300]}


@dataclass
class SearchResult:
    passages: list[Passage] = field(default_factory=list)
    indexed: int = 0                        # vectors searched
    below_threshold: bool = False           # something was found, but nothing was relevant enough


def doc_dict(d: RagDocument, *, admin: bool) -> dict:
    out = {"id": d.id, "title": d.title, "filename": d.filename, "file_type": d.file_type, "min_role": d.min_role,
           "status": d.status, "pages": d.pages, "chunks": d.chunk_count, "size_bytes": d.size_bytes,
           "created_at": d.created_at.isoformat() if d.created_at else None}
    if admin:
        out.update({"quarantined_chunks": d.quarantined_chunks, "uploaded_by": d.uploaded_by, "sha256": d.sha256,
                    "embedding_model": d.embedding_model})
    return out


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


class RagService:
    def __init__(self, embedder, data_dir: str | Path, cfg=settings):
        self.embedder, self.data_dir, self.cfg = embedder, Path(data_dir), cfg
        self._store: FaissStore | None = None
        self._ready = False
        self._seen = None                    # fingerprint of the index file as this process last wrote or read it
        self._lock = threading.RLock()

    # ------------------------------------------------------------ readiness
    def _embed_failure(self, exc: EmbedderUnavailable) -> RagError:
        return RagError("embedder_unavailable", str(exc), 503)

    def ensure_ready(self) -> FaissStore:
        with self._lock:
            if self._ready and self._store is not None:
                return self._store
            try:
                store = FaissStore(self.data_dir, self.embedder.dim, self.embedder.name)
            except EmbedderUnavailable as exc:
                raise self._embed_failure(exc) from None
            self._store = store
            try:
                store.load()
            except IndexMismatch as exc:
                logger.warning("%s Rebuilding it from the stored chunks.", exc)
                self._rebuild(store)
            self._reconcile(store)
            self._seen = self._stamp()
            self._ready = True
            return store

    def _stamp(self):
        try:
            st = self._store.index_path.stat()
            return (st.st_mtime_ns, st.st_size)
        except (OSError, AttributeError):
            return None

    def _save(self, store: FaissStore) -> None:
        store.save()
        self._seen = self._stamp()

    def _refresh_if_stale(self) -> None:
        """Several worker processes share one index folder. If another process saved a newer index file, load it and
        reconcile with MySQL (which also re-adds anything a concurrent write overwrote). Correctness of access control never
        depends on this: results are always resolved through MySQL."""
        store = self._store
        if store is None or self._stamp() == self._seen:
            return
        with self._lock:
            if self._stamp() == self._seen:
                return
            try:
                store.load()
            except IndexMismatch:
                self._rebuild(store)
            self._seen = self._stamp()
            self._reconcile(store)

    def _chunk_rows(self, only_ids: set[int] | None = None) -> list[tuple[int, str]]:
        """(chunk id, text to embed) for every stored chunk (or only those ids)."""
        with SessionLocal() as s:
            q = s.query(RagChunk.id, RagDocument.title, RagChunk.section, RagChunk.text, RagChunk.chunk_index, RagChunk.page) \
                .join(RagDocument, RagChunk.document_id == RagDocument.id).order_by(RagChunk.id)
            rows = q.all()
        return [(r[0], embedding_text(Chunk(r[4], r[3], r[5], r[2]), r[1])) for r in rows if only_ids is None or r[0] in only_ids]

    def _embed_in_batches(self, pairs: list[tuple[int, str]], store: FaissStore) -> None:
        step = 256
        for i in range(0, len(pairs), step):
            part = pairs[i:i + step]
            store.add([p[0] for p in part], self.embedder.embed_documents([p[1] for p in part]))

    def _rebuild(self, store: FaissStore) -> int:
        store.reset()
        pairs = self._chunk_rows()
        try:
            self._embed_in_batches(pairs, store)
        except EmbedderUnavailable as exc:
            raise self._embed_failure(exc) from None
        self._save(store)
        return len(pairs)

    def _reconcile(self, store: FaissStore) -> tuple[int, int]:
        with SessionLocal() as s:
            db_ids = {i for (i,) in s.query(RagChunk.id).all()}
        have = store.ids()
        orphans, missing = have - db_ids, db_ids - have
        if orphans:
            store.remove(sorted(orphans))
        if missing:
            try:
                self._embed_in_batches(self._chunk_rows(missing), store)
            except EmbedderUnavailable as exc:
                raise self._embed_failure(exc) from None
        if orphans or missing:
            self._save(store)
            logger.info("RAG index reconciled: removed %d stale vectors, added %d missing", len(orphans), len(missing))
        return len(orphans), len(missing)

    def reindex(self, actor: str) -> dict:
        """Admin: throw the vector index away and rebuild it from the stored chunk text with the configured model."""
        started = time.perf_counter()
        with self._lock:
            store = self.ensure_ready()
            self._refresh_if_stale()
            count = self._rebuild(store)
            audit.record(actor=actor, action="reindex", outcome="ok", details={"vectors": count, "model": self.embedder.name},
                         duration_ms=round((time.perf_counter() - started) * 1000, 2))
        return {"vectors": count, "model": self.embedder.name}

    # ------------------------------------------------------------ upload
    def ingest(self, *, data: bytes, filename: str, title: str | None, min_role: str, actor: str) -> dict:
        started = time.perf_counter()
        cfg = self.cfg
        if not access.is_level(min_role):
            raise RagError("bad_visibility", "min_role must be viewer, reviewer or admin.", 422)
        name = extractor.display_name(filename)
        label = (clean_text(title, 200, redact=False) or "").strip() or Path(name).stem[:200] or "document"
        try:
            ex = extractor.extract(name, data, max_pdf_pages=cfg.rag_max_pdf_pages, max_chars=cfg.rag_max_extracted_chars)
        except ExtractionError as exc:
            self._audit_refusal(actor, name, exc.code)
            raise RagError(exc.code, exc.message, exc.http_status) from None
        segments = [Segment(clean_extracted(s.text), s.page) for s in ex.segments]
        segments = [s for s in segments if s.text]
        if not segments:
            self._audit_refusal(actor, name, "empty")
            raise RagError("empty", "The document contains no usable text.", 422)

        chunks = chunk_segments(segments, cfg.rag_chunk_chars, cfg.rag_chunk_overlap_chars, cfg.rag_min_chunk_chars)
        kept = [c for c in chunks if not looks_like_injection(c.text)]
        quarantined = len(chunks) - len(kept)
        if not kept:
            self._audit_refusal(actor, name, "quarantined")
            raise RagError("quarantined", "Every passage in this document looks like an instruction to an AI system, so none was indexed.", 422)
        for n, c in enumerate(kept, 1):
            c.index = n                                     # citations count the passages that were actually indexed

        try:
            vectors = self.embedder.embed_documents([embedding_text(c, label) for c in kept])
        except EmbedderUnavailable as exc:
            raise self._embed_failure(exc) from None

        sha = hashlib.sha256(data).hexdigest()
        with self._lock:
            store = self.ensure_ready()
            self._refresh_if_stale()
            with SessionLocal() as s:
                if s.query(RagDocument.id).count() >= cfg.rag_max_documents:
                    raise RagError("document_limit", f"The document limit ({cfg.rag_max_documents}) is reached. Delete one first.", 409)
                dup = s.query(RagDocument.id).filter(RagDocument.sha256 == sha).first()
                if dup:
                    raise RagError("duplicate", f"This exact file is already stored as document {dup[0]}.", 409)
                doc = RagDocument(title=label, filename=name, file_type=ex.file_type, sha256=sha, size_bytes=len(data), min_role=min_role,
                                  pages=ex.pages, chunk_count=len(kept), quarantined_chunks=quarantined,
                                  embedding_model=self.embedder.name, uploaded_by=clean_text(actor, 100, redact=False) or "unknown")
                rows = []
                try:
                    s.add(doc)
                    s.flush()
                    rows = [RagChunk(document_id=doc.id, chunk_index=c.index, page=c.page, section=c.section, text=c.text) for c in kept]
                    s.add_all(rows)
                    s.flush()
                    audit.add_row(s, actor=actor, action="upload", outcome="ok", entity_id=doc.id,
                                  details={"title": label, "filename": name, "file_type": ex.file_type, "min_role": min_role,
                                           "chunks": len(kept), "quarantined_chunks": quarantined, "size_bytes": len(data)},
                                  duration_ms=round((time.perf_counter() - started) * 1000, 2))
                    s.commit()
                except IntegrityError:
                    s.rollback()
                    raise RagError("duplicate", "This exact file is already stored.", 409) from None
                ids, doc_id, out = [r.id for r in rows], doc.id, doc_dict(doc, admin=True)
            try:
                store.add(ids, vectors)
                self._save(store)
            except Exception:
                logger.exception("could not add document %s to the vector index; removing it again", doc_id)
                self._remove_rows(doc_id)
                store.remove(ids)
                self._audit_refusal(actor, name, "index_failed", entity_id=doc_id)
                raise RagError("index_failed", "The document could not be added to the search index and was not stored.", 500) from None
        return out

    def _audit_refusal(self, actor: str, filename: str, code: str, entity_id: int | None = None) -> None:
        try:
            audit.record(actor=actor, action="upload", outcome="refused", entity_id=entity_id,
                         details={"filename": clean_text(filename, 120, redact=False), "reason": code})
        except Exception:
            logger.exception("could not write the RAG audit row")

    def _remove_rows(self, doc_id: int) -> list[int]:
        with SessionLocal() as s:
            ids = [i for (i,) in s.query(RagChunk.id).filter(RagChunk.document_id == doc_id).all()]
            s.query(RagChunk).filter(RagChunk.document_id == doc_id).delete(synchronize_session=False)
            s.query(RagDocument).filter(RagDocument.id == doc_id).delete(synchronize_session=False)
            s.commit()
        return ids

    # ------------------------------------------------------------ delete
    def delete(self, doc_id: int, actor: str) -> dict:
        with self._lock:
            store = self.ensure_ready()
            self._refresh_if_stale()
            with SessionLocal() as s:
                doc = s.get(RagDocument, doc_id)
                if doc is None:
                    raise RagError("not_found", "No such document.", 404)
                ids = [i for (i,) in s.query(RagChunk.id).filter(RagChunk.document_id == doc_id).all()]
                title = doc.title
                s.query(RagChunk).filter(RagChunk.document_id == doc_id).delete(synchronize_session=False)
                s.delete(doc)
                audit.add_row(s, actor=actor, action="delete", outcome="ok", entity_id=doc_id, details={"title": title, "chunks": len(ids)})
                s.commit()
            try:
                store.remove(ids)
                self._save(store)
            except Exception:
                logger.exception("could not remove vectors of deleted document %s; the next reconcile will", doc_id)
        return {"deleted": doc_id, "chunks": len(ids)}

    # ------------------------------------------------------------ read side
    def list_documents(self, role: str) -> list[dict]:
        levels = access.visible_levels(role)
        if not levels:
            return []
        with SessionLocal() as s:
            rows = s.query(RagDocument).filter(RagDocument.status == "active", RagDocument.min_role.in_(levels)) \
                .order_by(RagDocument.id.desc()).all()
            return [doc_dict(d, admin=role == "admin") for d in rows]

    def status(self, role: str) -> dict:
        out = {"enabled": True, "documents": len(self.list_documents(role))}
        if role == "admin":
            with SessionLocal() as s:
                chunks = s.query(RagChunk.id).count()
            store = self._store if self._ready else None
            out.update({"embedding_model": self.embedder.name, "embedder_loaded": bool(getattr(self.embedder, "loaded", True)),
                        "chunks": chunks, "index_vectors": store.ntotal if store else None,
                        "index_in_sync": (store.ntotal == chunks) if store else None})
        return out

    def search(self, role: str, query: str, top_k: int | None = None, min_score: float | None = None) -> SearchResult:
        levels = access.visible_levels(role)
        query = (clean_text(query, MAX_QUERY_CHARS, redact=False) or "").strip()
        if not levels or not query:
            return SearchResult()
        top_k = max(1, min(top_k or self.cfg.rag_top_k, MAX_TOP_K))
        floor = self.cfg.rag_min_score if min_score is None else min_score
        try:
            qvec = self.embedder.embed_query(query)
        except EmbedderUnavailable as exc:
            raise self._embed_failure(exc) from None
        store = self.ensure_ready()
        self._refresh_if_stale()
        total = store.ntotal
        k = min(total, max(top_k * 6, 30))
        qterms = _terms(query)
        while True:
            ids, scores = store.search(qvec, k)
            score_of = dict(zip(ids, scores))
            with SessionLocal() as s:
                rows = s.query(RagChunk, RagDocument).join(RagDocument, RagChunk.document_id == RagDocument.id) \
                    .filter(RagChunk.id.in_(ids or [0]), RagDocument.status == "active", RagDocument.min_role.in_(levels)).all()
            found = [(c, d, score_of[c.id]) for c, d in rows if score_of.get(c.id, -1) >= floor]
            exhausted = k >= total or (scores and scores[-1] < floor)       # nothing further down the ranking can qualify
            if exhausted or len(found) >= top_k * 2:
                break
            k = min(total, k * 4)
        ranked = sorted(found, key=lambda t: t[2] + LEXICAL_WEIGHT * (len(qterms & _terms(t[0].text)) / len(qterms) if qterms else 0),
                        reverse=True)
        per_doc: dict[int, int] = {}
        passages: list[Passage] = []
        for c, d, sc in ranked:
            if per_doc.get(d.id, 0) >= self.cfg.rag_max_chunks_per_document:
                continue
            per_doc[d.id] = per_doc.get(d.id, 0) + 1
            passages.append(Passage(f"DOC-{d.id}-C{c.chunk_index}", d.id, c.id, c.chunk_index, d.title, c.page, c.section, c.text, round(sc, 4)))
            if len(passages) >= top_k:
                break
        return SearchResult(passages, total, below_threshold=not passages and bool(ids))


# ---------------------------------------------------------------- process-wide instance
_service: RagService | None = None
_service_lock = threading.Lock()


def get_service() -> RagService:
    global _service
    with _service_lock:
        if _service is None:
            try:
                embedder = build_embedder(settings)
            except EmbedderUnavailable as exc:
                raise RagError("embedder_unavailable", str(exc), 503) from None
            _service = RagService(embedder, settings.rag_data_dir)
        return _service


def set_service(service: RagService | None) -> None:
    """Tests (and scripts) install their own instance; None forgets the current one."""
    global _service
    with _service_lock:
        _service = service
