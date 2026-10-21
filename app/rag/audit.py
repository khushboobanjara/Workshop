"""Audit rows for the document store, in the existing append-only audit_log (entity_type='rag', stage='rag').

Recorded: who, what (upload, delete, reindex, search, refusals), the document id, and counts. NOT recorded: document text, file
contents, or the passages a search returned (they are in rag_chunks already; a second copy in the log would only be another
place for them to leak from). Search queries are stored cleaned and shortened, as chat questions are.
"""
from app.db import AuditLog, SessionLocal
from app.mcp_server.sanitize import clean_text, sanitize_json


def add_row(session, *, actor: str, action: str, outcome: str, entity_id: int | None = None, details: dict | None = None,
            duration_ms: float | None = None) -> None:
    """Adds the row to an open session, so it commits (or rolls back) together with the change it describes."""
    session.add(AuditLog(entity_type="rag", entity_id=entity_id, stage="rag", action=f"{action}:{outcome}"[:100],
                         actor=clean_text(actor, 100, redact=False) or "unknown", details=sanitize_json(details or {}),
                         duration_ms=duration_ms))


def record(*, actor: str, action: str, outcome: str, entity_id: int | None = None, details: dict | None = None,
           duration_ms: float | None = None) -> None:
    """Writes one row and commits. RAISES if it cannot: callers decide whether that is fatal (for a search it is)."""
    with SessionLocal() as s:
        add_row(s, actor=actor, action=action, outcome=outcome, entity_id=entity_id, details=details, duration_ms=duration_ms)
        s.commit()
