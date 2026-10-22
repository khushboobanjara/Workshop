"""One audit_log row per chat turn (entity_type='chat'), next to the per-tool rows the MCP core already writes
(stage 'mcp', transport 'chat', same request id), so a question can be traced to every lookup it caused.

Recorded: who, the question (cleaned, shortened), which tools ran, how many records, whether the answer verified, the
model and token use, and a SHA-256 of the answer. NOT recorded: the answer text itself, the Groq key, or any returned
data: the records are in the database already, and the hash lets the answer be checked later if it was kept elsewhere.
"""
import hashlib

from app.db import AuditLog, SessionLocal
from app.mcp_server.sanitize import clean_text, sanitize_json

OUTCOMES = ("ok", "blocked", "failed", "rate_limited")


def record_turn(*, principal, outcome: str, question: str, model: str, result=None, error_code: str | None = None,
                error_detail: str | None = None, duration_ms: float | None = None) -> None:
    """Writes one row and commits; RAISES if it cannot (the caller withholds the answer in that case)."""
    details = {"role": principal.role, "via": principal.via, "model": model, "outcome": outcome,
               "question": clean_text(question, 300)}
    if result is not None:
        details.update({
            "tools": [t["tool"] for t in result.tools_used], "tool_calls": len(result.tools_used),
            "outcomes": [t["outcome"] for t in result.tools_used], "records": len(result.sources),
            "verified": result.verified, "grounded": result.grounded, "blocked": result.blocked, "reason": result.reason,
            "fact_checked": result.fact_checked, "fact_check_unavailable": result.fact_check_unavailable,
            "retries": result.retries, "unsupported_count": len(result.unsupported), "groq_calls": result.groq_calls,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "answer_chars": len(result.answer), "answer_sha256": hashlib.sha256(result.answer.encode("utf-8")).hexdigest()})
    if result is not None and getattr(result, "rag", False):          # only turns where the document tool was offered
        details.update({"documents": True, "document_searches": result.document_searches, "passages_retrieved": result.passages_retrieved,
                        "citations": [c["ref"] for c in result.citations]})
        if result.failed_checks:                                      # which check rejected an attempt (kinds only, no text)
            details["failed_checks"] = list(result.failed_checks)
    if result is not None and getattr(result, "route", "llm") != "llm":    # answered without (or with a single) model call
        details["route"] = result.route
    if error_code:
        details["error_code"] = error_code
    if error_detail:
        details["error_detail"] = clean_text(error_detail, 300)
    with SessionLocal() as s:
        s.add(AuditLog(entity_type="chat", entity_id=None, stage="chat", action=f"ask:{outcome}",
                       actor=clean_text(principal.username, 100, redact=False) or "unknown",
                       details=sanitize_json(details), duration_ms=duration_ms))
        s.commit()
