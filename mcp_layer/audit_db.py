"""Audit sinks. Events carry argument NAMES only (already sanitized by the gateway), never values."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, List, Optional

log = logging.getLogger("mcp.audit.db")
fallback_log = logging.getLogger("mcp.audit")

_INSERT = (
    "INSERT INTO mcp_audit_logs (request_id, user_id, role, server_id, tool_name, argument_names, "
    "permission_result, security_result, denial_reason, success, error_code, source, verified, duration_ms) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)
_FK_VIOLATION = 1452   # user row vanished between login and audit: keep the event, drop the link


def _clip(value: Any, n: int) -> Optional[str]:
    return None if value is None else str(value)[:n]


def _row(event: dict, user_id: Optional[int]) -> tuple:
    names = ",".join(event.get("argument_names") or [])[:255] or None
    return (
        _clip(event.get("request_id"), 32) or "0" * 32, user_id, _clip(event.get("role"), 20),
        _clip(event.get("server_id"), 64), _clip(event.get("tool"), 64), names,
        event.get("permission_result") or "NOT_CHECKED", event.get("security_result") or "NOT_CHECKED",
        _clip(event.get("denial_reason"), 60), int(bool(event.get("success"))),
        _clip(event.get("error_code"), 64), _clip(event.get("source"), 64),
        int(bool(event.get("verified"))), event.get("duration_ms"),
    )


class DbAuditor:
    """Writes to mcp_audit_logs. If the database is unavailable the event is still written to the
    application log, so an outage never silently erases the audit trail."""

    def __init__(self, connection_factory: Optional[Callable[[], Any]] = None):
        self._factory = connection_factory

    def _connect(self):
        if self._factory:
            return self._factory()
        from src.database.database import get_db_connection
        return get_db_connection()

    def record(self, event: dict) -> None:
        try:
            self._insert(event, event.get("user_id"))
        except Exception as exc:
            if getattr(exc, "errno", None) == _FK_VIOLATION and event.get("user_id") is not None:
                try:
                    self._insert(event, None)
                    return
                except Exception:
                    pass
            log.error("audit DB write failed; event kept in log",
                      extra={"extra_data": {"exc_type": type(exc).__name__}})
            fallback_log.warning("mcp_call_unstored", extra={"extra_data": event})

    def _insert(self, event: dict, user_id: Optional[int]) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_INSERT, _row(event, user_id))
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()


class CompositeAuditor:
    """Fan-out; one failing sink never stops the others or the request."""

    def __init__(self, sinks: Iterable[Any]):
        self.sinks: List[Any] = list(sinks)

    def record(self, event: dict) -> None:
        for sink in self.sinks:
            try:
                sink.record(event)
            except Exception as exc:
                log.error("audit sink failed", extra={"extra_data": {"exc_type": type(exc).__name__}})
