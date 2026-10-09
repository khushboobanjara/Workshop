"""Database-held RESTRICTIONS on top of the code-defined permissions (table mcp_permissions).

A row with granted=0 revokes a role's access to a tool. A row can never grant more than the
code allows. If the table cannot be read and there is no cached copy, access is refused.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, Protocol, Set, Tuple

from .schemas import Role

log = logging.getLogger("mcp.overrides")

_REVOKED_SQL = (
    "SELECT p.role, t.server_id, t.tool_name FROM mcp_permissions p "
    "JOIN mcp_tools t ON t.tool_id = p.tool_id WHERE p.granted = 0"
)


class PermissionOverrides(Protocol):
    def is_revoked(self, role: Role, server_id: str, tool_name: str) -> bool: ...


class NoOverrides:
    def is_revoked(self, role, server_id, tool_name) -> bool:
        return False


class DbPermissionOverrides:
    def __init__(self, connection_factory: Optional[Callable[[], Any]] = None, ttl_seconds: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self._factory, self._ttl, self._clock = connection_factory, ttl_seconds, clock
        self._revoked: Optional[Set[Tuple[str, str, str]]] = None
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    def _connect(self):
        if self._factory:
            return self._factory()
        from src.database.database import get_db_connection
        return get_db_connection()

    def _refresh(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_REVOKED_SQL)
            self._revoked = {(str(r), s, t) for r, s, t in cur.fetchall()}
            self._loaded_at = self._clock()
        finally:
            conn.close()

    def is_revoked(self, role: Role, server_id: str, tool_name: str) -> bool:
        with self._lock:
            if self._revoked is None or self._clock() - self._loaded_at > self._ttl:
                try:
                    self._refresh()
                except Exception as exc:
                    if self._revoked is None:
                        raise                      # nothing cached -> caller must deny
                    log.error("override refresh failed, using last known copy",
                              extra={"extra_data": {"exc_type": type(exc).__name__}})
                    self._loaded_at = self._clock()   # back off for one TTL
            return (role.value, server_id, tool_name) in self._revoked
