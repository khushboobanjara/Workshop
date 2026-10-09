"""Writes the code-defined registry into mcp_servers / mcp_tools.

The CODE is the source of truth for what servers/tools exist and their metadata. The database
holds runtime state (status, health, is_enabled) which a sync must never overwrite.
Tools removed from code are reported, never auto-deleted or auto-disabled.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .registry import MCPRegistry

log = logging.getLogger("mcp.registry_sync")

_UPSERT_SERVER = (
    "INSERT INTO mcp_servers (server_id, server_name, server_type, description) "
    "VALUES (%s, %s, %s, %s) "
    "ON DUPLICATE KEY UPDATE server_name = VALUES(server_name), "
    "server_type = VALUES(server_type), description = VALUES(description)"
)
_UPSERT_TOOL = (
    "INSERT INTO mcp_tools (server_id, tool_name, permission, ownership, kind, "
    "audit_required, requires_verified_data) VALUES (%s, %s, %s, %s, %s, %s, %s) "
    "ON DUPLICATE KEY UPDATE permission = VALUES(permission), ownership = VALUES(ownership), "
    "kind = VALUES(kind), audit_required = VALUES(audit_required), "
    "requires_verified_data = VALUES(requires_verified_data)"
)
_LIST_TOOLS = "SELECT server_id, tool_name FROM mcp_tools"


def _default_connection():
    from src.database.database import get_db_connection   # lazy: keeps this module importable in tests
    return get_db_connection()


def sync_registry_to_db(registry: MCPRegistry,
                        connection_factory: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    """Idempotent. Returns {"servers": n, "tools": n, "orphaned_tools": [...]} or raises on DB error
    after rolling back (all-or-nothing)."""
    conn = (connection_factory or _default_connection)()
    try:
        cur = conn.cursor()
        servers = tools = 0
        known = set()
        for spec in registry.servers():
            cur.execute(_UPSERT_SERVER, (spec.server_id, spec.server_name,
                                         spec.server_type.value, spec.description[:255]))
            servers += 1
            for name, meta in spec.tools.items():
                cur.execute(_UPSERT_TOOL, (spec.server_id, name, meta.permission,
                                           meta.ownership.value, meta.kind.value,
                                           int(meta.audit_required), int(meta.requires_verified_data)))
                tools += 1
                known.add((spec.server_id, name))
        cur.execute(_LIST_TOOLS)
        orphaned: List[str] = sorted(f"{s}.{t}" for s, t in cur.fetchall() if (s, t) not in known)
        conn.commit()
        if orphaned:
            log.warning("tools in DB but not in code (left untouched): %s", orphaned)
        log.info("registry synced: %d servers, %d tools", servers, tools)
        return {"servers": servers, "tools": tools, "orphaned_tools": orphaned}
    except Exception:
        conn.rollback()
        log.exception("registry sync failed")
        raise
    finally:
        conn.close()