"""Assembles the production gateway and runs the startup checks (used by FastAPI in Phase 12)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from .audit_db import CompositeAuditor, DbAuditor
from .config import MCPSettings, load_settings
from .gateway import Auditor, LoggingAuditor, MCPGateway, SafetyGate
from .overrides import DbPermissionOverrides, NoOverrides
from .permissions import validate_registry_permissions
from .rbac import RBACAuthorizer
from .registry import MCPRegistry
from .registry_sync import sync_registry_to_db
from .safety import SafetyLayer
from .servers import build_registry

log = logging.getLogger("mcp.bootstrap")


def build_gateway(registry: Optional[MCPRegistry] = None, *, use_db_overrides: bool = True,
                  use_db_audit: bool = True, auditor: Optional[Auditor] = None,
                  settings: Optional[MCPSettings] = None, safety: Optional[SafetyGate] = None) -> MCPGateway:
    registry = registry or build_registry(with_tools=True)
    problems = validate_registry_permissions(registry)
    if problems:   # refuse to start with a mis-declared tool rather than run it half-protected
        raise RuntimeError("MCP permission matrix mismatch: " + "; ".join(problems))
    if auditor is None:
        # Always log to file (survives a DB outage); additionally store in mcp_audit_logs.
        auditor = CompositeAuditor([LoggingAuditor(), DbAuditor()] if use_db_audit else [LoggingAuditor()])
    settings = settings or load_settings()
    safety = safety or SafetyLayer.from_settings(settings)   # Phase 10: on by default; pass safety= to replace
    authorizer = RBACAuthorizer(DbPermissionOverrides() if use_db_overrides else NoOverrides())
    gateway = MCPGateway(registry, authorizer=authorizer, auditor=auditor, settings=settings, safety=safety)
    if registry.get("mcp_05_ai_assistant") and "get_verified_context" in registry.get("mcp_05_ai_assistant").tools:
        from .servers.mcp_05_assistant import bind_gateway   # the assistant calls the other MCPs via the gateway
        bind_gateway(gateway)
    if registry.get("mcp_09_security_audit") and "check_permission" in registry.get("mcp_09_security_audit").tools:
        from .servers.mcp_09_security_audit import bind_gateway as bind_security   # check_permission uses the real authorizer
        bind_security(gateway)
    if registry.get("mcp_10_llm_monitoring") and "get_mcp_health" in registry.get("mcp_10_llm_monitoring").tools:
        from .servers.mcp_10_llm_monitoring import bind_gateway as bind_monitoring   # get_mcp_health pings the servers
        bind_monitoring(gateway)
    return gateway


def database_available(connection_factory: Optional[Callable[[], Any]] = None) -> bool:
    try:
        if connection_factory is None:
            from src.database.database import get_db_connection as connection_factory
        conn = connection_factory()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchall()
            return True
        finally:
            conn.close()
    except Exception as exc:
        log.error("database probe failed", extra={"extra_data": {"exc_type": type(exc).__name__}})
        return False


async def initialize_mcp(gateway: MCPGateway, *, sync_registry: bool = True,
                         connection_factory: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    """Startup checks. NEVER raises: a database outage must not stop the website from starting;
    it only means MCP tools report DATABASE_UNAVAILABLE until the database returns."""
    report: Dict[str, Any] = {}
    db_up = await asyncio.to_thread(database_available, connection_factory)
    report["database"] = "up" if db_up else "down"

    if sync_registry and db_up:
        try:
            report["registry_sync"] = await asyncio.to_thread(sync_registry_to_db, gateway.registry, connection_factory)
        except Exception as exc:
            report["registry_sync"] = {"error": type(exc).__name__}
    else:
        report["registry_sync"] = {"skipped": "database down" if not db_up else "disabled"}

    report["health"] = await gateway.health_check()
    log.info("mcp initialized", extra={"extra_data": {"database": report["database"],
                                                      "servers": len(report["health"])}})
    return report
