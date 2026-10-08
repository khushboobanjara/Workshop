"""Central MCP registry: which servers exist, their roles, tools and health."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from fastmcp import FastMCP

from .schemas import Role, ServerType, ToolMeta

# A tool must never take identity or raw-SQL style parameters from the caller.
FORBIDDEN_PARAMS = frozenset({"role", "is_admin", "token", "password", "sql", "command", "code"})
OWN_DATA_FORBIDDEN_PARAMS = frozenset({"user_id", "patient_id", "owner_id"})


@dataclass
class MCPServerSpec:
    server_id: str
    server_name: str
    server_type: ServerType
    description: str
    allowed_roles: frozenset
    mcp: FastMCP = field(repr=False, default=None)
    tools: Dict[str, ToolMeta] = field(default_factory=dict)
    status: str = "active"            # active | disabled
    health_status: str = "unknown"    # healthy | degraded | down | unknown

    def tool(self, meta: ToolMeta) -> Callable:
        """Register a tool: metadata into the registry, function into FastMCP."""
        if meta.name in self.tools:
            raise ValueError(f"duplicate tool {meta.name} on {self.server_id}")
        if not meta.allowed_roles <= self.allowed_roles:
            raise ValueError(f"{meta.name}: tool roles exceed server roles on {self.server_id}")

        def decorator(fn: Callable) -> Callable:
            params = set(inspect.signature(fn).parameters)
            bad = params & FORBIDDEN_PARAMS
            if meta.ownership.value == "current_user_only":
                bad |= params & OWN_DATA_FORBIDDEN_PARAMS
            if bad:
                raise ValueError(f"{meta.name}: forbidden parameter(s) {sorted(bad)}")
            self.mcp.tool(name=meta.name, description=meta.description or fn.__doc__ or meta.name)(fn)
            self.tools[meta.name] = meta
            return fn

        return decorator

    def public_dict(self) -> dict:
        """Safe summary for the dashboard. No secrets, no callables."""
        return {
            "server_id": self.server_id,
            "server_name": self.server_name,
            "server_type": self.server_type.value,
            "description": self.description,
            "status": self.status,
            "health_status": self.health_status,
            "allowed_roles": sorted(r.value for r in self.allowed_roles),
            "tools": sorted(self.tools),
            "tool_count": len(self.tools),
        }


class MCPRegistry:
    def __init__(self) -> None:
        self._servers: Dict[str, MCPServerSpec] = {}

    def add(self, spec: MCPServerSpec) -> MCPServerSpec:
        if spec.server_id in self._servers:
            raise ValueError(f"duplicate server {spec.server_id}")
        self._servers[spec.server_id] = spec
        return spec

    def get(self, server_id: str) -> Optional[MCPServerSpec]:
        return self._servers.get(server_id)

    def servers(self, server_type: Optional[ServerType] = None) -> List[MCPServerSpec]:
        items: Iterable[MCPServerSpec] = self._servers.values()
        if server_type:
            items = (s for s in items if s.server_type == server_type)
        return list(items)

    def counts(self) -> dict:
        by = {t.value: 0 for t in ServerType}
        for s in self._servers.values():
            by[s.server_type.value] += 1
        return {"total": len(self._servers), **by}

    def tool_count(self) -> int:
        return sum(len(s.tools) for s in self._servers.values())

    def discover(self, role: Role) -> List[dict]:
        """Servers a role may see (coarse). Tool-level checks come in Phase 3."""
        return [s.public_dict() for s in self._servers.values() if role in s.allowed_roles]
