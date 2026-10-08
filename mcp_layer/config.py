"""MCP settings. Behaviour switches only - no secrets and no prices.

Pricing lives in the llm_pricing table (see migrations/001_mcp_foundation.sql).
"""
import os
from dataclasses import dataclass


def _int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class MCPSettings:
    enabled: bool
    tool_timeout_seconds: int
    max_argument_bytes: int
    rate_limit_per_minute: int


def load_settings() -> MCPSettings:
    return MCPSettings(
        enabled=os.getenv("MCP_ENABLED", "true").strip().lower() not in ("0", "false", "no"),
        tool_timeout_seconds=_int("MCP_TOOL_TIMEOUT_SECONDS", 15, 1, 120),
        max_argument_bytes=_int("MCP_MAX_ARGUMENT_BYTES", 20_000, 100, 1_000_000),
        rate_limit_per_minute=_int("MCP_RATE_LIMIT_PER_MINUTE", 60, 1, 10_000),
    )
