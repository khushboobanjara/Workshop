"""Glue for Phase 11: the chatbot calls track_llm_usage() after every LLM request.

It records through the gateway as the internal SYSTEM identity (MCP 10 record_llm_usage), so the
write is authorised and audited like any other call. It NEVER raises: if usage cannot be stored,
the user's reply must not break - the failure is logged and the gateway audit shows it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from .schemas import MCPResult, Principal

log = logging.getLogger("mcp.llm_tracking")
MONITORING_SERVER = "mcp_10_llm_monitoring"


def usage_from_response(response: Any) -> Optional[Tuple[int, int]]:
    """(input_tokens, output_tokens) from an OpenAI-style / Groq response, or None if the provider
    reported no usable numbers. Never guesses or estimates."""
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    ok = lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 0
    return (prompt, completion) if ok(prompt) and ok(completion) else None


async def track_llm_usage(gateway: Any, *, model: str, input_tokens: int, output_tokens: int,
                          status: str = "SUCCESS", request_ref: Optional[str] = None,
                          user_id: Optional[int] = None, server_id: Optional[str] = None,
                          tool_name: Optional[str] = None) -> Optional[MCPResult]:
    arguments: Dict[str, Any] = {"model": model, "input_tokens": input_tokens,
                                 "output_tokens": output_tokens, "status": status}
    for key, value in (("request_ref", request_ref), ("subject_user_id", user_id),
                       ("mcp_server_id", server_id), ("mcp_tool_name", tool_name)):
        if value is not None:
            arguments[key] = value
    try:
        result = await gateway.invoke(Principal.system(), MONITORING_SERVER, "record_llm_usage", arguments)
    except Exception as exc:
        log.error("LLM usage tracking failed", extra={"extra_data": {"exc_type": type(exc).__name__}})
        return None
    if not result.success:
        log.warning("LLM usage not recorded", extra={"extra_data": {"error_code": result.error_code}})
    return result
