"""MCP-backed data sources for the chatbot (Phase 11, slice A).

Strangler pattern: the existing handlers keep their wording, their option building and their booking
state machine. Only WHERE THE DATA COMES FROM changes, from a repository call to a gateway call. Each
fetch_* function returns (rows, failure) where `failure` is a ready-to-send chat response, or None.

This module imports chatbot.service lazily (inside functions) because service imports this module.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

from chatbot import mcp_bridge

log = logging.getLogger("chatbot.mcp")

MCP02, MCP05 = "mcp_02_doctor_appointment", "mcp_05_ai_assistant"
FailureOr = Tuple[Optional[List[dict]], Optional[dict]]


def _respond(text: str) -> dict:
    from chatbot import service
    return service.chatbot_response(text)


async def fetch_doctors(specialty: Any, principal) -> FailureOr:
    """Doctors for a specialty via MCP 02 find_doctors."""
    result = await mcp_bridge.call(principal, MCP02, "find_doctors", {"specialization": str(specialty).strip()})
    if not (result.success and result.verified):
        return None, _respond(mcp_bridge.user_message(result))
    rows = result.data if isinstance(result.data, list) else []
    return [row for row in rows if isinstance(row, dict)], None


async def fetch_my_appointments(principal) -> FailureOr:
    """The signed-in user's own appointments via MCP 02 get_my_appointments (identity is the principal's,
    never the session dict's)."""
    result = await mcp_bridge.call(principal, MCP02, "get_my_appointments")
    if not (result.success and result.verified):
        return None, _respond(mcp_bridge.user_message(result))
    rows = result.data if isinstance(result.data, list) else []
    return [row for row in rows if isinstance(row, dict)], None


# ----------------------------------------------------------- personal questions
# A question about the user's OWN records that the chatbot used to hand to the LLM with no data, so
# the LLM guessed ("your appointment is tomorrow"). These now go: MCP 05 -> grounding gate -> LLM.
# Whole words only: "add" must not match "address", nor "book" match "booking". "order" is a verb only in an
# imperative position ("order paracetamol", "I want to order"), never in "my order" / "order status".
_ACTION = re.compile(r"\b(?:cancel|reschedule|rebook|book|change|move|delete|remove|buy|add|update|edit|modify|"
                     r"pay|refund|replace|checkout)\b|"
                     r"(?:^|\b(?:to|please|can you|could you|let me|let's)\s+)order\b(?!\s+(?:status|history|id|number|details|summary)\b)",
                     re.I)
_WHEN = r"(?:next|upcoming|last|latest|previous|past|recent|current|earliest)"
_TOPICS = {
    "profile": re.compile(r"\bmy\s+(?:profile|account|details|phone(?:\s+number)?|mobile(?:\s+number)?|"
                          r"e-?mail(?:\s+address|\s+id)?|contact(?:\s+details|\s+info)?|"
                          r"registered\s+(?:name|e-?mail|phone|number))\b", re.I),
    "appointments": re.compile(rf"\bmy\s+(?:{_WHEN}\s+)?(?:appointments?|bookings?|visits?|consultations?)\b", re.I),
    "cart": re.compile(r"\bmy\s+(?:cart|basket)\b", re.I),
    "orders": re.compile(rf"\bmy\s+(?:(?:medicine|pharmacy|{_WHEN})\s+)?orders?\b", re.I),
}


def personal_topics(message: str) -> List[str]:
    """Topics of the user's own data that this message asks about; [] when it should take the normal path.
    Deliberately conservative: it needs 'my <thing>' and no action verb (cancel, book, order, ...)."""
    if not isinstance(message, str) or _ACTION.search(message):
        return []
    return [name for name, pattern in _TOPICS.items() if pattern.search(message)]


async def answer_personal_question(message: str, topics: List[str], principal, conversation_history=None) -> dict:
    """Fetch verified context from MCP 05, then let the grounding gate decide whether the LLM may speak."""
    from chatbot.llm import generate_response
    from chatbot.prompts import SYSTEM_PROMPT
    from chatbot import service

    runtime = mcp_bridge.get_runtime()
    if runtime is None or principal is None:
        return service.chatbot_response(mcp_bridge.unavailable_text() if principal else "Please log in again.")

    result = await mcp_bridge.call(principal, MCP05, "get_verified_context", {"topics": topics})
    reply = await runtime.gate.answer(
        results={"verified_context": result}, user_message=message, system_prompt=SYSTEM_PROMPT,
        llm=generate_response, history=conversation_history, principal=principal, server_id=MCP05)
    return service.chatbot_response(reply.text)
