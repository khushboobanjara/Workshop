"""Request-local LLM token accounting and per-user in-memory summaries."""

from __future__ import annotations

import contextvars
import os
import threading
from copy import deepcopy
from typing import Any

_request_usage: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "llm_request_usage", default=None
)
_usage_lock = threading.Lock()
_user_usage: dict[str, dict[str, Any]] = {}


def _rate(env_name: str) -> float | None:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _chat_model() -> str:
    return (os.getenv("GROQ_CHAT_MODEL") or "openai/gpt-oss-120b").strip()


def _vision_model() -> str:
    return (os.getenv("GROQ_VISION_MODEL") or "qwen/qwen3.8-27b").strip()


def pricing_rates(model: str | None = None) -> dict[str, Any]:
    """Return rates for the requested model; do not price a vision model at chat rates."""
    selected_model = (model or _chat_model()).strip()
    if selected_model == _chat_model():
        env = {
            "input_usd_per_million_tokens": "LLM_INPUT_PRICE_PER_1M_USD",
            "cached_input_usd_per_million_tokens": "LLM_CACHED_INPUT_PRICE_PER_1M_USD",
            "output_usd_per_million_tokens": "LLM_OUTPUT_PRICE_PER_1M_USD",
        }
    elif selected_model == _vision_model():
        env = {
            "input_usd_per_million_tokens": "GROQ_VISION_INPUT_PRICE_PER_1M_USD",
            "cached_input_usd_per_million_tokens": "GROQ_VISION_CACHED_INPUT_PRICE_PER_1M_USD",
            "output_usd_per_million_tokens": "GROQ_VISION_OUTPUT_PRICE_PER_1M_USD",
        }
    else:
        # Unknown model: require explicitly configured vision/alternate-model rates
        # rather than silently estimating with the default chat model's rates.
        env = {
            "input_usd_per_million_tokens": "GROQ_VISION_INPUT_PRICE_PER_1M_USD",
            "cached_input_usd_per_million_tokens": "GROQ_VISION_CACHED_INPUT_PRICE_PER_1M_USD",
            "output_usd_per_million_tokens": "GROQ_VISION_OUTPUT_PRICE_PER_1M_USD",
        }
    return {
        "model": selected_model,
        **{key: _rate(env_name) for key, env_name in env.items()},
    }


def reset_request_usage() -> None:
    _request_usage.set(None)


def set_request_usage(usage: dict[str, Any]) -> None:
    _request_usage.set(dict(usage))


def get_request_usage() -> dict[str, Any] | None:
    usage = _request_usage.get()
    return deepcopy(usage) if usage else None


def _price(usage: dict[str, Any]) -> float | None:
    rates = pricing_rates(str(usage.get("model") or _chat_model()))
    input_rate = rates["input_usd_per_million_tokens"]
    output_rate = rates["output_usd_per_million_tokens"]
    cached_rate = rates["cached_input_usd_per_million_tokens"]
    if input_rate is None or output_rate is None:
        return None

    input_tokens = max(0, int(usage.get("prompt_tokens", 0) or 0))
    cached_tokens = max(0, min(input_tokens, int(usage.get("cached_input_tokens", 0) or 0)))
    if cached_tokens and cached_rate is None:
        return None
    uncached_tokens = input_tokens - cached_tokens
    input_cost = uncached_tokens * input_rate
    if cached_tokens:
        input_cost += cached_tokens * float(cached_rate)
    output_cost = max(0, int(usage.get("completion_tokens", 0) or 0)) * output_rate
    return (input_cost + output_cost) / 1_000_000


def record_user_usage(user_id: Any, usage: dict[str, Any]) -> dict[str, Any]:
    """Record token counts and priced totals, without retaining prompt/answer text."""
    key = str(user_id)
    prompt_tokens = max(0, int(usage.get("prompt_tokens", 0) or 0))
    completion_tokens = max(0, int(usage.get("completion_tokens", 0) or 0))
    item = {
        "model": str(usage.get("model", "unknown")),
        "prompt_tokens": prompt_tokens,
        "cached_input_tokens": max(0, int(usage.get("cached_input_tokens", 0) or 0)),
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0),
    }
    rates = pricing_rates(item["model"])
    cost = _price(item)
    with _usage_lock:
        summary = _user_usage.setdefault(key, {
            "request_count": 0,
            "prompt_tokens_total": 0,
            "cached_input_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "priced_requests": 0,
            "unpriced_requests": 0,
            "pricing_configured": True,
            "last_request": None,
        })
        summary["request_count"] += 1
        summary["prompt_tokens_total"] += prompt_tokens
        summary["cached_input_tokens_total"] += item["cached_input_tokens"]
        summary["completion_tokens_total"] += completion_tokens
        summary["total_tokens"] += item["total_tokens"]
        if cost is None:
            summary["unpriced_requests"] += 1
            summary["pricing_configured"] = False
        else:
            summary["priced_requests"] += 1
            summary["estimated_cost_usd"] = round(
                float(summary["estimated_cost_usd"]) + cost, 8
            )
        item["estimated_cost_usd"] = round(cost, 8) if cost is not None else None
        item["pricing_rates"] = rates
        summary["last_request"] = item
        return deepcopy(summary)


def get_user_usage(user_id: Any) -> dict[str, Any]:
    with _usage_lock:
        summary = deepcopy(_user_usage.get(str(user_id), {
            "request_count": 0,
            "prompt_tokens_total": 0,
            "cached_input_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "priced_requests": 0,
            "unpriced_requests": 0,
            "pricing_configured": False,
            "last_request": None,
        }))
    summary["pricing_rates"] = pricing_rates()
    vision_rates = pricing_rates(_vision_model())
    summary["vision_pricing_rates"] = vision_rates
    if summary["unpriced_requests"] > 0:
        summary["pricing_configured"] = False
    elif summary["request_count"] == 0:
        rates = summary["pricing_rates"]
        summary["pricing_configured"] = (
            rates["input_usd_per_million_tokens"] is not None
            and rates["output_usd_per_million_tokens"] is not None
        )
    return summary
