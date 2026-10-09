"""LLM cost arithmetic. Pure functions: no database, no framework, no hard-coded prices.

    input_cost  = input_tokens  / 1,000,000 * input_price_per_million
    output_cost = output_tokens / 1,000,000 * output_price_per_million
    total_cost  = input_cost + output_cost

Decimal is used so money is exact. Results are rounded to 8 places (HALF_UP) because that is the
scale of the llm_usage cost columns, and the total is the sum of the ROUNDED parts so the three
stored numbers always add up.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, NamedTuple, Optional

MILLION = Decimal(1_000_000)
COST_SCALE = Decimal("0.00000001")      # 8 decimal places, matches DECIMAL(14,8)
MAX_TOKENS = 10_000_000


class Cost(NamedTuple):
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal


def _price(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid")
    price = value if isinstance(value, Decimal) else Decimal(str(value))
    if not price.is_finite() or price < 0:
        raise ValueError(f"{label} is invalid")
    return price


def _tokens(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= MAX_TOKENS):
        raise ValueError(f"{label} is invalid")
    return value


def calculate_cost(input_tokens: int, output_tokens: int, input_price_per_million: Any,
                   output_price_per_million: Any) -> Cost:
    tin, tout = _tokens(input_tokens, "input_tokens"), _tokens(output_tokens, "output_tokens")
    pin, pout = _price(input_price_per_million, "input price"), _price(output_price_per_million, "output price")
    input_cost = (Decimal(tin) / MILLION * pin).quantize(COST_SCALE, rounding=ROUND_HALF_UP)
    output_cost = (Decimal(tout) / MILLION * pout).quantize(COST_SCALE, rounding=ROUND_HALF_UP)
    return Cost(input_cost, output_cost, input_cost + output_cost)


def money_text(value: Optional[Decimal]) -> Optional[str]:
    """Exact, plain (never scientific) text, or None. Used in tool results so no precision is lost."""
    return None if value is None else format(value, "f")
