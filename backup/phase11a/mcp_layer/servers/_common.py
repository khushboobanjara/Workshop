"""Helpers shared by the tool modules. No database access, no FastMCP import."""
from __future__ import annotations

import functools
import inspect
import re
from datetime import date, datetime, time
from typing import Any, Callable, Iterable, Optional

from ..schemas import ErrorCode, MCPResult

SOURCE_DB = "mysql"

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PHONE = re.compile(r"^\+?[0-9]{7,15}$")


class BadInput(Exception):
    """A validation / business-rule failure whose message is safe to show the user."""

    def __init__(self, message: str, code: str = ErrorCode.INVALID_INPUT):
        super().__init__(message)
        self.message = message
        self.code = code


# ------------------------------------------------------------------ results
def ok(data: Any, source: str = SOURCE_DB, message: Optional[str] = None, **metadata) -> dict:
    return MCPResult.ok(data, source=source, message=message, metadata=metadata).model_dump(mode="json")


def ok_rows(rows: Optional[Iterable], empty_message: str, source: str = SOURCE_DB) -> dict:
    """Verified list result. An empty list is a real answer ("nothing found"), not an error,
    but it is flagged with row_count=0 so the grounding gate (Phase 9) can give a controlled reply."""
    items = list(rows or [])
    return MCPResult.ok(items, source=source, message=None if items else empty_message,
                        metadata={"row_count": len(items)}).model_dump(mode="json")


def fail(code: str, message: Optional[str] = None, source: str = SOURCE_DB) -> dict:
    return MCPResult.fail(code, source=source, message=message).model_dump(mode="json")


def rules(fn: Callable) -> Callable:
    """Turns BadInput into a clean failed MCPResult. Everything else is left to guard.guarded(),
    which classifies database outages. Preserves the signature for FastMCP."""
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except BadInput as exc:
                return fail(exc.code, exc.message)
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BadInput as exc:
            return fail(exc.code, exc.message)
    return wrapper


# --------------------------------------------------------------- validation
def clean_text(value: Any, label: str, min_len: int = 1, max_len: int = 100) -> str:
    if not isinstance(value, str):
        raise BadInput(f"{label} is required.")
    text = " ".join(_CONTROL.sub(" ", value).split())
    if not (min_len <= len(text) <= max_len):
        raise BadInput(f"{label} must be between {min_len} and {max_len} characters.")
    return text


def clean_phone(value: Any) -> str:
    if not isinstance(value, str):
        raise BadInput("A valid phone number is required.")
    phone = re.sub(r"[\s\-()]", "", value)
    if not _PHONE.match(phone):
        raise BadInput("A valid phone number is required.")
    return phone


def positive_int(value: Any, label: str, maximum: int = 2_000_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (0 < value <= maximum):
        raise BadInput(f"{label} is invalid.")
    return value


def parse_date(value: Any) -> date:
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError:
        raise BadInput("Date must be in YYYY-MM-DD format.") from None


def parse_time(value: Any) -> time:
    try:
        return datetime.strptime(str(value).strip(), "%H:%M").time()
    except ValueError:
        raise BadInput("Time must be in HH:MM (24-hour) format.") from None


def require_future(appointment_date: date, appointment_time: time, now: Optional[datetime] = None) -> None:
    now = now or datetime.now()
    if datetime.combine(appointment_date, appointment_time) <= now:
        raise BadInput("Appointments can only be booked for a future time.")


def public_rows(rows: Optional[Iterable[dict]], drop: Iterable[str] = ()) -> list:
    blocked = set(drop)
    return [{k: v for k, v in dict(r).items() if k not in blocked} for r in (rows or [])]


def opt_text(value: Any, label: str, min_len: int = 1, max_len: int = 100) -> Optional[str]:
    """None / blank -> None (leave the field alone); otherwise validated text."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return clean_text(value, label, min_len, max_len)


def money(value: Any, label: str, maximum: float = 1_000_000) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 <= float(value) <= maximum):
        raise BadInput(f"{label} must be a number between 0 and {maximum:g}.")
    return round(float(value), 2)


def whole_number(value: Any, label: str, minimum: int = 0, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (minimum <= value <= maximum):
        raise BadInput(f"{label} must be a whole number between {minimum} and {maximum}.")
    return value


def clean_email(value: Any) -> str:
    text = clean_text(value, "Email", 5, 120).lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", text):
        raise BadInput("A valid email address is required.")
    return text
