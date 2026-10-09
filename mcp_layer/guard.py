"""Tool guard: classifies failures INSIDE a tool and returns a clean MCPResult.

Why: if an exception escapes a FastMCP tool, FastMCP prints the traceback (including driver
text such as host names) and the gateway only learns "the tool failed". Catching here lets us
return the right error_code - above all DATABASE_UNAVAILABLE, which the grounding layer uses
to block the LLM from guessing - and keeps driver messages out of logs and responses.
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable

from .schemas import ErrorCode, MCPResult

log = logging.getLogger("mcp.guard")

# Class names (matched on the exception's MRO, so mysql-connector need not be importable here).
_DB_UNAVAILABLE_CLASSES = frozenset({"InterfaceError", "OperationalError", "PoolError"})
# MySQL client/server errnos that mean "cannot reach / use the database".
_DB_UNAVAILABLE_ERRNOS = frozenset({1040, 1045, 1049, 1053, 1129, 1130, 1205, 1213, 2002, 2003, 2006, 2013, 2055})

_SAFE_MESSAGES = {
    ErrorCode.DATABASE_UNAVAILABLE: "The required system data is currently unavailable.",
    ErrorCode.INVALID_INPUT: "The request could not be processed.",
    ErrorCode.ACCESS_DENIED: "You do not have access to this action.",
    ErrorCode.INTERNAL_ERROR: "Something went wrong. Please try again.",
}


def _is_db_exception(exc: BaseException) -> bool:
    return any((c.__module__ or "").startswith(("mysql.", "_mysql", "pymysql")) for c in type(exc).__mro__)


def classify_exception(exc: BaseException) -> str:
    if _is_db_exception(exc):
        names = {c.__name__ for c in type(exc).__mro__}
        errno = getattr(exc, "errno", None)
        if names & _DB_UNAVAILABLE_CLASSES or errno in _DB_UNAVAILABLE_ERRNOS:
            return ErrorCode.DATABASE_UNAVAILABLE
        return ErrorCode.INTERNAL_ERROR           # e.g. SQL bug, constraint violation
    if isinstance(exc, PermissionError):
        return ErrorCode.ACCESS_DENIED
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return ErrorCode.INVALID_INPUT
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ErrorCode.DATABASE_UNAVAILABLE      # tools only talk to the database / internal services
    return ErrorCode.INTERNAL_ERROR


def _failure(exc: BaseException, server_id: str, tool_name: str) -> dict:
    code = classify_exception(exc)
    log.error("tool failed", extra={"extra_data": {            # type + errno only: never the message
        "server": server_id, "tool": tool_name, "exc_type": type(exc).__name__,
        "errno": getattr(exc, "errno", None) if isinstance(getattr(exc, "errno", None), int) else None,
        "error_code": code}})
    return MCPResult.fail(code, source=server_id, message=_SAFE_MESSAGES[code]).model_dump(mode="json")


def guarded(fn: Callable[..., Any], server_id: str, tool_name: str) -> Callable[..., Any]:
    """Wrap a sync or async tool. Signature and annotations are preserved for FastMCP."""
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                return _failure(exc, server_id, tool_name)
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return _failure(exc, server_id, tool_name)
    return wrapper
