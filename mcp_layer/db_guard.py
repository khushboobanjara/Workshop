"""Fail-closed database health check used before AI/data operations."""

from __future__ import annotations

from typing import Any


def check_database() -> dict[str, Any]:
    """Check the configured clinic database without leaking connection details."""
    connection = None
    cursor = None
    try:
        # Lazy import means unit tests and static tooling can load this module
        # even before database dependencies/configuration have been installed.
        from src.database.database import get_db_connection

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        if not result or result[0] != 1:
            return {
                "available": False,
                "message": "Clinic database health check did not return a valid result.",
            }
        return {"available": True, "message": "Clinic database is reachable."}
    except Exception as exc:
        # Do not return exception details: database messages can include
        # infrastructure information that should not be exposed to users.
        return {
            "available": False,
            "message": "Clinic database is unavailable. No AI-generated answer was produced.",
            "error_type": type(exc).__name__,
        }
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
