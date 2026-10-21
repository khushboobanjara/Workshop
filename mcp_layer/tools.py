"""Database-backed tool implementations shared by FastMCP and FastAPI."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from mcp_layer.db_guard import check_database
from mcp_layer.security import (
    ALL_TOOLS,
    ADMIN_TOOLS,
    SYSTEM_TOOLS,
    USER_TOOLS,
    current_mcp_role,
    require_tool_access,
)


TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "name": "user_search_doctors",
        "group": "USER",
        "description": "Search currently available clinic doctors; results come only from the clinic database.",
        "inputs": [{"name": "specialization", "type": "text", "required": False, "placeholder": "e.g. Cardiologist"}],
    },
    {
        "name": "user_list_specializations",
        "group": "USER",
        "description": "List specializations present in the current doctor database.",
        "inputs": [],
    },
    {
        "name": "user_get_doctor_availability",
        "group": "USER",
        "description": "Return recorded upcoming availability for a doctor ID.",
        "inputs": [{"name": "doctor_id", "type": "number", "required": True, "placeholder": "Doctor ID"}],
    },
    {
        "name": "user_search_medicines",
        "group": "USER",
        "description": "Search the available medicine catalog by name, generic name, or category.",
        "inputs": [{"name": "query", "type": "text", "required": True, "placeholder": "Medicine or category"}],
    },
    {
        "name": "user_get_medicine_details",
        "group": "USER",
        "description": "Look up a medicine by its database ID.",
        "inputs": [{"name": "medicine_id", "type": "number", "required": True, "placeholder": "Medicine ID"}],
    },
    {
        "name": "admin_list_pending_doctors",
        "group": "ADMIN",
        "description": "List doctor registrations awaiting approval. Accessible to ADMIN and SUPER_ADMIN.",
        "inputs": [],
    },
    {
        "name": "admin_approve_doctor",
        "group": "ADMIN",
        "description": "Approve a pending doctor account by its registered email address.",
        "inputs": [{"name": "email", "type": "email", "required": True, "placeholder": "doctor@example.com"}],
    },
    {
        "name": "admin_clinic_metrics",
        "group": "ADMIN",
        "description": "Return aggregate clinic counts only; no patient names or records are exposed.",
        "inputs": [],
    },
    {
        "name": "system_database_health",
        "group": "SYSTEM",
        "description": "Check whether the configured clinic database is reachable with SELECT 1.",
        "inputs": [],
    },
    {
        "name": "system_mcp_status",
        "group": "SYSTEM",
        "description": "Return MCP configuration status without revealing credentials.",
        "inputs": [],
    },
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _database_failure() -> dict[str, Any] | None:
    health = check_database()
    if health.get("available"):
        return None
    return {
        "ok": False,
        "error": "database_unavailable",
        "message": "Clinic database is unavailable. No AI-generated answer or unverified clinic data was produced.",
        "source": "database_guard",
    }


def _user_search_doctors(specialization: str = "") -> dict[str, Any]:
    from src.database.doctor_repository import (
        find_doctors_by_specialization,
        get_all_available_doctors,
    )

    term = str(specialization or "").strip()
    if term:
        rows = find_doctors_by_specialization(term)
    else:
        rows = get_all_available_doctors()
    return {"ok": True, "source": "clinic_database", "count": len(rows), "data": _json_safe(rows)}


def _user_list_specializations() -> dict[str, Any]:
    from src.database.doctor_repository import get_specializations

    rows = get_specializations()
    return {"ok": True, "source": "clinic_database", "count": len(rows), "data": _json_safe(rows)}


def _user_get_doctor_availability(doctor_id: int) -> dict[str, Any]:
    from src.database.doctor_repository import get_doctor_by_id, get_doctor_availability

    doctor_id = int(doctor_id)
    if doctor_id <= 0:
        return {"ok": False, "error": "invalid_doctor_id", "message": "Doctor ID must be a positive integer."}
    doctor = get_doctor_by_id(doctor_id)
    if not doctor:
        return {"ok": True, "source": "clinic_database", "count": 0, "data": [], "message": "No doctor with that ID exists in the database."}
    rows = get_doctor_availability(doctor_id)
    return {"ok": True, "source": "clinic_database", "doctor": _json_safe(doctor), "count": len(rows), "data": _json_safe(rows)}


def _user_search_medicines(query: str) -> dict[str, Any]:
    from src.database.pharmacy_repository import search_medicines

    term = str(query or "").strip()
    if not term:
        return {"ok": False, "error": "query_required", "message": "Enter a medicine name or category."}
    rows = search_medicines(term)[:50]
    return {"ok": True, "source": "clinic_database", "count": len(rows), "data": _json_safe(rows)}


def _user_get_medicine_details(medicine_id: int) -> dict[str, Any]:
    from src.database.pharmacy_repository import get_medicine_by_id

    medicine_id = int(medicine_id)
    if medicine_id <= 0:
        return {"ok": False, "error": "invalid_medicine_id", "message": "Medicine ID must be a positive integer."}
    row = get_medicine_by_id(medicine_id)
    return {
        "ok": True,
        "source": "clinic_database",
        "count": 1 if row else 0,
        "data": _json_safe(row),
        "message": None if row else "No medicine with that ID exists in the database.",
    }


def _admin_list_pending_doctors() -> dict[str, Any]:
    from src.database.database import get_db_connection

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT user_id, full_name, email, phone, created_at
            FROM users
            WHERE role = 'DOCTOR' AND UPPER(COALESCE(doctor_status, 'PENDING')) = 'PENDING'
            ORDER BY created_at ASC
            LIMIT 200
            """
        )
        rows = cursor.fetchall()
        return {"ok": True, "source": "clinic_database", "count": len(rows), "data": _json_safe(rows)}
    finally:
        cursor.close()
        connection.close()


def _admin_approve_doctor(email: str) -> dict[str, Any]:
    from src.database.database import get_db_connection

    email = str(email or "").strip().lower()
    if not email or "@" not in email or len(email) > 254:
        return {"ok": False, "error": "invalid_email", "message": "Provide a valid doctor account email."}

    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            UPDATE users
            SET doctor_status = 'APPROVED'
            WHERE LOWER(email) = %s AND role = 'DOCTOR'
              AND UPPER(COALESCE(doctor_status, 'PENDING')) = 'PENDING'
            """,
            (email,),
        )
        changed = cursor.rowcount
        connection.commit()
        if changed:
            return {"ok": True, "source": "clinic_database", "approved": True, "email": email, "message": "Doctor account approved."}

        cursor.execute(
            "SELECT doctor_status FROM users WHERE LOWER(email) = %s AND role = 'DOCTOR' LIMIT 1",
            (email,),
        )
        row = cursor.fetchone()
        if not row:
            return {"ok": True, "source": "clinic_database", "approved": False, "message": "No doctor account with that email exists."}
        status = row[0]
        return {"ok": True, "source": "clinic_database", "approved": False, "doctor_status": status, "message": "Doctor account was not pending, so no change was made."}
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def _admin_clinic_metrics() -> dict[str, Any]:
    from src.database.database import get_db_connection

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    queries = {
        "total_users": "SELECT COUNT(*) AS value FROM users",
        "total_doctors": "SELECT COUNT(*) AS value FROM doctors",
        "available_doctors": "SELECT COUNT(*) AS value FROM doctors WHERE available = TRUE",
        "total_appointments": "SELECT COUNT(*) AS value FROM appointments",
        "pending_appointments": "SELECT COUNT(*) AS value FROM appointments WHERE status IN ('BOOKED', 'CONFIRMED')",
        "available_medicines": "SELECT COUNT(*) AS value FROM medicines WHERE is_available = TRUE",
    }
    try:
        metrics: dict[str, int] = {}
        for name, query in queries.items():
            cursor.execute(query)
            row = cursor.fetchone()
            metrics[name] = int(row["value"] if row else 0)
        return {"ok": True, "source": "clinic_database", "data": metrics}
    finally:
        cursor.close()
        connection.close()


def _system_database_health() -> dict[str, Any]:
    return {"ok": True, "source": "system_probe", **check_database()}


def _system_mcp_status() -> dict[str, Any]:
    from mcp_layer.security import TOKEN_ENV_BY_ROLE

    import os
    configured = [
        role for role, env_name in TOKEN_ENV_BY_ROLE.items()
        if os.getenv(env_name, "").strip()
    ]
    return {
        "ok": True,
        "server": "Sanjeevani Clinic MCP",
        "tool_count": len(TOOL_CATALOG),
        "tools_by_group": {
            "USER": len(USER_TOOLS),
            "ADMIN": len(ADMIN_TOOLS),
            "SYSTEM": len(SYSTEM_TOOLS),
        },
        "configured_token_roles": configured,
        "security_note": "Token values are never returned.",
    }


_HANDLERS = {
    "user_search_doctors": _user_search_doctors,
    "user_list_specializations": _user_list_specializations,
    "user_get_doctor_availability": _user_get_doctor_availability,
    "user_search_medicines": _user_search_medicines,
    "user_get_medicine_details": _user_get_medicine_details,
    "admin_list_pending_doctors": _admin_list_pending_doctors,
    "admin_approve_doctor": _admin_approve_doctor,
    "admin_clinic_metrics": _admin_clinic_metrics,
    "system_database_health": _system_database_health,
    "system_mcp_status": _system_mcp_status,
}


def list_tools_for_role(role: str | None) -> list[dict[str, Any]]:
    from copy import deepcopy
    from mcp_layer.security import can_access_tool

    return [deepcopy(tool) for tool in TOOL_CATALOG if can_access_tool(role, tool["name"])]


def invoke_tool(tool_name: str, arguments: dict[str, Any] | None = None, role: str | None = None) -> dict[str, Any]:
    """Invoke a registered MCP tool through shared RBAC and database guards."""
    arguments = arguments or {}
    if tool_name not in _HANDLERS:
        raise ValueError("Unknown MCP tool.")

    effective_role = role if role is not None else current_mcp_role()
    require_tool_access(effective_role, tool_name)

    # System health/status remain available during DB outages so operators can
    # diagnose the failure. Every other data tool fails closed before querying.
    if tool_name not in SYSTEM_TOOLS:
        failure = _database_failure()
        if failure:
            return failure

    try:
        result = _HANDLERS[tool_name](**arguments)
        return _json_safe(result)
    except PermissionError:
        raise
    except Exception as exc:
        # Fail closed and avoid leaking SQL, environment variables, or stack traces.
        if tool_name not in SYSTEM_TOOLS and not check_database().get("available"):
            return {
                "ok": False,
                "error": "database_unavailable",
                "message": "Clinic database is unavailable. No AI-generated answer or unverified clinic data was produced.",
                "source": "database_guard",
            }
        return {
            "ok": False,
            "error": "tool_execution_failed",
            "message": "The MCP tool could not complete the request. No unverified result was returned.",
            "detail_type": type(exc).__name__,
        }
