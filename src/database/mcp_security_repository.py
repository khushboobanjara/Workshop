"""Queries behind MCP 09 (Security / RBAC / Audit) - Phase 7. Existing repositories are untouched.

Rules followed here:
- every value is a bound parameter; the only SQL text that is ever concatenated is a fixed constant
  defined in this file (never anything that came from a caller);
- only audit metadata is read: argument NAMES, never values (the audit table holds no values);
- Decimal / aggregate values are converted to plain int / float so they serialise as numbers.
"""
from decimal import Decimal

from src.database.database import get_db_connection

_WINDOW = "created_at >= NOW() - INTERVAL %s HOUR"
_PROBLEM = "(permission_result = 'DENIED' OR security_result = 'BLOCKED' OR success = 0)"
_DENIED_OR_BLOCKED = "(permission_result = 'DENIED' OR security_result = 'BLOCKED')"
_LOG_COLUMNS = (
    "audit_id, request_id, user_id, role, server_id, tool_name, argument_names, permission_result, "
    "security_result, denial_reason, success, error_code, source, verified, duration_ms, created_at"
)
SECURITY_SERVER_ID = "mcp_09_security_audit"


def _number(value):
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


def _clean(row):
    return {k: _number(v) for k, v in dict(row).items()} if row else row


def _fetch(query, params=(), one=False):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, params)
        if one:
            return _clean(cursor.fetchone())
        return [_clean(r) for r in cursor.fetchall()]
    finally:
        cursor.close()
        connection.close()


# ------------------------------------------------------------------ reading
def query_audit_logs(hours, only_problems, server_id, tool_name, user_id, limit):
    where, params = [_WINDOW], [hours]
    if only_problems:
        where.append(_PROBLEM)
    if server_id:
        where.append("server_id = %s")
        params.append(server_id)
    if tool_name:
        where.append("tool_name = %s")
        params.append(tool_name)
    if user_id:
        where.append("user_id = %s")
        params.append(user_id)
    params.append(limit)
    return _fetch(
        f"SELECT {_LOG_COLUMNS} FROM mcp_audit_logs WHERE {' AND '.join(where)} "
        "ORDER BY audit_id DESC LIMIT %s",
        tuple(params),
    )


def audit_totals(hours):
    return _fetch(
        "SELECT COUNT(*) AS total_calls, "
        "COALESCE(SUM(permission_result = 'DENIED'), 0) AS denied, "
        "COALESCE(SUM(security_result = 'BLOCKED'), 0) AS blocked, "
        "COALESCE(SUM(success = 0), 0) AS failed, "
        "COUNT(DISTINCT user_id) AS distinct_users "
        f"FROM mcp_audit_logs WHERE {_WINDOW}",
        (hours,), one=True,
    )


def audit_by_server(hours):
    return _fetch(
        "SELECT server_id, COUNT(*) AS calls, "
        "COALESCE(SUM(permission_result = 'DENIED'), 0) AS denied, "
        "COALESCE(SUM(success = 0), 0) AS failed, MAX(created_at) AS last_activity "
        f"FROM mcp_audit_logs WHERE {_WINDOW} AND server_id IS NOT NULL "
        "GROUP BY server_id ORDER BY calls DESC LIMIT 20",
        (hours,),
    )


def top_denial_reasons(hours):
    return _fetch(
        "SELECT denial_reason, COUNT(*) AS occurrences FROM mcp_audit_logs "
        f"WHERE {_WINDOW} AND denial_reason IS NOT NULL "
        "GROUP BY denial_reason ORDER BY occurrences DESC LIMIT 10",
        (hours,),
    )


def security_events(hours, limit):
    return _fetch(
        f"SELECT {_LOG_COLUMNS} FROM mcp_audit_logs WHERE {_WINDOW} AND {_DENIED_OR_BLOCKED} "
        "ORDER BY audit_id DESC LIMIT %s",
        (hours, limit),
    )


def repeat_denials(hours, min_count):
    """Users with many denied / blocked calls in the window (possible probing)."""
    return _fetch(
        "SELECT user_id, role, COUNT(*) AS denials, MAX(created_at) AS last_seen FROM mcp_audit_logs "
        f"WHERE {_WINDOW} AND {_DENIED_OR_BLOCKED} AND user_id IS NOT NULL "
        "GROUP BY user_id, role HAVING COUNT(*) >= %s ORDER BY denials DESC LIMIT 20",
        (hours, min_count),
    )


# ------------------------------------------------------------------ writing
def insert_security_event(request_id, user_id, event_type, reason):
    """One row in mcp_audit_logs for an event that did not come from a gateway call
    (for example the chatbot's own prompt-injection filter). No new table is needed."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO mcp_audit_logs (request_id, user_id, server_id, tool_name, permission_result, "
            "security_result, denial_reason, success, error_code, source, verified) "
            "VALUES (%s, %s, %s, %s, 'NOT_CHECKED', 'BLOCKED', %s, 0, %s, 'mcp_internal', 0)",
            (request_id, user_id, SECURITY_SERVER_ID, event_type, reason, event_type),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()
