"""Queries behind MCP 10 (LLM / Token / Cost / Monitoring) - Phase 8. Existing repositories untouched.

Uses the llm_pricing and llm_usage tables from migration 001 (no new tables).
Rules: every value is a bound parameter; concatenated SQL text is only fixed constants from this
file; aggregates become plain int / float; costs that are stored per request stay exact Decimals.
"""
from decimal import Decimal

import mysql.connector

from src.database.database import get_db_connection

_WINDOW = "created_at >= NOW() - INTERVAL %s HOUR"
_USAGE_COLUMNS = (
    "usage_id, request_id, user_id, role, mcp_server_id, tool_name, model, input_tokens, output_tokens, "
    "total_tokens, input_cost, output_cost, total_cost, currency, status, created_at"
)
_DUPLICATE_KEY = 1062


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


# ------------------------------------------------------------------ pricing
def get_active_price(model):
    """The price in force today: the newest ACTIVE row whose effective_from has arrived.
    Returns the raw Decimal prices (not converted), or None if the model has no price."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT model_name, input_price_per_million, output_price_per_million, currency, effective_from "
            "FROM llm_pricing WHERE model_name = %s AND is_active = 1 AND effective_from <= CURDATE() "
            "ORDER BY effective_from DESC LIMIT 1",
            (model,),
        )
        return cursor.fetchone()
    finally:
        cursor.close()
        connection.close()


def list_pricing():
    return _fetch(
        "SELECT pricing_id, model_name, input_price_per_million, output_price_per_million, currency, "
        "effective_from, is_active FROM llm_pricing ORDER BY model_name, effective_from DESC LIMIT 200"
    )


def insert_pricing(model, input_price, output_price, currency, effective_from):
    """Adds a price row (history is kept; the newest effective row wins). Returns False if that
    model already has a price for that date."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO llm_pricing (model_name, input_price_per_million, output_price_per_million, "
            "currency, effective_from, is_active) VALUES (%s, %s, %s, %s, %s, 1)",
            (model, input_price, output_price, currency, effective_from),
        )
        connection.commit()
        return True
    except mysql.connector.IntegrityError as exc:
        connection.rollback()
        if getattr(exc, "errno", None) == _DUPLICATE_KEY:
            return False
        raise
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


# -------------------------------------------------------------------- usage
def insert_usage(request_id, user_id, role, server_id, tool_name, model, input_tokens, output_tokens,
                 input_cost, output_cost, total_cost, currency, status):
    """Stores one request. Returns False if that request_id was already recorded (never double-counts)."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO llm_usage (request_id, user_id, role, mcp_server_id, tool_name, model, input_tokens, "
            "output_tokens, total_tokens, input_cost, output_cost, total_cost, currency, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (request_id, user_id, role, server_id, tool_name, model, input_tokens, output_tokens,
             input_tokens + output_tokens, input_cost, output_cost, total_cost, currency, status),
        )
        connection.commit()
        return True
    except mysql.connector.IntegrityError as exc:
        connection.rollback()
        if getattr(exc, "errno", None) == _DUPLICATE_KEY:
            return False
        raise
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def usage_totals(hours):
    return _fetch(
        "SELECT COUNT(*) AS requests, COALESCE(SUM(input_tokens), 0) AS input_tokens, "
        "COALESCE(SUM(output_tokens), 0) AS output_tokens, COALESCE(SUM(total_tokens), 0) AS total_tokens, "
        "COALESCE(SUM(total_cost IS NULL), 0) AS unpriced_requests, "
        "COALESCE(SUM(status = 'FAILED'), 0) AS failed_requests, "
        "COALESCE(SUM(status = 'BLOCKED'), 0) AS blocked_requests "
        f"FROM llm_usage WHERE {_WINDOW}",
        (hours,), one=True,
    )


def cost_by_currency(hours):
    """Costs are never added across currencies."""
    return _fetch(
        f"SELECT currency, SUM(total_cost) AS total_cost FROM llm_usage WHERE {_WINDOW} "
        "AND total_cost IS NOT NULL GROUP BY currency ORDER BY currency",
        (hours,),
    )


def usage_by_model(hours):
    return _fetch(
        "SELECT model, COUNT(*) AS requests, SUM(input_tokens) AS input_tokens, "
        "SUM(output_tokens) AS output_tokens, SUM(total_tokens) AS total_tokens, "
        "SUM(total_cost) AS total_cost, MIN(currency) AS currency "
        f"FROM llm_usage WHERE {_WINDOW} GROUP BY model ORDER BY total_tokens DESC LIMIT 20",
        (hours,),
    )


def usage_by_tool(hours):
    return _fetch(
        "SELECT mcp_server_id, tool_name, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, "
        "SUM(total_cost) AS total_cost "
        f"FROM llm_usage WHERE {_WINDOW} GROUP BY mcp_server_id, tool_name "
        "ORDER BY total_tokens DESC LIMIT 20",
        (hours,),
    )


def usage_top_users(hours):
    return _fetch(
        "SELECT user_id, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(total_cost) AS total_cost "
        f"FROM llm_usage WHERE {_WINDOW} AND user_id IS NOT NULL GROUP BY user_id "
        "ORDER BY total_tokens DESC LIMIT 10",
        (hours,),
    )


def query_usage(hours, model, user_id, limit):
    where, params = [_WINDOW], [hours]
    if model:
        where.append("model = %s")
        params.append(model)
    if user_id:
        where.append("user_id = %s")
        params.append(user_id)
    params.append(limit)
    return _fetch(
        f"SELECT {_USAGE_COLUMNS} FROM llm_usage WHERE {' AND '.join(where)} ORDER BY usage_id DESC LIMIT %s",
        tuple(params),
    )
