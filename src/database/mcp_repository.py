"""Repository functions added for the MCP layer (Phase 5).

Kept in its own module so the existing repositories are not modified.
Every query is parameterised and returns only the columns MCP tools may expose
(never password_hash, never Cashfree session ids).
"""
from src.database.database import get_db_connection


# =========================================================
# USER PROFILE  (MCP 01)
# =========================================================

def get_user_profile(user_id: int):
    """Safe profile columns for one user. Returns a dict or None."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT user_id, full_name, email, phone, role
            FROM users
            WHERE user_id = %s
            """,
            (user_id,),
        )
        return cursor.fetchone()
    finally:
        cursor.close()
        connection.close()


def update_user_profile(user_id: int, full_name: str, phone: str) -> None:
    """Updates ONLY full_name and phone. Email, role and password are never touched here."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            UPDATE users
            SET full_name = %s, phone = %s
            WHERE user_id = %s
            """,
            (full_name, phone, user_id),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


# =========================================================
# PHARMACY ORDERS  (MCP 03)
# =========================================================

def list_user_pharmacy_orders(user_id: int):
    """The user's own orders, newest first. No payment-gateway fields."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT order_id, total_amount, delivery_method, payment_method,
                   payment_status, order_status, created_at
            FROM pharmacy_orders
            WHERE user_id = %s
            ORDER BY created_at DESC, order_id DESC
            """,
            (user_id,),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


def cancel_pharmacy_order(order_id: int, user_id: int) -> bool:
    """Cancel one of the user's OWN orders. Returns True if it was cancelled.

    Only an order that is still PLACED and not PAID can be cancelled here (a paid order needs
    a refund, which stays a manual / admin process). CASH orders already reduced stock, so the
    stock is put back inside the same transaction.

    REQUIRES: pharmacy_orders.order_status must accept the value 'CANCELLED'.
    Check with:  SHOW COLUMNS FROM pharmacy_orders LIKE 'order_status';
    """
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT order_id, payment_method, payment_status, order_status
            FROM pharmacy_orders
            WHERE order_id = %s AND user_id = %s
            FOR UPDATE
            """,
            (order_id, user_id),
        )
        order = cursor.fetchone()
        if (
            not order
            or order["order_status"] != "PLACED"
            or order["payment_status"] == "PAID"
        ):
            connection.rollback()
            return False

        if order["payment_method"] == "CASH":
            cursor.execute(
                """
                UPDATE medicines m
                JOIN pharmacy_order_items poi ON poi.medicine_id = m.medicine_id
                SET m.stock_quantity = m.stock_quantity + poi.quantity
                WHERE poi.order_id = %s
                """,
                (order_id,),
            )

        cursor.execute(
            """
            UPDATE pharmacy_orders
            SET order_status = 'CANCELLED'
            WHERE order_id = %s AND user_id = %s
            """,
            (order_id, user_id),
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()
