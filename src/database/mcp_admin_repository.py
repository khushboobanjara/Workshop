"""Admin queries for MCP 06-08 (Phase 6). Existing repositories are not modified.

Rules followed everywhere in this file:
- every value is a bound parameter;
- table / column names in dynamic UPDATEs come ONLY from the fixed whitelists below;
- password_hash and payment-gateway columns are never selected;
- multi-step changes run in one transaction.
"""
from src.database.database import get_db_connection

DOCTOR_EDITABLE = ("doctor_name", "specialization", "experience", "phone", "consultation_fee")
MEDICINE_EDITABLE = ("medicine_name", "generic_name", "category", "description", "price",
                     "manufacturer", "prescription_required")

_USER_COLUMNS = "user_id, full_name, email, phone, role, doctor_status, email_verified"


# ------------------------------------------------------------------ helpers
def _fetch(query, params=(), one=False):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, params)
        return cursor.fetchone() if one else cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


def _execute(query, params=()):
    """Run one write and return the affected row count."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        connection.commit()
        return cursor.rowcount
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def _update_whitelisted(table, key_column, key, fields, allowed):
    columns = [c for c in allowed if c in fields]
    if not columns:
        return 0
    assignments = ", ".join(f"{c} = %s" for c in columns)
    return _execute(
        f"UPDATE {table} SET {assignments} WHERE {key_column} = %s",
        [fields[c] for c in columns] + [key],
    )


def _like(term):
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


# ================================================================== DOCTORS (MCP 06)
def list_all_doctors():
    return _fetch(
        """
        SELECT doctor_id, doctor_name, specialization, experience, phone, email,
               consultation_fee, available
        FROM doctors
        ORDER BY doctor_name
        """
    )


def doctor_email_exists(email):
    return _fetch("SELECT doctor_id FROM doctors WHERE LOWER(email) = LOWER(%s) LIMIT 1", (email,), one=True) is not None


def insert_doctor(doctor_name, specialization, experience, phone, email, consultation_fee):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO doctors
                (doctor_name, specialization, experience, phone, email, consultation_fee, available)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
            """,
            (doctor_name, specialization, experience, phone, email, consultation_fee),
        )
        connection.commit()
        return cursor.lastrowid
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def update_doctor(doctor_id, fields):
    return _update_whitelisted("doctors", "doctor_id", doctor_id, fields, DOCTOR_EDITABLE)


def set_doctor_available(doctor_id, available):
    return _execute("UPDATE doctors SET available = %s WHERE doctor_id = %s", (1 if available else 0, doctor_id))


def list_pending_doctor_applications():
    return _fetch(
        """
        SELECT user_id, full_name, email, phone, doctor_status
        FROM users
        WHERE role = 'DOCTOR' AND doctor_status = 'PENDING'
        ORDER BY user_id
        """
    )


def decide_doctor_application(user_id, new_status):
    """Only a DOCTOR account that is still PENDING can be decided. Returns True if changed."""
    return _execute(
        "UPDATE users SET doctor_status = %s WHERE user_id = %s AND role = 'DOCTOR' AND doctor_status = 'PENDING'",
        (new_status, user_id),
    ) > 0


# ================================================================== APPOINTMENTS (MCP 06)
def list_appointments(status=None, day=None, limit=100):
    where, params = [], []
    if status:
        where.append("a.status = %s")
        params.append(status)
    if day:
        where.append("a.appointment_date = %s")
        params.append(day)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    return _fetch(
        f"""
        SELECT a.appointment_id, a.user_id, a.doctor_id, d.doctor_name, a.patient_name,
               a.appointment_date, a.appointment_time, a.status, a.payment_method, a.payment_status
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.doctor_id
        {clause}
        ORDER BY a.appointment_date DESC, a.appointment_time DESC
        LIMIT %s
        """,
        params,
    )


_APPOINTMENT_ACTIONS = {
    "confirm": ("CONFIRMED", "('BOOKED')"),
    "cancel": ("CANCELLED", "('BOOKED', 'CONFIRMED')"),
    "complete": ("COMPLETED", "('CONFIRMED')"),
}


def set_appointment_status(appointment_id, action):
    """The WHERE clause encodes the legal transitions, so an illegal one changes 0 rows."""
    new_status, allowed_from = _APPOINTMENT_ACTIONS[action]
    return _execute(
        f"UPDATE appointments SET status = %s WHERE appointment_id = %s AND status IN {allowed_from}",
        (new_status, appointment_id),
    ) > 0


# ================================================================== PHARMACY (MCP 07)
def list_medicines(search=None, only_available=False, limit=200):
    where, params = [], []
    if only_available:
        where.append("is_available = TRUE")
    if search:
        where.append("(medicine_name LIKE %s OR generic_name LIKE %s OR category LIKE %s)")
        params += [_like(search)] * 3
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    return _fetch(
        f"""
        SELECT medicine_id, medicine_name, generic_name, category, description, price,
               stock_quantity, manufacturer, prescription_required, is_available
        FROM medicines {clause}
        ORDER BY medicine_name
        LIMIT %s
        """,
        params,
    )


def get_medicine(medicine_id):
    return _fetch("SELECT medicine_id, medicine_name, stock_quantity, is_available FROM medicines WHERE medicine_id = %s",
                  (medicine_id,), one=True)


def insert_medicine(medicine_name, generic_name, category, description, price, stock_quantity,
                    manufacturer, prescription_required):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO medicines
                (medicine_name, generic_name, category, description, price, stock_quantity,
                 manufacturer, prescription_required, is_available)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            """,
            (medicine_name, generic_name, category, description, price, stock_quantity,
             manufacturer, 1 if prescription_required else 0),
        )
        connection.commit()
        return cursor.lastrowid
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def update_medicine(medicine_id, fields):
    return _update_whitelisted("medicines", "medicine_id", medicine_id, fields, MEDICINE_EDITABLE)


def set_medicine_stock(medicine_id, stock_quantity):
    return _execute("UPDATE medicines SET stock_quantity = %s WHERE medicine_id = %s", (stock_quantity, medicine_id))


def set_medicine_available(medicine_id, available):
    return _execute("UPDATE medicines SET is_available = %s WHERE medicine_id = %s", (1 if available else 0, medicine_id))


def delete_medicine_if_unused(medicine_id):
    """Hard delete, only when no order ever referenced it. Returns 'DELETED' | 'NOT_FOUND' | 'IN_USE'."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT medicine_id FROM medicines WHERE medicine_id = %s FOR UPDATE", (medicine_id,))
        if not cursor.fetchone():
            connection.rollback()
            return "NOT_FOUND"
        cursor.execute("SELECT COUNT(*) AS n FROM pharmacy_order_items WHERE medicine_id = %s", (medicine_id,))
        if cursor.fetchone()["n"] > 0:
            connection.rollback()
            return "IN_USE"
        cursor.execute("DELETE FROM pharmacy_cart WHERE medicine_id = %s", (medicine_id,))
        cursor.execute("DELETE FROM medicines WHERE medicine_id = %s", (medicine_id,))
        connection.commit()
        return "DELETED"
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def list_pharmacy_orders(status=None, limit=100):
    clause, params = "", []
    if status:
        clause, params = "WHERE order_status = %s", [status]
    params.append(limit)
    return _fetch(
        f"""
        SELECT order_id, user_id, total_amount, delivery_method, payment_method,
               payment_status, order_status, created_at
        FROM pharmacy_orders {clause}
        ORDER BY created_at DESC, order_id DESC
        LIMIT %s
        """,
        params,
    )


def set_pharmacy_order_status(order_id, new_status):
    """PLACED -> CONFIRMED (cash orders only; online orders are confirmed by the payment callback)
    and PLACED -> CANCELLED (never a PAID order: that needs a refund). Cancelling a CASH order puts
    its stock back, in the same transaction. Returns 'OK' | 'NOT_FOUND' | 'NOT_ALLOWED'.

    REQUIRES pharmacy_orders.order_status to accept 'CANCELLED'."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT payment_method, payment_status, order_status FROM pharmacy_orders WHERE order_id = %s FOR UPDATE",
            (order_id,),
        )
        order = cursor.fetchone()
        if not order:
            connection.rollback()
            return "NOT_FOUND"
        if order["order_status"] != "PLACED" or order["payment_status"] == "PAID":
            connection.rollback()
            return "NOT_ALLOWED"
        if new_status == "CONFIRMED" and order["payment_method"] != "CASH":
            connection.rollback()
            return "NOT_ALLOWED"
        if new_status == "CANCELLED" and order["payment_method"] == "CASH":
            cursor.execute(
                """
                UPDATE medicines m
                JOIN pharmacy_order_items poi ON poi.medicine_id = m.medicine_id
                SET m.stock_quantity = m.stock_quantity + poi.quantity
                WHERE poi.order_id = %s
                """,
                (order_id,),
            )
        cursor.execute("UPDATE pharmacy_orders SET order_status = %s WHERE order_id = %s", (new_status, order_id))
        connection.commit()
        return "OK"
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


# ================================================================== USERS (MCP 08)
def list_users(role_filter=None, search=None, exclude_super_admin=True, limit=50):
    where, params = [], []
    if exclude_super_admin:
        where.append("role <> 'SUPER_ADMIN'")
    if role_filter:
        where.append("role = %s")
        params.append(role_filter)
    if search:
        where.append("(full_name LIKE %s OR email LIKE %s)")
        params += [_like(search)] * 2
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    return _fetch(f"SELECT {_USER_COLUMNS} FROM users {clause} ORDER BY user_id DESC LIMIT %s", params)


def get_user(user_id):
    return _fetch(f"SELECT {_USER_COLUMNS} FROM users WHERE user_id = %s", (user_id,), one=True)


def set_user_role(user_id, new_role, pending_doctor):
    """Changes users.role. A new DOCTOR with no status becomes PENDING (a doctor with no explicit
    status is refused at login by the MCP role resolver). Never touches a SUPER_ADMIN row."""
    if pending_doctor:
        return _execute(
            """
            UPDATE users
            SET role = %s, doctor_status = COALESCE(NULLIF(doctor_status, ''), 'PENDING')
            WHERE user_id = %s AND role <> 'SUPER_ADMIN'
            """,
            (new_role, user_id),
        ) >= 0
    return _execute(
        "UPDATE users SET role = %s WHERE user_id = %s AND role <> 'SUPER_ADMIN'", (new_role, user_id)
    ) >= 0
