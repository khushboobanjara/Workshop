"""
Database access for the doctor dashboard.

Every query that reads or changes appointments / availability is scoped by
doctor_id, so a doctor can only ever touch their own records.
"""

from datetime import date, datetime, time, timedelta

from src.database.database import get_db_connection


ACTIVE_STATUSES = ("BOOKED", "CONFIRMED")


# =========================================================
# SMALL HELPERS
# =========================================================

def _to_time(value):
    """MySQL TIME columns come back as timedelta; normalise to datetime.time."""
    if value is None:
        return None
    if isinstance(value, timedelta):
        total = int(value.total_seconds())
        return time((total // 3600) % 24, (total % 3600) // 60)
    if isinstance(value, datetime):
        return value.time()
    return value


def _time_label(value):
    t = _to_time(value)
    return t.strftime("%I:%M %p").lstrip("0") if t else ""


def _time_iso(value):
    t = _to_time(value)
    return t.strftime("%H:%M") if t else ""


def _date_label(value):
    return value.strftime("%a, %d %b %Y") if value else ""


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
    """Run a write query and return the number of affected rows."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        connection.commit()
        return cursor.rowcount
    finally:
        cursor.close()
        connection.close()


# =========================================================
# DOCTOR <-> USER LINK
# =========================================================

def get_doctor_by_email(email: str):
    """
    The users table has no doctor_id, so a login is linked to a doctors row
    by matching e-mail addresses (case-insensitive).
    """
    if not email:
        return None

    return _fetch(
        """
        SELECT
            doctor_id,
            doctor_name,
            specialization,
            experience,
            phone,
            email,
            consultation_fee,
            available
        FROM doctors
        WHERE LOWER(email) = LOWER(%s)
        LIMIT 1
        """,
        (email.strip(),),
        one=True,
    )


def get_user_doctor_status(user_id: int):
    """Value of users.doctor_status (None when empty)."""
    row = _fetch(
        "SELECT doctor_status FROM users WHERE user_id = %s",
        (user_id,),
        one=True,
    )
    return row["doctor_status"] if row else None


def set_doctor_accepting(doctor_id: int, accepting: bool):
    """Toggle doctors.available (controls whether patients can find the doctor)."""
    return _execute(
        "UPDATE doctors SET available = %s WHERE doctor_id = %s",
        (1 if accepting else 0, doctor_id),
    ) >= 0


# =========================================================
# DASHBOARD NUMBERS
# =========================================================

def get_dashboard_stats(doctor_id: int, today: date):
    month_start = today.replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    month_end = next_month - timedelta(days=1)

    row = _fetch(
        """
        SELECT
            COALESCE(SUM(
                a.appointment_date = %s AND a.status <> 'CANCELLED'
            ), 0) AS today_total,

            COALESCE(SUM(
                a.appointment_date = %s
                AND a.status IN ('BOOKED', 'CONFIRMED')
            ), 0) AS today_remaining,

            COALESCE(SUM(
                a.appointment_date > %s
                AND a.status IN ('BOOKED', 'CONFIRMED')
            ), 0) AS upcoming,

            COALESCE(SUM(
                a.appointment_date BETWEEN %s AND %s
                AND a.status = 'COMPLETED'
            ), 0) AS completed_month,

            COALESCE(SUM(CASE
                WHEN a.appointment_date BETWEEN %s AND %s
                     AND a.status <> 'CANCELLED'
                     AND a.payment_status = 'PAID'
                THEN d.consultation_fee ELSE 0 END), 0) AS earned_month,

            COALESCE(SUM(CASE
                WHEN a.status = 'COMPLETED'
                     AND COALESCE(a.payment_status, 'PENDING') <> 'PAID'
                THEN d.consultation_fee ELSE 0 END), 0) AS outstanding
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.doctor_id
        WHERE a.doctor_id = %s
        """,
        (
            today,
            today,
            today,
            month_start, month_end,
            month_start, month_end,
            doctor_id,
        ),
        one=True,
    )

    return {
        "today_total": int(row["today_total"]),
        "today_remaining": int(row["today_remaining"]),
        "upcoming": int(row["upcoming"]),
        "completed_month": int(row["completed_month"]),
        "earned_month": float(row["earned_month"]),
        "outstanding": float(row["outstanding"]),
    }


# =========================================================
# APPOINTMENTS
# =========================================================

VALID_VIEWS = ("today", "upcoming", "past", "all")
VALID_STATUSES = ("BOOKED", "CONFIRMED", "COMPLETED", "CANCELLED")


def get_doctor_appointments(
    doctor_id: int,
    today: date,
    view: str = "today",
    status: str = "",
    search: str = "",
    limit: int = 200,
):
    where = ["a.doctor_id = %s"]
    params = [doctor_id]

    if view == "today":
        where.append("a.appointment_date = %s")
        params.append(today)
    elif view == "upcoming":
        where.append("a.appointment_date > %s")
        params.append(today)
    elif view == "past":
        where.append("a.appointment_date < %s")
        params.append(today)

    if status in VALID_STATUSES:
        where.append("a.status = %s")
        params.append(status)

    if search:
        where.append("(a.patient_name LIKE %s OR a.phone LIKE %s)")
        like = f"%{search}%"
        params.extend([like, like])

    direction = "ASC" if view in ("today", "upcoming") else "DESC"

    rows = _fetch(
        f"""
        SELECT
            a.appointment_id,
            a.appointment_date,
            a.appointment_time,
            a.patient_name,
            a.phone,
            a.status,
            a.payment_method,
            a.payment_status,
            d.consultation_fee
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.doctor_id
        WHERE {' AND '.join(where)}
        ORDER BY a.appointment_date {direction},
                 a.appointment_time {direction}
        LIMIT {int(limit)}
        """,
        tuple(params),
    )

    for row in rows:
        status_value = (row["status"] or "").upper()
        method = (row["payment_method"] or "").upper()
        paid = (row["payment_status"] or "").upper() == "PAID"
        is_due = row["appointment_date"] <= today

        row["status"] = status_value
        row["payment_method"] = method
        row["payment_status"] = (row["payment_status"] or "PENDING").upper()
        row["date_label"] = _date_label(row["appointment_date"])
        row["time_label"] = _time_label(row["appointment_time"])
        row["fee"] = float(row["consultation_fee"] or 0)

        row["can_confirm"] = (
            status_value == "BOOKED" and (method == "CASH" or paid)
        )
        row["awaiting_payment"] = (
            status_value == "BOOKED" and method != "CASH" and not paid
        )
        row["can_complete"] = status_value == "CONFIRMED" and is_due
        row["can_cancel"] = status_value in ACTIVE_STATUSES
        row["can_mark_paid"] = (
            method == "CASH"
            and not paid
            and status_value in ("CONFIRMED", "COMPLETED")
        )

    return rows


def update_appointment_status(
    doctor_id: int,
    appointment_id: int,
    action: str,
    today: date,
):
    """
    Apply one doctor action to one appointment. The WHERE clause encodes
    which transitions are legal, so an illegal request affects 0 rows.
    Returns (success, message).
    """
    actions = {
        "confirm": (
            """
            UPDATE appointments SET status = 'CONFIRMED'
            WHERE appointment_id = %s AND doctor_id = %s
              AND status = 'BOOKED'
              AND (payment_method = 'CASH' OR payment_status = 'PAID')
            """,
            (appointment_id, doctor_id),
            "Appointment confirmed.",
        ),
        "complete": (
            """
            UPDATE appointments SET status = 'COMPLETED'
            WHERE appointment_id = %s AND doctor_id = %s
              AND status = 'CONFIRMED'
              AND appointment_date <= %s
            """,
            (appointment_id, doctor_id, today),
            "Appointment marked as completed.",
        ),
        "cancel": (
            """
            UPDATE appointments SET status = 'CANCELLED'
            WHERE appointment_id = %s AND doctor_id = %s
              AND status IN ('BOOKED', 'CONFIRMED')
            """,
            (appointment_id, doctor_id),
            "Appointment cancelled.",
        ),
        "mark-paid": (
            """
            UPDATE appointments SET payment_status = 'PAID'
            WHERE appointment_id = %s AND doctor_id = %s
              AND payment_method = 'CASH'
              AND COALESCE(payment_status, 'PENDING') <> 'PAID'
              AND status IN ('CONFIRMED', 'COMPLETED')
            """,
            (appointment_id, doctor_id),
            "Cash payment recorded.",
        ),
    }

    if action not in actions:
        return False, "Unknown action."

    query, params, success_message = actions[action]

    if _execute(query, params) > 0:
        return True, success_message

    return False, "That change isn't allowed for this appointment right now."


# =========================================================
# AVAILABILITY SLOTS
# =========================================================

def get_upcoming_slots(doctor_id: int, today: date):
    rows = _fetch(
        """
        SELECT
            s.availability_id,
            s.available_date,
            s.start_time,
            s.end_time,
            (
                SELECT COUNT(*)
                FROM appointments a
                WHERE a.doctor_id = s.doctor_id
                  AND a.appointment_date = s.available_date
                  AND a.appointment_time >= s.start_time
                  AND a.appointment_time < s.end_time
                  AND a.status IN ('BOOKED', 'CONFIRMED')
            ) AS booked_count
        FROM doctor_availability s
        WHERE s.doctor_id = %s
          AND s.is_available = TRUE
          AND s.available_date >= %s
        ORDER BY s.available_date, s.start_time
        LIMIT 60
        """,
        (doctor_id, today),
    )

    for row in rows:
        row["date_label"] = _date_label(row["available_date"])
        row["start_label"] = _time_label(row["start_time"])
        row["end_label"] = _time_label(row["end_time"])
        row["booked_count"] = int(row["booked_count"])

    return rows


def add_slot(
    doctor_id: int,
    slot_date: date,
    start: time,
    end: time,
    today: date,
):
    """Returns (success, message)."""
    if slot_date < today:
        return False, "Choose today or a future date."

    if start >= end:
        return False, "End time must be after start time."

    overlap = _fetch(
        """
        SELECT availability_id
        FROM doctor_availability
        WHERE doctor_id = %s
          AND available_date = %s
          AND is_available = TRUE
          AND %s < end_time
          AND %s > start_time
        LIMIT 1
        """,
        (doctor_id, slot_date, start, end),
        one=True,
    )

    if overlap:
        return False, "That overlaps a slot you already have on this date."

    _execute(
        """
        INSERT INTO doctor_availability (
            doctor_id, available_date, start_time, end_time, is_available
        )
        VALUES (%s, %s, %s, %s, TRUE)
        """,
        (doctor_id, slot_date, start, end),
    )

    return True, "Availability added."


def remove_slot(doctor_id: int, availability_id: int):
    """
    Soft-remove a slot (is_available = FALSE). Refused while active
    appointments fall inside it. Returns (success, message).
    """
    slot = _fetch(
        """
        SELECT availability_id, doctor_id, available_date, start_time, end_time
        FROM doctor_availability
        WHERE availability_id = %s AND doctor_id = %s AND is_available = TRUE
        """,
        (availability_id, doctor_id),
        one=True,
    )

    if not slot:
        return False, "Slot not found."

    booked = _fetch(
        """
        SELECT COUNT(*) AS n
        FROM appointments
        WHERE doctor_id = %s
          AND appointment_date = %s
          AND appointment_time >= %s
          AND appointment_time < %s
          AND status IN ('BOOKED', 'CONFIRMED')
        """,
        (doctor_id, slot["available_date"], slot["start_time"], slot["end_time"]),
        one=True,
    )

    if booked["n"] > 0:
        return False, (
            f"This slot has {booked['n']} active appointment(s). "
            "Cancel or complete them first."
        )

    _execute(
        """
        UPDATE doctor_availability SET is_available = FALSE
        WHERE availability_id = %s AND doctor_id = %s
        """,
        (availability_id, doctor_id),
    )

    return True, "Availability removed."