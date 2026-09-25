from src.database.database import get_db_connection


# =========================================================
# GET AVAILABILITY BY ID
# =========================================================

def get_availability_by_id(
    availability_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    query = """
        SELECT
            availability_id,
            doctor_id,
            available_date,
            start_time,
            end_time,
            is_available
        FROM doctor_availability
        WHERE availability_id = %s
        AND is_available = TRUE
    """

    cursor.execute(
        query,
        (availability_id,)
    )

    availability = cursor.fetchone()

    cursor.close()
    connection.close()

    return availability


# =========================================================
# CHECK TIME INSIDE AVAILABILITY
# =========================================================

def is_time_available(
    availability_id: int,
    appointment_time
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        SELECT availability_id
        FROM doctor_availability
        WHERE availability_id = %s
        AND is_available = TRUE
        AND %s >= start_time
        AND %s < end_time
    """

    cursor.execute(
        query,
        (
            availability_id,
            appointment_time,
            appointment_time
        )
    )

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    return result is not None


# =========================================================
# CHECK WHETHER SLOT IS ALREADY BOOKED
# =========================================================

def is_slot_booked(
    doctor_id: int,
    appointment_date,
    appointment_time
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        SELECT appointment_id
        FROM appointments
        WHERE doctor_id = %s
        AND appointment_date = %s
        AND appointment_time = %s
        AND status IN ('BOOKED', 'CONFIRMED')
        LIMIT 1
    """

    cursor.execute(
        query,
        (
            doctor_id,
            appointment_date,
            appointment_time
        )
    )

    appointment = cursor.fetchone()

    cursor.close()
    connection.close()

    return appointment is not None


# =========================================================
# CREATE APPOINTMENT
# =========================================================

def create_appointment(
    user_id: int,
    doctor_id: int,
    appointment_date,
    appointment_time,
    patient_name: str,
    phone: str,
    payment_method: str = "CASH"
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        INSERT INTO appointments (
            user_id,
            doctor_id,
            appointment_date,
            appointment_time,
            patient_name,
            phone,
            status,
            payment_method,
            payment_status
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            'BOOKED',
            %s,
            'PENDING'
        )
    """

    cursor.execute(
        query,
        (
            user_id,
            doctor_id,
            appointment_date,
            appointment_time,
            patient_name,
            phone,
            payment_method
        )
    )

    connection.commit()

    appointment_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return appointment_id


# =========================================================
# GET USER APPOINTMENTS
# =========================================================

def get_user_appointments(
    user_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    query = """
        SELECT
            a.appointment_id,
            a.appointment_date,
            a.appointment_time,
            a.status,
            a.payment_method,
            a.payment_status,
            a.created_at,
            d.doctor_name,
            d.specialization,
            d.consultation_fee
        FROM appointments a
        JOIN doctors d
            ON a.doctor_id = d.doctor_id
        WHERE a.user_id = %s
        ORDER BY
            a.appointment_date DESC,
            a.appointment_time DESC
    """

    cursor.execute(
        query,
        (user_id,)
    )

    appointments = cursor.fetchall()

    cursor.close()
    connection.close()

    return appointments


# =========================================================
# CANCEL APPOINTMENT
# =========================================================

def cancel_appointment(
    appointment_id: int,
    user_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        UPDATE appointments
        SET status = 'CANCELLED'
        WHERE appointment_id = %s
        AND user_id = %s
        AND status IN ('BOOKED', 'CONFIRMED')
    """

    cursor.execute(
        query,
        (
            appointment_id,
            user_id
        )
    )

    connection.commit()

    updated_rows = cursor.rowcount

    cursor.close()
    connection.close()

    return updated_rows > 0


# =========================================================
# RESCHEDULE APPOINTMENT
# =========================================================

def reschedule_appointment(
    appointment_id: int,
    user_id: int,
    appointment_date,
    appointment_time
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        UPDATE appointments
        SET
            appointment_date = %s,
            appointment_time = %s,
            status = 'BOOKED'
        WHERE appointment_id = %s
        AND user_id = %s
        AND status IN ('BOOKED', 'CONFIRMED')
    """

    cursor.execute(
        query,
        (
            appointment_date,
            appointment_time,
            appointment_id,
            user_id
        )
    )

    connection.commit()

    updated_rows = cursor.rowcount

    cursor.close()
    connection.close()

    return updated_rows > 0


# =========================================================
# GET APPOINTMENT BY ID
# =========================================================

def get_appointment_by_id(
    appointment_id: int,
    user_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    query = """
        SELECT
            a.appointment_id,
            a.user_id,
            a.doctor_id,
            a.appointment_date,
            a.appointment_time,
            a.patient_name,
            a.phone,
            a.status,
            a.payment_method,
            a.payment_status,
            d.doctor_name,
            d.specialization,
            d.consultation_fee
        FROM appointments a
        JOIN doctors d
            ON a.doctor_id = d.doctor_id
        WHERE a.appointment_id = %s
        AND a.user_id = %s
    """

    cursor.execute(
        query,
        (
            appointment_id,
            user_id
        )
    )

    appointment = cursor.fetchone()

    cursor.close()
    connection.close()

    return appointment


# =========================================================
# SAVE CASHFREE ORDER
# =========================================================

def save_cashfree_order(
    appointment_id: int,
    user_id: int,
    cashfree_order_id: str,
    payment_session_id: str
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        UPDATE appointments
        SET
            cashfree_order_id = %s,
            cashfree_payment_session_id = %s
        WHERE appointment_id = %s
        AND user_id = %s
    """

    cursor.execute(
        query,
        (
            cashfree_order_id,
            payment_session_id,
            appointment_id,
            user_id
        )
    )

    connection.commit()

    updated_rows = cursor.rowcount

    cursor.close()
    connection.close()

    return updated_rows > 0


# =========================================================
# GET APPOINTMENT FOR CASHFREE PAYMENT
# =========================================================

def get_appointment_for_cashfree_payment(
    appointment_id: int,
    user_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    query = """
        SELECT
            a.appointment_id,
            a.user_id,
            a.doctor_id,
            a.appointment_date,
            a.appointment_time,
            a.patient_name,
            a.phone,
            a.status,
            a.payment_method,
            a.payment_status,
            a.cashfree_order_id,
            a.cashfree_payment_session_id,
            d.doctor_name,
            d.specialization,
            d.consultation_fee
        FROM appointments a
        JOIN doctors d
            ON a.doctor_id = d.doctor_id
        WHERE a.appointment_id = %s
        AND a.user_id = %s
    """

    cursor.execute(
        query,
        (
            appointment_id,
            user_id
        )
    )

    appointment = cursor.fetchone()

    cursor.close()
    connection.close()

    return appointment


# =========================================================
# GET APPOINTMENT BY CASHFREE ORDER ID
# =========================================================

def get_appointment_by_cashfree_order_id(
    cashfree_order_id: str,
    user_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    query = """
        SELECT
            a.appointment_id,
            a.user_id,
            a.doctor_id,
            a.appointment_date,
            a.appointment_time,
            a.patient_name,
            a.phone,
            a.status,
            a.payment_method,
            a.payment_status,
            a.cashfree_order_id,
            a.cashfree_payment_session_id,
            d.doctor_name,
            d.specialization,
            d.consultation_fee
        FROM appointments a
        JOIN doctors d
            ON a.doctor_id = d.doctor_id
        WHERE a.cashfree_order_id = %s
        AND a.user_id = %s
        LIMIT 1
    """

    cursor.execute(
        query,
        (
            cashfree_order_id,
            user_id
        )
    )

    appointment = cursor.fetchone()

    cursor.close()
    connection.close()

    return appointment


# =========================================================
# MARK PAYMENT SUCCESS
# =========================================================

def update_payment_success(
    appointment_id: int,
    user_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        UPDATE appointments
        SET
            payment_status = 'PAID',
            status = 'CONFIRMED'
        WHERE appointment_id = %s
        AND user_id = %s
    """

    cursor.execute(
        query,
        (
            appointment_id,
            user_id
        )
    )

    connection.commit()

    updated_rows = cursor.rowcount

    cursor.close()
    connection.close()

    return updated_rows > 0


# =========================================================
# MARK PAYMENT FAILED
# =========================================================

def update_payment_failed(
    appointment_id: int,
    user_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        UPDATE appointments
        SET
            payment_status = 'FAILED'
        WHERE appointment_id = %s
        AND user_id = %s
    """

    cursor.execute(
        query,
        (
            appointment_id,
            user_id
        )
    )

    connection.commit()

    updated_rows = cursor.rowcount

    cursor.close()
    connection.close()

    return updated_rows > 0


# =========================================================
# KEEP PAYMENT PENDING
# =========================================================

def update_payment_pending(
    appointment_id: int,
    user_id: int
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        UPDATE appointments
        SET
            payment_status = 'PENDING'
        WHERE appointment_id = %s
        AND user_id = %s
    """

    cursor.execute(
        query,
        (
            appointment_id,
            user_id
        )
    )

    connection.commit()

    updated_rows = cursor.rowcount

    cursor.close()
    connection.close()

    return updated_rows > 0