from src.database.database import get_db_connection


def find_doctors_by_specialization(
    specialization: str
):
    connection = get_db_connection()

    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            doctor_id,
            doctor_name,
            specialization,
            experience,
            consultation_fee,
            available
        FROM doctors
        WHERE LOWER(specialization) = LOWER(%s)
        AND available = TRUE
    """

    cursor.execute(
        query,
        (specialization,)
    )

    doctors = cursor.fetchall()

    cursor.close()
    connection.close()

    return doctors


def get_doctor_by_id(doctor_id: int):
    connection = get_db_connection()

    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            doctor_id,
            doctor_name,
            specialization,
            experience,
            consultation_fee,
            available
        FROM doctors
        WHERE doctor_id = %s
    """

    cursor.execute(
        query,
        (doctor_id,)
    )

    doctor = cursor.fetchone()

    cursor.close()
    connection.close()

    return doctor


def get_doctor_availability(doctor_id: int):
    connection = get_db_connection()

    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            availability_id,
            doctor_id,
            available_date,
            start_time,
            end_time
        FROM doctor_availability
        WHERE doctor_id = %s
        AND is_available = TRUE
        AND available_date >= CURDATE()
        ORDER BY available_date, start_time
    """

    cursor.execute(
        query,
        (doctor_id,)
    )

    availability = cursor.fetchall()

    cursor.close()
    connection.close()

    return availability