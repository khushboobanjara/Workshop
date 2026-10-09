from src.database.database import get_db_connection


def create_user(
    full_name: str,
    email: str,
    phone: str,
    password_hash: str,
    role: str = "PATIENT"
):
    """
    Create a new patient or doctor account.

    Patients are registered directly.
    Doctors require admin approval.
    """

    role = role.upper().strip()

    if role not in ("PATIENT", "DOCTOR"):
        raise ValueError("Invalid user role.")

    doctor_status = "PENDING" if role == "DOCTOR" else None

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        query = """
            INSERT INTO users
            (
                full_name,
                email,
                phone,
                password_hash,
                role,
                doctor_status,
                email_verified
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            query,
            (
                full_name,
                email,
                phone,
                password_hash,
                role,
                doctor_status,
                1
            )
        )

        connection.commit()

        return cursor.lastrowid

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def get_user_by_email(email: str):
    """
    Retrieve a user using their email address.
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        query = """
            SELECT
                user_id,
                full_name,
                email,
                phone,
                password_hash,
                role,
                doctor_status,
                email_verified,
                created_at
            FROM users
            WHERE email = %s
        """

        cursor.execute(query, (email,))

        return cursor.fetchone()

    finally:
        cursor.close()
        connection.close()


def get_user_auth_by_id(user_id: int):
    """
    Minimal identity row for authorization (used by the MCP layer).

    Deliberately excludes password_hash and every other sensitive column.
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT user_id, email, role, doctor_status
            FROM users
            WHERE user_id = %s
            """,
            (user_id,),
        )

        return cursor.fetchone()

    finally:
        cursor.close()
        connection.close()
