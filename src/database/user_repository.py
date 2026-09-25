from src.database.database import get_db_connection


def create_user(
    full_name: str,
    email: str,
    phone: str,
    password_hash: str
):

    connection = get_db_connection()

    cursor = connection.cursor()

    query = """
        INSERT INTO users
        (
            full_name,
            email,
            phone,
            password_hash
        )
        VALUES (%s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (
            full_name,
            email,
            phone,
            password_hash
        )
    )

    connection.commit()

    user_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return user_id

def get_user_by_email(email: str):

    connection = get_db_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    query = """
        SELECT
            user_id,
            full_name,
            email,
            phone,
            password_hash,
            role
        FROM users
        WHERE email = %s
    """

    cursor.execute(
        query,
        (email,)
    )

    user = cursor.fetchone()

    cursor.close()
    connection.close()

    return user