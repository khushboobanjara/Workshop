from datetime import datetime, timedelta

from src.database.database import get_db_connection


def save_otp(email: str, otp_hash: str):
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Remove previous OTPs for this email
        cursor.execute(
            "DELETE FROM email_otps WHERE email = %s",
            (email,)
        )

        # Store the new OTP hash
        cursor.execute(
            """
            INSERT INTO email_otps
                (email, otp_hash, expires_at, is_verified)
            VALUES (%s, %s, %s, FALSE)
            """,
            (
                email,
                otp_hash,
                datetime.now() + timedelta(minutes=5)
            )
        )

        connection.commit()

    finally:
        cursor.close()
        connection.close()


def get_otp_by_email(email: str):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT otp_id, email, otp_hash, expires_at, is_verified
            FROM email_otps
            WHERE email = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email,)
        )

        return cursor.fetchone()

    finally:
        cursor.close()
        connection.close()


def delete_otp(email: str):
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            "DELETE FROM email_otps WHERE email = %s",
            (email,)
        )
        connection.commit()

    finally:
        cursor.close()
        connection.close()