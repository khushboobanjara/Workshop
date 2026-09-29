from auth.password import hash_password, verify_password

from src.database.user_repository import (
    create_user,
    get_user_by_email
)

from src.database.otp_repository import (
    save_otp,
    get_otp_by_email,
    delete_otp
)

from src.services.email_service import send_otp_email

import secrets
from datetime import datetime


def generate_otp():
    """Generate a six-digit OTP."""
    return str(secrets.randbelow(900000) + 100000)


def login_user(email, password):
    """Authenticate a user using email and password."""

    user = get_user_by_email(email)

    if not user:
        return {
            "success": False,
            "message": "Invalid email or password.",
            "user": None
        }

    if not verify_password(password, user["password_hash"]):
        return {
            "success": False,
            "message": "Invalid email or password.",
            "user": None
        }

    return {
        "success": True,
        "message": "Login successful.",
        "user": user
    }


def request_registration_otp(
    full_name,
    email,
    phone,
    password,
    role="PATIENT"
):
    """Validate registration details and send an OTP."""

    role = role.upper().strip()

    if role not in ("PATIENT", "DOCTOR"):
        return False, "Invalid user role."

    if get_user_by_email(email):
        return False, "Email is already registered."

    otp = generate_otp()
    otp_hash = hash_password(otp)

    save_otp(email, otp_hash)

    if not send_otp_email(email, otp):
        delete_otp(email)
        return False, "Failed to send OTP. Please try again."

    return True, "OTP sent successfully."


def complete_registration(
    full_name,
    email,
    phone,
    password_hash,
    otp,
    role="PATIENT"
):
    """Verify OTP and create a patient or doctor account."""

    role = role.upper().strip()

    if role not in ("PATIENT", "DOCTOR"):
        return False, "Invalid user role."

    otp_record = get_otp_by_email(email)

    if not otp_record:
        return False, "OTP not found. Please request a new OTP."

    if otp_record["expires_at"] < datetime.now():
        delete_otp(email)
        return False, "OTP has expired. Please request a new OTP."

    if not verify_password(otp, otp_record["otp_hash"]):
        return False, "Invalid OTP."

    user_id = create_user(
        full_name=full_name,
        email=email,
        phone=phone,
        password_hash=password_hash,
        role=role
    )

    if not user_id:
        return False, "Registration failed."

    delete_otp(email)

    return True, "Registration successful."