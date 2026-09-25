from auth.password import (
    hash_password,
    verify_password
)

from src.database.user_repository import (
    create_user,
    get_user_by_email
)


def register_user(
    full_name: str,
    email: str,
    phone: str,
    password: str
):

    existing_user = get_user_by_email(email)

    if existing_user:

        return {
            "success": False,
            "message": "Email already registered."
        }

    password_hash = hash_password(password)

    user_id = create_user(
        full_name=full_name,
        email=email,
        phone=phone,
        password_hash=password_hash
    )

    return {
        "success": True,
        "user_id": user_id,
        "message": "Registration successful."
    }


def login_user(
    email: str,
    password: str
):

    user = get_user_by_email(email)

    if not user:

        return {
            "success": False,
            "message": "Invalid email or password."
        }

    password_valid = verify_password(
        password,
        user["password_hash"]
    )

    if not password_valid:

        return {
            "success": False,
            "message": "Invalid email or password."
        }

    return {
        "success": True,
        "user": {
            "user_id": user["user_id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "phone": user["phone"],
            "role": user["role"]
        }
    }