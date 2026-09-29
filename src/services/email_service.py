import os
import smtplib

from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


def send_otp_email(to_email: str, otp: str) -> bool:
    """
    Send an OTP email to the user.

    Args:
        to_email: Recipient's email address.
        otp: One-time password to send.

    Returns:
        True if the email was sent successfully,
        otherwise False.
    """

    if not SMTP_EMAIL or not SMTP_PASSWORD:
        raise ValueError(
            "SMTP_EMAIL or SMTP_PASSWORD is missing in .env"
        )

    message = EmailMessage()

    message["Subject"] = "Sanjeevani Clinic - Email Verification"
    message["From"] = SMTP_EMAIL
    message["To"] = to_email

    message.set_content(
        f"""
Hello,

Thank you for registering with Sanjeevani Clinic.

Your email verification OTP is:

{otp}

This OTP is valid for 5 minutes.

Please do not share this OTP with anyone.

If you did not request this verification, please ignore this email.

Regards,
Sanjeevani Clinic
"""
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(message)

        return True

    except (smtplib.SMTPException, OSError) as error:
        print(f"Failed to send OTP email: {error}")
        return False