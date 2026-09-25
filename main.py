import os
import sys
import uvicorn

from uuid import uuid4

from datetime import date, time

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    Form,
    Request
)

from fastapi.responses import (
    HTMLResponse,
    RedirectResponse
)

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

from starlette.middleware.sessions import SessionMiddleware


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.exception import CustomException
from src.logger import get_logger

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

from nlp_pretrained.ner_tagger import (
    get_pos_tags,
    extract_entities
)

from nlp_pretrained.embedding import (
    most_similar_words
)

from nlp_pretrained.sentiment_analyzer import (
    analyze_sentiment
)

from chatbot.service import process_message

from auth.auth_service import (
    register_user,
    login_user
)


# =========================================================
# DATABASE IMPORTS
# =========================================================

from src.database.database import get_db_connection

from src.database.doctor_repository import (
    find_doctors_by_specialization,
    get_doctor_by_id,
    get_doctor_availability
)

from src.database.appointment_repository import (
    get_availability_by_id,
    is_time_available,
    is_slot_booked,
    create_appointment,
    get_user_appointments,
    cancel_appointment,
    reschedule_appointment,
    get_appointment_by_id,
    save_cashfree_order,
    get_appointment_for_cashfree_payment,
    get_appointment_by_cashfree_order_id,
    update_payment_success,
    update_payment_failed,
    update_payment_pending
)


# =========================================================
# PHARMACY DATABASE
# =========================================================

from src.database.pharmacy_repository import (
    get_all_medicines,
    search_medicines,
    get_medicine_by_id,
    add_to_cart,
    get_user_cart,
    remove_from_cart,
    update_cart_quantity,
    clear_cart,
    create_pharmacy_order,
    get_pharmacy_order,
    save_pharmacy_cashfree_order,
    get_pharmacy_order_by_cashfree_order_id,
    update_pharmacy_payment_success,
    update_pharmacy_payment_failed
)


# =========================================================
# CASHFREE PAYMENT
# =========================================================

from src.payment.cashfree_service import (
    create_cashfree_order,
    get_cashfree_order_payments
)


# =========================================================
# LOGGER
# =========================================================

logger = get_logger(__name__)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Sanjeevani Clinic",
    description="Digital Healthcare Assistance Platform",
    version="1.0.0"
)


# =========================================================
# SESSION MIDDLEWARE
# =========================================================

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET"),
    max_age=60 * 60 * 24 * 7
)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory="templates"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_current_user(request: Request):

    user_id = request.session.get("user_id")

    if not user_id:
        return None

    return {
        "user_id": user_id,
        "full_name": request.session.get("full_name"),
        "email": request.session.get("email"),
        "phone": request.session.get("phone"),
        "role": request.session.get("role")
    }


def require_login(request: Request):

    user = get_current_user(request)

    if not user:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    return user


# =========================================================
# HOME
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    try:

        appointments = get_user_appointments(
            user["user_id"]
        )

        # Only show active/upcoming appointments
        upcoming_appointments = [
            appointment
            for appointment in appointments
            if appointment["status"] in (
                "BOOKED",
                "CONFIRMED"
            )
        ]

        # Sort by date and time
        upcoming_appointments.sort(
            key=lambda appointment: (
                appointment["appointment_date"],
                appointment["appointment_time"]
            )
        )

        # Show only first 3
        upcoming_appointments = upcoming_appointments[:3]

    except Exception as e:

        logger.exception(
            "Failed to load dashboard appointments: %s",
            e
        )

        upcoming_appointments = []

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": user,
            "upcoming_appointments": upcoming_appointments
        }
    )


# =========================================================
# LOGIN PAGE
# =========================================================

@app.get(
    "/login",
    response_class=HTMLResponse
)
async def login_page(request: Request):

    if request.session.get("user_id"):

        return RedirectResponse(
            url="/",
            status_code=303
        )

    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "error": None
        }
    )


# =========================================================
# LOGIN
# =========================================================

@app.post(
    "/login",
    response_class=HTMLResponse
)
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):

    result = login_user(
        email=email,
        password=password
    )

    if not result["success"]:

        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": result["message"]
            }
        )

    user = result["user"]

    request.session["user_id"] = user["user_id"]
    request.session["full_name"] = user["full_name"]
    request.session["email"] = user["email"]
    request.session["phone"] = user["phone"]
    request.session["role"] = user["role"]

    return RedirectResponse(
        url="/",
        status_code=303
    )


# =========================================================
# REGISTER PAGE
# =========================================================

@app.get(
    "/register",
    response_class=HTMLResponse
)
async def register_page(request: Request):

    if request.session.get("user_id"):

        return RedirectResponse(
            url="/",
            status_code=303
        )

    return templates.TemplateResponse(
        request,
        "register.html",
        {
            "error": None
        }
    )


# =========================================================
# REGISTER
# =========================================================

@app.post(
    "/register",
    response_class=HTMLResponse
)
async def register(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    password: str = Form(...)
):

    result = register_user(
        full_name=full_name,
        email=email,
        phone=phone,
        password=password
    )

    if not result["success"]:

        return templates.TemplateResponse(
            request,
            "register.html",
            {
                "error": result["message"]
            }
        )

    return RedirectResponse(
        url="/login",
        status_code=303
    )


# =========================================================
# LOGOUT
# =========================================================

@app.get("/logout")
async def logout(request: Request):

    request.session.clear()

    return RedirectResponse(
        url="/login",
        status_code=303
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "application": "Sanjeevani Clinic"
    }


# =========================================================
# SCREENING - GET
# =========================================================

@app.get(
    "/predict",
    response_class=HTMLResponse
)
async def predict_form(request: Request):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    logger.info(
        "Predict form page accessed....."
    )

    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "user": user
        }
    )


# =========================================================
# SCREENING - POST
# =========================================================

@app.post(
    "/predict",
    response_class=HTMLResponse
)
async def predict_data(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    try:

        form_data = await request.form()

        # -------------------------------------------------
        # Existing screening fields
        # -------------------------------------------------
        #
        # IMPORTANT:
        # This section reads the fields from your existing
        # predict.html dynamically so the existing screening
        # form remains compatible.
        #

        data = dict(form_data)

        logger.info(
            f"Screening request received from user "
            f"{user['user_id']}"
        )

        # -------------------------------------------------
        # Try to create CustomData using the existing
        # project pipeline.
        #
        # Your existing predict.html / CustomData mapping
        # should remain the source of the actual screening
        # fields.
        # -------------------------------------------------

        try:

            custom_data = CustomData(
                **data
            )

            final_data = custom_data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()

            prediction = predict_pipeline.predict(
                final_data
            )

            return templates.TemplateResponse(
                request,
                "result.html",
                {
                    "user": user,
                    "prediction": prediction
                }
            )

        except TypeError:

            # -------------------------------------------------
            # If the existing CustomData class does not accept
            # the exact form field names as **kwargs, redirect
            # back to the existing screening form.
            # -------------------------------------------------

            logger.warning(
                "CustomData field mapping requires the "
                "existing project-specific implementation."
            )

            return templates.TemplateResponse(
                request,
                "predict.html",
                {
                    "user": user,
                    "error": (
                        "Please check the screening form "
                        "field names with CustomData."
                    )
                }
            )

    except Exception as e:

        logger.error(
            f"Screening error: {str(e)}"
        )

        return templates.TemplateResponse(
            request,
            "predict.html",
            {
                "user": user,
                "error": "Unable to process screening."
            }
        )


# =========================================================
# PRETRAINED NLP - GET
# =========================================================

@app.get(
    "/pretrained-nlp",
    response_class=HTMLResponse
)
async def pretrained_nlp_page(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    return templates.TemplateResponse(
        request,
        "pretrained_nlp.html",
        {
            "user": user
        }
    )


# =========================================================
# PRETRAINED NLP - POST
# =========================================================

@app.post(
    "/pretrained-nlp",
    response_class=HTMLResponse
)
async def pretrained_nlp_form(
    request: Request,
    text: str = Form(...)
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    try:

        pos_tags = get_pos_tags(text)

        entities = extract_entities(text)

        similar_words = most_similar_words(text)

        sentiment = analyze_sentiment(text)

        return templates.TemplateResponse(
            request,
            "pretrained_nlp.html",
            {
                "user": user,
                "text": text,
                "pos_tags": pos_tags,
                "entities": entities,
                "similar_words": similar_words,
                "sentiment": sentiment
            }
        )

    except Exception as e:

        logger.error(
            f"NLP error: {str(e)}"
        )

        return templates.TemplateResponse(
            request,
            "pretrained_nlp.html",
            {
                "user": user,
                "text": text,
                "error": "Unable to process NLP request."
            }
        )


# =========================================================
# CHATBOT
# =========================================================

# =========================================================
# CHATBOT
# =========================================================

@app.post("/chat")
async def chat(request: Request):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    try:

        data = await request.json()

        if not isinstance(data, dict):

            return {
                "success": False,
                "response": "Invalid chatbot request."
            }

        message = str(
            data.get("message", "") or ""
        ).strip()

        if not message:

            return {
                "success": False,
                "response": "Please enter a message."
            }

        # Pass the authenticated user into the chatbot.
        # The chatbot can then keep booking state tied to the
        # correct logged-in patient.
        response = await process_message(
            message,
            conversation_history=[],
            user=user
        )

        # service.py can return structured data such as:
        # response + type + options. Return it unchanged so the
        # frontend can render doctor/date/time/medicine options.
        if isinstance(response, dict):

            return response

        return {
            "success": True,
            "response": str(response)
        }

    except Exception as e:

        logger.exception(
            f"Chatbot error: {str(e)}"
        )

        return {
            "success": False,
            "response": (
                "Sorry, I am unable to process "
                "your request right now."
            )
        }

# =========================================================
# DOCTORS
# =========================================================

@app.get(
    "/doctors",
    response_class=HTMLResponse
)
async def doctors_page(
    request: Request,
    specialization: str = ""
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    doctors = []

    if specialization.strip():

        doctors = find_doctors_by_specialization(
            specialization.strip()
        )

    return templates.TemplateResponse(
        request,
        "doctors.html",
        {
            "user": user,
            "doctors": doctors,
            "specialization": specialization
        }
    )


# =========================================================
# DOCTOR DETAILS
# =========================================================

@app.get(
    "/doctors/{doctor_id}",
    response_class=HTMLResponse
)
async def doctor_details(
    request: Request,
    doctor_id: int
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    doctor = get_doctor_by_id(
        doctor_id
    )

    if not doctor:

        return HTMLResponse(
            content="Doctor not found",
            status_code=404
        )

    availability = get_doctor_availability(
        doctor_id
    )

    return templates.TemplateResponse(
        request,
        "doctor_details.html",
        {
            "user": user,
            "doctor": doctor,
            "availability": availability
        }
    )


# =========================================================
# BOOK APPOINTMENT - GET
# =========================================================

@app.get(
    "/appointments/book",
    response_class=HTMLResponse
)
async def book_appointment_page(
    request: Request,
    doctor_id: int,
    availability_id: int
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    doctor = get_doctor_by_id(
        doctor_id
    )

    if not doctor:

        return HTMLResponse(
            content="Doctor not found.",
            status_code=404
        )

    availability = get_availability_by_id(
        availability_id
    )

    if not availability:

        return HTMLResponse(
            content="Availability slot not found.",
            status_code=404
        )

    if availability["doctor_id"] != doctor_id:

        return HTMLResponse(
            content="Invalid doctor availability.",
            status_code=400
        )

    return templates.TemplateResponse(
        request,
        "book_appointment.html",
        {
            "user": user,
            "doctor": doctor,
            "availability": availability,
            "error": None
        }
    )


# =========================================================
# BOOK APPOINTMENT - POST
# =========================================================

@app.post(
    "/appointments/book",
    response_class=HTMLResponse
)
async def book_appointment(
    request: Request,

    doctor_id: int = Form(...),

    availability_id: int = Form(...),

    appointment_date: str = Form(...),

    appointment_time: str = Form(...),

    patient_name: str = Form(...),

    phone: str = Form(...),

    payment_method: str = Form(...)
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    # -----------------------------------------------------
    # Payment method validation
    # -----------------------------------------------------

    payment_method = payment_method.upper().strip()

    if payment_method not in (
        "CASH",
        "ONLINE"
    ):

        return HTMLResponse(
            content="Invalid payment method.",
            status_code=400
        )

    # -----------------------------------------------------
    # Doctor validation
    # -----------------------------------------------------

    doctor = get_doctor_by_id(
        doctor_id
    )

    if not doctor:

        return HTMLResponse(
            content="Doctor not found.",
            status_code=404
        )

    # -----------------------------------------------------
    # Availability validation
    # -----------------------------------------------------

    availability = get_availability_by_id(
        availability_id
    )

    if not availability:

        return HTMLResponse(
            content="Selected availability slot is not available.",
            status_code=400
        )

    if availability["doctor_id"] != doctor_id:

        return HTMLResponse(
            content="Invalid doctor availability.",
            status_code=400
        )

    # -----------------------------------------------------
    # Date validation
    # -----------------------------------------------------

    try:

        selected_date = date.fromisoformat(
            appointment_date
        )

    except ValueError:

        return HTMLResponse(
            content="Invalid appointment date.",
            status_code=400
        )

    availability_date = availability[
        "available_date"
    ]

    if selected_date != availability_date:

        return HTMLResponse(
            content=(
                "Selected date does not match "
                "the doctor's availability."
            ),
            status_code=400
        )

    # -----------------------------------------------------
    # Time validation
    # -----------------------------------------------------

    try:

        selected_time = time.fromisoformat(
            appointment_time
        )

    except ValueError:

        return HTMLResponse(
            content="Invalid appointment time.",
            status_code=400
        )

    if not is_time_available(
        availability_id,
        selected_time
    ):

        return HTMLResponse(
            content=(
                "Selected time is outside "
                "the doctor's availability."
            ),
            status_code=400
        )

    # -----------------------------------------------------
    # Check duplicate booking
    # -----------------------------------------------------

    if is_slot_booked(
        doctor_id,
        selected_date,
        selected_time
    ):

        return HTMLResponse(
            content=(
                "This appointment slot has "
                "already been booked."
            ),
            status_code=409
        )

    # -----------------------------------------------------
    # Patient validation
    # -----------------------------------------------------

    patient_name = patient_name.strip()
    phone = phone.strip()

    if not patient_name:

        return HTMLResponse(
            content="Patient name is required.",
            status_code=400
        )

    if not phone:

        return HTMLResponse(
            content="Phone number is required.",
            status_code=400
        )

    # -----------------------------------------------------
    # Create appointment
    # -----------------------------------------------------

    appointment_id = create_appointment(
        user_id=user["user_id"],
        doctor_id=doctor_id,
        appointment_date=selected_date,
        appointment_time=selected_time,
        patient_name=patient_name,
        phone=phone,
        payment_method=payment_method
    )

    logger.info(
        f"Appointment {appointment_id} created "
        f"for user {user['user_id']}"
    )

    # =====================================================
    # CASH PAYMENT
    # =====================================================

    if payment_method == "CASH":

        return templates.TemplateResponse(
            request,
            "appointment_success.html",
            {
                "user": user,
                "appointment_id": appointment_id,
                "doctor": doctor,
                "appointment_date": selected_date,
                "appointment_time": selected_time,
                "patient_name": patient_name,
                "phone": phone,
                "payment_method": payment_method,
                "payment_status": "PENDING"
            }
        )

    # =====================================================
    # CASHFREE ONLINE PAYMENT
    # =====================================================

    try:

        # -------------------------------------------------
        # Cashfree order ID
        # -------------------------------------------------

        cashfree_order_id = (
            f"appointment_{appointment_id}"
        )

        # -------------------------------------------------
        # CashFree Return URL
        # -------------------------------------------------

        return_url = (
        "http://127.0.0.1:8000"
        "/payments/cashfree/return"
        f"?order_id={cashfree_order_id}"
        )

        # -------------------------------------------------
        # Create Cashfree order
        # -------------------------------------------------

        cashfree_order = create_cashfree_order(
            order_id=cashfree_order_id,

            amount=float(
                doctor["consultation_fee"]
            ),

            customer_id=str(
                user["user_id"]
            ),

            customer_name=patient_name,

            customer_email=user["email"],

            customer_phone=phone,

            return_url=return_url
        )

        # -------------------------------------------------
        # Save Cashfree order information
        # -------------------------------------------------

        save_cashfree_order(
            appointment_id=appointment_id,

            user_id=user["user_id"],

            cashfree_order_id=(
                cashfree_order["order_id"]
            ),

            payment_session_id=(
                cashfree_order[
                    "payment_session_id"
                ]
            )
        )

        logger.info(
            f"Cashfree order created: "
            f"{cashfree_order['order_id']}"
        )

        # -------------------------------------------------
        # Open Cashfree checkout
        # -------------------------------------------------

        return templates.TemplateResponse(
            request,
            "cashfree_checkout.html",
            {
                "user": user,
                "doctor": doctor,

                "appointment_id":
                    appointment_id,

                "appointment_date":
                    selected_date,

                "appointment_time":
                    selected_time,

                "patient_name":
                    patient_name,

                "phone":
                    phone,

                "payment_amount":
                    doctor["consultation_fee"],

                "payment_session_id":
                    cashfree_order[
                        "payment_session_id"
                    ],

                "cashfree_order_id":
                    cashfree_order[
                        "order_id"
                    ]
            }
        )

    except Exception as e:

        logger.exception(
        "Cashfree appointment payment initialization failed"
        )

        return HTMLResponse(
            content=(
                "Appointment was created, but online payment "
                "could not be initialized. Check the terminal "
                "for the exact Cashfree error."
            ),
            status_code=500
    )


# =========================================================
# CASHFREE RETURN URL
# =========================================================

@app.get(
    "/payments/cashfree/return",
    response_class=HTMLResponse
)
async def cashfree_return(
    request: Request,
    order_id: str = ""
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    if not order_id:
        return HTMLResponse(
            content="Cashfree order ID missing.",
            status_code=400
        )

    logger.info(
        f"Cashfree returned order: {order_id}"
    )

    # -----------------------------------------------------
    # Find the appointment belonging to this Cashfree order
    # and the currently logged-in user.
    # -----------------------------------------------------

    appointment = get_appointment_by_cashfree_order_id(
        cashfree_order_id=order_id,
        user_id=user["user_id"]
    )

    if not appointment:
        logger.error(
            f"No appointment found for Cashfree order: "
            f"{order_id}"
        )

        return HTMLResponse(
            content=(
                "Appointment associated with this "
                "payment was not found."
            ),
            status_code=404
        )

    appointment_id = appointment["appointment_id"]

    logger.info(
        f"Appointment found: {appointment_id}"
    )

    # -----------------------------------------------------
    # Ask Cashfree for the actual payment transactions.
    # Do NOT mark the appointment PAID just because the
    # browser returned to this URL.
    # -----------------------------------------------------

    try:
        payments = get_cashfree_order_payments(
            order_id
        )

    except Exception:
        logger.exception(
            "Unable to verify Cashfree payment."
        )

        return templates.TemplateResponse(
            request,
            "cashfree_payment_result.html",
            {
                "user": user,
                "appointment": appointment,
                "payment_status": "PENDING",
                "message": (
                    "We could not verify the payment "
                    "right now. Please check your "
                    "profile again shortly."
                )
            }
        )

    logger.info(
        f"Cashfree payments received for order "
        f"{order_id}: {payments}"
    )

    # -----------------------------------------------------
    # Cashfree's Python SDK returns PaymentEntity objects.
    # This helper also supports dictionary responses.
    # -----------------------------------------------------

    def extract_payment_status(payment):

        if isinstance(payment, dict):
            return str(
                payment.get(
                    "payment_status",
                    ""
                )
            ).upper()

        return str(
            getattr(
                payment,
                "payment_status",
                ""
            )
        ).upper()

    final_status = "PENDING"

    if payments:

        # SUCCESS has priority.
        for payment in payments:

            status = extract_payment_status(
                payment
            )

            logger.info(
                f"Cashfree payment status: {status}"
            )

            if status == "SUCCESS":
                final_status = "PAID"
                break

        # If no successful transaction exists,
        # check whether the payment failed.
        if final_status != "PAID":

            for payment in payments:

                status = extract_payment_status(
                    payment
                )

                if status in (
                    "FAILED",
                    "FAILURE",
                    "USER_DROPPED"
                ):
                    final_status = "FAILED"
                    break

    # -----------------------------------------------------
    # Update MySQL according to the verified Cashfree
    # payment status.
    # -----------------------------------------------------

    if final_status == "PAID":

        update_payment_success(
            appointment_id=appointment_id,
            user_id=user["user_id"]
        )

        message = (
            "Payment successful. "
            "Your appointment is confirmed."
        )

    elif final_status == "FAILED":

        update_payment_failed(
            appointment_id=appointment_id,
            user_id=user["user_id"]
        )

        message = (
            "Payment was not successful. "
            "Your payment status is marked as failed."
        )

    else:

        update_payment_pending(
            appointment_id=appointment_id,
            user_id=user["user_id"]
        )

        message = (
            "Payment is still being processed. "
            "Please check your appointment status "
            "again shortly."
        )

    # -----------------------------------------------------
    # Fetch the updated appointment.
    # -----------------------------------------------------

    appointment = get_appointment_by_cashfree_order_id(
        cashfree_order_id=order_id,
        user_id=user["user_id"]
    )

    # -----------------------------------------------------
    # Show final payment result.
    # -----------------------------------------------------

    return templates.TemplateResponse(
        request,
        "cashfree_payment_result.html",
        {
            "user": user,
            "appointment": appointment,
            "payment_status": final_status,
            "message": message
        }
    )


# =========================================================
# PROFILE - PHARMACY ORDERS HELPER
# =========================================================

def get_user_pharmacy_orders(user_id):

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT
                order_id,
                total_amount,
                delivery_method,
                address,
                phone,
                payment_method,
                payment_status,
                order_status,
                created_at
            FROM pharmacy_orders
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))

        return cursor.fetchall()

    finally:
        cursor.close()
        connection.close()


# =========================================================
# PROFILE
# =========================================================

@app.get(
    "/profile",
    response_class=HTMLResponse
)
async def profile_page(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    appointments = get_user_appointments(
        user["user_id"]
    )

    pharmacy_orders = get_user_pharmacy_orders(
        user["user_id"]
    )

    return templates.TemplateResponse(
        request,
        "profile.html",
        {
            "user": user,
            "appointments": appointments,
            "pharmacy_orders": pharmacy_orders
        }
    )


# =========================================================
# CANCEL APPOINTMENT
# =========================================================

@app.post(
    "/appointments/{appointment_id}/cancel"
)
async def cancel_user_appointment(
    request: Request,
    appointment_id: int
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    success = cancel_appointment(
        appointment_id=appointment_id,
        user_id=user["user_id"]
    )

    return RedirectResponse(
        url="/profile",
        status_code=303
    )


# =========================================================
# RESCHEDULE PAGE
# =========================================================

@app.get(
    "/appointments/{appointment_id}/reschedule",
    response_class=HTMLResponse
)
async def reschedule_page(
    request: Request,
    appointment_id: int
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    appointment = get_appointment_by_id(
        appointment_id,
        user["user_id"]
    )

    if not appointment:

        return HTMLResponse(
            content="Appointment not found.",
            status_code=404
        )

    if appointment["status"] not in (
        "BOOKED",
        "CONFIRMED"
    ):

        return HTMLResponse(
            content=(
                "This appointment cannot "
                "be rescheduled."
            ),
            status_code=400
        )

    availability = get_doctor_availability(
        appointment["doctor_id"]
    )

    return templates.TemplateResponse(
        request,
        "reschedule_appointment.html",
        {
            "user": user,
            "appointment": appointment,
            "availability": availability,
            "error": None
        }
    )


# =========================================================
# RESCHEDULE APPOINTMENT
# =========================================================

@app.post(
    "/appointments/{appointment_id}/reschedule",
    response_class=HTMLResponse
)
async def reschedule_user_appointment(
    request: Request,

    appointment_id: int,

    appointment_date: str = Form(...),

    appointment_time: str = Form(...)
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    # -----------------------------------------------------
    # Get existing appointment
    # -----------------------------------------------------

    appointment = get_appointment_by_id(
        appointment_id,
        user["user_id"]
    )

    if not appointment:

        return HTMLResponse(
            content="Appointment not found.",
            status_code=404
        )

    if appointment["status"] not in (
        "BOOKED",
        "CONFIRMED"
    ):

        return HTMLResponse(
            content=(
                "This appointment cannot "
                "be rescheduled."
            ),
            status_code=400
        )

    # -----------------------------------------------------
    # Parse date
    # -----------------------------------------------------

    try:

        selected_date = date.fromisoformat(
            appointment_date
        )

    except ValueError:

        return HTMLResponse(
            content="Invalid appointment date.",
            status_code=400
        )

    # -----------------------------------------------------
    # Parse time
    # -----------------------------------------------------

    try:

        selected_time = time.fromisoformat(
            appointment_time
        )

    except ValueError:

        return HTMLResponse(
            content="Invalid appointment time.",
            status_code=400
        )

    # -----------------------------------------------------
    # Get doctor's availability
    # -----------------------------------------------------

    availability = get_doctor_availability(
        appointment["doctor_id"]
    )

    valid_slot = False

    for slot in availability:

        if slot["available_date"] != selected_date:
            continue

        if (
            selected_time >= slot["start_time"]
            and
            selected_time < slot["end_time"]
        ):

            valid_slot = True
            break

    if not valid_slot:

        return HTMLResponse(
            content=(
                "The selected date and time "
                "are not available for this doctor."
            ),
            status_code=400
        )

    # -----------------------------------------------------
    # Check duplicate booking
    # -----------------------------------------------------

    if is_slot_booked(
        appointment["doctor_id"],
        selected_date,
        selected_time
    ):

        return HTMLResponse(
            content=(
                "This appointment slot is "
                "already booked."
            ),
            status_code=409
        )

    # -----------------------------------------------------
    # Update appointment
    # -----------------------------------------------------

    success = reschedule_appointment(
        appointment_id=appointment_id,
        user_id=user["user_id"],
        appointment_date=selected_date,
        appointment_time=selected_time
    )

    if not success:

        return HTMLResponse(
            content=(
                "Unable to reschedule "
                "the appointment."
            ),
            status_code=500
        )

    return RedirectResponse(
        url="/profile",
        status_code=303
    )


# =========================================================
# PHARMACY
# =========================================================
@app.get(
    "/pharmacy",
    response_class=HTMLResponse
)
async def pharmacy_page(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    medicines = get_all_medicines()

    return templates.TemplateResponse(
        request,
        "pharmacy.html",
        {
            "user": user,
            "medicines": medicines,
            "search_term": ""
        }
    )


@app.get(
    "/pharmacy/search",
    response_class=HTMLResponse
)
async def pharmacy_search(
    request: Request,
    q: str = ""
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    q = q.strip()

    if not q:
        return RedirectResponse(
            url="/pharmacy",
            status_code=303
        )

    medicines = search_medicines(q)

    return templates.TemplateResponse(
        request,
        "pharmacy.html",
        {
            "user": user,
            "medicines": medicines,
            "search_term": q
        }
    )


@app.get(
    "/pharmacy/medicine/{medicine_id}",
    response_class=HTMLResponse
)
async def medicine_details(
    request: Request,
    medicine_id: int
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    medicine = get_medicine_by_id(medicine_id)

    if not medicine:
        return HTMLResponse(
            content="Medicine not found",
            status_code=404
        )

    return templates.TemplateResponse(
        request,
        "medicine_details.html",
        {
            "user": user,
            "medicine": medicine
        }
    )


# =========================================================
# PHARMACY CART
# =========================================================

@app.post("/pharmacy/cart/add")
async def pharmacy_add_to_cart(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    form = await request.form()

    try:
        medicine_id = int(form.get("medicine_id"))
        quantity = int(form.get("quantity", 1))
    except (TypeError, ValueError):
        return HTMLResponse(
            content="Invalid medicine or quantity.",
            status_code=400
        )

    if quantity <= 0:
        quantity = 1

    medicine = get_medicine_by_id(medicine_id)

    if not medicine:
        return HTMLResponse(
            content="Medicine not found",
            status_code=404
        )

    stock = int(medicine["stock_quantity"])

    if stock <= 0:
        return HTMLResponse(
            content="Medicine is out of stock",
            status_code=400
        )

    # Validate requested quantity against current stock.
    # The repository increments an existing cart quantity.
    existing_cart = get_user_cart(user["user_id"])
    existing_quantity = 0

    for item in existing_cart:
        if int(item["medicine_id"]) == medicine_id:
            existing_quantity = int(item["quantity"])
            break

    if existing_quantity + quantity > stock:
        return HTMLResponse(
            content=(
                f"Only {stock} units are available. "
                f"You already have {existing_quantity} "
                "in your cart."
            ),
            status_code=400
        )

    add_to_cart(
        user["user_id"],
        medicine_id,
        quantity
    )

    return RedirectResponse(
        url="/cart",
        status_code=303
    )


@app.get(
    "/cart",
    response_class=HTMLResponse
)
async def cart_page(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    cart_items = get_user_cart(
        user["user_id"]
    )

    total_amount = sum(
        float(item["item_total"])
        for item in cart_items
    )

    return templates.TemplateResponse(
        request,
        "cart.html",
        {
            "user": user,
            "cart_items": cart_items,
            "total_amount": total_amount
        }
    )


@app.post("/pharmacy/cart/update")
async def pharmacy_update_cart(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    form = await request.form()

    try:
        medicine_id = int(form.get("medicine_id"))
        quantity = int(form.get("quantity"))
    except (TypeError, ValueError):
        return HTMLResponse(
            content="Invalid medicine or quantity.",
            status_code=400
        )

    medicine = get_medicine_by_id(medicine_id)

    if not medicine:
        return HTMLResponse(
            content="Medicine not found",
            status_code=404
        )

    stock = int(medicine["stock_quantity"])

    if quantity > stock:
        return HTMLResponse(
            content=f"Only {stock} units are available.",
            status_code=400
        )

    update_cart_quantity(
        user["user_id"],
        medicine_id,
        quantity
    )

    return RedirectResponse(
        url="/cart",
        status_code=303
    )


@app.post("/pharmacy/cart/remove")
async def pharmacy_remove_from_cart(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    form = await request.form()

    try:
        medicine_id = int(form.get("medicine_id"))
    except (TypeError, ValueError):
        return HTMLResponse(
            content="Invalid medicine.",
            status_code=400
        )

    remove_from_cart(
        user["user_id"],
        medicine_id
    )

    return RedirectResponse(
        url="/cart",
        status_code=303
    )


@app.post("/pharmacy/cart/clear")
async def pharmacy_clear_cart(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    clear_cart(
        user["user_id"]
    )

    return RedirectResponse(
        url="/cart",
        status_code=303
    )


# =========================================================
# PHARMACY CHECKOUT - GET
# =========================================================

@app.get(
    "/pharmacy/checkout",
    response_class=HTMLResponse
)
async def pharmacy_checkout(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    cart_items = get_user_cart(
        user["user_id"]
    )

    if not cart_items:
        return RedirectResponse(
            url="/cart",
            status_code=303
        )

    total_amount = sum(
        float(item["item_total"])
        for item in cart_items
    )

    return templates.TemplateResponse(
        request,
        "pharmacy_checkout.html",
        {
            "user": user,
            "cart_items": cart_items,
            "total_amount": total_amount
        }
    )


# =========================================================
# PHARMACY CHECKOUT - PLACE ORDER
# =========================================================

@app.post("/pharmacy/checkout")
async def pharmacy_place_order(
    request: Request
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    form = await request.form()

    patient_name = (form.get("patient_name") or "").strip()
    phone = (form.get("phone") or "").strip()
    delivery_method = (form.get("delivery_method") or "").strip().upper()
    address = (form.get("delivery_address") or form.get("address") or "").strip()
    payment_method = (form.get("payment_method") or "").strip().upper()

    if not patient_name:
        return HTMLResponse(
            "Patient name is required.",
            status_code=400
        )

    if not phone:
        return HTMLResponse(
            "Phone number is required.",
            status_code=400
        )

    try:
        order = create_pharmacy_order(
            user_id=user["user_id"],
            patient_name=patient_name,
            phone=phone,
            delivery_method=delivery_method,
            address=address,
            payment_method=payment_method
        )
    except ValueError as e:
        return HTMLResponse(
            str(e),
            status_code=400
        )
    except Exception as e:
        logger.exception(
            f"Pharmacy order error: {e}"
        )
        return HTMLResponse(
            "Unable to place pharmacy order.",
            status_code=500
        )

    order_id = order["order_id"]
    total_amount = float(order["total_amount"])

    # -----------------------------------------------------
    # CASH PAYMENT
    # -----------------------------------------------------
    if payment_method == "CASH":
        return RedirectResponse(
            url=f"/pharmacy/order/{order_id}",
            status_code=303
        )

    # -----------------------------------------------------
    # ONLINE PAYMENT
    # -----------------------------------------------------
    if payment_method == "ONLINE":

        # Cashfree order IDs must be unique for every payment attempt.
        # This also prevents a previous failed/retried attempt from
        # reusing an already-created Cashfree order.
        cashfree_order_id = (
            f"pharmacy_{order_id}_{uuid4().hex[:10]}"
        )

        return_url = (
            "http://127.0.0.1:8000"
            "/pharmacy/payment/return"
            f"?order_id={order_id}"
        )

        try:
            cashfree_order = create_cashfree_order(
                order_id=cashfree_order_id,
                amount=total_amount,
                customer_id=str(user["user_id"]),
                customer_name=patient_name,
                customer_email=user["email"],
                customer_phone=phone,
                return_url=return_url
            )

            logger.info(
                "Pharmacy Cashfree order created: %s",
                cashfree_order["order_id"]
            )

        except Exception as e:
            logger.exception(
                "Cashfree pharmacy order creation failed: %s",
                e
            )
            return HTMLResponse(
                content=(
                    "Unable to create Cashfree payment order. "
                    "Check the server terminal for the exact error."
                ),
                status_code=500
            )

        # Save the Cashfree identifiers separately so a database
        # error is not confused with a Cashfree API error.
        try:
            save_pharmacy_cashfree_order(
                order_id=order_id,
                cashfree_order_id=cashfree_order["order_id"],
                payment_session_id=cashfree_order["payment_session_id"]
            )

        except Exception as e:
            logger.exception(
                "Cashfree order was created but could not be saved "
                "to pharmacy_orders: %s",
                e
            )
            return HTMLResponse(
                content=(
                    "Cashfree order was created, but the payment "
                    "information could not be saved. Check the server terminal."
                ),
                status_code=500
            )

        return templates.TemplateResponse(
            request,
            "pharmacy_payment.html",
            {
                "user": user,
                "order": order,
                "payment_session_id": cashfree_order["payment_session_id"]
            }
        )

    return HTMLResponse(
        "Invalid payment method.",
        status_code=400
    )


# =========================================================
# PHARMACY ORDER DETAILS
# =========================================================

@app.get(
    "/pharmacy/order/{order_id}",
    response_class=HTMLResponse
)
async def pharmacy_order_details(
    request: Request,
    order_id: int
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    order = get_pharmacy_order(
        order_id,
        user["user_id"]
    )

    if not order:
        return HTMLResponse(
            "Order not found.",
            status_code=404
        )

    return templates.TemplateResponse(
        request,
        "pharmacy_order.html",
        {
            "user": user,
            "order": order
        }
    )


# =========================================================
# PHARMACY CASHFREE RETURN / VERIFICATION
# =========================================================

@app.get(
    "/pharmacy/payment/return",
    response_class=HTMLResponse
)
async def pharmacy_cashfree_return(
    request: Request,
    order_id: int
):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    order = get_pharmacy_order(
        order_id,
        user["user_id"]
    )

    if not order:
        return HTMLResponse(
            "Pharmacy order not found.",
            status_code=404
        )

    cashfree_order_id = order.get("cashfree_order_id")

    if not cashfree_order_id:
        return HTMLResponse(
            "Cashfree order ID is missing.",
            status_code=400
        )

    try:
        payments = get_cashfree_order_payments(
            cashfree_order_id
        )
    except Exception:
        logger.exception(
            "Unable to verify pharmacy Cashfree payment."
        )

        return templates.TemplateResponse(
            request,
            "pharmacy_order.html",
            {
                "user": user,
                "order": order,
                "payment_message": (
                    "Payment could not be verified right now. "
                    "Please check your order again shortly."
                )
            }
        )

    def extract_payment_status(payment):
        if isinstance(payment, dict):
            return str(
                payment.get("payment_status", "")
            ).upper()

        return str(
            getattr(payment, "payment_status", "")
        ).upper()

    final_status = "PENDING"

    if payments:
        for payment in payments:
            status = extract_payment_status(payment)
            if status == "SUCCESS":
                final_status = "PAID"
                break

        if final_status != "PAID":
            for payment in payments:
                status = extract_payment_status(payment)
                if status in (
                    "FAILED",
                    "FAILURE",
                    "USER_DROPPED"
                ):
                    final_status = "FAILED"
                    break

    logger.info(
        "Pharmacy Cashfree order %s payment status: %s",
        cashfree_order_id,
        final_status
    )

    if final_status == "PAID":
        update_pharmacy_payment_success(
            order_id=order_id,
            user_id=user["user_id"]
        )
        message = "Payment successful. Your pharmacy order is confirmed."

    elif final_status == "FAILED":
        update_pharmacy_payment_failed(
            order_id=order_id,
            user_id=user["user_id"]
        )
        message = "Payment failed. Please try again or choose another payment method."

    else:
        message = "Payment is still being processed. Please check your order again shortly."

    order = get_pharmacy_order(
        order_id,
        user["user_id"]
    )

    return templates.TemplateResponse(
        request,
        "pharmacy_order.html",
        {
            "user": user,
            "order": order,
            "payment_message": message
        }
    )


# =========================================================
# APPLICATION START
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

