import os
import json
import sys
import base64
import secrets
from pathlib import Path
import uvicorn
from uuid import uuid4

from datetime import date, time

from dotenv import load_dotenv
import httpx

from fastapi import (
    FastAPI,
    Form,
    Request,
    UploadFile,
    File,
    HTTPException
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

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


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


from auth.auth_service import (
    login_user,
    request_registration_otp,
    complete_registration
)

from auth.password import hash_password


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
    update_pharmacy_payment_failed,
    get_nearby_pharmacies
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

# Keep this value stable across restarts. Set SESSION_SECRET in .env.
SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError(
        "SESSION_SECRET is missing. Add a long random value to your .env file."
    )

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=60 * 60 * 24 * 7,
    same_site="lax",
    https_only=False  # Set to True when deployed behind HTTPS.
)

# Doctor dashboard routes (/doctor/...)
from doctor_dashboard import router as doctor_dashboard_router
app.include_router(doctor_dashboard_router)


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
    """Return the identity stored in the authenticated session.

    Never accept a user ID from a query parameter, form, or URL for
    ownership checks. Normalize the session value before passing it to
    database repository functions.
    """
    raw_user_id = request.session.get("user_id")

    if raw_user_id is None:
        return None

    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        request.session.clear()
        return None

    if user_id <= 0:
        request.session.clear()
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

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    if (user.get("role") or "").upper() == "DOCTOR":
        return RedirectResponse(
            url="/doctor/dashboard",
            status_code=303
        )

    logger.info("Home page accessed...")

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": user
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

    # Prevent stale session fields from a previous account being retained.
    request.session.clear()
    request.session["user_id"] = user["user_id"]
    request.session["full_name"] = user["full_name"]
    request.session["email"] = user["email"]
    request.session["phone"] = user["phone"]
    request.session["role"] = user["role"]

    if (user["role"] or "").upper() == "DOCTOR":
        return RedirectResponse(
            url="/doctor/dashboard",
            status_code=303
        )

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
# REGISTER - SEND EMAIL OTP
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
    password: str = Form(...),
    role: str = Form("PATIENT")
):
    if request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=303)

    full_name = full_name.strip()
    email = email.strip().lower()
    phone = phone.strip()
    role = role.strip().upper()

    if role not in ("PATIENT", "DOCTOR"):
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "Please select a valid account type."},
            status_code=400
        )

    if not full_name or not email or not phone or not password:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "All fields are required."},
            status_code=400
        )

    success, message = request_registration_otp(
        full_name=full_name,
        email=email,
        phone=phone,
        password=password,
        role=role
    )

    if not success:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": message},
            status_code=400
        )

    # Starlette's default session is a signed client-side cookie.
    # Store only the password hash, never the plain-text password.
    request.session["pending_registration"] = {
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "password_hash": hash_password(password),
        "role": role
    }

    return RedirectResponse(
        url="/verify-otp",
        status_code=303
    )


# =========================================================
# VERIFY REGISTRATION OTP
# =========================================================

@app.get(
    "/verify-otp",
    response_class=HTMLResponse
)
async def verify_otp_page(request: Request):
    pending = request.session.get("pending_registration")

    if not pending:
        return RedirectResponse(url="/register", status_code=303)

    return templates.TemplateResponse(
        request,
        "verify_otp.html",
        {
            "email": pending["email"],
            "error": None
        }
    )


@app.post(
    "/verify-otp",
    response_class=HTMLResponse
)
async def verify_otp(
    request: Request,
    otp: str = Form(...)
):
    pending = request.session.get("pending_registration")

    if not pending:
        return RedirectResponse(url="/register", status_code=303)

    otp = otp.strip()
    if len(otp) != 6 or not otp.isdigit():
        return templates.TemplateResponse(
            request,
            "verify_otp.html",
            {
                "email": pending["email"],
                "error": "Enter a valid six-digit OTP."
            },
            status_code=400
        )

    success, message = complete_registration(
        full_name=pending["full_name"],
        email=pending["email"],
        phone=pending["phone"],
        password_hash=pending["password_hash"],
        otp=otp,
        role=pending.get("role", "PATIENT")
    )

    if not success:
        return templates.TemplateResponse(
            request,
            "verify_otp.html",
            {
                "email": pending["email"],
                "error": message
            },
            status_code=400
        )

    request.session.pop("pending_registration", None)

    return RedirectResponse(
        url="/login?registered=true",
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
async def predict_data(request: Request):

    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    try:
        form_data = await request.form()

        logger.info(
            f"Screening request received from user {user['user_id']}"
        )

        custom_data = CustomData(
            age=int(form_data.get("age")),
            gender=form_data.get("gender"),
            fever=float(form_data.get("fever")),
            cough=form_data.get("cough"),
            city=form_data.get("city")
        )

        final_data = custom_data.get_data_as_dataframe()

        prediction = PredictPipeline().predict(final_data)

        return templates.TemplateResponse(
            request,
            "result.html",
            {
                "user": user,
                "prediction": prediction
            }
        )

    except (ValueError, TypeError) as e:
        logger.warning(f"Screening validation error: {e}")
        return templates.TemplateResponse(
            request,
            "predict.html",
            {
                "user": user,
                "error": str(e)
            }
        )

    except Exception:
        logger.exception("Screening error")
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

@app.post("/chat")
async def chat(request: Request):
    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    try:
        data = await request.json()
        message = data.get("message", "")

        if not isinstance(message, str):
            message = ""
        message = message.strip()

        if not message:
            return {"response": "Please enter a message."}

        # Import the chatbot only when a chat request arrives.
        # IMPORTANT: appointment_repository.py must not import chatbot.service;
        # chatbot.service may import appointment_repository.py, not vice versa.
        from chatbot.service import process_message

        # process_message returns a structured response dictionary.
        # Return it directly so the frontend receives response/type/options
        # at the expected top level instead of a nested object.
        result = await process_message(
            message,
            user=user
        )

        if isinstance(result, dict):
            return result

        return {"response": str(result)}

    except Exception as e:
        logger.exception(f"Chatbot error: {str(e)}")
        return {
            "response": (
                "Sorry, I am unable to process "
                "your request right now."
            )
        }


# =========================================================
# INJURY / INFECTION / ALLERGY IMAGE ANALYSIS (GROQ)
# =========================================================

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_FOLLOWUP_QUESTIONS = 4
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
_vision_client = None


def _get_vision_client():
    """Create the Groq async client lazily, using the existing GROQ_API_KEY."""
    global _vision_client
    if _vision_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            logger.error("GROQ_API_KEY is missing. Add it to the .env file beside main.py.")
            raise HTTPException(
                status_code=503,
                detail="Image analysis is not configured. Add GROQ_API_KEY to the .env file beside main.py, then restart the server."
            )
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            logger.error(f"Groq package is not installed: {exc}")
            raise HTTPException(
                status_code=503,
                detail="Groq is not installed. Run: pip install groq"
            )
        _vision_client = AsyncGroq(api_key=api_key)
    return _vision_client


def _medical_image_system_prompt() -> str:
    return (
        "You are a cautious healthcare information assistant for a clinic website. Review only visible "
        "injuries, wounds, burns, bruises, rashes, and possible skin concerns. If unrelated or unclear, say "
        "this feature supports only those concerns and do not guess.\n\n"
        "Return ONLY valid JSON (no markdown fences) with this schema: "
        "{\"headings\":{\"1. Visible observations\":\"...\","
        "\"2. Possible explanations\":\"...\",\"3. General care\":\"...\","
        "\"4. When to seek care\":\"...\"},"
        "\"follow_up_question\":null or one question,\"options\":[...]}\n"
        "For an initial review, include all four headings. Decide whether one medically useful follow-up is needed. "
        "If yes, ask exactly one question and provide 2-5 short answer options (include Not sure where useful). "
        "If no, set follow_up_question to null and options to []. For follow-up turns, use the image and conversation "
        "history, do not repeat answered questions, and either ask one useful next question or finish with guidance. "
        "When finishing, set follow_up_question to null and options to [].\n\n"
        "Describe only visible features such as color, shape, location, and whether the skin appears broken. Do not "
        "infer timing, cause, depth, healing, infection, or absence of infection from a photo. Possibilities are not "
        "a diagnosis. Do not prescribe medicines or dosages. Give conservative, relevant first-aid advice and explain "
        "warning signs. Recommend urgent care for trouble breathing, facial/throat swelling, uncontrolled bleeding, "
        "deep/gaping wounds, severe burns, rapidly spreading redness, fever with worsening skin symptoms, or severe pain. "
        "For emergencies, advise local emergency services. Never infer or suggest self-harm, intent, mental state, or "
        "the cause of an injury from an image. Be calm and clear that this is not a diagnosis and cannot replace a clinician."
    )


def _parse_model_json(text: str) -> dict:
    """Extract the JSON object from the model reply, including if fences slip in."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model reply")
    return json.loads(text[start:end + 1])


def _format_result(data: dict, force_final: bool) -> dict:
    """Convert model JSON to the response shape expected by index.html."""
    headings = data.get("headings")
    if not isinstance(headings, dict) or not headings:
        raise ValueError("Model reply has no headings")

    response_text = "\n\n".join(
        f"{title}\n{str(body).strip()}" for title, body in headings.items()
    )
    question = data.get("follow_up_question")
    options = data.get("options")
    if not isinstance(question, str) or not question.strip():
        question = None
    if not isinstance(options, list):
        options = []
    options = [str(option).strip() for option in options if str(option).strip()][:5]

    if force_final or question is None or len(options) < 2:
        return {"response": response_text, "question": None, "options": [], "is_final": True}
    return {"response": response_text, "question": question.strip(), "options": options, "is_final": False}


async def _read_and_validate_image(image: UploadFile) -> tuple[bytes, str]:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a JPG, PNG or WEBP image.")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="The image file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image must be smaller than 5 MB.")
    return data, image.content_type


async def _run_image_analysis(
    image_bytes: bytes,
    media_type: str,
    prompt_text: str,
    force_final: bool
) -> dict:
    try:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        response = await _get_vision_client().chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": _medical_image_system_prompt()},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{encoded_image}"
                    }}
                ]}
            ],
            temperature=0.2,
            max_tokens=1200
        )
        reply = response.choices[0].message.content or ""
        return _format_result(_parse_model_json(reply), force_final)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Groq image analysis error: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Unable to analyze the image right now. Check your Groq vision model configuration and try again."
        )


@app.post("/chat/image-analysis")
async def chat_image_analysis(request: Request, image: UploadFile = File(...)):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    image_bytes, media_type = await _read_and_validate_image(image)
    logger.info(f"Image analysis requested by user {user['user_id']}")
    return await _run_image_analysis(
        image_bytes, media_type,
        "Please review this image. This is the initial review.",
        force_final=False
    )


@app.post("/chat/image-followup")
async def chat_image_followup(
    request: Request,
    image: UploadFile = File(...),
    answer: str = Form(...),
    initial_analysis: str = Form(""),
    history: str = Form("[]")
):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    answer = answer.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="Please select an answer.")
    image_bytes, media_type = await _read_and_validate_image(image)

    try:
        turns = json.loads(history)
        if not isinstance(turns, list):
            turns = []
    except (json.JSONDecodeError, TypeError):
        turns = []
    clean_turns = [
        turn for turn in turns
        if isinstance(turn, dict)
        and turn.get("role") in ("assistant", "user")
        and isinstance(turn.get("content"), str)
    ][-12:]
    answers_given = sum(1 for turn in clean_turns if turn["role"] == "user") + 1
    force_final = answers_given >= MAX_FOLLOWUP_QUESTIONS

    conversation = "\n".join(
        f"{'Assistant' if turn['role'] == 'assistant' else 'Patient'}: {turn['content'][:1500]}"
        for turn in clean_turns
    )
    prompt = (
        "This is a follow-up turn.\n\n"
        f"Initial analysis:\n{initial_analysis[:3000]}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Latest patient answer: {answer[:300]}\n\n"
    )
    if force_final:
        prompt += "You have enough information. Do not ask another question: finish with updated guidance using all four headings."
    else:
        prompt += "Ask one more useful question only if truly needed; otherwise finish with updated guidance using all four headings."

    logger.info(f"Image follow-up from user {user['user_id']}")
    return await _run_image_analysis(image_bytes, media_type, prompt, force_final=force_final)


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

def prepare_slots_for_display(slots):
    """
    Turn raw availability rows into display-ready rows:
    readable date/time labels, and drop slots that already ended today.
    (MySQL TIME columns arrive as timedelta objects.)
    """
    from datetime import datetime as _datetime
    from datetime import time as _time
    from datetime import timedelta as _timedelta

    def to_clock(value):
        if isinstance(value, _timedelta):
            total = int(value.total_seconds())
            return _time((total // 3600) % 24, (total % 3600) // 60)
        if isinstance(value, _datetime):
            return value.time()
        return value

    now = _datetime.now()
    prepared = []

    for slot in slots:
        start = to_clock(slot["start_time"])
        end = to_clock(slot["end_time"])

        if slot["available_date"] == now.date() and end <= now.time():
            continue

        item = dict(slot)
        item["date_label"] = slot["available_date"].strftime("%a, %d %b %Y")
        item["start_label"] = start.strftime("%I:%M %p").lstrip("0")
        item["end_label"] = end.strftime("%I:%M %p").lstrip("0")
        prepared.append(item)

    return prepared


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

    availability = prepare_slots_for_display(
        get_doctor_availability(
            doctor_id
        )
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

        # Use a unique Cashfree order ID for every payment attempt.
        # This avoids duplicate-order errors when a previous attempt
        # reached Cashfree but failed before the session was saved.
        cashfree_order_id = (
            f"appointment_{appointment_id}_{uuid4().hex[:8]}"
        )

        # Build the return URL from the current application host and
        # include the Cashfree order ID expected by cashfree_return().
        return_url = (
            f"{request.url_for('cashfree_return')}"
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

        saved = save_cashfree_order(
            appointment_id=appointment_id,
            user_id=user["user_id"],
            cashfree_order_id=cashfree_order["order_id"],
            payment_session_id=cashfree_order["payment_session_id"]
        )

        if not saved:
            raise RuntimeError(
                "Cashfree order was created, but its payment session "
                "could not be saved for this appointment."
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

    except Exception:
        # Keep the appointment so the patient can retry payment later.
        # Log the full traceback; the previous code hid the actual cause.
        logger.exception(
            "Cashfree initialization failed for appointment %s "
            "(user_id=%s)",
            appointment_id,
            user["user_id"]
        )

        retry_url = f"/appointments/{appointment_id}/pay"
        return HTMLResponse(
            content=f"""
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>Payment could not start</title>
              <style>
                body {{ font-family: Arial, sans-serif; background: #f4f8f7;
                       color: #20312f; margin: 0; padding: 32px; }}
                main {{ max-width: 650px; margin: 8vh auto; background: white;
                        padding: 32px; border: 1px solid #dce8e5;
                        border-radius: 16px; }}
                a {{ display: inline-block; margin-top: 16px; padding: 12px 18px;
                     background: #0d5148; color: white; border-radius: 8px;
                     text-decoration: none; }}
              </style>
            </head>
            <body><main>
              <h1>Appointment created</h1>
              <p>Your appointment number is <strong>{appointment_id}</strong>.</p>
              <p>Online payment could not be started. Your appointment is still
                 saved with payment pending. You can retry payment from here.</p>
              <a href="{retry_url}">Retry online payment</a>
              <p><a href="/profile#appointments" style="background:#e7f1ef;color:#0d5148">
                 View my appointments</a></p>
            </main></body></html>
            """,
            status_code=502
        )


# =========================================================
# CHATBOT APPOINTMENT PAYMENT PAGE
# =========================================================

@app.get(
    "/appointments/{appointment_id}/pay",
    response_class=HTMLResponse
)
async def appointment_pay(
    request: Request,
    appointment_id: int
):
    user = require_login(request)

    if isinstance(user, RedirectResponse):
        return user

    # Only allow the logged-in user to pay for their own appointment.
    appointment = get_appointment_for_cashfree_payment(
        appointment_id=appointment_id,
        user_id=user["user_id"]
    )

    if not appointment:
        return HTMLResponse(
            content="Appointment not found.",
            status_code=404
        )

    appointment_status = str(appointment.get("status", "")).upper()
    payment_status = str(appointment.get("payment_status", "")).upper()

    if appointment_status == "CANCELLED":
        return HTMLResponse(
            content="Payment is not available for cancelled appointments.",
            status_code=400
        )

    if payment_status == "PAID":
        return RedirectResponse(url="/profile#appointments", status_code=303)

    try:
        amount = float(appointment.get("consultation_fee") or 0)
        patient_name = appointment.get("patient_name") or user.get("full_name") or "Patient"
        phone = appointment.get("phone") or user.get("phone")

        if amount <= 0:
            return HTMLResponse(
                content="A valid consultation fee was not found for this appointment.",
                status_code=400
            )

        payment_session_id = appointment.get("cashfree_payment_session_id")
        cashfree_order_id = appointment.get("cashfree_order_id")

        # Create a Cashfree order if the chatbot-created appointment
        # does not already have a payment session.
        if not payment_session_id:
            if not phone:
                return HTMLResponse(
                    content="A phone number is required to start online payment. Please update your profile.",
                    status_code=400
                )

            cashfree_order_id = f"appointment_{appointment_id}_{uuid4().hex[:8]}"
            return_url = (
                "http://127.0.0.1:8000/payments/cashfree/return"
                f"?order_id={cashfree_order_id}"
            )

            cashfree_order = create_cashfree_order(
                order_id=cashfree_order_id,
                amount=amount,
                customer_id=str(user["user_id"]),
                customer_name=patient_name,
                customer_email=user["email"],
                customer_phone=phone,
                return_url=return_url
            )

            cashfree_order_id = cashfree_order["order_id"]
            payment_session_id = cashfree_order["payment_session_id"]

            save_cashfree_order(
                appointment_id=appointment_id,
                user_id=user["user_id"],
                cashfree_order_id=cashfree_order_id,
                payment_session_id=payment_session_id
            )

        doctor = {
            "doctor_name": appointment.get("doctor_name"),
            "specialization": appointment.get("specialization"),
            "consultation_fee": amount
        }

        return templates.TemplateResponse(
            request,
            "cashfree_checkout.html",
            {
                "user": user,
                "doctor": doctor,
                "appointment_id": appointment_id,
                "appointment_date": appointment.get("appointment_date"),
                "appointment_time": appointment.get("appointment_time"),
                "patient_name": patient_name,
                "phone": phone,
                "payment_amount": amount,
                "payment_session_id": payment_session_id,
                "cashfree_order_id": cashfree_order_id
            }
        )

    except Exception:
        logger.exception("Failed to initialize chatbot appointment payment")
        return HTMLResponse(
            content="Unable to start online payment. Please check the server logs and try again.",
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

    # The repository receives only the authenticated user's ID from
    # the session. It applies WHERE appointments.user_id = %s.
    appointments = get_user_appointments(
        user_id=int(user["user_id"])
    )

    return templates.TemplateResponse(
        request,
        "profile.html",
        {
            "user": user,
            "appointments": appointments
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

    return templates.TemplateResponse(
        request,
        "pharmacy.html",
        {
            "user": user,
            "medicines": [],
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


# =========================================================
# PHARMACY - FIND NEARBY PHARMACIES
# =========================================================

@app.post("/pharmacy/nearby")
async def find_nearby_pharmacies(request: Request):
    """Find pharmacies near the patient's current location.

    Accepts JSON, form data, or query parameters. Expected location fields
    are latitude/longitude (also accepts lat/lng); radius_km is optional.
    """

    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Please log in to find nearby pharmacies."
        )

    try:
        content_type = request.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            payload = await request.json()
        elif (
            "application/x-www-form-urlencoded" in content_type
            or "multipart/form-data" in content_type
        ):
            form = await request.form()
            payload = dict(form)
        else:
            payload = dict(request.query_params)

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid location data."
            )

        latitude = payload.get("latitude", payload.get("lat"))
        longitude = payload.get("longitude", payload.get("lng"))
        radius_km = payload.get("radius_km", 10)

        if latitude in (None, "") or longitude in (None, ""):
            raise HTTPException(
                status_code=400,
                detail="Your latitude and longitude are required. Please allow location access."
            )

        try:
            latitude = float(latitude)
            longitude = float(longitude)
            radius_km = float(radius_km)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Latitude, longitude, and radius must be numbers."
            )

        pharmacies = get_nearby_pharmacies(
            user_lat=latitude,
            user_lng=longitude,
            radius_km=radius_km
        )

        return {
            "success": True,
            "pharmacies": pharmacies,
            "count": len(pharmacies)
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception(
            "Nearby pharmacy search failed for user %s",
            user["user_id"]
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to find nearby pharmacies right now. "
                "Please check the pharmacy location data and Google Maps API configuration."
            )
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
    address = (
        form.get("delivery_address")
        or form.get("address")
        or ""
    ).strip()    
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

        cashfree_order_id = f"pharmacy_{order_id}"

        return_url = (
            "http://localhost:8000"
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

            save_pharmacy_cashfree_order(
                order_id=order_id,
                cashfree_order_id=cashfree_order["order_id"],
                payment_session_id=cashfree_order["payment_session_id"]
            )

        except Exception as e:
            logger.exception(
                f"Cashfree pharmacy order error: {e}"
            )
            return HTMLResponse(
                "Unable to start online payment.",
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
    order_id: str = "",
    cf_order_id: str = ""
):
    """Verify Cashfree payment, then show the order only to its owner."""

    # Cashfree can return after the browser session has changed or expired.
    # Verify and update the payment using the database order owner; do not
    # block payment verification just because the current session differs.
    user = get_current_user(request)

    supplied_order_id = (order_id or cf_order_id or "").strip()
    if not supplied_order_id:
        return HTMLResponse(
            "Payment return is missing the order ID. Please open your order from My Profile.",
            status_code=400
        )

    # Resolve the order and its real owner from the database. Do not use
    # the current session user to locate the order: Cashfree may return to
    # a browser session that has changed since checkout.
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        if supplied_order_id.isdigit():
            cursor.execute(
                """
                SELECT order_id, user_id, cashfree_order_id,
                       payment_method, payment_status, order_status
                FROM pharmacy_orders
                WHERE order_id = %s
                """,
                (int(supplied_order_id),)
            )
        else:
            cursor.execute(
                """
                SELECT order_id, user_id, cashfree_order_id,
                       payment_method, payment_status, order_status
                FROM pharmacy_orders
                WHERE cashfree_order_id = %s
                """,
                (supplied_order_id,)
            )
        order = cursor.fetchone()
    except Exception:
        logger.exception(
            "Could not load pharmacy order for Cashfree return: %s",
            supplied_order_id
        )
        return HTMLResponse(
            "Unable to load your order right now. Please try again.",
            status_code=500
        )
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

    if not order:
        logger.warning(
            "Pharmacy order not found for payment return: %s",
            supplied_order_id
        )
        return HTMLResponse("Pharmacy order not found.", status_code=404)

    if str(order.get("payment_method", "")).upper() != "ONLINE":
        return HTMLResponse(
            "This order is not configured for online payment.",
            status_code=400
        )

    cashfree_order_id = order.get("cashfree_order_id")
    if not cashfree_order_id:
        return HTMLResponse(
            "Cashfree order ID is missing. Please contact support.",
            status_code=400
        )

    # The return URL is not proof of payment. Verify with Cashfree.
    try:
        payments = get_cashfree_order_payments(cashfree_order_id)
    except Exception:
        logger.exception(
            "Cashfree verification failed for pharmacy order %s",
            order["order_id"]
        )
        if not user or str(user["user_id"]) != str(order["user_id"]):
            return HTMLResponse(
                "Payment could not be verified right now. Please sign in with the account used to place the order and check the order status from My Profile.",
                status_code=503
            )
        owned_order = get_pharmacy_order(order["order_id"], user["user_id"])
        return templates.TemplateResponse(
            request,
            "pharmacy_order.html",
            {
                "user": user,
                "order": owned_order,
                "payment_message": (
                    "Payment could not be verified right now. "
                    "Your order has not been marked as paid. Please check again shortly."
                )
            },
            status_code=503
        )

    def extract_payment_status(payment):
        if isinstance(payment, dict):
            value = payment.get("payment_status", "")
        else:
            value = getattr(payment, "payment_status", "")
        return str(value or "").strip().upper()

    statuses = [extract_payment_status(payment) for payment in (payments or [])]
    if "SUCCESS" in statuses:
        final_status = "PAID"
    elif any(status in ("FAILED", "FAILURE", "USER_DROPPED") for status in statuses):
        final_status = "FAILED"
    else:
        final_status = "PENDING"

    owner_id = order["user_id"]
    try:
        if final_status == "PAID":
            update_pharmacy_payment_success(
                order_id=order["order_id"],
                user_id=owner_id
            )

            # update_pharmacy_payment_success() atomically confirms the order,
            # reduces stock once, and removes only the purchased quantities
            # from the owning user's cart. Do not clear the entire cart here:
            # the user may have added new items after checkout.
            message = "Payment successful. Your pharmacy order is confirmed."
        elif final_status == "FAILED":
            update_pharmacy_payment_failed(
                order_id=order["order_id"],
                user_id=owner_id
            )
            message = "Payment failed. You can retry payment from your order page."
        else:
            message = (
                "Payment is still being processed. Your order has not been "
                "marked as paid. Please check again shortly."
            )
    except Exception:
        logger.exception(
            "Failed to update payment for pharmacy order %s",
            order["order_id"]
        )
        return HTMLResponse(
            "Payment was checked, but the order could not be updated. Please contact support.",
            status_code=500
        )

    # Never display another account's order. Payment verification and order
    # updates are complete even if this browser is logged into a different
    # account; provide a safe message instead of returning a misleading 403.
    if not user or str(user["user_id"]) != str(owner_id):
        logger.warning(
            "Payment return session mismatch: order=%s owner=%s session_user=%s",
            order["order_id"], owner_id, user["user_id"] if user else None
        )
        # Do not expose order details to a different signed-in account.
        # Give the user a direct way to end the current session and sign in
        # with the account that owns the order. The order is still verified
        # and updated above using the database owner, not the session user.
        return HTMLResponse(
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment verified | Sanjeevani</title>
    <style>
        body { margin: 0; padding: 24px; font-family: Arial, sans-serif;
               background: #f4f8f7; color: #183b36; }
        .card { max-width: 620px; margin: 8vh auto; padding: 32px;
                background: #fff; border: 1px solid #dce9e5;
                border-radius: 16px; box-shadow: 0 8px 28px #12352b12; }
        h1 { margin-top: 0; font-size: 26px; }
        p { line-height: 1.6; color: #526a65; }
        a { display: inline-block; margin-top: 12px; padding: 12px 20px;
            border-radius: 9px; background: #0d5148; color: white;
            text-decoration: none; }
    </style>
</head>
<body>
    <main class="card">
        <h1>Payment verification complete</h1>
        <p>Your payment has been checked and the order has been updated.</p>
        <p>This order belongs to a different account. For your privacy, its
           details are not shown in this session.</p>
        <p>Sign out, then sign in with the account used to place the order
           to view it in My Profile.</p>
        <a href="/logout">Sign out and continue</a>
    </main>
</body>
</html>""",
            status_code=200
        )

    owned_order = get_pharmacy_order(order["order_id"], user["user_id"])
    if not owned_order:
        return HTMLResponse(
            "Payment was checked, but the order could not be loaded. Please open My Profile.",
            status_code=500
        )

    return templates.TemplateResponse(
        request,
        "pharmacy_order.html",
        {
            "user": user,
            "order": owned_order,
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