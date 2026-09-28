# chatbot/service.py
import os
import requests
import inspect
import logging
import re
from datetime import datetime, date, time
from difflib import SequenceMatcher


# ============================================================
# LLM / ROUTER
# ============================================================

from chatbot.llm import generate_response
from chatbot.router import detect_intent, is_greeting
from chatbot.prompts import SYSTEM_PROMPT


# ============================================================
# DATABASE
# ============================================================

from src.database.doctor_repository import (
    find_doctors_by_specialization,
    get_doctor_by_id,
    get_doctor_availability
)

from src.database.appointment_repository import (
    create_appointment,
    is_slot_booked,
    get_user_appointments,
    cancel_appointment,
    reschedule_appointment,
    save_cashfree_order
)

from src.payment.cashfree_service import (
    create_cashfree_order,
    CASHFREE_CONFIGURED
)

from src.database.pharmacy_repository import (
    get_all_medicines,
    search_medicines,
    add_to_cart
)


# ============================================================
# BOOKING STATES
# ============================================================

# State is maintained separately for each logged-in user.
#
# Example:
#
# booking_states = {
#     1: {
#         "active": True,
#         "step": "doctor",
#         ...
#     }
# }
#
# This prevents two users from sharing the same booking state.

booking_states = {}


# Keep this name for compatibility with older code/tests.
booking_state = {
    "active": False,
    "step": None,
    "specialty": None,
    "doctor_id": None,
    "doctor_name": None,
    "appointment_date": None,
    "appointment_time": None,
    "patient_name": None,
    "phone": None,
    "payment_method": None,
    "doctors": [],
    "available_dates": [],
    "available_slots": []
}


# ============================================================
# PENDING ACTION STATES
# (cancel / reschedule / pharmacy-selection follow-ups)
# ============================================================

pending_states = {}


def _empty_pending_state():

    return {
        "action": None,
        "step": None,
        "appointments": [],
        "appointment_id": None,
        "doctor_id": None,
        "doctor_name": None,
        "available_dates": [],
        "available_slots": [],
        "appointment_date": None,
        "appointment_time": None,
        "medicines": []
    }


def _get_pending_state(user=None):

    key = _get_user_key(user)

    if key not in pending_states:
        pending_states[key] = _empty_pending_state()

    return pending_states[key]


def _reset_pending_state(user=None):

    key = _get_user_key(user)

    pending_states[key] = _empty_pending_state()


# ============================================================
# RESPONSE HELPER
# ============================================================

def chatbot_response(
    message,
    response_type=None,
    options=None
):
    """
    Creates the structured response expected by index.html.
    """

    result = {
        "response": str(message)
    }

    if response_type:
        result["type"] = response_type

    if options:
        result["options"] = options

    return result


# ============================================================
# STATE HELPERS
# ============================================================

def _get_user_key(user=None):

    if isinstance(user, dict):

        user_id = user.get("user_id")

        if user_id is not None:
            return str(user_id)

    return "anonymous"


def _create_empty_state():

    return {
        "active": False,

        "step": None,

        "specialty": None,

        "doctor_id": None,

        "doctor_name": None,

        "appointment_date": None,

        "appointment_time": None,

        "patient_name": None,

        "phone": None,

        "payment_method": None,

        "doctors": [],

        "available_dates": [],

        "available_slots": []
    }


def get_booking_state(user=None):

    key = _get_user_key(user)

    if key not in booking_states:

        booking_states[key] = _create_empty_state()

    return booking_states[key]


def reset_booking(user=None):

    key = _get_user_key(user)

    booking_states[key] = _create_empty_state()

    # Compatibility with older code.
    booking_state.clear()
    booking_state.update(
        booking_states[key]
    )


def start_booking(user=None):

    state = get_booking_state(user)

    state["active"] = True

    state["step"] = "specialty"

    state["specialty"] = None
    state["doctor_id"] = None
    state["doctor_name"] = None
    state["appointment_date"] = None
    state["appointment_time"] = None
    state["patient_name"] = None
    state["phone"] = None
    state["payment_method"] = None

    state["doctors"] = []
    state["available_dates"] = []
    state["available_slots"] = []

    booking_state.clear()
    booking_state.update(state)

    return state


def set_booking_value(
    key,
    value,
    user=None
):

    state = get_booking_state(user)

    if key not in state:

        return False

    state[key] = value

    booking_state.clear()
    booking_state.update(state)

    return True


def get_booking_step(user=None):

    state = get_booking_state(user)

    return state.get("step")


def get_booking_data(user=None):

    state = get_booking_state(user)

    return {
        "specialty": state.get("specialty"),
        "doctor_id": state.get("doctor_id"),
        "doctor_name": state.get("doctor_name"),
        "appointment_date": state.get("appointment_date"),
        "appointment_time": state.get("appointment_time"),
        "patient_name": state.get("patient_name"),
        "phone": state.get("phone"),
        "payment_method": state.get("payment_method")
    }


# ============================================================
# ASYNC HELPER
# ============================================================

async def _resolve_result(result):

    if inspect.isawaitable(result):

        return await result

    return result


# ============================================================
# LLM HELPERS
# ============================================================

async def _detect_intent(
    message,
    conversation_history=None
):

    """
    Calls router.detect_intent safely whether the
    router function is sync or async.
    """

    try:

        result = detect_intent(
            message
        )

        result = await _resolve_result(
            result
        )

        return result

    except TypeError:

        try:

            result = detect_intent(
                message,
                conversation_history or []
            )

            result = await _resolve_result(
                result
            )

            return result

        except Exception:

            return {
                "intent": "GENERAL",
                "specialty": None,
                "medicine": None,
                "date": None,
                "time": None,
                "doctor_name": None,
                "confidence": 0
            }

    except Exception:

        return {
            "intent": "GENERAL",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0
        }


async def _generate_llm_response(
    message,
    conversation_history=None
):

    """
    Calls generate_response with the REAL medical-assistant
    system prompt (chatbot/prompts.SYSTEM_PROMPT), and passes
    the user's message as a proper "user" role turn appended
    to any prior conversation history.

    Previously this passed the raw user message as the system
    prompt and never included SYSTEM_PROMPT at all, which is
    why general/symptom questions got generic or off-target
    answers - the model had no medical-safety instructions and
    often no actual user turn to respond to.
    """

    try:

        messages = list(
            conversation_history or []
        )

        messages.append({
            "role": "user",
            "content": message
        })

        result = generate_response(
            SYSTEM_PROMPT,
            messages
        )

        result = await _resolve_result(
            result
        )

        return result

    except Exception:

        return (
            "Sorry, I am unable to process "
            "your request right now."
        )


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_text(value):

    if value is None:

        return ""

    return str(value).strip()


def _normalize(value):

    return (
        _clean_text(value)
        .lower()
        .strip()
    )


def _clean_doctor_name(name):

    name = _clean_text(name)

    # Prevent:
    #
    # Dr. Dr. Suresh Sharma
    #
    # If database already contains Dr.

    name = re.sub(
        r"^(dr\.?\s*)+",
        "",
        name,
        flags=re.IGNORECASE
    )

    return f"Dr. {name}"


def _format_money(value):

    try:

        return f"₹{float(value):,.2f}"

    except Exception:

        return f"₹{value}"


def _format_time(value):

    if value is None:

        return ""

    if isinstance(value, datetime):

        value = value.time()

    if isinstance(value, time):

        return value.strftime("%I:%M %p").lstrip("0")

    text = str(value)

    # Handle:
    # 09:00:00
    # 14:00:00

    try:

        parsed = datetime.strptime(
            text,
            "%H:%M:%S"
        )

        return parsed.strftime(
            "%I:%M %p"
        ).lstrip("0")

    except Exception:

        pass

    try:

        parsed = datetime.strptime(
            text,
            "%H:%M"
        )

        return parsed.strftime(
            "%I:%M %p"
        ).lstrip("0")

    except Exception:

        return text


# ============================================================
# SPECIALIZATION NORMALIZATION
# ============================================================

SPECIALIZATION_MAP = {

    "cardioligist": "Cardiologist",
    "cardiologist": "Cardiologist",
    "cardiology": "Cardiologist",
    "heart doctor": "Cardiologist",
    "heart specialist": "Cardiologist",

    "dermatologist": "Dermatologist",
    "dermatologist doctor": "Dermatologist",
    "skin doctor": "Dermatologist",
    "skin specialist": "Dermatologist",

    "pediatrician": "Pediatrician",
    "paediatrician": "Pediatrician",
    "child doctor": "Pediatrician",
    "kids doctor": "Pediatrician",
    "children doctor": "Pediatrician",

    "orthopedic": "Orthopedic",
    "orthopaedic": "Orthopedic",
    "orthopedist": "Orthopedic",
    "bone doctor": "Orthopedic",
    "bone specialist": "Orthopedic",

    "ophthalmologist": "Ophthalmologist",
    "eye doctor": "Ophthalmologist",
    "eye specialist": "Ophthalmologist",

    "neurologist": "Neurologist",
    "neurology": "Neurologist",
    "brain doctor": "Neurologist",
    "brain specialist": "Neurologist",

    "dentist": "Dentist",
    "dental doctor": "Dentist",
    "tooth doctor": "Dentist",

    "gynecologist": "Gynecologist",
    "gynaecologist": "Gynecologist",
    "women doctor": "Gynecologist",

    "physician": "Physician",
    "general physician": "General Physician",

    "psychiatrist": "Psychiatrist",

    "ent": "ENT",
    "ent specialist": "ENT",
    "ear doctor": "ENT",

    "urologist": "Urologist",

    "nephrologist": "Nephrologist",

    "gastroenterologist": "Gastroenterologist"
}


def normalize_specialty(value):

    text = _normalize(value)

    if not text:

        return None

    # Exact match first.
    if text in SPECIALIZATION_MAP:

        return SPECIALIZATION_MAP[text]

    # Safe complete-word / phrase matching.
    # This prevents values such as "appointment" from
    # accidentally matching "ent".
    for key, specialty in SPECIALIZATION_MAP.items():

        pattern = (
            r"(?<!\w)"
            + re.escape(key)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            return specialty

    return None


# ============================================================
# NUMBER / ORDINAL HELPERS
# ============================================================

NUMBER_WORDS = {

    "zero": 0,

    "one": 1,
    "first": 1,
    "frist": 1,
    "fir st": 1,

    "two": 2,
    "second": 2,
    "secod": 2,
    "secon": 2,
    "seond": 2,

    "three": 3,
    "third": 3,
    "thrid": 3,

    "four": 4,
    "fourth": 4,
    "forth": 4,

    "five": 5,
    "fifth": 5,

    "six": 6,
    "sixth": 6,

    "seven": 7,
    "seventh": 7,

    "eight": 8,
    "eighth": 8,

    "nine": 9,
    "ninth": 9,

    "ten": 10,
    "tenth": 10
}


def parse_number_choice(
    value,
    maximum=None
):

    text = _normalize(value)

    if not text:

        return None

    # Direct integer.

    if text.isdigit():

        number = int(text)

        if number >= 1:

            if maximum is None:

                return number

            if number <= maximum:

                return number

        return None

    # Remove punctuation.

    clean = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    if clean in NUMBER_WORDS:

        number = NUMBER_WORDS[clean]

        if maximum is None:

            return number

        if number <= maximum:

            return number

    # Fuzzy matching for common typing mistakes.

    candidates = list(
        NUMBER_WORDS.keys()
    )

    best_word = None
    best_score = 0

    for word in candidates:

        score = SequenceMatcher(
            None,
            clean,
            word
        ).ratio()

        if score > best_score:

            best_score = score
            best_word = word

    if best_word and best_score >= 0.72:

        number = NUMBER_WORDS[
            best_word
        ]

        if maximum is None:

            return number

        if number <= maximum:

            return number

    return None


# ============================================================
# DATE HELPERS
# ============================================================

def _date_from_value(value):

    if value is None:

        return None

    if isinstance(value, datetime):

        return value.date()

    if isinstance(value, date):

        return value

    text = _clean_text(value)

    if not text:

        return None

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d %B %Y",
        "%d %b %Y"
    ]

    for fmt in formats:

        try:

            return datetime.strptime(
                text,
                fmt
            ).date()

        except ValueError:

            continue

    return None


def _get_date_from_availability(row):

    if not isinstance(row, dict):

        return None

    possible_keys = [
        "availability_date",
        "available_date",
        "appointment_date",
        "date",
        "slot_date"
    ]

    for key in possible_keys:

        if key in row:

            parsed = _date_from_value(
                row[key]
            )

            if parsed:

                return parsed

    return None


def _get_start_time(row):

    if not isinstance(row, dict):

        return None

    possible_keys = [
        "start_time",
        "available_from",
        "from_time",
        "time"
    ]

    for key in possible_keys:

        if key in row:

            value = row[key]

            if isinstance(value, time):

                return value

            if isinstance(value, datetime):

                return value.time()

            text = _clean_text(value)

            for fmt in [
                "%H:%M:%S",
                "%H:%M",
                "%I:%M %p",
                "%I:%M%p"
            ]:

                try:

                    return datetime.strptime(
                        text,
                        fmt
                    ).time()

                except ValueError:

                    continue

    return None


def _get_end_time(row):

    if not isinstance(row, dict):

        return None

    possible_keys = [
        "end_time",
        "available_to",
        "to_time"
    ]

    for key in possible_keys:

        if key in row:

            value = row[key]

            if isinstance(value, time):

                return value

            if isinstance(value, datetime):

                return value.time()

            text = _clean_text(value)

            for fmt in [
                "%H:%M:%S",
                "%H:%M",
                "%I:%M %p",
                "%I:%M%p"
            ]:

                try:

                    return datetime.strptime(
                        text,
                        fmt
                    ).time()

                except ValueError:

                    continue

    return None


# ============================================================
# DOCTOR OPTIONS
# ============================================================

def _build_doctor_options(doctors):

    options = []

    for index, doctor in enumerate(
        doctors,
        start=1
    ):

        doctor_name = _clean_doctor_name(
            doctor.get("doctor_name")
        )

        experience = doctor.get(
            "experience",
            0
        )

        fee = doctor.get(
            "consultation_fee",
            0
        )

        options.append(
            {
                "value": str(index),

                "title": doctor_name,

                "subtitle":
                    f"{experience} years experience",

                "price":
                    _format_money(fee)
            }
        )

    return options


# ============================================================
# DATE OPTIONS
# ============================================================

def _filter_booked_slots(
    doctor_id,
    rows
):

    """
    Removes availability rows whose exact (date, start_time)
    is already booked, so a taken slot is never shown to the
    user as "Available" in the first place - instead of only
    being caught after they've re-entered their name, phone,
    and payment method.
    """

    available = []

    for row in rows:

        row_date = _get_date_from_availability(
            row
        )

        row_time = _get_start_time(
            row
        )

        if row_date is None or row_time is None:
            continue

        try:
            taken = is_slot_booked(
                doctor_id,
                row_date,
                row_time
            )
        except Exception:
            taken = False

        if not taken:
            available.append(row)

    return available


def _build_date_options(
    available_dates
):

    options = []

    for index, available_date in enumerate(
        available_dates,
        start=1
    ):

        parsed = _date_from_value(
            available_date
        )

        if parsed:

            title = parsed.strftime(
                "%a, %d %b %Y"
            )

        else:

            title = str(
                available_date
            )

        options.append(
            {
                "value": str(index),

                "title": title,

                "subtitle": "Available"
            }
        )

    return options


# ============================================================
# TIME OPTIONS
# ============================================================

def _build_time_options(
    matching_slots
):

    options = []

    for index, slot in enumerate(
        matching_slots,
        start=1
    ):

        start_time = _get_start_time(
            slot
        )

        end_time = _get_end_time(
            slot
        )

        if start_time:

            title = _format_time(
                start_time
            )

        else:

            title = "Available time"

        subtitle = "Available"

        if end_time:

            subtitle = (
                f"Available until "
                f"{_format_time(end_time)}"
            )

        options.append(
            {
                "value": str(index),

                "title": title,

                "subtitle": subtitle
            }
        )

    return options


# ============================================================
# PAYMENT OPTIONS
# ============================================================

def _payment_options():

    return [

        {
            "value": "CASH",

            "title": "Cash at Clinic",

            "subtitle":
                "Pay when you visit"
        },

        {
            "value": "ONLINE",

            "title": "Online Payment",

            "subtitle":
                "UPI, Card or Net Banking"
        }

    ]


# ============================================================
# FIND DOCTOR BY USER INPUT
# ============================================================

def _find_selected_doctor(
    state,
    value
):

    doctors = state.get(
        "doctors",
        []
    )

    if not doctors:

        return None

    choice = parse_number_choice(
        value,
        maximum=len(doctors)
    )

    if choice:

        return doctors[
            choice - 1
        ]

    text = _normalize(value)

    for doctor in doctors:

        name = _normalize(
            doctor.get(
                "doctor_name",
                ""
            )
        )

        if text == name:

            return doctor

        if text in name:

            return doctor

    # Fuzzy name matching.

    best_doctor = None
    best_score = 0

    for doctor in doctors:

        name = _normalize(
            doctor.get(
                "doctor_name",
                ""
            )
        )

        score = SequenceMatcher(
            None,
            text,
            name
        ).ratio()

        if score > best_score:

            best_score = score
            best_doctor = doctor

    if best_score >= 0.60:

        return best_doctor

    return None


# ============================================================
# FIND SELECTED DATE
# ============================================================

def _find_selected_date(
    state,
    value
):

    available_dates = state.get(
        "available_dates",
        []
    )

    if not available_dates:

        return None

    choice = parse_number_choice(
        value,
        maximum=len(available_dates)
    )

    if choice:

        return _date_from_value(
            available_dates[
                choice - 1
            ]
        )

    parsed = _date_from_value(
        value
    )

    if parsed:

        for available_date in available_dates:

            current = _date_from_value(
                available_date
            )

            if current == parsed:

                return current

    text = _normalize(value)

    for available_date in available_dates:

        current = _date_from_value(
            available_date
        )

        if current:

            formatted = _normalize(
                current.strftime(
                    "%Y-%m-%d"
                )
            )

            if text == formatted:

                return current

    return None


# ============================================================
# FIND SELECTED TIME
# ============================================================

def _find_selected_time(
    state,
    value
):

    slots = state.get(
        "available_slots",
        []
    )

    if not slots:
        return None

    choice = parse_number_choice(
        value,
        maximum=len(slots)
    )

    if choice:
        return _get_start_time(
            slots[choice - 1]
        )

    text = _normalize(value)

    # Accept both 10:00 AM and 10 AM / 10am.
    time_formats = [
        "%H:%M",
        "%H:%M:%S",
        "%H",
        "%I:%M %p",
        "%I:%M%p",
        "%I %p",
        "%I%p"
    ]

    parsed_time = None

    for fmt in time_formats:
        try:
            parsed_time = datetime.strptime(
                text.upper(),
                fmt
            ).time()
            break
        except ValueError:
            continue

    if not parsed_time:
        return None

    # Match an exact slot start first.
    for slot in slots:
        start = _get_start_time(slot)
        if start == parsed_time:
            return start

    # Some availability rows represent a continuous window
    # such as 09:00 AM - 12:00 PM. In that case accept a time
    # inside the real availability window.
    for slot in slots:
        start = _get_start_time(slot)
        end = _get_end_time(slot)

        if not start or not end:
            continue

        if start <= parsed_time <= end:
            return parsed_time

    return None


# ============================================================
# SPECIALTY BOOKING START
# ============================================================

async def _start_appointment_booking(
    message,
    intent_data,
    user
):

    state = start_booking(user)

    specialty = None

    # Get specialty from the router first.
    if isinstance(
        intent_data,
        dict
    ):

        specialty = intent_data.get(
            "specialty"
        )

    if specialty:

        specialty = normalize_specialty(
            specialty
        )

        if specialty:

            return await _select_doctor(
                specialty,
                user
            )

    # Detect specialty directly from the user's message.
    specialty = normalize_specialty(
        message
    )

    if specialty:

        return await _select_doctor(
            specialty,
            user
        )

    state["step"] = "specialty"

    return chatbot_response(
        "Which type of doctor would you "
        "like to book an appointment with?",
        "specialty_selection",
        [
            {
                "value": "Cardiologist",
                "title": "Cardiologist",
                "subtitle":
                    "Heart specialist"
            },
            {
                "value": "Neurologist",
                "title": "Neurologist",
                "subtitle":
                    "Brain and nervous system"
            },
            {
                "value": "Dermatologist",
                "title": "Dermatologist",
                "subtitle":
                    "Skin specialist"
            },
            {
                "value": "Pediatrician",
                "title": "Pediatrician",
                "subtitle":
                    "Child specialist"
            },
            {
                "value": "Orthopedic",
                "title": "Orthopedic",
                "subtitle":
                    "Bone and joint specialist"
            }
        ]
    )


# ============================================================
# SELECT DOCTOR
# ============================================================

async def _select_doctor(
    specialty,
    user
):

    specialty = normalize_specialty(
        specialty
    )

    if not specialty:

        return chatbot_response(
            "Please tell me the type of "
            "doctor you need."
        )

    doctors = find_doctors_by_specialization(
        specialty
    )

    if not doctors:

        state = get_booking_state(user)

        state["step"] = "specialty"

        return chatbot_response(
            f"I couldn't find any available "
            f"{specialty} right now. "
            f"Please try another specialization."
        )

    state = get_booking_state(user)

    state["active"] = True

    state["specialty"] = specialty

    state["doctors"] = doctors

    state["step"] = "doctor"

    booking_state.clear()
    booking_state.update(state)

    options = _build_doctor_options(
        doctors
    )

    return chatbot_response(
        f"Here are the available "
        f"{specialty}s. Please choose a doctor.",
        "doctor_selection",
        options
    )


# ============================================================
# HANDLE DOCTOR SELECTION
# ============================================================

async def _handle_doctor_selection(
    message,
    user
):

    state = get_booking_state(user)

    doctor = _find_selected_doctor(
        state,
        message
    )

    if not doctor:

        return chatbot_response(
            "Please choose a doctor by "
            "typing the doctor number or "
            "doctor name."
        )

    doctor_id = doctor.get(
        "doctor_id"
    )

    if not doctor_id:

        return chatbot_response(
            "I could not identify that doctor. "
            "Please choose another doctor."
        )

    doctor_name = _clean_doctor_name(
        doctor.get(
            "doctor_name"
        )
    )

    state["doctor_id"] = doctor_id

    state["doctor_name"] = doctor_name

    # Get real availability from database.

    availability = get_doctor_availability(
        doctor_id
    )

    # Never show a date/time that's already booked.
    availability = _filter_booked_slots(
        doctor_id,
        availability
    )

    if not availability:

        state["step"] = "doctor"

        return chatbot_response(
            f"{doctor_name} currently has "
            f"no available appointment dates."
        )

    available_dates = []

    for row in availability:

        available_date = (
            _get_date_from_availability(
                row
            )
        )

        if available_date:

            if available_date not in available_dates:

                available_dates.append(
                    available_date
                )

    available_dates.sort()

    if not available_dates:

        state["step"] = "doctor"

        return chatbot_response(
            f"I couldn't find available dates "
            f"for {doctor_name}."
        )

    state["available_dates"] = (
        available_dates
    )

    state["available_slots"] = (
        availability
    )

    state["step"] = "date"

    booking_state.clear()
    booking_state.update(state)

    options = _build_date_options(
        available_dates
    )

    return chatbot_response(
        f"You selected {doctor_name}. "
        f"Please choose an available date.",
        "date_selection",
        options
    )


# ============================================================
# HANDLE DATE SELECTION
# ============================================================

async def _handle_date_selection(
    message,
    user
):

    state = get_booking_state(user)

    selected_date = _find_selected_date(
        state,
        message
    )

    if not selected_date:

        return chatbot_response(
            "Please choose a valid date "
            "from the available options."
        )

    state["appointment_date"] = (
        selected_date
    )

    doctor_id = state.get(
        "doctor_id"
    )

    if not doctor_id:

        reset_booking(user)

        return chatbot_response(
            "Let's start the appointment "
            "booking again. Please choose "
            "a doctor."
        )

    availability = get_doctor_availability(
        doctor_id
    )

    matching_slots = []

    for row in availability:

        row_date = (
            _get_date_from_availability(
                row
            )
        )

        if row_date == selected_date:

            matching_slots.append(
                row
            )

    # Never show a time that's already booked.
    matching_slots = _filter_booked_slots(
        doctor_id,
        matching_slots
    )

    if not matching_slots:

        state["step"] = "date"

        return chatbot_response(
            "There are no available times "
            "for that date. Please choose "
            "another date."
        )

    state["available_slots"] = (
        matching_slots
    )

    state["step"] = "time"

    booking_state.clear()
    booking_state.update(state)

    options = _build_time_options(
        matching_slots
    )

    formatted_date = (
        selected_date.strftime(
            "%d %b %Y"
        )
    )

    return chatbot_response(
        f"Dr. {state['doctor_name'].replace('Dr. ', '')} "
        f"is available on {formatted_date}. "
        f"Please choose an appointment time.",
        "time_selection",
        options
    )


# ============================================================
# HANDLE TIME SELECTION
# ============================================================

async def _handle_time_selection(
    message,
    user
):

    state = get_booking_state(user)

    selected_time = _find_selected_time(
        state,
        message
    )

    if not selected_time:

        return chatbot_response(
            "Please choose a valid appointment "
            "time from the available options."
        )

    doctor_id = state.get("doctor_id")
    appointment_date = state.get("appointment_date")

    # Defensive re-check: the slot was free when listed,
    # but may have just been taken by someone else in the
    # meantime. Catch that HERE, before collecting name,
    # phone, and payment method all over again.
    try:
        already_booked = is_slot_booked(
            doctor_id,
            appointment_date,
            selected_time
        )
    except Exception:
        already_booked = False

    if already_booked:

        remaining_slots = _filter_booked_slots(
            doctor_id,
            state.get("available_slots", [])
        )

        state["available_slots"] = remaining_slots

        booking_state.clear()
        booking_state.update(state)

        if not remaining_slots:

            state["step"] = "date"

            return chatbot_response(
                "Sorry, that was the last "
                "available time for this date "
                "and it was just booked. Please "
                "choose another date.",
                "date_selection",
                _build_date_options(
                    state.get("available_dates", [])
                )
            )

        return chatbot_response(
            "Sorry, that time was just booked "
            "by someone else. Please choose "
            "another available time.",
            "time_selection",
            _build_time_options(remaining_slots)
        )

    state["appointment_time"] = (
        selected_time
    )

    state["step"] = "name"

    booking_state.clear()
    booking_state.update(state)

    return chatbot_response(
        "What is your full name?"
    )


# ============================================================
# HANDLE PATIENT NAME
# ============================================================

async def _handle_name(
    message,
    user
):

    name = _clean_text(message)

    if len(name) < 2:

        return chatbot_response(
            "Please enter your full name."
        )

    state = get_booking_state(user)

    state["patient_name"] = name

    state["step"] = "phone"

    booking_state.clear()
    booking_state.update(state)

    return chatbot_response(
        "Please provide your 10-digit "
        "phone number."
    )


# ============================================================
# HANDLE PHONE
# ============================================================

async def _handle_phone(
    message,
    user
):

    phone = re.sub(
        r"\D",
        "",
        _clean_text(message)
    )

    if len(phone) != 10:

        return chatbot_response(
            "Please enter a valid 10-digit "
            "phone number."
        )

    state = get_booking_state(user)

    state["phone"] = phone

    state["step"] = "payment"

    booking_state.clear()
    booking_state.update(state)

    return chatbot_response(
        "How would you like to pay?",
        "payment_selection",
        _payment_options()
    )


# ============================================================
# HANDLE PAYMENT
# ============================================================

async def _handle_payment(
    message,
    user
):

    text = _normalize(message)

    payment_method = None

    if text in [
        "cash",
        "cash at clinic",
        "1",
        "one"
    ]:

        payment_method = "CASH"

    elif text in [
        "online",
        "online payment",
        "upi",
        "card",
        "net banking",
        "netbanking",
        "2",
        "two"
    ]:

        payment_method = "ONLINE"

    else:

        # Fuzzy matching.

        if "cash" in text:

            payment_method = "CASH"

        elif (
            "online" in text
            or "upi" in text
            or "card" in text
        ):

            payment_method = "ONLINE"

    if not payment_method:

        return chatbot_response(
            "Please choose Cash at Clinic "
            "or Online Payment."
        )

    state = get_booking_state(user)

    state["payment_method"] = (
        payment_method
    )

    return await _finalize_booking(
        user
    )


# ============================================================
# FINALIZE APPOINTMENT
# ============================================================

async def _finalize_booking(user):

    state = get_booking_state(user)

    doctor_id = state.get(
        "doctor_id"
    )

    appointment_date = state.get(
        "appointment_date"
    )

    appointment_time = state.get(
        "appointment_time"
    )

    patient_name = state.get(
        "patient_name"
    )

    phone = state.get(
        "phone"
    )

    payment_method = state.get(
        "payment_method"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not doctor_id:

        return chatbot_response(
            "Doctor information is missing. "
            "Please start the booking again."
        )

    if not appointment_date:

        return chatbot_response(
            "Appointment date is missing. "
            "Please start the booking again."
        )

    if not appointment_time:

        return chatbot_response(
            "Appointment time is missing. "
            "Please start the booking again."
        )

    if not patient_name:

        return chatbot_response(
            "Patient name is missing."
        )

    if not phone:

        return chatbot_response(
            "Phone number is missing."
        )

    if payment_method not in [
        "CASH",
        "ONLINE"
    ]:

        return chatbot_response(
            "Please select a valid payment method."
        )

    # --------------------------------------------------------
    # Verify doctor exists
    # --------------------------------------------------------

    doctor = get_doctor_by_id(
        doctor_id
    )

    if not doctor:

        reset_booking(user)

        return chatbot_response(
            "The selected doctor is no longer "
            "available. Please start again."
        )

    # --------------------------------------------------------
    # Verify availability again
    # --------------------------------------------------------

    availability = get_doctor_availability(
        doctor_id
    )

    valid_time = False

    for row in availability:

        row_date = (
            _get_date_from_availability(
                row
            )
        )

        start_time = _get_start_time(
            row
        )

        end_time = _get_end_time(
            row
        )

        if (
            row_date == appointment_date
            and (
                start_time == appointment_time
                or (
                    start_time
                    and end_time
                    and start_time <= appointment_time <= end_time
                )
            )
        ):

            valid_time = True
            break

    if not valid_time:

        return chatbot_response(
            "That appointment time is no longer "
            "available. Please choose another time."
        )

    # --------------------------------------------------------
    # Duplicate booking check
    # --------------------------------------------------------

    try:

        already_booked = is_slot_booked(
            doctor_id,
            appointment_date,
            appointment_time
        )

    except TypeError:

        already_booked = is_slot_booked(
            doctor_id=doctor_id,
            appointment_date=appointment_date,
            appointment_time=appointment_time
        )

    if already_booked:

        return chatbot_response(
            "Sorry, that appointment slot has "
            "already been booked. Please choose "
            "another time."
        )

    # --------------------------------------------------------
    # Create appointment
    # --------------------------------------------------------

    try:

        appointment_id = create_appointment(
            user_id=user["user_id"],

            doctor_id=doctor_id,

            appointment_date=appointment_date,

            appointment_time=appointment_time,

            patient_name=patient_name,

            phone=phone,

            payment_method=payment_method
        )

    except TypeError:

        # Compatibility with older repository
        # function signatures.

        appointment_id = create_appointment(
            user["user_id"],
            doctor_id,
            appointment_date,
            appointment_time,
            patient_name,
            phone,
            payment_method
        )

    # --------------------------------------------------------
    # Format response
    # --------------------------------------------------------

    doctor_name = _clean_doctor_name(
        doctor.get(
            "doctor_name"
        )
    )

    fee = _format_money(
        doctor.get(
            "consultation_fee",
            0
        )
    )

    formatted_date = (
        appointment_date.strftime(
            "%d %b %Y"
        )
        if isinstance(
            appointment_date,
            date
        )
        else str(
            appointment_date
        )
    )

    formatted_time = _format_time(
        appointment_time
    )

    # --------------------------------------------------------
    # Reset booking state
    # --------------------------------------------------------

    reset_booking(user)

    # --------------------------------------------------------
    # Cash
    # --------------------------------------------------------

    if payment_method == "CASH":

        return chatbot_response(
            (
                "Appointment booked successfully.\n\n"
                f"Appointment ID: {appointment_id}\n"
                f"Doctor: {doctor_name}\n"
                f"Date: {formatted_date}\n"
                f"Time: {formatted_time}\n"
                f"Amount: {fee}\n"
                "Payment: Cash at Clinic"
            ),
            "booking_success"
        )

    # --------------------------------------------------------
    # Online (Cashfree)
    #
    # Mirrors the same create_cashfree_order() +
    # save_cashfree_order() calls the existing web booking
    # page (main.py) already uses successfully, so the
    # appointment ends up with a real cashfree_order_id and
    # payment_session_id either way - not just when booked
    # from the web form.
    # --------------------------------------------------------

    if not CASHFREE_CONFIGURED:

        return chatbot_response(
            (
                "Appointment created successfully.\n\n"
                f"Appointment ID: {appointment_id}\n"
                f"Doctor: {doctor_name}\n"
                f"Date: {formatted_date}\n"
                f"Time: {formatted_time}\n"
                f"Amount: {fee}\n\n"
                "Online payment isn't configured on this "
                "clinic yet. Please pay cash at the clinic, "
                "or ask the clinic to enable online payment."
            ),
            "booking_success"
        )

    try:

        cashfree_order_id = (
            f"appointment_{appointment_id}"
        )

        return_url = (
            "http://127.0.0.1:8000"
            "/payments/cashfree/return"
            f"?order_id={cashfree_order_id}"
        )

        cashfree_order = create_cashfree_order(
            order_id=cashfree_order_id,
            amount=float(
                doctor.get(
                    "consultation_fee",
                    0
                )
            ),
            customer_id=str(
                user["user_id"]
            ),
            customer_name=patient_name,
            customer_email=user.get(
                "email",
                ""
            ),
            customer_phone=phone,
            return_url=return_url
        )

        save_cashfree_order(
            appointment_id=appointment_id,
            user_id=user["user_id"],
            cashfree_order_id=cashfree_order["order_id"],
            payment_session_id=(
                cashfree_order["payment_session_id"]
            )
        )

        pay_link = (
            f"/appointments/{appointment_id}/pay"
        )

        return chatbot_response(
            (
                "Appointment created successfully.\n\n"
                f"Appointment ID: {appointment_id}\n"
                f"Doctor: {doctor_name}\n"
                f"Date: {formatted_date}\n"
                f"Time: {formatted_time}\n"
                f"Amount: {fee}\n\n"
                "Tap below to complete your payment."
            ),
            "booking_success",
            [
                {
                    "value": pay_link,
                    "title": "💳 Pay Now",
                    "subtitle": fee,
                    "url": pay_link
                }
            ]
        )

    except Exception as e:

        import logging
        import traceback

        logging.getLogger(__name__).error(
            "Chat-initiated Cashfree order failed "
            "for appointment_id=%s: %s",
            appointment_id,
            repr(e)
        )

        traceback.print_exc()

        return chatbot_response(
            (
                "Appointment created successfully.\n\n"
                f"Appointment ID: {appointment_id}\n"
                f"Doctor: {doctor_name}\n"
                f"Date: {formatted_date}\n"
                f"Time: {formatted_time}\n"
                f"Amount: {fee}\n\n"
                "I couldn't start online payment right now. "
                "Please try again from your appointment "
                "page, or pay cash at the clinic."
            ),
            "booking_success"
        )


# ============================================================
# BOOKING MESSAGE HANDLER
# ============================================================

async def handle_booking_message(
    message,
    user,
    intent_data=None
):

    state = get_booking_state(user)

    if not state.get("active"):

        start_booking(user)

        state = get_booking_state(user)

    step = state.get(
        "step"
    )

    # --------------------------------------------------------
    # Allow the patient to change specialization while
    # selecting a doctor. Examples:
    #   any other neurologist
    #   i need a neurologist
    #   actually i want a dermatologist
    # --------------------------------------------------------

    new_specialty = normalize_specialty(
        message
    )

    if (
        new_specialty
        and step in (
            "specialty",
            "doctor"
        )
        and not parse_number_choice(message)
    ):

        return await _select_doctor(
            new_specialty,
            user
        )

    # --------------------------------------------------------
    # SPECIALTY
    # --------------------------------------------------------

    if step == "specialty":

        specialty = None

        if isinstance(
            intent_data,
            dict
        ):

            specialty = intent_data.get(
                "specialty"
            )

        if not specialty:

            specialty = normalize_specialty(
                message
            )

        # If the message is still a generic booking request,
        # ask the patient to choose a specialty.
        if (
            not specialty
            or _normalize(message)
            in [
                "book",
                "booking",
                "appointment",
                "book appointment",
                "i want to book an appointment"
            ]
        ):

            return chatbot_response(
                "Which type of doctor would you "
                "like to book an appointment with?",
                "specialty_selection",
                [
                    {
                        "value": "Cardiologist",
                        "title": "Cardiologist",
                        "subtitle":
                            "Heart specialist"
                    },
                    {
                        "value": "Neurologist",
                        "title": "Neurologist",
                        "subtitle":
                            "Brain and nervous system"
                    },
                    {
                        "value": "Dermatologist",
                        "title": "Dermatologist",
                        "subtitle":
                            "Skin specialist"
                    },
                    {
                        "value": "Pediatrician",
                        "title": "Pediatrician",
                        "subtitle":
                            "Child specialist"
                    },
                    {
                        "value": "Orthopedic",
                        "title": "Orthopedic",
                        "subtitle":
                            "Bone and joint specialist"
                    }
                ]
            )

        return await _select_doctor(
            specialty,
            user
        )

    # --------------------------------------------------------
    # DOCTOR
    # --------------------------------------------------------

    if step == "doctor":

        return await _handle_doctor_selection(
            message,
            user
        )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if step == "date":

        return await _handle_date_selection(
            message,
            user
        )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if step == "time":

        return await _handle_time_selection(
            message,
            user
        )

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    if step == "name":

        return await _handle_name(
            message,
            user
        )

    # --------------------------------------------------------
    # PHONE
    # --------------------------------------------------------

    if step == "phone":

        return await _handle_phone(
            message,
            user
        )

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    if step == "payment":

        return await _handle_payment(
            message,
            user
        )

    # --------------------------------------------------------
    # Unknown state
    # --------------------------------------------------------

    reset_booking(user)

    return chatbot_response(
        "Let's start the appointment booking "
        "again. Which type of doctor would "
        "you like to see?"
    )


# ============================================================
# DOCTOR SEARCH
# ============================================================

async def handle_doctor_search(
    message,
    intent_data,
    user=None
):

    specialty = None

    if isinstance(
        intent_data,
        dict
    ):

        specialty = intent_data.get(
            "specialty"
        )

    if not specialty:

        specialty = normalize_specialty(message)

    doctors = find_doctors_by_specialization(
        specialty
    )

    if not doctors:

        return chatbot_response(
            f"I couldn't find any available "
            f"{specialty or 'doctors'}."
        )

    # ========================================================
    # IMPORTANT:
    #
    # Use the SAME option-building helper as the normal
    # booking flow (_build_doctor_options), which uses an
    # ordinal position ("1", "2", ...) as each option's
    # "value". _find_selected_doctor() - used by the booking
    # flow's doctor-selection step - expects that same ordinal
    # convention. The previous version of this function built
    # its own options using the real doctor_id as "value",
    # which would silently break or select the wrong doctor
    # once the state machine below is entered.
    # ========================================================

    options = _build_doctor_options(
        doctors
    )

    # ========================================================
    # IMPORTANT:
    #
    # Enter the same booking state machine that
    # _start_appointment_booking() uses, so that when the
    # user replies with a doctor name/number next (e.g.
    # "Dr. Neha Singh"), process_message() sees
    # state["active"] == True and routes it through
    # handle_booking_message -> _handle_doctor_selection,
    # continuing straight into date/time/booking - instead
    # of falling through to a generic LLM reply.
    # ========================================================

    state = get_booking_state(user)

    state["active"] = True
    state["specialty"] = specialty
    state["doctors"] = doctors
    state["step"] = "doctor"

    booking_state.clear()
    booking_state.update(state)

    return chatbot_response(
        f"Here are the available "
        f"{specialty}s. Please choose a doctor "
        f"if you'd like to book an appointment.",
        "doctor_search",
        options
    )


# ============================================================
# PATIENT HISTORY
# ============================================================

async def handle_patient_history(
    user
):

    try:

        appointments = get_user_appointments(
            user["user_id"]
        )

    except Exception:

        return chatbot_response(
            "I could not retrieve your "
            "appointment history right now."
        )

    if not appointments:

        return chatbot_response(
            "You don't have any appointments yet."
        )

    lines = [
        "Your recent appointments:"
    ]

    for appointment in appointments[:5]:

        doctor_name = _clean_doctor_name(
            appointment.get(
                "doctor_name",
                "Doctor"
            )
        )

        appointment_date = (
            appointment.get(
                "appointment_date"
            )
        )

        appointment_time = (
            appointment.get(
                "appointment_time"
            )
        )

        status = appointment.get(
            "status",
            "UNKNOWN"
        )

        lines.append(
            (
                f"\nAppointment #{appointment.get('appointment_id')}"
                f"\nDoctor: {doctor_name}"
                f"\nDate: {appointment_date}"
                f"\nTime: {_format_time(appointment_time)}"
                f"\nStatus: {status}"
            )
        )

    return chatbot_response(
        "\n".join(lines)
    )


# ============================================================
# PHARMACY
# ============================================================

# ============================================================
# GOOGLE MAPS - NEARBY PHARMACY SEARCH
# ============================================================

GOOGLE_MAPS_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY"
)


def find_nearby_pharmacies(
    latitude,
    longitude,
    radius=5000,
    max_results=5
):
    """
    Find nearby pharmacies using Google Places API (New).

    Parameters:
        latitude    : User latitude.
        longitude   : User longitude.
        radius      : Search radius in meters.
        max_results : Maximum number of pharmacies to return.

    Returns:
        A list of dictionaries containing pharmacy name,
        address, phone, coordinates and Google Maps URL.
    """

    if not GOOGLE_MAPS_API_KEY:

        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not configured. "
            "Add it to your .env file."
        )

    try:
        latitude = float(latitude)
        longitude = float(longitude)
        radius = float(radius)
        max_results = int(max_results)
    except (TypeError, ValueError):

        raise ValueError(
            "Latitude, longitude, radius and max_results "
            "must contain valid numeric values."
        )

    if not -90 <= latitude <= 90:

        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if not -180 <= longitude <= 180:

        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    radius = max(1.0, min(radius, 50000.0))
    max_results = max(1, min(max_results, 20))

    url = (
        "https://places.googleapis.com/v1/"
        "places:searchNearby"
    )

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,"
            "places.formattedAddress,"
            "places.location,"
            "places.googleMapsUri,"
            "places.nationalPhoneNumber"
        )
    }

    payload = {
        "includedTypes": [
            "pharmacy"
        ],
        "maxResultCount": max_results,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "radius": radius
            }
        },
        "regionCode": "IN",
        "languageCode": "en"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=10
        )
    except requests.RequestException as e:

        raise RuntimeError(
            f"Unable to connect to Google Places API: {e}"
        ) from e

    if not response.ok:

        try:
            error_details = response.json()
        except ValueError:
            error_details = response.text

        raise RuntimeError(
            "Google Places API request failed "
            f"({response.status_code}): {error_details}"
        )

    data = response.json()
    pharmacies = []

    for place in data.get("places", []):

        display_name = place.get(
            "displayName",
            {}
        )

        location = place.get(
            "location",
            {}
        )

        pharmacies.append(
            {
                "name": display_name.get(
                    "text",
                    "Pharmacy"
                ),
                "address": place.get(
                    "formattedAddress",
                    "Address unavailable"
                ),
                "latitude": location.get(
                    "latitude"
                ),
                "longitude": location.get(
                    "longitude"
                ),
                "phone": place.get(
                    "nationalPhoneNumber"
                ),
                "google_maps_url": place.get(
                    "googleMapsUri"
                )
            }
        )

    return pharmacies


# ============================================================
# SPECIAL CHAT COMMAND - NEARBY PHARMACY
# ============================================================

def _handle_nearby_pharmacy_command(message):
    """
    Handle a frontend-generated command carrying the user's
    browser latitude/longitude.

    Format:
        __NEARBY_PHARMACY__|latitude|longitude
    """

    prefix = "__NEARBY_PHARMACY__|"

    if not message.startswith(prefix):
        return None

    payload = message[len(prefix):].split("|", 1)

    if len(payload) != 2:
        return chatbot_response(
            "I could not read your location. Please try again."
        )

    latitude, longitude = payload

    try:
        pharmacies = find_nearby_pharmacies(
            latitude=latitude,
            longitude=longitude,
            radius=5000,
            max_results=5
        )
    except Exception as e:
        logging.getLogger(__name__).exception(
            "Nearby pharmacy Google Places search failed: %s",
            e
        )

        return chatbot_response(
            "I could not find nearby pharmacies right now. Please try again."
        )

    if not pharmacies:
        return chatbot_response(
            "I could not find any nearby pharmacies within 5 km."
        )

    options = []

    for pharmacy in pharmacies:

        name = pharmacy.get(
            "name",
            "Pharmacy"
        )

        address = pharmacy.get(
            "address",
            "Address unavailable"
        )

        phone = pharmacy.get("phone")

        subtitle = address

        if phone:
            subtitle = f"{address} • {phone}"

        option = {
            "value": name,
            "title": f"📍 {name}",
            "subtitle": subtitle
        }

        maps_url = pharmacy.get(
            "google_maps_url"
        )

        if maps_url:
            option["url"] = maps_url

        options.append(option)

    return chatbot_response(
        "Here are nearby pharmacies. Select one to open its location in Google Maps.",
        "nearby_pharmacy",
        options
    )


def _clean_medicine_query(
    message,
    intent_data=None
):

    medicine = None

    if isinstance(intent_data, dict):
        medicine = intent_data.get("medicine")

    if not medicine:
        medicine = _clean_text(message)

    # Remove common request phrases.
    medicine = re.sub(
        r"""\b(
        i\s+need|
        i\s+want|
        can\s+i\s+get|
        please\s+give\s+me|
        buy|
        order|
        medicine|
        medicines|
        pharmacy|
        tablet|
        tablets|
        pill|
        pills
        )\b""",
        "",
        medicine,
        flags=re.IGNORECASE | re.VERBOSE
    )

    medicine = re.sub(
        r"\b(some|any|a|an|to|for|me|please|the|of)\b",
        "",
        medicine,
        flags=re.IGNORECASE
    )

    medicine = re.sub(
        r"\s+",
        " ",
        medicine
    ).strip()

    return medicine


def _medicine_similarity(
    query,
    medicine
):

    query = _normalize(query)

    if not query:
        return 0.0

    names = [
        medicine.get("medicine_name", ""),
        medicine.get("generic_name", "")
    ]

    best_score = 0.0

    for name in names:

        name_text = _normalize(name)

        if not name_text:
            continue

        # Compare against complete name and individual words.
        scores = [
            SequenceMatcher(
                None,
                query,
                name_text
            ).ratio()
        ]

        for word in name_text.split():
            cleaned_word = re.sub(
                r"[^a-z0-9]+",
                "",
                word
            )

            if cleaned_word:
                scores.append(
                    SequenceMatcher(
                        None,
                        query,
                        cleaned_word
                    ).ratio()
                )

        best_score = max(
            best_score,
            *scores
        )

    return best_score


def _find_fuzzy_medicine(
    query
):

    if not query:
        return None

    try:
        medicines = get_all_medicines()
    except Exception:
        return None

    best_item = None
    best_score = 0.0

    for item in medicines or []:

        score = _medicine_similarity(
            query,
            item
        )

        if score > best_score:
            best_score = score
            best_item = item

    # Do not guess on very short or weak matches.
    if len(_normalize(query)) >= 5 and best_score >= 0.70:
        return best_item

    return None


def _looks_like_pharmacy_request(
    message
):

    text = _normalize(message)

    pharmacy_words = [
        "medicine",
        "medicines",
        "pharmacy",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "syrup",
        "ointment",
        "pill",
        "pills",
        "drug"
    ]

    if any(
        word in text.split()
        for word in pharmacy_words
    ):
        return True

    # Also catch misspelled medicine names such as
    # "paracitamol".
    return _find_fuzzy_medicine(text) is not None


async def handle_pharmacy(
    message,
    intent_data,
    user=None
):

    medicine = _clean_medicine_query(
        message,
        intent_data
    )

    # Generic request: do not search for words like
    # "some" or "medicines" as the medicine name.
    if not medicine:

        return chatbot_response(
            "Which medicine would you like to search for?"
        )

    try:
        medicines = search_medicines(
            medicine
        )
    except Exception:
        medicines = []

    # Search failed or no exact/partial DB result.
    # Try typo-tolerant matching against the real medicine list.
    if not medicines:

        fuzzy_medicine = _find_fuzzy_medicine(
            medicine
        )

        if fuzzy_medicine:
            medicines = [
                fuzzy_medicine
            ]

    # A medicine is considered available only when at least one
    # matching item has positive stock.
    available_medicines = [
        item
        for item in (medicines or [])
        if float(item.get("stock_quantity", 0) or 0) > 0
    ]

    if not available_medicines:

        return chatbot_response(
            (
                f"'{medicine}' is not available at Sanjeevani Clinic.\n\n"
                "You can find nearby pharmacies using your current location."
            ),
            "pharmacy_unavailable",
            [
                {
                    "value": "NEARBY_PHARMACY",
                    "title": "📍 Find Nearby Pharmacy",
                    "subtitle": "Use my current location"
                }
            ]
        )

    medicines = available_medicines

    options = []

    for item in medicines[:10]:

        stock = item.get(
            "stock_quantity",
            0
        )

        availability = (
            "In stock"
            if stock > 0
            else "Out of stock"
        )

        options.append(
            {
                "value": str(
                    item.get(
                        "medicine_id"
                    )
                ),
                "title": item.get(
                    "medicine_name",
                    "Medicine"
                ),
                "subtitle": availability,
                "price": _format_money(
                    item.get(
                        "price",
                        0
                    )
                )
            }
        )

    display_name = item.get(
        "medicine_name",
        medicine
    )

    if len(medicines) == 1:
        response_text = (
            f"I found {display_name}. "
            "Please select it to add it to your cart."
        )
    else:
        response_text = (
            f"Here are the medicines matching "
            f"'{medicine}'. Select one to add it "
            f"to your cart."
        )

    # ========================================================
    # Remember these results so that the user's NEXT message
    # (clicking an option, or typing the medicine name/number)
    # can be resolved as an add-to-cart action instead of
    # falling through to a generic LLM reply.
    # ========================================================

    pending = _get_pending_state(user)
    pending["action"] = "PHARMACY_SELECT"
    pending["medicines"] = medicines[:10]

    return chatbot_response(
        response_text,
        "medicine_selection",
        options
    )


# ============================================================
# APPOINTMENT FORMATTING HELPER
# ============================================================

def _describe_appointment(appointment):

    doctor_name = _clean_doctor_name(
        appointment.get(
            "doctor_name",
            "Doctor"
        )
    )

    appointment_date = appointment.get(
        "appointment_date"
    )

    appointment_time = appointment.get(
        "appointment_time"
    )

    return (
        f"{doctor_name} on {appointment_date} "
        f"at {_format_time(appointment_time)}"
    )


# ============================================================
# CANCEL APPOINTMENT
# ============================================================

async def handle_cancel_appointment(
    message,
    user
):

    try:
        appointments = get_user_appointments(
            user["user_id"]
        )
    except Exception:
        return chatbot_response(
            "I could not retrieve your appointments "
            "right now. Please try again."
        )

    active_appointments = [
        a for a in (appointments or [])
        if str(a.get("status", "")).upper()
        in ("BOOKED", "CONFIRMED")
    ]

    if not active_appointments:

        return chatbot_response(
            "You don't have any active "
            "appointments to cancel."
        )

    if len(active_appointments) == 1:

        appt = active_appointments[0]

        pending = _get_pending_state(user)
        pending["action"] = "CANCEL_CONFIRM"
        pending["appointment_id"] = appt.get(
            "appointment_id"
        )
        pending["appointments"] = [appt]

        return chatbot_response(
            f"Cancel your appointment with "
            f"{_describe_appointment(appt)}? "
            f"Reply 'yes' to confirm or 'no' "
            f"to keep it."
        )

    pending = _get_pending_state(user)
    pending["action"] = "CANCEL_SELECT"
    pending["appointments"] = active_appointments

    lines = [
        "Which appointment would you like to "
        "cancel? Reply with the number:"
    ]

    for index, appt in enumerate(
        active_appointments,
        start=1
    ):
        lines.append(
            f"{index}. {_describe_appointment(appt)} "
            f"({appt.get('status')})"
        )

    return chatbot_response(
        "\n".join(lines)
    )


# ============================================================
# RESCHEDULE APPOINTMENT
# ============================================================

async def handle_reschedule_appointment(
    message,
    user
):

    try:
        appointments = get_user_appointments(
            user["user_id"]
        )
    except Exception:
        return chatbot_response(
            "I could not retrieve your appointments "
            "right now. Please try again."
        )

    active_appointments = [
        a for a in (appointments or [])
        if str(a.get("status", "")).upper()
        in ("BOOKED", "CONFIRMED")
    ]

    if not active_appointments:

        return chatbot_response(
            "You don't have any active "
            "appointments to reschedule."
        )

    if len(active_appointments) == 1:

        return _begin_reschedule_date_step(
            active_appointments[0],
            user
        )

    pending = _get_pending_state(user)
    pending["action"] = "RESCHEDULE_SELECT"
    pending["appointments"] = active_appointments

    lines = [
        "Which appointment would you like to "
        "reschedule? Reply with the number:"
    ]

    for index, appt in enumerate(
        active_appointments,
        start=1
    ):
        lines.append(
            f"{index}. {_describe_appointment(appt)} "
            f"({appt.get('status')})"
        )

    return chatbot_response(
        "\n".join(lines)
    )


def _begin_reschedule_date_step(
    appointment,
    user
):

    doctor_id = appointment.get("doctor_id")

    try:
        availability = get_doctor_availability(
            doctor_id
        )
    except Exception:
        availability = []

    # Never show a date/time that's already booked.
    availability = _filter_booked_slots(
        doctor_id,
        availability
    )

    if not availability:

        _reset_pending_state(user)

        return chatbot_response(
            "This doctor currently has no "
            "available slots to reschedule into. "
            "Please try again later."
        )

    available_dates = []

    for row in availability:

        available_date = _get_date_from_availability(
            row
        )

        if available_date and (
            available_date not in available_dates
        ):
            available_dates.append(available_date)

    available_dates.sort()

    if not available_dates:

        _reset_pending_state(user)

        return chatbot_response(
            "I couldn't find available dates "
            "for this doctor right now."
        )

    pending = _get_pending_state(user)
    pending["action"] = "RESCHEDULE_DATE"
    pending["appointment_id"] = appointment.get(
        "appointment_id"
    )
    pending["doctor_id"] = doctor_id
    pending["doctor_name"] = appointment.get(
        "doctor_name"
    )
    pending["available_dates"] = available_dates
    pending["available_slots"] = availability

    options = _build_date_options(
        available_dates
    )

    doctor_name = _clean_doctor_name(
        appointment.get(
            "doctor_name",
            "the doctor"
        )
    )

    return chatbot_response(
        f"Rescheduling your appointment with "
        f"{doctor_name}. Please choose a new date.",
        "date_selection",
        options
    )


# ============================================================
# PENDING ACTION MESSAGE HANDLER
# (cancel / reschedule / pharmacy-selection follow-ups)
# ============================================================

def _find_pending_medicine(
    medicines,
    value
):

    if not medicines:
        return None

    text = _normalize(value)

    if not text:
        return None

    # Direct medicine_id match (e.g. option button click).

    if text.isdigit():

        target_id = int(text)

        for item in medicines:

            if item.get("medicine_id") == target_id:
                return item

    # Exact or partial name match.

    for item in medicines:

        name = _normalize(
            item.get("medicine_name", "")
        )

        if not name:
            continue

        if text == name or text in name:
            return item

    # Fuzzy fallback.

    best_item = None
    best_score = 0.0

    for item in medicines:

        name = _normalize(
            item.get("medicine_name", "")
        )

        if not name:
            continue

        score = SequenceMatcher(
            None,
            text,
            name
        ).ratio()

        if score > best_score:
            best_score = score
            best_item = item

    if best_item and best_score >= 0.6:
        return best_item

    return None


async def _handle_pending_action_message(
    message,
    user
):

    """
    Handles a follow-up reply while a cancel, reschedule, or
    pharmacy-selection flow is in progress for this user.

    Returns a chatbot_response dict if the message was
    consumed by the pending flow, or None if the pending
    state should be abandoned and the message should be
    processed normally instead (e.g. the user typed something
    unrelated, or explicitly asked to stop).
    """

    pending = _get_pending_state(user)
    action = pending.get("action")

    if not action:
        return None

    text = _normalize(message)

    if text in (
        "stop",
        "cancel this",
        "never mind",
        "nevermind",
        "menu"
    ):
        _reset_pending_state(user)

        return chatbot_response(
            "No problem, let me know if you "
            "need anything else."
        )

    # --------------------------------------------------------
    # PHARMACY SELECTION
    # --------------------------------------------------------

    if action == "PHARMACY_SELECT":

        medicine = _find_pending_medicine(
            pending.get("medicines", []),
            message
        )

        if not medicine:
            _reset_pending_state(user)
            return None

        try:
            add_to_cart(
                user["user_id"],
                medicine.get("medicine_id"),
                1
            )
        except Exception:
            _reset_pending_state(user)
            return chatbot_response(
                "I couldn't add that to your "
                "cart right now. Please try "
                "again from the Pharmacy page."
            )

        _reset_pending_state(user)

        return chatbot_response(
            f"Added {medicine.get('medicine_name')} "
            f"({_format_money(medicine.get('price', 0))}) "
            f"to your cart. Go to Pharmacy → Cart "
            f"to check out, or tell me another "
            f"medicine to add."
        )

    # --------------------------------------------------------
    # CANCEL: choose which appointment
    # --------------------------------------------------------

    if action == "CANCEL_SELECT":

        appointments = pending.get("appointments", [])

        choice = parse_number_choice(
            message,
            maximum=len(appointments)
        )

        if not choice:

            return chatbot_response(
                "Please reply with the number of "
                "the appointment you'd like to "
                "cancel."
            )

        appt = appointments[choice - 1]

        pending["action"] = "CANCEL_CONFIRM"
        pending["appointment_id"] = appt.get(
            "appointment_id"
        )

        return chatbot_response(
            f"Cancel your appointment with "
            f"{_describe_appointment(appt)}? "
            f"Reply 'yes' to confirm or 'no' "
            f"to keep it."
        )

    # --------------------------------------------------------
    # CANCEL: confirm
    # --------------------------------------------------------

    if action == "CANCEL_CONFIRM":

        if text in ("yes", "y", "confirm", "confirmed"):

            appointment_id = pending.get(
                "appointment_id"
            )

            try:
                success = cancel_appointment(
                    appointment_id,
                    user["user_id"]
                )
            except Exception:
                success = False

            _reset_pending_state(user)

            if success:
                return chatbot_response(
                    "Your appointment has been "
                    "cancelled."
                )

            return chatbot_response(
                "I couldn't cancel that "
                "appointment. It may have "
                "already been cancelled or "
                "completed."
            )

        if text in ("no", "n", "keep it"):

            _reset_pending_state(user)

            return chatbot_response(
                "No changes made. Your "
                "appointment is still booked."
            )

        return chatbot_response(
            "Please reply 'yes' to cancel or "
            "'no' to keep your appointment."
        )

    # --------------------------------------------------------
    # RESCHEDULE: choose which appointment
    # --------------------------------------------------------

    if action == "RESCHEDULE_SELECT":

        appointments = pending.get("appointments", [])

        choice = parse_number_choice(
            message,
            maximum=len(appointments)
        )

        if not choice:

            return chatbot_response(
                "Please reply with the number of "
                "the appointment you'd like to "
                "reschedule."
            )

        appt = appointments[choice - 1]

        return _begin_reschedule_date_step(
            appt,
            user
        )

    # --------------------------------------------------------
    # RESCHEDULE: choose new date
    # --------------------------------------------------------

    if action == "RESCHEDULE_DATE":

        selected_date = _find_selected_date(
            pending,
            message
        )

        if not selected_date:

            return chatbot_response(
                "Please choose one of the "
                "available dates."
            )

        matching_slots = [
            slot for slot in pending.get(
                "available_slots", []
            )
            if _get_date_from_availability(slot)
            == selected_date
        ]

        if not matching_slots:

            return chatbot_response(
                "No time slots found for that "
                "date. Please choose another date."
            )

        pending["appointment_date"] = selected_date
        pending["available_slots"] = matching_slots
        pending["action"] = "RESCHEDULE_TIME"

        options = _build_time_options(
            matching_slots
        )

        return chatbot_response(
            f"Please choose a new time for "
            f"{selected_date.strftime('%a, %d %b %Y')}.",
            "time_selection",
            options
        )

    # --------------------------------------------------------
    # RESCHEDULE: choose new time + execute
    # --------------------------------------------------------

    if action == "RESCHEDULE_TIME":

        selected_time = _find_selected_time(
            pending,
            message
        )

        if not selected_time:

            return chatbot_response(
                "Please choose one of the "
                "available times."
            )

        doctor_id = pending.get("doctor_id")
        appointment_date = pending.get(
            "appointment_date"
        )
        appointment_id = pending.get(
            "appointment_id"
        )

        try:
            already_booked = is_slot_booked(
                doctor_id,
                appointment_date,
                selected_time
            )
        except Exception:
            already_booked = False

        if already_booked:

            return chatbot_response(
                "That slot has just been booked "
                "by someone else. Please choose "
                "another time."
            )

        try:
            success = reschedule_appointment(
                appointment_id,
                user["user_id"],
                appointment_date,
                selected_time
            )
        except Exception:
            success = False

        _reset_pending_state(user)

        if success:

            return chatbot_response(
                f"Your appointment has been "
                f"rescheduled to "
                f"{appointment_date.strftime('%a, %d %b %Y')} "
                f"at {_format_time(selected_time)}."
            )

        return chatbot_response(
            "I couldn't reschedule that "
            "appointment. It may have already "
            "been cancelled or completed."
        )

    # Unknown pending action - clear it defensively.

    _reset_pending_state(user)

    return None


# ============================================================
# SCREENING
# ============================================================

async def handle_screening():

    return chatbot_response(
        "You can use Sanjeevani's Health "
        "Screening module for preliminary "
        "health-risk assessment.\n\n"
        "Open Health Screening from the "
        "dashboard to continue."
    )


# ============================================================
# EMERGENCY
# ============================================================

async def handle_emergency():

    return chatbot_response(
        "If you are experiencing a medical "
        "emergency, contact your local emergency "
        "service or go to the nearest emergency "
        "department immediately."
    )


# ============================================================
# MAIN PROCESS MESSAGE
# ============================================================

async def process_message(
    user_message,
    conversation_history=None,
    user=None
):

    """
    Main chatbot entry point.

    IMPORTANT:
    This function is async because the LLM/router
    functions are async.
    """

    message = _clean_text(
        user_message
    )

    if not message:

        return chatbot_response(
            "Please enter a message."
        )

    if user is None:

        user = {
            "user_id": "anonymous",
            "full_name": "Patient",
            "email": "",
            "phone": ""
        }

    # Frontend-generated nearby pharmacy command. This must run
    # before pending/booking handling because it is an explicit
    # user action coming from the location button.
    nearby_result = _handle_nearby_pharmacy_command(message)

    if nearby_result is not None:
        return nearby_result

    state = get_booking_state(
        user
    )

    # ========================================================
    # GREETING ALWAYS BREAKS OUT OF A STUCK FLOW
    #
    # If the user is mid-way through booking/cancelling/
    # rescheduling/pharmacy-selection (state left over from
    # an earlier session, a restart, or just changing their
    # mind), a plain "hi"/"hello" should always reset that
    # and greet normally - not be swallowed as an invalid
    # answer to whatever step they were stuck on.
    # ========================================================

    if is_greeting(message):

        reset_booking(user)
        _reset_pending_state(user)

        name = (
            user.get("full_name")
            if isinstance(user, dict)
            else None
        )

        greeting = (
            f"Hello {name}. How can I help you today?"
            if name
            else "Hello! How can I help you today?"
        )

        return chatbot_response(
            greeting
        )

    # ========================================================
    # ACTIVE BOOKING
    #
    # Once the booking has started, do NOT let the general
    # LLM override the current booking step.
    # ========================================================

    if state.get("active"):

        return await handle_booking_message(
            message,
            user
        )

    # ========================================================
    # PENDING ACTION FOLLOW-UP
    #
    # If a cancel / reschedule / pharmacy-selection flow is
    # in progress for this user, let it consume this message
    # first. If it returns None, the flow decided this
    # message doesn't belong to it (e.g. an unrelated new
    # question) and processing falls through to normal intent
    # handling below.
    # ========================================================

    pending = _get_pending_state(user)

    if pending.get("action"):

        pending_result = await _handle_pending_action_message(
            message,
            user
        )

        if pending_result is not None:

            return pending_result

    # ========================================================
    # DIRECT PHARMACY ROUTING
    #
    # Pharmacy requests are deterministic because a misspelled
    # medicine name should still reach the real medicine DB.
    # ========================================================

    if _looks_like_pharmacy_request(message):

        return await handle_pharmacy(
            message,
            {"medicine": None},
            user
        )

    # ========================================================
    # DETECT INTENT
    # ========================================================

    intent_data = await _detect_intent(
        message,
        conversation_history
    )

    if not isinstance(
        intent_data,
        dict
    ):

        intent_data = {
            "intent": "GENERAL",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0
        }

    intent = str(
        intent_data.get(
            "intent",
            "GENERAL"
        )
    ).upper().strip()

    # ========================================================
    # EMERGENCY
    # ========================================================

    if intent == "EMERGENCY":

        return await handle_emergency()

    # ========================================================
    # BOOK APPOINTMENT
    # ========================================================

    if intent == "BOOK_APPOINTMENT":

        return await _start_appointment_booking(
            message,
            intent_data,
            user
        )

    # ========================================================
    # DOCTOR SEARCH
    # ========================================================

    if intent == "DOCTOR_SEARCH":

        return await handle_doctor_search(
            message,
            intent_data,
            user
        )

    # ========================================================
    # CANCEL
    # ========================================================

    if intent == "CANCEL_APPOINTMENT":

        return await handle_cancel_appointment(
            message,
            user
        )

    # ========================================================
    # RESCHEDULE
    # ========================================================

    if intent == "RESCHEDULE_APPOINTMENT":

        return await handle_reschedule_appointment(
            message,
            user
        )

    # ========================================================
    # PATIENT HISTORY
    # ========================================================

    if intent == "PATIENT_HISTORY":

        return await handle_patient_history(
            user
        )

    # ========================================================
    # PHARMACY
    # ========================================================

    if intent == "PHARMACY":

        return await handle_pharmacy(
            message,
            intent_data,
            user
        )

    # ========================================================
    # SCREENING
    # ========================================================

    if intent == "SCREENING":

        return await handle_screening()

    # ========================================================
    # GENERAL
    # ========================================================

    response = await _generate_llm_response(
        message,
        conversation_history
    )

    if isinstance(
        response,
        dict
    ):

        # If your LLM already returns
        # structured data, preserve it.

        return response

    return chatbot_response(
        response
    )


# ============================================================
# COMPATIBILITY FUNCTIONS
# ============================================================

def get_current_booking_state(
    user=None
):

    return get_booking_state(
        user
    )


def clear_booking_state(
    user=None
):

    reset_booking(
        user
    )