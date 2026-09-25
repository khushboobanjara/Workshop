# chatbot/conversation.py

from datetime import datetime


# ============================================================
# IN-MEMORY CONVERSATION STORE
# ============================================================

_conversations = {}


# ============================================================
# DEFAULT STATE
# ============================================================

def create_state():

    return {
        "intent": None,

        "step": None,

        "specialty": None,

        "doctor_id": None,

        "doctor_name": None,

        "appointment_date": None,

        "appointment_time": None,

        "patient_name": None,

        "phone": None,

        "payment_method": None,

        "medicine_id": None,

        "medicine": None,

        "delivery_method": None,

        "delivery_address": None,

        "pharmacy_order_id": None,

        "appointment_id": None,

        "payment_session_id": None,

        "last_updated": datetime.now().isoformat()
    }


# ============================================================
# USER KEY
# ============================================================

def _user_key(user):

    if isinstance(
        user,
        dict
    ):

        user_id = user.get(
            "user_id"
        )

        if user_id is not None:

            return str(user_id)

    return "anonymous"


# ============================================================
# GET STATE
# ============================================================

def get_state(user):

    key = _user_key(
        user
    )

    if key not in _conversations:

        _conversations[key] = {
            "state": create_state(),
            "history": []
        }

    return _conversations[key]


# ============================================================
# STATE ONLY
# ============================================================

def get_current_state(user):

    conversation = get_state(
        user
    )

    return conversation["state"]


# ============================================================
# HISTORY
# ============================================================

def get_history(user):

    conversation = get_state(
        user
    )

    return conversation["history"]


# ============================================================
# ADD MESSAGE
# ============================================================

def add_message(
    user,
    role,
    content
):

    conversation = get_state(
        user
    )

    conversation["history"].append(
        {
            "role": role,
            "content": str(content)
        }
    )

    # Keep only recent conversation.
    conversation["history"] = (
        conversation["history"][-20:]
    )


# ============================================================
# UPDATE STATE
# ============================================================

def update_state(
    user,
    **values
):

    state = get_current_state(
        user
    )

    for key, value in values.items():

        if key in state:

            state[key] = value

    state["last_updated"] = (
        datetime.now().isoformat()
    )

    return state


# ============================================================
# RESET
# ============================================================

def reset_state(user):

    key = _user_key(
        user
    )

    _conversations[key] = {
        "state": create_state(),
        "history": []
    }


# ============================================================
# RESET ONLY BOOKING
# ============================================================

def reset_booking(user):

    state = get_current_state(
        user
    )

    booking_fields = [
        "intent",
        "step",
        "specialty",
        "doctor_id",
        "doctor_name",
        "appointment_date",
        "appointment_time",
        "patient_name",
        "phone",
        "payment_method",
        "appointment_id",
        "payment_session_id"
    ]

    for field in booking_fields:

        state[field] = None

    state["last_updated"] = (
        datetime.now().isoformat()
    )

    return state