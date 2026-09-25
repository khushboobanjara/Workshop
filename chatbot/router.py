# =========================================================
# SANJEEVANI CLINIC
# CHATBOT INTENT ROUTER
# =========================================================

import inspect
import json
import re

from chatbot.llm import generate_response


# =========================================================
# SYSTEM PROMPT
# =========================================================

ROUTER_SYSTEM_PROMPT = """
You are the intent router for Sanjeevani Clinic.

Your job is ONLY to identify the user's intent and extract
information that is explicitly present in the user's message.

IMPORTANT RULES:

1. NEVER invent information.
2. NEVER guess a medical specialization that the user did not mention.
3. If the user says "I want to book an appointment" without
   mentioning a specialty, specialty MUST be null.
4. Understand spelling mistakes, typing mistakes, grammar mistakes,
   incomplete sentences, Hinglish, informal English, abbreviations,
   and phonetic typing.
5. "i want to bok apointmnt" means BOOK_APPOINTMENT.
6. "mujhe apointment book krni h" means BOOK_APPOINTMENT.
7. "i need cardioligist" means DOCTOR_SEARCH with specialty Cardiologist.
8. "skin doctor" means Dermatologist.
9. "heart doctor" means Cardiologist.
10. "child doctor" or "kids doctor" means Pediatrician.
11. "bone doctor" means Orthopedic.
12. "eye doctor" means Ophthalmologist.
13. "brain doctor" means Neurologist.
14. "dentist" means Dentist.
15. "ENT doctor" means ENT Specialist.
16. "general physician" means General Physician.
17. Greetings like "hi", "hello", "hey", "namaste" are GENERAL.
18. Return ONLY valid JSON.
19. Do not add markdown.
20. Do not add explanations.

Allowed intents:

EMERGENCY
BOOK_APPOINTMENT
DOCTOR_SEARCH
CANCEL_APPOINTMENT
RESCHEDULE_APPOINTMENT
PATIENT_HISTORY
PHARMACY
SCREENING
GENERAL

Return exactly this structure:

{
    "intent": "GENERAL",
    "specialty": null,
    "medicine": null,
    "date": null,
    "time": null,
    "doctor_name": null,
    "confidence": 0.0
}
"""


# =========================================================
# DEFAULT RESULT
# =========================================================

def default_result():
    return {
        "intent": "GENERAL",
        "specialty": None,
        "medicine": None,
        "date": None,
        "time": None,
        "doctor_name": None,
        "confidence": 0.0
    }


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize_text(message: str) -> str:
    """
    Normalize user text for deterministic matching.
    """

    message = message or ""

    message = message.strip().lower()

    message = re.sub(
        r"\s+",
        " ",
        message
    )

    return message


# =========================================================
# NORMALIZE SPECIALTY
# =========================================================

def normalize_specialty(specialty):
    """
    Convert common spelling mistakes and informal names
    into the specialization names used by the database.
    """

    if not specialty:
        return None

    value = normalize_text(
        str(specialty)
    )

    specialty_map = {

        # Cardiologist
        "cardiologist": "Cardiologist",
        "cardiology": "Cardiologist",
        "cardioligist": "Cardiologist",
        "cardiologyst": "Cardiologist",
        "heart doctor": "Cardiologist",
        "heart specialist": "Cardiologist",

        # Dermatologist
        "dermatologist": "Dermatologist",
        "dermatology": "Dermatologist",
        "dermatologyst": "Dermatologist",
        "skin doctor": "Dermatologist",
        "skin specialist": "Dermatologist",

        # Pediatrician
        "pediatrician": "Pediatrician",
        "pediatrics": "Pediatrician",
        "paediatrician": "Pediatrician",
        "child doctor": "Pediatrician",
        "kids doctor": "Pediatrician",
        "children doctor": "Pediatrician",

        # Orthopedic
        "orthopedic": "Orthopedic",
        "orthopaedic": "Orthopedic",
        "orthopedist": "Orthopedic",
        "bone doctor": "Orthopedic",
        "bone specialist": "Orthopedic",

        # Ophthalmologist
        "ophthalmologist": "Ophthalmologist",
        "ophthalmology": "Ophthalmologist",
        "eye doctor": "Ophthalmologist",
        "eye specialist": "Ophthalmologist",

        # Neurologist
        "neurologist": "Neurologist",
        "neurology": "Neurologist",
        "neuroligist": "Neurologist",
        "brain doctor": "Neurologist",
        "brain specialist": "Neurologist",

        # Dentist
        "dentist": "Dentist",
        "dental doctor": "Dentist",
        "dental specialist": "Dentist",

        # ENT
        "ent": "ENT Specialist",
        "ent doctor": "ENT Specialist",
        "ent specialist": "ENT Specialist",
        "ear nose throat doctor": "ENT Specialist",
        "ear nose throat specialist": "ENT Specialist",

        # General Physician
        "general physician": "General Physician",
        "general doctor": "General Physician",
        "physician": "General Physician",
        "gp": "General Physician",
        "doctor": "General Physician"
    }

    return specialty_map.get(
        value,
        specialty
    )


# =========================================================
# EXTRACT SPECIALTY DIRECTLY FROM USER MESSAGE
# =========================================================

def extract_specialty_from_message(message: str):
    """
    Extract a known specialty directly from the user's message.

    This prevents the LLM from inventing a specialty.
    """

    text = normalize_text(message)

    specialty_patterns = [

        (
            r"\bcardiologist\b",
            "Cardiologist"
        ),
        (
            r"\bcardiology\b",
            "Cardiologist"
        ),
        (
            r"\bcardioligist\b",
            "Cardiologist"
        ),
        (
            r"\bheart doctor\b",
            "Cardiologist"
        ),
        (
            r"\bheart specialist\b",
            "Cardiologist"
        ),

        (
            r"\bdermatologist\b",
            "Dermatologist"
        ),
        (
            r"\bdermatology\b",
            "Dermatologist"
        ),
        (
            r"\bskin doctor\b",
            "Dermatologist"
        ),
        (
            r"\bskin specialist\b",
            "Dermatologist"
        ),

        (
            r"\bpediatrician\b",
            "Pediatrician"
        ),
        (
            r"\bpaediatrician\b",
            "Pediatrician"
        ),
        (
            r"\bchild doctor\b",
            "Pediatrician"
        ),
        (
            r"\bkids doctor\b",
            "Pediatrician"
        ),
        (
            r"\bchildren doctor\b",
            "Pediatrician"
        ),

        (
            r"\borthopedic\b",
            "Orthopedic"
        ),
        (
            r"\borthopaedic\b",
            "Orthopedic"
        ),
        (
            r"\borthopedist\b",
            "Orthopedic"
        ),
        (
            r"\bbone doctor\b",
            "Orthopedic"
        ),

        (
            r"\bophthalmologist\b",
            "Ophthalmologist"
        ),
        (
            r"\bophthalmology\b",
            "Ophthalmologist"
        ),
        (
            r"\beye doctor\b",
            "Ophthalmologist"
        ),
        (
            r"\beye specialist\b",
            "Ophthalmologist"
        ),

        (
            r"\bneurologist\b",
            "Neurologist"
        ),
        (
            r"\bneurology\b",
            "Neurologist"
        ),
        (
            r"\bneuroligist\b",
            "Neurologist"
        ),
        (
            r"\bbrain doctor\b",
            "Neurologist"
        ),

        (
            r"\bdentist\b",
            "Dentist"
        ),
        (
            r"\bdental doctor\b",
            "Dentist"
        ),

        (
            r"\bent doctor\b",
            "ENT Specialist"
        ),
        (
            r"\bent specialist\b",
            "ENT Specialist"
        ),
        (
            r"\bear nose throat doctor\b",
            "ENT Specialist"
        ),
        (
            r"\bear nose throat specialist\b",
            "ENT Specialist"
        ),
        (
            r"\bent\b",
            "ENT Specialist"
        ),

        (
            r"\bgeneral physician\b",
            "General Physician"
        ),
        (
            r"\bgeneral doctor\b",
            "General Physician"
        ),
        (
            r"\bphysician\b",
            "General Physician"
        )
    ]

    for pattern, specialty in specialty_patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            return specialty

    return None


# =========================================================
# EXTRACT MEDICINE
# =========================================================

def extract_medicine_from_message(message: str):
    """
    Basic medicine extraction.
    More detailed pharmacy handling remains in service.py.
    """

    text = normalize_text(message)

    patterns = [
        r"(?:medicine|medicines|tablet|tablets|drug|capsule)\s+(.+)$",
        r"(?:buy|order|need|want)\s+(.+?)\s+(?:medicine|medicines|tablet|tablets|drug|capsule)$"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            medicine = match.group(
                1
            ).strip()

            if medicine:
                return medicine

    return None


# =========================================================
# GREETINGS
# =========================================================

def is_greeting(text: str) -> bool:

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
        "heloo",
        "hy",
        "namaste",
        "namaskar",
        "good morning",
        "good afternoon",
        "good evening"
    }

    return text in greetings


# =========================================================
# BOOKING REQUEST DETECTION
# =========================================================

def is_booking_request(text: str) -> bool:

    booking_patterns = [

        r"\bbook an appointment\b",
        r"\bbook appointment\b",
        r"\bwant to book an appointment\b",
        r"\bwant to book appointment\b",
        r"\bneed to book an appointment\b",
        r"\bneed to book appointment\b",

        r"\bi need an appointment\b",
        r"\bi need appointment\b",
        r"\bi want an appointment\b",
        r"\bi want appointment\b",

        r"\bappointment booking\b",
        r"\bbook my appointment\b",
        r"\bappointment book\b",

        r"\bappointment lena hai\b",
        r"\bappointment leni hai\b",
        r"\bappointment chahiye\b",
        r"\bappointment book karna hai\b",
        r"\bappointment book krna hai\b",
        r"\bappointment book karni hai\b",
        r"\bappointment book krni hai\b",

        r"\bappointment\b.*\bbook\b",
        r"\bbook\b.*\bappointment\b",

        # common spelling mistakes
        r"\bbok\b.*\bapointmnt\b",
        r"\bapointment\b.*\bbok\b"
    ]

    for pattern in booking_patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            return True

    return False


# =========================================================
# CANCEL REQUEST
# =========================================================

def is_cancel_request(text: str) -> bool:

    patterns = (
        "cancel my appointment",
        "cancel appointment",
        "cancel an appointment",
        "i want to cancel",
        "i need to cancel",
        "appointment cancel",
        "appointment cancel karna hai",
        "appointment cancel krna hai"
    )

    return any(
        pattern in text
        for pattern in patterns
    )


# =========================================================
# RESCHEDULE REQUEST
# =========================================================

def is_reschedule_request(text: str) -> bool:

    patterns = (
        "reschedule my appointment",
        "reschedule appointment",
        "i want to reschedule",
        "i need to reschedule",
        "change my appointment",
        "change appointment",
        "appointment reschedule",
        "appointment change"
    )

    return any(
        pattern in text
        for pattern in patterns
    )


# =========================================================
# EMERGENCY REQUEST
# =========================================================

def is_emergency_request(text: str) -> bool:

    emergency_words = (
        "emergency",
        "severe chest pain",
        "difficulty breathing",
        "cannot breathe",
        "can't breathe",
        "unconscious",
        "heavy bleeding",
        "stroke",
        "heart attack"
    )

    return any(
        phrase in text
        for phrase in emergency_words
    )


# =========================================================
# SCREENING REQUEST
# =========================================================

def is_screening_request(text: str) -> bool:

    screening_patterns = (
        "health screening",
        "screening",
        "health check",
        "health checkup",
        "check my health",
        "screen my health",
        "do screening",
        "screening test"
    )

    return any(
        pattern in text
        for pattern in screening_patterns
    )


# =========================================================
# PHARMACY REQUEST
# =========================================================

def is_pharmacy_request(text: str) -> bool:

    pharmacy_patterns = (
        "pharmacy",
        "medicine",
        "medicines",
        "tablet",
        "tablets",
        "capsule",
        "drug",
        "buy medicine",
        "order medicine",
        "need medicine",
        "want medicine"
    )

    return any(
        pattern in text
        for pattern in pharmacy_patterns
    )


# =========================================================
# DOCTOR SEARCH REQUEST
# =========================================================

def is_doctor_search_request(text: str) -> bool:

    doctor_patterns = (
        "find a doctor",
        "find doctor",
        "search doctor",
        "doctor search",
        "need a doctor",
        "need doctor",
        "looking for a doctor",
        "specialist",
        "doctor chahiye",
        "doctor dikhana hai"
    )

    return any(
        pattern in text
        for pattern in doctor_patterns
    )


# =========================================================
# PATIENT HISTORY
# =========================================================

def is_history_request(text: str) -> bool:

    history_patterns = (
        "my history",
        "patient history",
        "medical history",
        "appointment history",
        "my appointments",
        "previous appointments",
        "past appointments",
        "my records",
        "medical records"
    )

    return any(
        pattern in text
        for pattern in history_patterns
    )


# =========================================================
# PARSE JSON FROM LLM
# =========================================================

def parse_llm_json(raw_response):

    if isinstance(
        raw_response,
        dict
    ):
        return raw_response

    if raw_response is None:
        return default_result()

    text = str(
        raw_response
    ).strip()

    # Remove markdown code fences if present
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # First try direct JSON
    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:
        pass

    # Find JSON object inside response
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if not match:
        return default_result()

    try:

        return json.loads(
            match.group(0)
        )

    except json.JSONDecodeError:

        return default_result()


# =========================================================
# CLEAN ROUTER RESULT
# =========================================================

def clean_router_result(result):

    if not isinstance(
        result,
        dict
    ):
        result = default_result()

    cleaned = default_result()

    # -----------------------------------------------------
    # Intent
    # -----------------------------------------------------

    intent = result.get(
        "intent"
    )

    if intent:

        intent = str(
            intent
        ).strip().upper()

    allowed_intents = {
        "EMERGENCY",
        "BOOK_APPOINTMENT",
        "DOCTOR_SEARCH",
        "CANCEL_APPOINTMENT",
        "RESCHEDULE_APPOINTMENT",
        "PATIENT_HISTORY",
        "PHARMACY",
        "SCREENING",
        "GENERAL"
    }

    if intent in allowed_intents:

        cleaned["intent"] = intent

    # -----------------------------------------------------
    # Specialty
    # -----------------------------------------------------

    specialty = result.get(
        "specialty"
    )

    if specialty:
        cleaned["specialty"] = normalize_specialty(
            specialty
        )

    # -----------------------------------------------------
    # Medicine
    # -----------------------------------------------------

    medicine = result.get(
        "medicine"
    )

    if medicine:
        cleaned["medicine"] = str(
            medicine
        ).strip()

    # -----------------------------------------------------
    # Date
    # -----------------------------------------------------

    value = result.get(
        "date"
    )

    if value:

        cleaned["date"] = str(
            value
        ).strip()

    # -----------------------------------------------------
    # Time
    # -----------------------------------------------------

    value = result.get(
        "time"
    )

    if value:

        cleaned["time"] = str(
            value
        ).strip()

    # -----------------------------------------------------
    # Doctor name
    # -----------------------------------------------------

    value = result.get(
        "doctor_name"
    )

    if value:

        cleaned["doctor_name"] = str(
            value
        ).strip()

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = result.get(
        "confidence",
        0.0
    )

    try:

        confidence = float(
            confidence
        )

    except (
        TypeError,
        ValueError
    ):

        confidence = 0.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence
        )
    )

    cleaned["confidence"] = confidence

    return cleaned


# =========================================================
# CALL LLM
# =========================================================

async def _call_llm(message: str):

    """
    Call generate_response safely.

    The existing llm.py implementation may expose slightly
    different argument styles, so this function supports the
    common forms without changing llm.py.
    """

    prompt = (
        ROUTER_SYSTEM_PROMPT
        + "\n\nUSER MESSAGE:\n"
        + message
        + "\n\nRETURN JSON ONLY."
    )

    try:

        # Most common implementation:
        # generate_response(prompt)

        result = generate_response(
            prompt
        )

        if inspect.isawaitable(
            result
        ):

            result = await result

        return result

    except TypeError:

        # Compatibility fallback for an implementation
        # using system_prompt + user_message.

        try:

            result = generate_response(
                ROUTER_SYSTEM_PROMPT,
                message
            )

            if inspect.isawaitable(
                result
            ):

                result = await result

            return result

        except Exception:

            return None

    except Exception:

        return None


# =========================================================
# DETECT INTENT
# =========================================================

async def detect_intent(message: str):

    """
    Main intent detection function.
    """

    text = normalize_text(
        message
    )

    # Empty message
    if not text:

        return default_result()

    # =====================================================
    # 1. GREETING
    # =====================================================

    if is_greeting(text):

        return {
            "intent": "GENERAL",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.99
        }

    # =====================================================
    # 2. EMERGENCY
    # =====================================================

    if is_emergency_request(text):

        return {
            "intent": "EMERGENCY",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.99
        }

    # =====================================================
    # 3. CANCEL
    # =====================================================

    if is_cancel_request(text):

        return {
            "intent": "CANCEL_APPOINTMENT",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.99
        }

    # =====================================================
    # 4. RESCHEDULE
    # =====================================================

    if is_reschedule_request(text):

        return {
            "intent": "RESCHEDULE_APPOINTMENT",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.99
        }

    # =====================================================
    # 5. BOOK APPOINTMENT
    #
    # IMPORTANT:
    #
    # If user says:
    # "i want to book an appointment"
    #
    # we return:
    #
    # BOOK_APPOINTMENT
    # specialty = None
    #
    # We DO NOT allow the LLM to invent ENT/Cardiology/etc.
    # =====================================================

    if is_booking_request(text):

        specialty = extract_specialty_from_message(
            text
        )

        return {
            "intent": "BOOK_APPOINTMENT",
            "specialty": specialty,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.99
        }

    # =====================================================
    # 6. SCREENING
    # =====================================================

    if is_screening_request(text):

        return {
            "intent": "SCREENING",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.98
        }

    # =====================================================
    # 7. PATIENT HISTORY
    # =====================================================

    if is_history_request(text):

        return {
            "intent": "PATIENT_HISTORY",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.98
        }

    # =====================================================
    # 8. PHARMACY
    # =====================================================

    if is_pharmacy_request(text):

        medicine = extract_medicine_from_message(
            text
        )

        return {
            "intent": "PHARMACY",
            "specialty": None,
            "medicine": medicine,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.95
        }

    # =====================================================
    # 9. DIRECT SPECIALTY DETECTION
    # =====================================================

    specialty = extract_specialty_from_message(
        text
    )

    if specialty and is_doctor_search_request(text):

        return {
            "intent": "DOCTOR_SEARCH",
            "specialty": specialty,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.99
        }

    # =====================================================
    # 10. GENERAL DOCTOR SEARCH
    # =====================================================

    if is_doctor_search_request(text):

        return {
            "intent": "DOCTOR_SEARCH",
            "specialty": None,
            "medicine": None,
            "date": None,
            "time": None,
            "doctor_name": None,
            "confidence": 0.95
        }

    # =====================================================
    # 11. LLM FALLBACK
    #
    # Used for natural/informal sentences that the
    # deterministic rules do not recognize.
    # =====================================================

    raw_response = await _call_llm(
        message
    )

    parsed = parse_llm_json(
        raw_response
    )

    result = clean_router_result(
        parsed
    )

    # =====================================================
    # SAFETY RULE:
    #
    # If the original message is clearly a booking request,
    # but the LLM somehow assigns another intent, force
    # BOOK_APPOINTMENT.
    # =====================================================

    if is_booking_request(text):

        result["intent"] = "BOOK_APPOINTMENT"

        explicit_specialty = extract_specialty_from_message(
            text
        )

        if explicit_specialty:

            result["specialty"] = explicit_specialty

        else:

            result["specialty"] = None

        result["confidence"] = max(
            result["confidence"],
            0.99
        )

    # =====================================================
    # SAFETY RULE:
    #
    # Never keep a hallucinated specialty when the user
    # did not mention one for a generic booking request.
    # =====================================================

    if (
        result["intent"] == "BOOK_APPOINTMENT"
        and not extract_specialty_from_message(text)
    ):

        result["specialty"] = None

    return result


# =========================================================
# OPTIONAL SYNC WRAPPER
# =========================================================

def detect_intent_sync(message: str):
    """
    Compatibility helper for code that needs a synchronous
    wrapper.

    Prefer detect_intent() in async code.
    """

    import asyncio

    try:

        loop = asyncio.get_running_loop()

    except RuntimeError:

        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "detect_intent_sync() cannot be used "
            "inside a running event loop. "
            "Use await detect_intent() instead."
        )

    return asyncio.run(
        detect_intent(message)
    )


# =========================================================
# TEST EXAMPLES
# =========================================================

if __name__ == "__main__":

    import asyncio

    async def _test():

        test_messages = [

            "hi",

            "hello",

            "i want to book an appointment",

            "i want to bok an apointmnt",

            "mujhe apointment book krni h",

            "i want to book an appointment with a neurologist",

            "i need a cardioligist",

            "i need a skin doctor",

            "find me a doctor",

            "i want to cancel my appointment",

            "reschedule my appointment",

            "i want medicine",

            "i need paracetamol medicine",

            "check my health"
        ]

        for message in test_messages:

            result = await detect_intent(
                message
            )

            print(
                "\nUSER:",
                message
            )

            print(
                "ROUTER:",
                result
            )

    asyncio.run(
        _test()
    )