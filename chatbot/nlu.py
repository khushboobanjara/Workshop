# chatbot/nlu.py

import json
import re
import inspect

from chatbot.llm import generate_response


# ============================================================
# ALLOWED INTENTS
# ============================================================

ALLOWED_INTENTS = {
    "GENERAL",
    "BOOK_APPOINTMENT",
    "DOCTOR_SEARCH",
    "PHARMACY",
    "SCREENING",
    "PATIENT_HISTORY",
    "CANCEL_APPOINTMENT",
    "RESCHEDULE_APPOINTMENT",
    "EMERGENCY",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

NLU_SYSTEM_PROMPT = """
You are the Natural Language Understanding engine for
Sanjeevani Clinic.

Your job is NOT to answer the user.

Your only job is to understand what the user wants and return
structured JSON.

The application has these intents:

GENERAL
BOOK_APPOINTMENT
DOCTOR_SEARCH
PHARMACY
SCREENING
PATIENT_HISTORY
CANCEL_APPOINTMENT
RESCHEDULE_APPOINTMENT
EMERGENCY

Return ONLY valid JSON.

Required JSON structure:

{
    "intent": "GENERAL",
    "specialty": null,
    "doctor_name": null,
    "medicine": null,
    "date": null,
    "time": null,
    "payment_method": null,
    "patient_name": null,
    "phone": null,
    "confidence": 0.0
}


IMPORTANT RULES:

1. "I need a neurologist" means:
   intent = BOOK_APPOINTMENT
   specialty = Neurologist

2. "I want to see a neurologist" means:
   intent = BOOK_APPOINTMENT
   specialty = Neurologist

3. "Can you find me a neurologist?" means:
   intent = DOCTOR_SEARCH
   specialty = Neurologist

4. "I want to book with Dr. Suresh Sharma" means:
   intent = BOOK_APPOINTMENT
   doctor_name = Dr. Suresh Sharma

5. "I need headache treatment" is GENERAL unless the user
   explicitly asks to find/book a doctor.

6. "I need medicine" means:
   intent = PHARMACY
   medicine = null

7. "I need paracetamol" means:
   intent = PHARMACY
   medicine = Paracetamol

8. Preserve misspelled medicine names.
   Example:
   "paracitamol" -> medicine = "paracitamol"

9. "book appointment", "book an appointment",
   "I need an appointment", "I want to see a doctor"
   should become BOOK_APPOINTMENT.

10. If the user is answering a previous question, use the
    conversation context.

11. Do not invent missing information.

12. Unknown fields MUST be null.

13. confidence must be between 0 and 1.

14. Do not return markdown.
15. Do not return explanations.
16. Do not return code fences.
"""


# ============================================================
# DEFAULT RESULT
# ============================================================

def empty_nlu():

    return {
        "intent": "GENERAL",
        "specialty": None,
        "doctor_name": None,
        "medicine": None,
        "date": None,
        "time": None,
        "payment_method": None,
        "patient_name": None,
        "phone": None,
        "confidence": 0.0,
    }


# ============================================================
# CLEAN JSON
# ============================================================

def _extract_json(text):

    if not text:
        return None

    text = str(text).strip()

    # Remove markdown code fences if model returns them.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find JSON object inside response.
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if not match:
        return None

    try:
        return json.loads(
            match.group(0)
        )
    except Exception:
        return None


# ============================================================
# NORMALIZE RESULT
# ============================================================

def _normalize_result(result):

    output = empty_nlu()

    if not isinstance(result, dict):
        return output

    intent = str(
        result.get(
            "intent",
            "GENERAL"
        )
    ).upper().strip()

    if intent not in ALLOWED_INTENTS:
        intent = "GENERAL"

    output["intent"] = intent

    for key in [
        "specialty",
        "doctor_name",
        "medicine",
        "date",
        "time",
        "payment_method",
        "patient_name",
        "phone",
    ]:

        value = result.get(key)

        if value is not None:

            value = str(value).strip()

            if value:
                output[key] = value

    try:

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        output["confidence"] = confidence

    except Exception:

        output["confidence"] = 0.0

    return output


# ============================================================
# CALL LLM
# ============================================================

async def _call_llm(prompt):

    try:

        result = generate_response(
            prompt,
            []
        )

    except TypeError:

        result = generate_response(
            prompt
        )

    if inspect.isawaitable(result):

        result = await result

    return result


# ============================================================
# MAIN NLU FUNCTION
# ============================================================

async def understand_message(
    message,
    conversation_history=None,
    current_state=None
):

    history = (
        conversation_history
        if isinstance(
            conversation_history,
            list
        )
        else []
    )

    state = (
        current_state
        if isinstance(
            current_state,
            dict
        )
        else {}
    )

    history_text = ""

    for item in history[-10:]:

        if not isinstance(
            item,
            dict
        ):
            continue

        role = item.get(
            "role",
            ""
        )

        content = item.get(
            "content",
            ""
        )

        if content:

            history_text += (
                f"{role}: {content}\n"
            )

    prompt = f"""
{NLU_SYSTEM_PROMPT}

CURRENT CONVERSATION STATE:

{json.dumps(
    state,
    ensure_ascii=False,
    default=str
)}

RECENT CONVERSATION:

{history_text}

CURRENT USER MESSAGE:

{message}

Return ONLY the JSON object.
"""

    try:

        raw_result = await _call_llm(
            prompt
        )

        parsed = _extract_json(
            raw_result
        )

        return _normalize_result(
            parsed
        )

    except Exception:

        # NLU failure must never crash
        # the application.
        return empty_nlu()
    