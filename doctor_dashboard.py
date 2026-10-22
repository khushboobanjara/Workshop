"""
Doctor dashboard routes.

Wire-up in main.py:

    from doctor_dashboard import router as doctor_dashboard_router
    app.include_router(doctor_dashboard_router)
"""

from datetime import date, time
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.logger import get_logger
from src.database.doctor_dashboard_repository import (
    VALID_STATUSES,
    VALID_VIEWS,
    add_slot,
    get_dashboard_stats,
    get_doctor_appointments,
    get_doctor_by_email,
    get_upcoming_slots,
    get_user_doctor_status,
    remove_slot,
    set_doctor_accepting,
    update_appointment_status,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/doctor")
templates = Jinja2Templates(directory="templates")

DASHBOARD_URL = "/doctor/dashboard"

# users.doctor_status values that should NOT get into the dashboard.
# Adjust to whatever values your app actually stores.
BLOCKED_DOCTOR_STATUSES = {"PENDING", "REJECTED", "SUSPENDED", "BLOCKED"}


# =========================================================
# ACCESS CONTROL
# =========================================================

def _session_user(request: Request):
    raw = request.session.get("user_id")
    try:
        user_id = int(raw)
    except (TypeError, ValueError):
        return None
    if user_id <= 0:
        return None
    return {
        "user_id": user_id,
        "full_name": request.session.get("full_name"),
        "email": request.session.get("email"),
        "role": (request.session.get("role") or "").upper(),
    }


def _doctor_or_response(request: Request):
    """
    Returns (user, doctor, None) when access is fine, otherwise
    (None, None, response) with the response to send back.
    """
    user = _session_user(request)
    if not user:
        return None, None, RedirectResponse("/login", status_code=303)

    if user["role"] != "DOCTOR":
        return None, None, HTMLResponse(
            "<h2>Doctor access only</h2>"
            "<p>This area is for doctor accounts.</p>"
            "<a href='/'>Back to home</a>",
            status_code=403,
        )

    status = (get_user_doctor_status(user["user_id"]) or "").upper()
    if status in BLOCKED_DOCTOR_STATUSES:
        return user, None, templates.TemplateResponse(
            request,
            "doctor_dashboard.html",
            {"user": user, "state": "pending", "doctor_status": status},
        )

    doctor = get_doctor_by_email(user["email"])
    if not doctor:
        return user, None, templates.TemplateResponse(
            request,
            "doctor_dashboard.html",
            {"user": user, "state": "unlinked"},
        )

    return user, doctor, None


def _flash(request: Request, ok: bool, text: str):
    request.session["doctor_flash"] = {"ok": ok, "text": text}


def _safe_next(value: str) -> str:
    """Only allow redirects back into the dashboard."""
    if value and value.startswith(DASHBOARD_URL):
        return value
    return DASHBOARD_URL


# =========================================================
# DASHBOARD
# =========================================================

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    view: str = "today",
    status: str = "",
    q: str = "",
):
    user, doctor, early = _doctor_or_response(request)
    if early:
        return early

    today = date.today()

    view = view if view in VALID_VIEWS else "today"
    status = status.upper() if status.upper() in VALID_STATUSES else ""
    q = q.strip()[:100]

    query_string = urlencode(
        {k: v for k, v in (("view", view), ("status", status), ("q", q)) if v}
    )

    return templates.TemplateResponse(
        request,
        "doctor_dashboard.html",
        {
            "user": user,
            "state": "ok",
            "doctor": doctor,
            "today": today,
            "today_label": today.strftime("%A, %d %B %Y"),
            "stats": get_dashboard_stats(doctor["doctor_id"], today),
            "appointments": get_doctor_appointments(
                doctor["doctor_id"], today, view, status, q
            ),
            "slots": get_upcoming_slots(doctor["doctor_id"], today),
            "view": view,
            "status_filter": status,
            "q": q,
            "statuses": VALID_STATUSES,
            "current_url": DASHBOARD_URL + ("?" + query_string if query_string else ""),
            "flash": request.session.pop("doctor_flash", None),
        },
    )


# =========================================================
# APPOINTMENT ACTIONS
# =========================================================

@router.post("/appointments/{appointment_id}/{action}")
async def appointment_action(
    request: Request,
    appointment_id: int,
    action: str,
    next: str = Form(DASHBOARD_URL),
):
    user, doctor, early = _doctor_or_response(request)
    if early:
        return early

    ok, message = update_appointment_status(
        doctor["doctor_id"], appointment_id, action, date.today()
    )

    logger.info(
        f"Doctor {doctor['doctor_id']} action={action} "
        f"appointment={appointment_id} ok={ok}"
    )

    _flash(request, ok, message)
    return RedirectResponse(_safe_next(next), status_code=303)


# =========================================================
# AVAILABILITY
# =========================================================

@router.post("/availability/add")
async def availability_add(
    request: Request,
    slot_date: str = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    next: str = Form(DASHBOARD_URL),
):
    user, doctor, early = _doctor_or_response(request)
    if early:
        return early

    try:
        parsed_date = date.fromisoformat(slot_date)
        parsed_start = time.fromisoformat(start_time)
        parsed_end = time.fromisoformat(end_time)
    except ValueError:
        _flash(request, False, "Please enter a valid date and times.")
        return RedirectResponse(_safe_next(next), status_code=303)

    ok, message = add_slot(
        doctor["doctor_id"], parsed_date, parsed_start, parsed_end, date.today()
    )
    _flash(request, ok, message)
    return RedirectResponse(_safe_next(next), status_code=303)


@router.post("/availability/{availability_id}/remove")
async def availability_remove(
    request: Request,
    availability_id: int,
    next: str = Form(DASHBOARD_URL),
):
    user, doctor, early = _doctor_or_response(request)
    if early:
        return early

    ok, message = remove_slot(doctor["doctor_id"], availability_id)
    _flash(request, ok, message)
    return RedirectResponse(_safe_next(next), status_code=303)


@router.post("/accepting")
async def toggle_accepting(
    request: Request,
    accepting: str = Form(...),
    next: str = Form(DASHBOARD_URL),
):
    user, doctor, early = _doctor_or_response(request)
    if early:
        return early

    turn_on = accepting == "1"
    set_doctor_accepting(doctor["doctor_id"], turn_on)
    _flash(
        request,
        True,
        "You are now visible to patients."
        if turn_on
        else "You are hidden from patient search.",
    )
    return RedirectResponse(_safe_next(next), status_code=303)