"""MCP 02 - Doctor / Appointment.

Own-data tools (get / cancel / reschedule) pass principal.user_id into repositories that already
filter on `appointments.user_id = %s`, so another user's appointment simply is not found.

book_appointment and get_doctor_availability take a doctor_id (the doctor being booked), which
is why they are declared Ownership.NONE: the rbac layer refuses a `doctor_id` argument on
CURRENT_USER_ONLY tools. The patient is never an argument - it is always the signed-in user.
"""
from typing import Optional

from src.database import appointment_repository as appointments
from src.database import doctor_repository as doctors

from ..gateway import current_principal
from ..registry import MCPServerSpec
from ..schemas import ErrorCode, Ownership, Role, ToolKind, ToolMeta
from ._common import (BadInput, clean_phone, clean_text, fail, ok, ok_rows, parse_date, parse_time,
                      positive_int, require_future, rules)

ROLES = frozenset({Role.USER, Role.ADMIN, Role.SUPER_ADMIN})
CANCELLABLE = {"BOOKED", "CONFIRMED"}


def _slot_is_open(doctor_id: int, day, at) -> None:
    """Raises BadInput unless the doctor works at that time and nobody has booked it."""
    windows = [w for w in doctors.get_doctor_availability(doctor_id) if w["available_date"] == day]
    if not any(appointments.is_time_available(w["availability_id"], at) for w in windows):
        raise BadInput("The doctor is not available at that date and time.", ErrorCode.CONFLICT)
    if appointments.is_slot_booked(doctor_id, day, at):
        raise BadInput("That time slot is already booked.", ErrorCode.CONFLICT)


def register(spec: MCPServerSpec) -> None:
    @spec.tool(ToolMeta(
        name="find_doctors", allowed_roles=ROLES, permission="doctor.search",
        ownership=Ownership.NONE, kind=ToolKind.READ, requires_verified_data=True,
        description="List available doctors, optionally for one specialization (e.g. 'Cardiologist')."))
    @rules
    def find_doctors(specialization: Optional[str] = None) -> dict:
        if specialization is not None and specialization.strip():
            rows = doctors.find_doctors_by_specialization(clean_text(specialization, "Specialization", 2, 60))
        else:
            rows = doctors.get_all_available_doctors()
        return ok_rows(rows, "No matching doctors are available in our records.")

    @spec.tool(ToolMeta(
        name="get_doctor_availability", allowed_roles=ROLES, permission="doctor.availability.read",
        ownership=Ownership.NONE, kind=ToolKind.READ, requires_verified_data=True,
        description="Upcoming availability windows for one doctor."))
    @rules
    def get_doctor_availability(doctor_id: int) -> dict:
        doctor_id = positive_int(doctor_id, "Doctor")
        if not doctors.get_doctor_by_id(doctor_id):
            return fail(ErrorCode.NOT_FOUND, "That doctor could not be found.")
        return ok_rows(doctors.get_doctor_availability(doctor_id), "This doctor has no upcoming availability.")

    @spec.tool(ToolMeta(
        name="book_appointment", allowed_roles=ROLES, permission="appointment.create.own",
        ownership=Ownership.NONE, kind=ToolKind.WRITE,
        description="Book a pay-at-clinic appointment for the signed-in user. Date YYYY-MM-DD, time HH:MM."))
    @rules
    def book_appointment(doctor_id: int, appointment_date: str, appointment_time: str,
                         patient_name: str, phone: str) -> dict:
        user_id = current_principal().user_id
        doctor_id = positive_int(doctor_id, "Doctor")
        day, at = parse_date(appointment_date), parse_time(appointment_time)
        require_future(day, at)
        name, number = clean_text(patient_name, "Patient name", 2, 100), clean_phone(phone)

        doctor = doctors.get_doctor_by_id(doctor_id)
        if not doctor or not doctor.get("available"):
            return fail(ErrorCode.NOT_FOUND, "That doctor is not available for booking.")
        _slot_is_open(doctor_id, day, at)

        # Online payment stays in the existing Cashfree flow; MCP books pay-at-clinic only.
        appointment_id = appointments.create_appointment(user_id, doctor_id, day, at, name, number, "CASH")
        created = appointments.get_appointment_by_id(appointment_id, user_id)
        if not created:   # the insert reported success but the row cannot be read back: do not claim it
            return fail(ErrorCode.INTERNAL_ERROR, "The booking could not be confirmed. Please check your appointments.")
        return ok(created, message="Appointment booked.")

    @spec.tool(ToolMeta(
        name="get_my_appointments", allowed_roles=ROLES, permission="appointment.read.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.READ, requires_verified_data=True,
        description="The signed-in user's own appointments, newest first."))
    @rules
    def get_my_appointments() -> dict:
        rows = appointments.get_user_appointments(current_principal().user_id)
        return ok_rows(rows, "You have no appointments on record.")

    @spec.tool(ToolMeta(
        name="cancel_appointment", allowed_roles=ROLES, permission="appointment.cancel.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.WRITE,
        description="Cancel one of the signed-in user's own upcoming appointments."))
    @rules
    def cancel_appointment(appointment_id: int) -> dict:
        user_id = current_principal().user_id
        appointment_id = positive_int(appointment_id, "Appointment")
        if not appointments.get_appointment_by_id(appointment_id, user_id):
            return fail(ErrorCode.NOT_FOUND, "That appointment could not be found.")
        if not appointments.cancel_appointment(appointment_id, user_id):
            return fail(ErrorCode.CONFLICT, "This appointment can no longer be cancelled.")
        return ok({"appointment_id": appointment_id, "status": "CANCELLED"}, message="Appointment cancelled.")

    @spec.tool(ToolMeta(
        name="reschedule_appointment", allowed_roles=ROLES, permission="appointment.reschedule.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.WRITE,
        description="Move one of the signed-in user's own appointments to a new date (YYYY-MM-DD) and time (HH:MM)."))
    @rules
    def reschedule_appointment(appointment_id: int, new_date: str, new_time: str) -> dict:
        user_id = current_principal().user_id
        appointment_id = positive_int(appointment_id, "Appointment")
        day, at = parse_date(new_date), parse_time(new_time)
        require_future(day, at)

        current = appointments.get_appointment_by_id(appointment_id, user_id)
        if not current:
            return fail(ErrorCode.NOT_FOUND, "That appointment could not be found.")
        if current["status"] not in CANCELLABLE:
            return fail(ErrorCode.CONFLICT, "This appointment can no longer be rescheduled.")
        _slot_is_open(int(current["doctor_id"]), day, at)
        if not appointments.reschedule_appointment(appointment_id, user_id, day, at):
            return fail(ErrorCode.CONFLICT, "This appointment can no longer be rescheduled.")
        return ok(appointments.get_appointment_by_id(appointment_id, user_id), message="Appointment rescheduled.")
