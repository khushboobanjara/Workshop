"""MCP 06 - Doctor / Appointment Administration.

Two families of tools live here and never mix:
  * ADMIN / SUPER_ADMIN tools (Ownership.ADMINISTRATIVE): manage ANY doctor or appointment.
  * DOCTOR tools (Ownership.DOCTOR_OWN): the doctor identity comes from principal.doctor_id,
    never from an argument, so a doctor can only touch their own schedule and patients.
A doctor holds none of the admin permissions, so a doctor can never reach an admin tool.
"""
import os
from datetime import date
from typing import Optional

from src.database import doctor_dashboard_repository as dash
from src.database import doctor_repository as doctors
from src.database import mcp_admin_repository as admin_repo

from ..gateway import current_principal
from ..registry import MCPServerSpec
from ..roles import BLOCKED_DOCTOR_STATUSES
from ..schemas import ErrorCode, Ownership, Role, ToolKind, ToolMeta
from ._common import (BadInput, clean_email, clean_phone, clean_text, fail, money, ok, ok_rows, opt_text,
                      parse_date, parse_time, positive_int, rules, whole_number)

ADMINS = frozenset({Role.ADMIN, Role.SUPER_ADMIN})
DOCTORS = frozenset({Role.DOCTOR})
APPOINTMENT_STATUSES = {"BOOKED", "CONFIRMED", "COMPLETED", "CANCELLED"}
ADMIN_ACTIONS = {"confirm", "cancel", "complete"}
DOCTOR_ACTIONS = {"confirm", "cancel", "complete", "mark-paid"}   # same actions as your doctor dashboard
VIEWS = {"today", "upcoming", "past", "all"}


def approved_status() -> str:
    """The users.doctor_status value meaning 'approved'. Set MCP_DOCTOR_APPROVED_STATUS if your
    database uses another word; a value the role resolver blocks is refused."""
    value = os.getenv("MCP_DOCTOR_APPROVED_STATUS", "APPROVED").strip().upper()
    if not value or value in BLOCKED_DOCTOR_STATUSES:
        raise RuntimeError("MCP_DOCTOR_APPROVED_STATUS is invalid")
    return value


def _doctor_or_fail(doctor_id: int):
    return doctors.get_doctor_by_id(doctor_id)


def register(spec: MCPServerSpec) -> None:
    # ------------------------------------------------------------------ admin: doctors
    @spec.tool(ToolMeta(
        name="admin_list_doctors", allowed_roles=ADMINS, permission="doctor.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="All doctors, including ones switched off."))
    @rules
    def admin_list_doctors() -> dict:
        return ok_rows(admin_repo.list_all_doctors(), "No doctors are on record.")

    @spec.tool(ToolMeta(
        name="add_doctor", allowed_roles=ADMINS, permission="doctor.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Add a doctor record (available immediately)."))
    @rules
    def add_doctor(doctor_name: str, specialization: str, experience: int, phone: str, email: str,
                   consultation_fee: float) -> dict:
        name = clean_text(doctor_name, "Doctor name", 2, 100)
        special = clean_text(specialization, "Specialization", 2, 60)
        years = whole_number(experience, "Experience (years)", 0, 70)
        number, mail = clean_phone(phone), clean_email(email)
        fee = money(consultation_fee, "Consultation fee", 100_000)
        if admin_repo.doctor_email_exists(mail):
            return fail(ErrorCode.CONFLICT, "A doctor with that email already exists.")
        new_id = admin_repo.insert_doctor(name, special, years, number, mail, fee)
        return ok({"doctor_id": new_id, "doctor_name": name, "specialization": special, "available": True},
                  message="Doctor added.")

    @spec.tool(ToolMeta(
        name="update_doctor", allowed_roles=ADMINS, permission="doctor.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Change a doctor's name, specialization, experience, phone or fee. Leave a field empty to keep it."))
    @rules
    def update_doctor(doctor_id: int, doctor_name: Optional[str] = None, specialization: Optional[str] = None,
                      experience: Optional[int] = None, phone: Optional[str] = None,
                      consultation_fee: Optional[float] = None) -> dict:
        doctor_id = positive_int(doctor_id, "Doctor")
        fields = {}
        if (v := opt_text(doctor_name, "Doctor name", 2, 100)) is not None: fields["doctor_name"] = v
        if (v := opt_text(specialization, "Specialization", 2, 60)) is not None: fields["specialization"] = v
        if experience is not None: fields["experience"] = whole_number(experience, "Experience (years)", 0, 70)
        if phone is not None and str(phone).strip(): fields["phone"] = clean_phone(phone)
        if consultation_fee is not None: fields["consultation_fee"] = money(consultation_fee, "Consultation fee", 100_000)
        if not fields:
            raise BadInput("Nothing to update.")
        if not _doctor_or_fail(doctor_id):
            return fail(ErrorCode.NOT_FOUND, "That doctor could not be found.")
        admin_repo.update_doctor(doctor_id, fields)
        return ok(_doctor_or_fail(doctor_id), message="Doctor updated.")

    @spec.tool(ToolMeta(
        name="set_doctor_enabled", allowed_roles=ADMINS, permission="doctor.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Enable or disable a doctor (disabled doctors are hidden from patients)."))
    @rules
    def set_doctor_enabled(doctor_id: int, enabled: bool) -> dict:
        doctor_id = positive_int(doctor_id, "Doctor")
        if not _doctor_or_fail(doctor_id):
            return fail(ErrorCode.NOT_FOUND, "That doctor could not be found.")
        admin_repo.set_doctor_available(doctor_id, bool(enabled))
        return ok({"doctor_id": doctor_id, "available": bool(enabled)}, message="Doctor updated.")

    @spec.tool(ToolMeta(
        name="list_doctor_applications", allowed_roles=ADMINS, permission="doctor.approve",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Doctor sign-ups waiting for approval."))
    @rules
    def list_doctor_applications() -> dict:
        return ok_rows(admin_repo.list_pending_doctor_applications(), "No doctor applications are pending.")

    @spec.tool(ToolMeta(
        name="review_doctor_application", allowed_roles=ADMINS, permission="doctor.approve",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Approve or reject a pending doctor sign-up. decision: approve | reject."))
    @rules
    def review_doctor_application(applicant_user_id: int, decision: str) -> dict:
        applicant = positive_int(applicant_user_id, "Applicant")
        choice = clean_text(decision, "Decision", 4, 10).lower()
        if choice not in ("approve", "reject"):
            raise BadInput("Decision must be approve or reject.")
        new_status = approved_status() if choice == "approve" else "REJECTED"
        if not admin_repo.decide_doctor_application(applicant, new_status):
            return fail(ErrorCode.NOT_FOUND, "No pending doctor application was found for that user.")
        return ok({"user_id": applicant, "doctor_status": new_status}, message="Application updated.")

    # ------------------------------------------------------------------ admin: appointments, availability
    @spec.tool(ToolMeta(
        name="admin_list_appointments", allowed_roles=ADMINS, permission="appointment.read.all",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Appointments across all doctors. Optional status and date (YYYY-MM-DD); newest first, max 100."))
    @rules
    def admin_list_appointments(status: Optional[str] = None, appointment_date: Optional[str] = None) -> dict:
        wanted = None
        if status and status.strip():
            wanted = status.strip().upper()
            if wanted not in APPOINTMENT_STATUSES:
                raise BadInput("Status must be BOOKED, CONFIRMED, COMPLETED or CANCELLED.")
        day = parse_date(appointment_date) if appointment_date and appointment_date.strip() else None
        return ok_rows(admin_repo.list_appointments(wanted, day, 100), "No appointments match.")

    @spec.tool(ToolMeta(
        name="admin_set_appointment_status", allowed_roles=ADMINS, permission="appointment.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="confirm (BOOKED), cancel (BOOKED/CONFIRMED) or complete (CONFIRMED) any appointment."))
    @rules
    def admin_set_appointment_status(appointment_id: int, action: str) -> dict:
        appointment_id = positive_int(appointment_id, "Appointment")
        act = clean_text(action, "Action", 4, 10).lower()
        if act not in ADMIN_ACTIONS:
            raise BadInput("Action must be confirm, cancel or complete.")
        if not admin_repo.set_appointment_status(appointment_id, act):
            return fail(ErrorCode.CONFLICT, "That change isn't allowed for this appointment right now.")
        return ok({"appointment_id": appointment_id, "action": act}, message="Appointment updated.")

    @spec.tool(ToolMeta(
        name="admin_get_doctor_availability", allowed_roles=ADMINS, permission="doctor.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Upcoming availability slots for any doctor."))
    @rules
    def admin_get_doctor_availability(doctor_id: int) -> dict:
        doctor_id = positive_int(doctor_id, "Doctor")
        if not _doctor_or_fail(doctor_id):
            return fail(ErrorCode.NOT_FOUND, "That doctor could not be found.")
        return ok_rows(dash.get_upcoming_slots(doctor_id, date.today()), "This doctor has no upcoming availability.")

    @spec.tool(ToolMeta(
        name="admin_add_availability", allowed_roles=ADMINS, permission="doctor.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Add an availability slot for a doctor (date YYYY-MM-DD, times HH:MM)."))
    @rules
    def admin_add_availability(doctor_id: int, slot_date: str, start_time: str, end_time: str) -> dict:
        doctor_id = positive_int(doctor_id, "Doctor")
        day, start, end = parse_date(slot_date), parse_time(start_time), parse_time(end_time)
        if not _doctor_or_fail(doctor_id):
            return fail(ErrorCode.NOT_FOUND, "That doctor could not be found.")
        done, message = dash.add_slot(doctor_id, day, start, end, date.today())
        return ok({"doctor_id": doctor_id}, message=message) if done else fail(ErrorCode.CONFLICT, message)

    @spec.tool(ToolMeta(
        name="admin_remove_availability", allowed_roles=ADMINS, permission="doctor.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Remove a doctor's availability slot (refused while active appointments fall inside it)."))
    @rules
    def admin_remove_availability(doctor_id: int, availability_id: int) -> dict:
        doctor_id, slot_id = positive_int(doctor_id, "Doctor"), positive_int(availability_id, "Slot")
        done, message = dash.remove_slot(doctor_id, slot_id)
        return ok({"doctor_id": doctor_id, "availability_id": slot_id}, message=message) if done \
            else fail(ErrorCode.CONFLICT, message)

    # ------------------------------------------------------------------ doctor: own schedule only
    @spec.tool(ToolMeta(
        name="get_my_doctor_profile", allowed_roles=DOCTORS, permission="doctor.profile.read.own",
        ownership=Ownership.DOCTOR_OWN, kind=ToolKind.READ, requires_verified_data=True,
        description="The signed-in doctor's own doctor record."))
    @rules
    def get_my_doctor_profile() -> dict:
        row = doctors.get_doctor_by_id(current_principal().doctor_id)
        return ok(row) if row else fail(ErrorCode.NOT_FOUND, "Your doctor record could not be found.")

    @spec.tool(ToolMeta(
        name="get_my_assigned_appointments", allowed_roles=DOCTORS, permission="appointment.read.assigned",
        ownership=Ownership.DOCTOR_OWN, kind=ToolKind.READ, requires_verified_data=True,
        description="Appointments assigned to the signed-in doctor. view: today | upcoming | past | all. "
                    "Optional status filter."))
    @rules
    def get_my_assigned_appointments(view: str = "today", status: Optional[str] = None) -> dict:
        chosen = (view or "today").strip().lower()
        if chosen not in VIEWS:
            raise BadInput("View must be today, upcoming, past or all.")
        wanted = ""
        if status and status.strip():
            wanted = status.strip().upper()
            if wanted not in APPOINTMENT_STATUSES:
                raise BadInput("Status must be BOOKED, CONFIRMED, COMPLETED or CANCELLED.")
        rows = dash.get_doctor_appointments(current_principal().doctor_id, date.today(), chosen, wanted, "", 200)
        return ok_rows(rows, "No appointments match.")

    @spec.tool(ToolMeta(
        name="update_my_appointment_status", allowed_roles=DOCTORS, permission="appointment.manage.assigned",
        ownership=Ownership.DOCTOR_OWN, kind=ToolKind.WRITE,
        description="confirm | complete | cancel | mark-paid for an appointment assigned to the signed-in doctor."))
    @rules
    def update_my_appointment_status(appointment_id: int, action: str) -> dict:
        appointment_id = positive_int(appointment_id, "Appointment")
        act = clean_text(action, "Action", 4, 10).lower()
        if act not in DOCTOR_ACTIONS:
            raise BadInput("Action must be confirm, complete, cancel or mark-paid.")
        done, message = dash.update_appointment_status(current_principal().doctor_id, appointment_id, act, date.today())
        return ok({"appointment_id": appointment_id, "action": act}, message=message) if done \
            else fail(ErrorCode.CONFLICT, message)

    @spec.tool(ToolMeta(
        name="get_my_availability", allowed_roles=DOCTORS, permission="doctor.profile.read.own",
        ownership=Ownership.DOCTOR_OWN, kind=ToolKind.READ, requires_verified_data=True,
        description="The signed-in doctor's upcoming availability slots."))
    @rules
    def get_my_availability() -> dict:
        return ok_rows(dash.get_upcoming_slots(current_principal().doctor_id, date.today()),
                       "You have no upcoming availability.")

    @spec.tool(ToolMeta(
        name="add_my_availability", allowed_roles=DOCTORS, permission="doctor.availability.update.own",
        ownership=Ownership.DOCTOR_OWN, kind=ToolKind.WRITE,
        description="Add an availability slot for the signed-in doctor (date YYYY-MM-DD, times HH:MM)."))
    @rules
    def add_my_availability(slot_date: str, start_time: str, end_time: str) -> dict:
        day, start, end = parse_date(slot_date), parse_time(start_time), parse_time(end_time)
        done, message = dash.add_slot(current_principal().doctor_id, day, start, end, date.today())
        return ok({"slot_date": day.isoformat()}, message=message) if done else fail(ErrorCode.CONFLICT, message)

    @spec.tool(ToolMeta(
        name="remove_my_availability", allowed_roles=DOCTORS, permission="doctor.availability.update.own",
        ownership=Ownership.DOCTOR_OWN, kind=ToolKind.WRITE,
        description="Remove one of the signed-in doctor's availability slots."))
    @rules
    def remove_my_availability(availability_id: int) -> dict:
        slot_id = positive_int(availability_id, "Slot")
        done, message = dash.remove_slot(current_principal().doctor_id, slot_id)
        return ok({"availability_id": slot_id}, message=message) if done else fail(ErrorCode.CONFLICT, message)

    @spec.tool(ToolMeta(
        name="set_my_accepting_appointments", allowed_roles=DOCTORS, permission="doctor.availability.update.own",
        ownership=Ownership.DOCTOR_OWN, kind=ToolKind.WRITE,
        description="Turn the signed-in doctor's visibility to patients on or off."))
    @rules
    def set_my_accepting_appointments(accepting: bool) -> dict:
        dash.set_doctor_accepting(current_principal().doctor_id, bool(accepting))
        return ok({"available": bool(accepting)}, message="Updated.")
