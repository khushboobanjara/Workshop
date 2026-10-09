"""Phase 5: the user MCP servers (MCP 01-05), run end to end through the real gateway + RBAC.

The database is replaced by in-memory fakes (monkeypatched onto the repository modules), so these
tests need no MySQL. They DO need fastmcp, pydantic and mysql-connector installed (as the earlier
phase tests do).
"""
import asyncio
import sys
import types
from datetime import date, datetime, time, timedelta

import mysql.connector.errors as mye
import pytest

from mcp_layer.gateway import MCPGateway
from mcp_layer.overrides import NoOverrides
from mcp_layer.permissions import validate_registry_permissions
from mcp_layer.rbac import RBACAuthorizer
from mcp_layer.schemas import ErrorCode, Principal, Role
from mcp_layer.servers import build_registry
from mcp_layer.servers.mcp_05_assistant import bind_gateway
from src.database import appointment_repository as arepo
from src.database import doctor_repository as drepo
from src.database import mcp_repository as mrepo
from src.database import pharmacy_repository as prepo

run = asyncio.run
P01, P02, P03, P04, P05 = ("mcp_01_patient_profile", "mcp_02_doctor_appointment", "mcp_03_pharmacy",
                           "mcp_04_health_screening", "mcp_05_ai_assistant")

USER = Principal(user_id=7, role=Role.USER)
OTHER = Principal(user_id=8, role=Role.USER)
ADMIN = Principal(user_id=1, role=Role.ADMIN)
DOCTOR = Principal(user_id=20, role=Role.DOCTOR, doctor_id=3)

DAY = date.today() + timedelta(days=3)
DAY_S = DAY.isoformat()


class Capture:
    def __init__(self): self.events = []
    def record(self, e): self.events.append(e)


def make_gateway():
    cap = Capture()
    gw = MCPGateway(build_registry(with_tools=True), authorizer=RBACAuthorizer(NoOverrides()), auditor=cap)
    bind_gateway(gw)
    return gw, cap


def call(gw, who, server, tool, args=None):
    return run(gw.invoke(who, server, tool, args or {}))


def db_down(*a, **k):
    raise mye.InterfaceError("2003: Can't connect to MySQL server on 'prod-db.internal:3306' (hunter2)", errno=2003)


# ===================================================================== fake database
class World:
    """Tiny in-memory stand-in for the tables the tools touch."""

    def __init__(self):
        self.users = {7: {"user_id": 7, "full_name": "Asha Rao", "email": "asha@example.com", "phone": "9876543210", "role": "PATIENT"},
                      8: {"user_id": 8, "full_name": "Ben Lee", "email": "ben@example.com", "phone": "9123456780", "role": "PATIENT"}}
        self.doctors = {1: {"doctor_id": 1, "doctor_name": "Dr. Mehta", "specialization": "Cardiologist", "experience": 10,
                            "consultation_fee": 500, "available": True},
                        2: {"doctor_id": 2, "doctor_name": "Dr. Off", "specialization": "Dentist", "experience": 3,
                            "consultation_fee": 300, "available": False}}
        self.windows = [{"availability_id": 11, "doctor_id": 1, "available_date": DAY, "start_time": timedelta(hours=9),
                         "end_time": timedelta(hours=17)}]
        self.appts, self.next_appt = {}, 100
        self.medicines = {5: {"medicine_id": 5, "medicine_name": "Paracetamol 500", "generic_name": "paracetamol", "category": "Fever",
                              "description": "", "price": 20, "stock_quantity": 10, "manufacturer": "X",
                              "prescription_required": False, "is_available": True},
                          6: {"medicine_id": 6, "medicine_name": "Out of stock", "generic_name": "g", "category": "c",
                              "description": "", "price": 5, "stock_quantity": 0, "manufacturer": "X",
                              "prescription_required": False, "is_available": False}}
        self.carts, self.orders, self.next_order = {}, {}, 500

    def install(self, mp):
        w = self
        mp.setattr(mrepo, "get_user_profile", lambda uid: dict(w.users[uid]) if uid in w.users else None)
        def upd(uid, name, phone): w.users[uid].update(full_name=name, phone=phone)
        mp.setattr(mrepo, "update_user_profile", upd)

        mp.setattr(drepo, "get_all_available_doctors", lambda: [d for d in w.doctors.values() if d["available"]])
        mp.setattr(drepo, "find_doctors_by_specialization",
                   lambda s: [d for d in w.doctors.values() if d["available"] and d["specialization"].lower() == s.lower()])
        mp.setattr(drepo, "get_doctor_by_id", lambda i: w.doctors.get(i))
        mp.setattr(drepo, "get_doctor_availability", lambda i: [x for x in w.windows if x["doctor_id"] == i])

        mp.setattr(arepo, "is_time_available",
                   lambda avail_id, t: time(9, 0) <= t < time(17, 0) and any(x["availability_id"] == avail_id for x in w.windows))
        mp.setattr(arepo, "is_slot_booked", lambda d, day, t: any(
            a["doctor_id"] == d and a["appointment_date"] == day and a["appointment_time"] == t and a["status"] in ("BOOKED", "CONFIRMED")
            for a in w.appts.values()))
        def create(uid, did, day, t, name, phone, pay="CASH"):
            w.next_appt += 1
            w.appts[w.next_appt] = {"appointment_id": w.next_appt, "user_id": uid, "doctor_id": did, "appointment_date": day,
                                    "appointment_time": t, "patient_name": name, "phone": phone, "status": "BOOKED",
                                    "payment_method": pay, "payment_status": "PENDING"}
            return w.next_appt
        mp.setattr(arepo, "create_appointment", create)
        mp.setattr(arepo, "get_appointment_by_id",
                   lambda aid, uid: dict(w.appts[aid]) if aid in w.appts and w.appts[aid]["user_id"] == uid else None)
        mp.setattr(arepo, "get_user_appointments", lambda uid: [dict(a) for a in w.appts.values() if a["user_id"] == uid])
        def cancel(aid, uid):
            a = w.appts.get(aid)
            if a and a["user_id"] == uid and a["status"] in ("BOOKED", "CONFIRMED"):
                a["status"] = "CANCELLED"
                return True
            return False
        mp.setattr(arepo, "cancel_appointment", cancel)
        def resched(aid, uid, day, t):
            a = w.appts.get(aid)
            if a and a["user_id"] == uid and a["status"] in ("BOOKED", "CONFIRMED"):
                a.update(appointment_date=day, appointment_time=t)
                return True
            return False
        mp.setattr(arepo, "reschedule_appointment", resched)

        mp.setattr(prepo, "search_medicines", lambda q: [m for m in w.medicines.values() if m["is_available"] and q.lower() in m["medicine_name"].lower()])
        mp.setattr(prepo, "get_medicine_by_id", lambda i: w.medicines.get(i))
        mp.setattr(prepo, "get_user_cart", lambda uid: list(w.carts.get(uid, [])))
        def add(uid, mid, qty=1):
            m = w.medicines.get(mid)
            if not m or not m["is_available"]:
                raise ValueError("Medicine is not available.")
            if qty > m["stock_quantity"]:
                raise ValueError(f"Only {m['stock_quantity']} units of {m['medicine_name']} are available.")
            w.carts.setdefault(uid, []).append({"medicine_id": mid, "medicine_name": m["medicine_name"], "quantity": qty})
        mp.setattr(prepo, "add_to_cart", add)
        def order(uid, name, phone, method, address, pay):
            if not w.carts.get(uid):
                raise ValueError("Your cart is empty.")
            w.next_order += 1
            w.orders[w.next_order] = {"order_id": w.next_order, "user_id": uid, "total_amount": 40, "delivery_method": method,
                                      "address": address, "phone": phone, "payment_method": pay, "payment_status": "PENDING",
                                      "order_status": "PLACED", "cashfree_order_id": "cf_secret", "cashfree_payment_session_id": "sess_secret",
                                      "created_at": datetime(2026, 10, 8), "items": []}
            w.carts[uid] = []
            return {"order_id": w.next_order, "total_amount": 40}
        mp.setattr(prepo, "create_pharmacy_order", order)
        mp.setattr(prepo, "get_pharmacy_order",
                   lambda oid, uid: dict(w.orders[oid]) if oid in w.orders and w.orders[oid]["user_id"] == uid else None)
        mp.setattr(mrepo, "list_user_pharmacy_orders", lambda uid: [dict(o) for o in w.orders.values() if o["user_id"] == uid])
        def cancel_order(oid, uid):
            o = w.orders.get(oid)
            if o and o["user_id"] == uid and o["order_status"] == "PLACED" and o["payment_status"] != "PAID":
                o["order_status"] = "CANCELLED"
                return True
            return False
        mp.setattr(mrepo, "cancel_pharmacy_order", cancel_order)


@pytest.fixture
def env(monkeypatch):
    world = World()
    world.install(monkeypatch)
    gw, cap = make_gateway()
    yield world, gw, cap
    bind_gateway(None)


# ============================================================================ registry
def test_permission_matrix_covers_every_real_tool():
    assert validate_registry_permissions(build_registry(with_tools=True)) == []


def test_default_registry_is_still_ping_only():
    assert build_registry().tool_count() == 10   # earlier phases' tests rely on this


def test_user_servers_have_real_tools_and_system_servers_do_not_yet():
    reg = build_registry(with_tools=True)
    for sid in (P01, P02, P03, P04, P05):
        assert len(reg.get(sid).tools) > 1
    assert len(reg.get("mcp_10_llm_monitoring").tools) > 1   # 06-08 got tools in Phase 6, 09 in 7, 10 in 8


# ============================================================================ RBAC on the new tools
@pytest.mark.parametrize("server,tool", [(P02, "find_doctors"), (P03, "search_medicine"), (P04, "run_screening"),
                                         (P05, "get_verified_context")])
def test_doctor_role_cannot_use_patient_servers(env, server, tool):
    _, gw, _ = env
    assert call(gw, DOCTOR, server, tool).error_code == ErrorCode.ACCESS_DENIED


def test_doctor_can_use_own_profile_tool(env):
    world, gw, _ = env
    world.users[20] = {"user_id": 20, "full_name": "Dr. Who", "email": "d@example.com", "phone": "9000000000", "role": "DOCTOR"}
    assert call(gw, DOCTOR, P01, "get_my_profile").success


def test_unauthenticated_is_rejected(env):
    _, gw, _ = env
    assert call(gw, None, P01, "get_my_profile").error_code == ErrorCode.UNAUTHENTICATED


@pytest.mark.parametrize("key", ["user_id", "patient_id", "owner_id", "doctor_id"])
def test_cannot_name_another_user_on_own_data_tools(env, key):
    _, gw, _ = env
    r = call(gw, USER, P01, "get_my_profile", {key: 8})
    assert r.error_code == ErrorCode.ACCESS_DENIED


def test_booking_is_always_for_the_signed_in_user_even_if_user_id_is_sent(env):
    world, gw, _ = env
    r = call(gw, USER, P02, "book_appointment", {"doctor_id": 1, "appointment_date": DAY_S, "appointment_time": "10:00",
                                                 "patient_name": "Asha Rao", "phone": "9876543210", "user_id": 8})
    if r.success:   # FastMCP may ignore the unknown argument; either way it must never be honoured
        assert all(a["user_id"] == 7 for a in world.appts.values())
    else:
        assert not world.appts


# ============================================================================ MCP 01
def test_get_my_profile_returns_only_own_row(env):
    _, gw, _ = env
    r = call(gw, USER, P01, "get_my_profile")
    assert r.success and r.verified and r.source == "mysql"
    assert r.data["user_id"] == 7 and "password_hash" not in r.data
    assert call(gw, OTHER, P01, "get_my_profile").data["user_id"] == 8


def test_update_my_profile_changes_only_the_caller(env):
    world, gw, _ = env
    r = call(gw, USER, P01, "update_my_profile", {"full_name": "  Asha  R ", "phone": "+91 98765-43211"})
    assert r.success and r.data["full_name"] == "Asha R" and r.data["phone"] == "+919876543211"
    assert world.users[8]["full_name"] == "Ben Lee"


@pytest.mark.parametrize("name,phone", [("A", "9876543210"), ("Asha", "abc"), ("Asha", "123"), ("", "9876543210")])
def test_update_my_profile_validates_input(env, name, phone):
    world, gw, _ = env
    r = call(gw, USER, P01, "update_my_profile", {"full_name": name, "phone": phone})
    assert r.error_code == ErrorCode.INVALID_INPUT and world.users[7]["full_name"] == "Asha Rao"


# ============================================================================ MCP 02
def test_find_doctors_lists_only_available_doctors(env):
    _, gw, _ = env
    r = call(gw, USER, P02, "find_doctors")
    assert [d["doctor_id"] for d in r.data] == [1] and r.metadata["row_count"] == 1
    assert call(gw, USER, P02, "find_doctors", {"specialization": "cardiologist"}).data[0]["doctor_id"] == 1


def test_empty_search_is_verified_empty_with_controlled_message(env):
    _, gw, _ = env
    r = call(gw, USER, P02, "find_doctors", {"specialization": "Neurologist"})
    assert r.success and r.verified and r.data == [] and r.metadata["row_count"] == 0 and r.message


def test_availability_for_unknown_doctor_is_not_found(env):
    _, gw, _ = env
    assert call(gw, USER, P02, "get_doctor_availability", {"doctor_id": 99}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, USER, P02, "get_doctor_availability", {"doctor_id": 1}).data[0]["availability_id"] == 11


BOOK = {"doctor_id": 1, "appointment_date": DAY_S, "appointment_time": "10:00", "patient_name": "Asha Rao", "phone": "9876543210"}


def test_book_appointment_success_is_pay_at_clinic(env):
    world, gw, _ = env
    r = call(gw, USER, P02, "book_appointment", BOOK)
    assert r.success and r.verified and r.data["status"] == "BOOKED" and r.data["payment_method"] == "CASH"
    assert world.appts[r.data["appointment_id"]]["user_id"] == 7


def test_double_booking_the_same_slot_is_a_conflict(env):
    _, gw, _ = env
    assert call(gw, USER, P02, "book_appointment", BOOK).success
    assert call(gw, OTHER, P02, "book_appointment", BOOK).error_code == ErrorCode.CONFLICT


@pytest.mark.parametrize("patch,code", [
    ({"appointment_date": (date.today() - timedelta(days=1)).isoformat()}, ErrorCode.INVALID_INPUT),
    ({"appointment_date": "31/12/2030"}, ErrorCode.INVALID_INPUT),
    ({"appointment_time": "25:99"}, ErrorCode.INVALID_INPUT),
    ({"appointment_time": "20:00"}, ErrorCode.CONFLICT),      # outside the doctor's window
    ({"doctor_id": 2}, ErrorCode.NOT_FOUND),                  # doctor switched off
    ({"doctor_id": 99}, ErrorCode.NOT_FOUND),
    ({"phone": "nope"}, ErrorCode.INVALID_INPUT),
    ({"patient_name": ""}, ErrorCode.INVALID_INPUT),
])
def test_book_appointment_rejections(env, patch, code):
    world, gw, _ = env
    r = call(gw, USER, P02, "book_appointment", {**BOOK, **patch})
    assert not r.success and not r.verified and r.error_code == code and not world.appts


def test_my_appointments_only_shows_own(env):
    _, gw, _ = env
    call(gw, USER, P02, "book_appointment", BOOK)
    assert len(call(gw, USER, P02, "get_my_appointments").data) == 1
    other = call(gw, OTHER, P02, "get_my_appointments")
    assert other.success and other.data == [] and other.metadata["row_count"] == 0


def test_cannot_cancel_or_reschedule_someone_elses_appointment(env):
    world, gw, _ = env
    aid = call(gw, USER, P02, "book_appointment", BOOK).data["appointment_id"]
    assert call(gw, OTHER, P02, "cancel_appointment", {"appointment_id": aid}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, OTHER, P02, "reschedule_appointment",
                {"appointment_id": aid, "new_date": DAY_S, "new_time": "11:00"}).error_code == ErrorCode.NOT_FOUND
    assert world.appts[aid]["status"] == "BOOKED" and world.appts[aid]["appointment_time"] == time(10, 0)


def test_cancel_then_cancel_again(env):
    _, gw, _ = env
    aid = call(gw, USER, P02, "book_appointment", BOOK).data["appointment_id"]
    assert call(gw, USER, P02, "cancel_appointment", {"appointment_id": aid}).success
    assert call(gw, USER, P02, "cancel_appointment", {"appointment_id": aid}).error_code == ErrorCode.CONFLICT


def test_reschedule_moves_the_appointment_and_blocks_taken_slots(env):
    world, gw, _ = env
    aid = call(gw, USER, P02, "book_appointment", BOOK).data["appointment_id"]
    call(gw, OTHER, P02, "book_appointment", {**BOOK, "appointment_time": "11:00"})
    taken = call(gw, USER, P02, "reschedule_appointment", {"appointment_id": aid, "new_date": DAY_S, "new_time": "11:00"})
    assert taken.error_code == ErrorCode.CONFLICT
    moved = call(gw, USER, P02, "reschedule_appointment", {"appointment_id": aid, "new_date": DAY_S, "new_time": "12:00"})
    assert moved.success and world.appts[aid]["appointment_time"] == time(12, 0)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_ids_are_rejected(env, bad):
    _, gw, _ = env
    assert call(gw, USER, P02, "cancel_appointment", {"appointment_id": bad}).error_code == ErrorCode.INVALID_INPUT


# ============================================================================ database failure
def test_database_outage_gives_unverified_database_unavailable_and_no_leak(env, monkeypatch):
    _, gw, cap = env
    monkeypatch.setattr(drepo, "get_all_available_doctors", db_down)
    r = call(gw, USER, P02, "find_doctors")
    assert not r.success and not r.verified and r.data is None and r.error_code == ErrorCode.DATABASE_UNAVAILABLE
    blob = r.model_dump_json() + repr(cap.events)
    assert "hunter2" not in blob and "prod-db" not in blob


def test_outage_during_write_does_not_claim_success(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(arepo, "create_appointment", db_down)
    r = call(gw, USER, P02, "book_appointment", BOOK)
    assert not r.success and r.error_code == ErrorCode.DATABASE_UNAVAILABLE


def test_booking_that_cannot_be_read_back_is_not_reported_as_success(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(arepo, "get_appointment_by_id", lambda a, u: None)
    r = call(gw, USER, P02, "book_appointment", BOOK)
    assert not r.success and not r.verified


# ============================================================================ MCP 03
def test_search_medicine_found_and_not_found(env):
    _, gw, _ = env
    assert call(gw, USER, P03, "search_medicine", {"query": "parace"}).data[0]["medicine_id"] == 5
    r = call(gw, USER, P03, "search_medicine", {"query": "unobtainium"})
    assert r.success and r.verified and r.data == [] and r.metadata["row_count"] == 0 and r.message


@pytest.mark.parametrize("q", ["%", "a%b", "x_y", "a", "", "z" * 61])
def test_search_medicine_rejects_odd_input(env, q):
    _, gw, _ = env
    assert call(gw, USER, P03, "search_medicine", {"query": q}).error_code == ErrorCode.INVALID_INPUT


def test_medicine_availability_never_invents_stock(env):
    _, gw, _ = env
    assert call(gw, USER, P03, "check_medicine_availability", {"medicine_id": 999}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, USER, P03, "check_medicine_availability", {"medicine_id": 5, "quantity": 3}).data["available"] is True
    assert call(gw, USER, P03, "check_medicine_availability", {"medicine_id": 5, "quantity": 11}).data["available"] is False
    assert call(gw, USER, P03, "check_medicine_availability", {"medicine_id": 6}).data["available"] is False


def test_nearby_pharmacies_range_errors_become_invalid_input(env, monkeypatch):
    _, gw, _ = env
    def bad(lat, lng, radius): raise ValueError("Latitude must be between -90 and 90.")
    monkeypatch.setattr(prepo, "get_nearby_pharmacies", bad)
    r = call(gw, USER, P03, "find_nearby_pharmacies", {"latitude": 200.0, "longitude": 0.0})
    assert r.error_code == ErrorCode.INVALID_INPUT and "Latitude" in r.message


def test_nearby_pharmacies_names_google_as_the_source(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(prepo, "get_nearby_pharmacies", lambda a, b, c: [{"name": "Apollo", "distance_km": 1.2}])
    r = call(gw, USER, P03, "find_nearby_pharmacies", {"latitude": 26.9, "longitude": 75.8})
    assert r.success and r.source == "google_places"


def test_cart_stock_message_comes_through_cleanly(env):
    _, gw, _ = env
    r = call(gw, USER, P03, "add_to_my_cart", {"medicine_id": 5, "quantity": 99})
    assert r.error_code == ErrorCode.INVALID_INPUT and "Only 10 units" in r.message
    assert call(gw, USER, P03, "add_to_my_cart", {"medicine_id": 5, "quantity": 2}).data[0]["quantity"] == 2
    assert call(gw, OTHER, P03, "get_my_cart").data == []


def test_order_flow_hides_payment_gateway_fields_and_scopes_to_owner(env):
    world, gw, _ = env
    call(gw, USER, P03, "add_to_my_cart", {"medicine_id": 5, "quantity": 2})
    placed = call(gw, USER, P03, "create_medicine_order",
                  {"patient_name": "Asha Rao", "phone": "9876543210", "delivery_method": "clinic_pickup"})
    assert placed.success and placed.data["payment_method"] == "CASH"
    assert "cashfree_payment_session_id" not in placed.data and "cashfree_order_id" not in placed.data
    oid = placed.data["order_id"]
    listing = call(gw, USER, P03, "get_my_orders")
    assert listing.data[0]["order_id"] == oid and "cashfree_order_id" not in listing.data[0]
    detail = call(gw, USER, P03, "get_my_order_details", {"order_id": oid})
    assert detail.success and "cashfree_payment_session_id" not in detail.data
    assert call(gw, OTHER, P03, "get_my_order_details", {"order_id": oid}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, OTHER, P03, "get_my_orders").data == []


def test_order_validation(env):
    _, gw, _ = env
    base = {"patient_name": "Asha Rao", "phone": "9876543210", "delivery_method": "HOME_DELIVERY"}
    assert call(gw, USER, P03, "create_medicine_order", base).error_code == ErrorCode.INVALID_INPUT          # no address
    assert call(gw, USER, P03, "create_medicine_order", {**base, "delivery_method": "DRONE"}).error_code == ErrorCode.INVALID_INPUT
    empty = call(gw, USER, P03, "create_medicine_order", {**base, "address": "12 MG Road, Jaipur"})
    assert empty.error_code == ErrorCode.INVALID_INPUT and "cart is empty" in empty.message


def test_cancel_order_rules(env):
    world, gw, _ = env
    call(gw, USER, P03, "add_to_my_cart", {"medicine_id": 5, "quantity": 1})
    oid = call(gw, USER, P03, "create_medicine_order",
               {"patient_name": "Asha Rao", "phone": "9876543210", "delivery_method": "CLINIC_PICKUP"}).data["order_id"]
    assert call(gw, OTHER, P03, "cancel_order", {"order_id": oid}).error_code == ErrorCode.NOT_FOUND
    assert world.orders[oid]["order_status"] == "PLACED"
    assert call(gw, USER, P03, "cancel_order", {"order_id": oid}).success
    assert call(gw, USER, P03, "cancel_order", {"order_id": oid}).error_code == ErrorCode.CONFLICT


def test_paid_order_cannot_be_cancelled(env):
    world, gw, _ = env
    call(gw, USER, P03, "add_to_my_cart", {"medicine_id": 5, "quantity": 1})
    oid = call(gw, USER, P03, "create_medicine_order",
               {"patient_name": "Asha Rao", "phone": "9876543210", "delivery_method": "CLINIC_PICKUP"}).data["order_id"]
    world.orders[oid]["payment_status"] = "PAID"
    assert call(gw, USER, P03, "cancel_order", {"order_id": oid}).error_code == ErrorCode.CONFLICT


# ============================================================================ MCP 04
@pytest.fixture
def fake_pipeline(monkeypatch):
    mod = types.ModuleType("src.pipeline.predict_pipeline")

    class CustomData:
        def __init__(self, age, gender, fever, cough, city): self.row = (age, gender, fever, cough, city)
        def get_data_as_dataframe(self): return self.row

    class PredictPipeline:
        def predict(self, features):
            if features[4] == "Atlantis":
                raise ValueError("'Atlantis' is not a supported value for city. Supported values: Jaipur, Delhi.")
            return "Positive", 83.5

    mod.CustomData, mod.PredictPipeline = CustomData, PredictPipeline
    pkg = types.ModuleType("src.pipeline"); pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "src.pipeline", pkg)
    monkeypatch.setitem(sys.modules, "src.pipeline.predict_pipeline", mod)


SCREEN = {"age": 30, "gender": "Female", "fever": 101.5, "cough": "Mild", "city": "Jaipur"}


def test_screening_result_is_labelled_screening_not_diagnosis(env, fake_pipeline):
    _, gw, _ = env
    r = call(gw, USER, P04, "run_screening", SCREEN)
    assert r.success and r.verified and r.source == "ml_model"
    assert r.data["result_type"] == "screening" and r.data["is_diagnosis"] is False
    assert "not a medical diagnosis" in r.data["disclaimer"] and r.data["screening_result"] == "Positive"


@pytest.mark.parametrize("patch", [{"age": 500}, {"age": -1}, {"fever": 5}, {"fever": 900}, {"city": ""}, {"gender": "x" * 31}])
def test_screening_rejects_out_of_range_input(env, fake_pipeline, patch):
    _, gw, _ = env
    assert call(gw, USER, P04, "run_screening", {**SCREEN, **patch}).error_code == ErrorCode.INVALID_INPUT


def test_screening_unsupported_category_gives_the_models_message(env, fake_pipeline):
    _, gw, _ = env
    r = call(gw, USER, P04, "run_screening", {**SCREEN, "city": "Atlantis"})
    assert r.error_code == ErrorCode.INVALID_INPUT and "Supported values" in r.message


# ============================================================================ MCP 05
def test_assistant_collects_verified_context_through_the_gateway(env):
    world, gw, cap = env
    call(gw, USER, P02, "book_appointment", BOOK)
    r = call(gw, USER, P05, "get_verified_context", {"topics": ["profile", "appointments", "profile"]})
    assert r.success and r.verified and set(r.data) == {"profile", "appointments"}
    assert r.data["profile"]["user_id"] == 7 and len(r.data["appointments"]) == 1
    called = {(e["server_id"], e["tool"]) for e in cap.events}
    assert (P01, "get_my_profile") in called and (P02, "get_my_appointments") in called   # sub-calls were audited


def test_assistant_context_is_scoped_to_the_caller(env):
    _, gw, _ = env
    call(gw, USER, P02, "book_appointment", BOOK)
    r = call(gw, OTHER, P05, "get_verified_context", {"topics": ["appointments"]})
    assert r.success and r.data["appointments"] == [] and r.metadata["row_counts"]["appointments"] == 0


def test_assistant_fails_closed_when_any_part_fails(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(arepo, "get_user_appointments", db_down)
    r = call(gw, USER, P05, "get_verified_context", {"topics": ["profile", "appointments"]})
    assert not r.success and not r.verified and r.data is None and r.error_code == ErrorCode.DATABASE_UNAVAILABLE


@pytest.mark.parametrize("topics", [[], ["passwords"], ["profile", "orders", "cart", "appointments", "profile2"], "profile"])
def test_assistant_rejects_bad_topics(env, topics):
    _, gw, _ = env
    assert not call(gw, USER, P05, "get_verified_context", {"topics": topics}).success


def test_assistant_without_a_gateway_is_unavailable(env):
    _, gw, _ = env
    bind_gateway(None)
    assert call(gw, USER, P05, "get_verified_context", {"topics": ["profile"]}).error_code == ErrorCode.MCP_UNAVAILABLE


# ============================================================================ audit
def test_audit_logs_argument_names_never_values(env):
    _, gw, cap = env
    call(gw, USER, P02, "book_appointment", {**BOOK, "patient_name": "Secret Person", "phone": "9876500000"})
    blob = repr(cap.events)
    assert "Secret Person" not in blob and "9876500000" not in blob
    event = [e for e in cap.events if e["tool"] == "book_appointment"][-1]
    assert event["permission_result"] == "ALLOWED" and "patient_name" in event["argument_names"]
