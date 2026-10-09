"""Phase 6: admin MCP servers (MCP 06 doctor/appointment admin, MCP 07 pharmacy admin, MCP 08 user admin),
run end to end through the real gateway + RBAC. The database is replaced by in-memory fakes.
Needs fastmcp, pydantic and mysql-connector installed (as the earlier phase tests do)."""
import asyncio
from datetime import date, timedelta

import mysql.connector.errors as mye
import pytest

from mcp_layer.gateway import MCPGateway
from mcp_layer.overrides import NoOverrides
from mcp_layer.permissions import validate_registry_permissions
from mcp_layer.rbac import RBACAuthorizer
from mcp_layer.schemas import ErrorCode, Principal, Role, ToolKind
from mcp_layer.servers import build_registry
from src.database import doctor_dashboard_repository as dash
from src.database import doctor_repository as drepo
from src.database import mcp_admin_repository as arepo

run = asyncio.run
P06, P07, P08 = "mcp_06_doctor_admin", "mcp_07_pharmacy_admin", "mcp_08_user_system_admin"

USER = Principal(user_id=7, role=Role.USER)
DOCTOR = Principal(user_id=20, role=Role.DOCTOR, doctor_id=3)
DOCTOR_UNLINKED = Principal(user_id=21, role=Role.DOCTOR, doctor_id=None)
ADMIN = Principal(user_id=1, role=Role.ADMIN)
SUPER = Principal(user_id=2, role=Role.SUPER_ADMIN)
FUTURE = (date.today() + timedelta(days=2)).isoformat()


class Capture:
    def __init__(self): self.events = []
    def record(self, e): self.events.append(e)


def make_gateway():
    cap = Capture()
    return MCPGateway(build_registry(with_tools=True), authorizer=RBACAuthorizer(NoOverrides()), auditor=cap), cap


def call(gw, who, server, tool, args=None):
    return run(gw.invoke(who, server, tool, args or {}))


def db_down(*a, **k):
    raise mye.InterfaceError("2003: Can't connect to MySQL server on 'prod-db.internal:3306'", errno=2003)


class World:
    def __init__(self):
        self.calls = []
        self.doctors = {1: {"doctor_id": 1, "doctor_name": "Dr. Mehta", "specialization": "Cardiologist", "experience": 10,
                            "phone": "9000000001", "email": "m@example.com", "consultation_fee": 500, "available": True}}
        self.users = {
            1: {"user_id": 1, "full_name": "Admin One", "email": "a@x.com", "phone": "9", "role": "ADMIN", "doctor_status": None, "email_verified": 1},
            2: {"user_id": 2, "full_name": "Boss", "email": "b@x.com", "phone": "9", "role": "SUPER_ADMIN", "doctor_status": None, "email_verified": 1},
            7: {"user_id": 7, "full_name": "Asha", "email": "s@x.com", "phone": "9", "role": "PATIENT", "doctor_status": None, "email_verified": 1},
            30: {"user_id": 30, "full_name": "Dr. New", "email": "n@x.com", "phone": "9", "role": "DOCTOR", "doctor_status": "PENDING", "email_verified": 1},
        }
        self.medicines = {5: {"medicine_id": 5, "medicine_name": "Paracetamol", "stock_quantity": 10, "is_available": True}}
        self.medicine_in_use = {5}
        self.orders = {500: {"order_id": 500, "order_status": "PLACED", "payment_method": "CASH", "payment_status": "PENDING"}}

    def install(self, mp):
        w = self
        rec = lambda name: (lambda *a, **k: w.calls.append((name, a, k)))

        mp.setattr(drepo, "get_doctor_by_id", lambda i: w.doctors.get(i))
        mp.setattr(arepo, "list_all_doctors", lambda: list(w.doctors.values()))
        mp.setattr(arepo, "doctor_email_exists", lambda e: any(d["email"] == e for d in w.doctors.values()))
        def insert(name, spec, exp, phone, email, fee):
            w.calls.append(("insert_doctor", (name, spec, exp, phone, email, fee), {}))
            w.doctors[9] = {"doctor_id": 9, "doctor_name": name, "specialization": spec, "email": email, "available": True}
            return 9
        mp.setattr(arepo, "insert_doctor", insert)
        def upd(i, fields): w.calls.append(("update_doctor", (i, fields), {})); w.doctors[i].update(fields)
        mp.setattr(arepo, "update_doctor", upd)
        mp.setattr(arepo, "set_doctor_available", lambda i, f: w.doctors[i].update(available=f))
        mp.setattr(arepo, "list_pending_doctor_applications",
                   lambda: [u for u in w.users.values() if u["role"] == "DOCTOR" and u["doctor_status"] == "PENDING"])
        def decide(uid, status):
            u = w.users.get(uid)
            if u and u["role"] == "DOCTOR" and u["doctor_status"] == "PENDING":
                u["doctor_status"] = status
                return True
            return False
        mp.setattr(arepo, "decide_doctor_application", decide)
        mp.setattr(arepo, "list_appointments", lambda s, d, n: [{"appointment_id": 1, "status": s or "BOOKED"}])
        mp.setattr(arepo, "set_appointment_status", lambda i, act: i == 1)

        mp.setattr(dash, "get_upcoming_slots", lambda did, today: [{"availability_id": 11, "doctor_id": did}])
        def add_slot(did, day, start, end, today):
            w.calls.append(("add_slot", (did, day, start, end), {}))
            return (False, "That overlaps a slot you already have on this date.") if start.hour == 9 else (True, "Availability added.")
        mp.setattr(dash, "add_slot", add_slot)
        def remove_slot(did, sid):
            w.calls.append(("remove_slot", (did, sid), {}))
            return (True, "Availability removed.") if sid == 11 else (False, "Slot not found.")
        mp.setattr(dash, "remove_slot", remove_slot)
        mp.setattr(dash, "get_doctor_appointments", lambda did, today, view, status, search, limit: (
            w.calls.append(("get_doctor_appointments", (did, view, status), {})) or [{"appointment_id": 1, "doctor_id": did}]))
        def dstatus(did, aid, act, today):
            w.calls.append(("update_appointment_status", (did, aid, act), {}))
            return (True, "Appointment confirmed.") if aid == 1 else (False, "That change isn't allowed for this appointment right now.")
        mp.setattr(dash, "update_appointment_status", dstatus)
        mp.setattr(dash, "set_doctor_accepting", lambda did, f: w.calls.append(("accepting", (did, f), {})))

        mp.setattr(arepo, "list_medicines", lambda s, a, n: list(w.medicines.values()))
        mp.setattr(arepo, "get_medicine", lambda i: w.medicines.get(i))
        mp.setattr(arepo, "insert_medicine", lambda *a: w.calls.append(("insert_medicine", a, {})) or 77)
        mp.setattr(arepo, "update_medicine", lambda i, f: w.calls.append(("update_medicine", (i, f), {})))
        mp.setattr(arepo, "set_medicine_stock", lambda i, q: w.medicines[i].update(stock_quantity=q))
        mp.setattr(arepo, "set_medicine_available", lambda i, f: w.medicines[i].update(is_available=f))
        def delete(i):
            w.calls.append(("delete_medicine", (i,), {}))
            if i not in w.medicines: return "NOT_FOUND"
            return "IN_USE" if i in w.medicine_in_use else "DELETED"
        mp.setattr(arepo, "delete_medicine_if_unused", delete)
        mp.setattr(arepo, "list_pharmacy_orders", lambda s, n: list(w.orders.values()))
        def order_status(i, status):
            o = w.orders.get(i)
            if not o: return "NOT_FOUND"
            if o["order_status"] != "PLACED": return "NOT_ALLOWED"
            o["order_status"] = status
            return "OK"
        mp.setattr(arepo, "set_pharmacy_order_status", order_status)

        mp.setattr(arepo, "list_users", lambda role, search, hide_super, n: (
            w.calls.append(("list_users", (role, hide_super), {})) or
            [u for u in w.users.values() if not (hide_super and u["role"] == "SUPER_ADMIN")]))
        mp.setattr(arepo, "get_user", lambda i: dict(w.users[i]) if i in w.users else None)
        def set_role(uid, role, pending_doctor):
            w.calls.append(("set_user_role", (uid, role, pending_doctor), {}))
            w.users[uid]["role"] = role
            if pending_doctor and not w.users[uid]["doctor_status"]: w.users[uid]["doctor_status"] = "PENDING"
        mp.setattr(arepo, "set_user_role", set_role)


@pytest.fixture
def env(monkeypatch):
    world = World()
    world.install(monkeypatch)
    gw, cap = make_gateway()
    return world, gw, cap


def names(world): return [c[0] for c in world.calls]


# ===================================================================== registry
def test_permission_matrix_covers_every_tool_including_admin_ones():
    assert validate_registry_permissions(build_registry(with_tools=True)) == []


def test_only_system_servers_are_still_ping_only():
    reg = build_registry(with_tools=True)
    assert set(reg.get("mcp_10_llm_monitoring").tools) == {"server_ping"}   # MCP 09 got tools in Phase 7
    for sid in (P06, P07, P08):
        assert len(reg.get(sid).tools) > 1


def test_exactly_one_destructive_tool_and_users_cannot_reach_it():
    destructive = [(s.server_id, n, m) for s in build_registry(with_tools=True).servers()
                   for n, m in s.tools.items() if m.kind is ToolKind.DESTRUCTIVE]
    assert [(s, n) for s, n, _ in destructive] == [(P07, "remove_medicine")]
    assert Role.USER not in destructive[0][2].allowed_roles and Role.DOCTOR not in destructive[0][2].allowed_roles


ADMIN_ONLY = [(P06, "admin_list_doctors"), (P06, "add_doctor"), (P06, "update_doctor"), (P06, "set_doctor_enabled"),
              (P06, "list_doctor_applications"), (P06, "review_doctor_application"), (P06, "admin_list_appointments"),
              (P06, "admin_set_appointment_status"), (P06, "admin_get_doctor_availability"), (P06, "admin_add_availability"),
              (P06, "admin_remove_availability"),
              (P07, "admin_list_medicines"), (P07, "add_medicine"), (P07, "update_medicine"), (P07, "set_medicine_stock"),
              (P07, "set_medicine_availability"), (P07, "remove_medicine"), (P07, "admin_list_pharmacy_orders"),
              (P07, "update_pharmacy_order_status"),
              (P08, "list_users"), (P08, "get_user_details")]


@pytest.mark.parametrize("server,tool", ADMIN_ONLY + [(P08, "set_user_role")])
@pytest.mark.parametrize("who", [USER, DOCTOR])
def test_users_and_doctors_cannot_call_any_admin_tool(env, who, server, tool):
    world, gw, _ = env
    assert call(gw, who, server, tool).error_code == ErrorCode.ACCESS_DENIED
    assert not world.calls


@pytest.mark.parametrize("server,tool", [(P06, "admin_list_doctors"), (P07, "admin_list_medicines"), (P08, "list_users")])
@pytest.mark.parametrize("who", [ADMIN, SUPER])
def test_admins_can_call_read_tools(env, who, server, tool):
    _, gw, _ = env
    assert call(gw, who, server, tool).success


def test_set_user_role_is_super_admin_only(env):
    world, gw, cap = env
    assert call(gw, ADMIN, P08, "set_user_role", {"target_user_id": 7, "new_role": "DOCTOR"}).error_code == ErrorCode.ACCESS_DENIED
    assert world.users[7]["role"] == "PATIENT"
    assert cap.events[-1]["denial_reason"] in ("role_not_allowed_on_tool", "permission_not_held")


def test_admins_cannot_use_doctor_only_tools(env):
    _, gw, _ = env
    for who in (ADMIN, SUPER, USER):
        assert call(gw, who, P06, "get_my_doctor_profile").error_code == ErrorCode.ACCESS_DENIED


# ===================================================================== doctor-own tools
def test_doctor_profile_uses_the_session_doctor_id(env):
    _, gw, _ = env
    world, _, _ = env
    world.doctors[3] = {"doctor_id": 3, "doctor_name": "Dr. Own", "available": True}
    r = call(gw, DOCTOR, P06, "get_my_doctor_profile")
    assert r.success and r.data["doctor_id"] == 3


def test_doctor_without_doctor_row_is_denied(env):
    _, gw, cap = env
    assert call(gw, DOCTOR_UNLINKED, P06, "get_my_availability").error_code == ErrorCode.ACCESS_DENIED
    assert cap.events[-1]["denial_reason"] == "no_doctor_identity"


@pytest.mark.parametrize("key", ["doctor_id", "user_id", "patient_id", "owner_id"])
def test_doctor_cannot_choose_whose_schedule_to_touch(env, key):
    world, gw, _ = env
    r = call(gw, DOCTOR, P06, "add_my_availability", {key: 99, "slot_date": FUTURE, "start_time": "10:00", "end_time": "12:00"})
    assert r.error_code == ErrorCode.ACCESS_DENIED and not world.calls


def test_doctor_actions_are_always_scoped_to_their_own_doctor_id(env):
    world, gw, _ = env
    assert call(gw, DOCTOR, P06, "update_my_appointment_status", {"appointment_id": 1, "action": "confirm"}).success
    assert call(gw, DOCTOR, P06, "get_my_assigned_appointments", {"view": "upcoming"}).success
    assert call(gw, DOCTOR, P06, "add_my_availability", {"slot_date": FUTURE, "start_time": "10:00", "end_time": "12:00"}).success
    assert call(gw, DOCTOR, P06, "remove_my_availability", {"availability_id": 11}).success
    assert call(gw, DOCTOR, P06, "set_my_accepting_appointments", {"accepting": False}).success
    assert call(gw, DOCTOR, P06, "get_my_availability").success
    assert world.calls and all(c[1][0] == 3 for c in world.calls)       # first argument of each repo call is doctor_id 3


def test_doctor_appointment_action_validation_and_refusal(env):
    _, gw, _ = env
    assert call(gw, DOCTOR, P06, "update_my_appointment_status", {"appointment_id": 1, "action": "delete"}).error_code == ErrorCode.INVALID_INPUT
    refused = call(gw, DOCTOR, P06, "update_my_appointment_status", {"appointment_id": 2, "action": "confirm"})
    assert refused.error_code == ErrorCode.CONFLICT and "isn't allowed" in refused.message
    assert call(gw, DOCTOR, P06, "get_my_assigned_appointments", {"view": "forever"}).error_code == ErrorCode.INVALID_INPUT
    assert call(gw, DOCTOR, P06, "get_my_assigned_appointments", {"status": "MAYBE"}).error_code == ErrorCode.INVALID_INPUT


def test_doctor_availability_repo_messages_come_through(env):
    _, gw, _ = env
    overlap = call(gw, DOCTOR, P06, "add_my_availability", {"slot_date": FUTURE, "start_time": "09:00", "end_time": "12:00"})
    assert overlap.error_code == ErrorCode.CONFLICT and "overlaps" in overlap.message
    assert call(gw, DOCTOR, P06, "remove_my_availability", {"availability_id": 999}).error_code == ErrorCode.CONFLICT
    assert call(gw, DOCTOR, P06, "add_my_availability", {"slot_date": "tomorrow", "start_time": "10:00", "end_time": "12:00"}).error_code == ErrorCode.INVALID_INPUT


# ===================================================================== MCP 06 admin
ADD = {"doctor_name": "Dr. Rao", "specialization": "Dentist", "experience": 5, "phone": "9876543210",
       "email": "Rao@Example.com", "consultation_fee": 400}


def test_add_doctor_normalises_and_blocks_duplicates(env):
    world, gw, _ = env
    r = call(gw, ADMIN, P06, "add_doctor", ADD)
    assert r.success and r.data["doctor_id"] == 9 and world.calls[-1][1][4] == "rao@example.com"
    assert call(gw, ADMIN, P06, "add_doctor", {**ADD, "email": "m@example.com"}).error_code == ErrorCode.CONFLICT


@pytest.mark.parametrize("patch", [{"doctor_name": ""}, {"experience": -1}, {"experience": 90}, {"phone": "x"},
                                   {"email": "not-an-email"}, {"consultation_fee": -5}, {"consultation_fee": 10**7}])
def test_add_doctor_validation(env, patch):
    world, gw, _ = env
    assert call(gw, ADMIN, P06, "add_doctor", {**ADD, **patch}).error_code == ErrorCode.INVALID_INPUT
    assert "insert_doctor" not in names(world)


def test_update_doctor(env):
    world, gw, _ = env
    assert call(gw, ADMIN, P06, "update_doctor", {"doctor_id": 1}).error_code == ErrorCode.INVALID_INPUT      # nothing to change
    assert call(gw, ADMIN, P06, "update_doctor", {"doctor_id": 99, "doctor_name": "X Y"}).error_code == ErrorCode.NOT_FOUND
    r = call(gw, ADMIN, P06, "update_doctor", {"doctor_id": 1, "consultation_fee": 650.5})
    assert r.success and world.doctors[1]["consultation_fee"] == 650.5


def test_enable_disable_doctor(env):
    world, gw, _ = env
    assert call(gw, ADMIN, P06, "set_doctor_enabled", {"doctor_id": 1, "enabled": False}).success
    assert world.doctors[1]["available"] is False
    assert call(gw, ADMIN, P06, "set_doctor_enabled", {"doctor_id": 99, "enabled": True}).error_code == ErrorCode.NOT_FOUND


def test_doctor_application_review(env, monkeypatch):
    world, gw, _ = env
    monkeypatch.delenv("MCP_DOCTOR_APPROVED_STATUS", raising=False)
    assert [a["user_id"] for a in call(gw, ADMIN, P06, "list_doctor_applications").data] == [30]
    assert call(gw, ADMIN, P06, "review_doctor_application", {"applicant_user_id": 7, "decision": "approve"}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, ADMIN, P06, "review_doctor_application", {"applicant_user_id": 30, "decision": "maybe"}).error_code == ErrorCode.INVALID_INPUT
    r = call(gw, ADMIN, P06, "review_doctor_application", {"applicant_user_id": 30, "decision": "Approve"})
    assert r.success and world.users[30]["doctor_status"] == "APPROVED"
    assert call(gw, ADMIN, P06, "review_doctor_application", {"applicant_user_id": 30, "decision": "reject"}).error_code == ErrorCode.NOT_FOUND


def test_reject_application_and_configured_approved_word(env, monkeypatch):
    world, gw, _ = env
    assert call(gw, ADMIN, P06, "review_doctor_application", {"applicant_user_id": 30, "decision": "reject"}).data["doctor_status"] == "REJECTED"
    world.users[30]["doctor_status"] = "PENDING"
    monkeypatch.setenv("MCP_DOCTOR_APPROVED_STATUS", "active")
    assert call(gw, ADMIN, P06, "review_doctor_application", {"applicant_user_id": 30, "decision": "approve"}).data["doctor_status"] == "ACTIVE"


def test_a_blocked_approved_word_is_refused_not_applied(env, monkeypatch):
    world, gw, _ = env
    monkeypatch.setenv("MCP_DOCTOR_APPROVED_STATUS", "PENDING")      # would leave the doctor locked out
    r = call(gw, ADMIN, P06, "review_doctor_application", {"applicant_user_id": 30, "decision": "approve"})
    assert not r.success and world.users[30]["doctor_status"] == "PENDING"


def test_admin_appointments(env):
    _, gw, _ = env
    assert call(gw, ADMIN, P06, "admin_list_appointments").success
    assert call(gw, ADMIN, P06, "admin_list_appointments", {"status": "weird"}).error_code == ErrorCode.INVALID_INPUT
    assert call(gw, ADMIN, P06, "admin_list_appointments", {"appointment_date": "31/12/2030"}).error_code == ErrorCode.INVALID_INPUT
    assert call(gw, ADMIN, P06, "admin_set_appointment_status", {"appointment_id": 1, "action": "complete"}).success
    assert call(gw, ADMIN, P06, "admin_set_appointment_status", {"appointment_id": 2, "action": "cancel"}).error_code == ErrorCode.CONFLICT
    assert call(gw, ADMIN, P06, "admin_set_appointment_status", {"appointment_id": 1, "action": "mark-paid"}).error_code == ErrorCode.INVALID_INPUT


def test_admin_manages_any_doctors_availability(env):
    world, gw, _ = env
    assert call(gw, ADMIN, P06, "admin_get_doctor_availability", {"doctor_id": 1}).success
    assert call(gw, ADMIN, P06, "admin_get_doctor_availability", {"doctor_id": 99}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, ADMIN, P06, "admin_add_availability",
                {"doctor_id": 1, "slot_date": FUTURE, "start_time": "10:00", "end_time": "12:00"}).success
    assert world.calls[-1][1][0] == 1
    assert call(gw, ADMIN, P06, "admin_remove_availability", {"doctor_id": 1, "availability_id": 999}).error_code == ErrorCode.CONFLICT


# ===================================================================== MCP 07
MED = {"medicine_name": "Cetirizine", "generic_name": "cetirizine", "category": "Allergy", "description": "Antihistamine",
       "price": 35.5, "stock_quantity": 40, "manufacturer": "Acme"}


def test_add_medicine_and_validation(env):
    world, gw, _ = env
    assert call(gw, ADMIN, P07, "add_medicine", MED).data["medicine_id"] == 77
    for patch in ({"price": -1}, {"stock_quantity": -3}, {"medicine_name": " "}, {"price": "free"}):
        assert call(gw, ADMIN, P07, "add_medicine", {**MED, **patch}).error_code == ErrorCode.INVALID_INPUT
    assert names(world).count("insert_medicine") == 1


def test_update_medicine(env):
    world, gw, _ = env
    assert call(gw, ADMIN, P07, "update_medicine", {"medicine_id": 5}).error_code == ErrorCode.INVALID_INPUT
    assert call(gw, ADMIN, P07, "update_medicine", {"medicine_id": 99, "price": 3}).error_code == ErrorCode.NOT_FOUND
    r = call(gw, ADMIN, P07, "update_medicine", {"medicine_id": 5, "price": 12, "prescription_required": True})
    assert r.success and r.data["updated"] == ["prescription_required", "price"]


def test_stock_and_availability(env):
    world, gw, _ = env
    r = call(gw, ADMIN, P07, "set_medicine_stock", {"medicine_id": 5, "stock_quantity": 25})
    assert r.data["previous_stock"] == 10 and world.medicines[5]["stock_quantity"] == 25
    assert call(gw, ADMIN, P07, "set_medicine_stock", {"medicine_id": 5, "stock_quantity": -1}).error_code == ErrorCode.INVALID_INPUT
    assert call(gw, ADMIN, P07, "set_medicine_stock", {"medicine_id": 99, "stock_quantity": 1}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, ADMIN, P07, "set_medicine_availability", {"medicine_id": 5, "is_available": False}).success
    assert world.medicines[5]["is_available"] is False


def test_remove_medicine_needs_explicit_confirmation(env):
    world, gw, _ = env
    assert call(gw, ADMIN, P07, "remove_medicine", {"medicine_id": 5}).error_code == ErrorCode.INVALID_INPUT
    assert call(gw, ADMIN, P07, "remove_medicine", {"medicine_id": 5, "confirm": False}).error_code == ErrorCode.INVALID_INPUT
    assert "delete_medicine" not in names(world)


def test_remove_medicine_outcomes(env):
    world, gw, cap = env
    assert call(gw, SUPER, P07, "remove_medicine", {"medicine_id": 5, "confirm": True}).error_code == ErrorCode.CONFLICT   # used by orders
    world.medicines[6] = {"medicine_id": 6, "medicine_name": "Unused", "stock_quantity": 0, "is_available": True}
    assert call(gw, ADMIN, P07, "remove_medicine", {"medicine_id": 6, "confirm": True}).success
    assert call(gw, ADMIN, P07, "remove_medicine", {"medicine_id": 99, "confirm": True}).error_code == ErrorCode.NOT_FOUND
    assert any(e["tool"] == "remove_medicine" for e in cap.events)                                      # destructive calls are audited


def test_pharmacy_order_admin(env):
    world, gw, _ = env
    assert call(gw, ADMIN, P07, "admin_list_pharmacy_orders").success
    assert call(gw, ADMIN, P07, "update_pharmacy_order_status", {"order_id": 500, "new_status": "DELIVERED"}).error_code == ErrorCode.INVALID_INPUT
    assert call(gw, ADMIN, P07, "update_pharmacy_order_status", {"order_id": 999, "new_status": "confirmed"}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, ADMIN, P07, "update_pharmacy_order_status", {"order_id": 500, "new_status": "confirmed"}).success
    assert call(gw, ADMIN, P07, "update_pharmacy_order_status", {"order_id": 500, "new_status": "CANCELLED"}).error_code == ErrorCode.CONFLICT


# ===================================================================== MCP 08
def test_admin_never_sees_super_admin_accounts(env):
    _, gw, _ = env
    seen = {u["user_id"] for u in call(gw, ADMIN, P08, "list_users").data}
    assert 2 not in seen and {1, 7, 30} <= seen
    assert call(gw, SUPER, P08, "list_users").data and 2 in {u["user_id"] for u in call(gw, SUPER, P08, "list_users").data}
    assert call(gw, ADMIN, P08, "get_user_details", {"target_user_id": 2}).error_code == ErrorCode.NOT_FOUND
    assert call(gw, SUPER, P08, "get_user_details", {"target_user_id": 2}).success


def test_user_listing_never_contains_secrets_and_validates_filters(env):
    _, gw, _ = env
    blob = call(gw, ADMIN, P08, "list_users").model_dump_json().lower()
    assert "password" not in blob
    assert call(gw, ADMIN, P08, "list_users", {"role_filter": "OWNER"}).error_code == ErrorCode.INVALID_INPUT


def test_set_user_role_rules(env):
    world, gw, _ = env
    ok = call(gw, SUPER, P08, "set_user_role", {"target_user_id": 7, "new_role": "admin"})
    assert ok.success and world.users[7]["role"] == "ADMIN"
    assert call(gw, SUPER, P08, "set_user_role", {"target_user_id": 2, "new_role": "PATIENT"}).error_code == ErrorCode.CONFLICT   # yourself
    other_boss = Principal(user_id=3, role=Role.SUPER_ADMIN)
    assert call(gw, other_boss, P08, "set_user_role", {"target_user_id": 2, "new_role": "PATIENT"}).error_code == ErrorCode.CONFLICT  # a super admin row
    assert world.users[2]["role"] == "SUPER_ADMIN"
    assert call(gw, SUPER, P08, "set_user_role", {"target_user_id": 99, "new_role": "ADMIN"}).error_code == ErrorCode.NOT_FOUND
    for bad in ("SUPER_ADMIN", "SYSTEM", "root", ""):
        assert call(gw, SUPER, P08, "set_user_role", {"target_user_id": 7, "new_role": bad}).error_code == ErrorCode.INVALID_INPUT


def test_new_doctor_role_starts_pending(env):
    world, gw, _ = env
    r = call(gw, SUPER, P08, "set_user_role", {"target_user_id": 7, "new_role": "DOCTOR"})
    assert r.success and world.calls[-1][1] == (7, "DOCTOR", True) and world.users[7]["doctor_status"] == "PENDING"


def test_role_argument_smuggling_is_blocked(env):
    world, gw, _ = env
    r = call(gw, SUPER, P08, "set_user_role", {"target_user_id": 7, "new_role": "ADMIN", "role": "SUPER_ADMIN"})
    assert r.error_code == ErrorCode.ACCESS_DENIED and world.users[7]["role"] == "PATIENT"


# ===================================================================== failures and audit
@pytest.mark.parametrize("who,server,tool,target", [
    (ADMIN, P06, "admin_list_doctors", "list_all_doctors"),
    (ADMIN, P07, "admin_list_medicines", "list_medicines"),
    (ADMIN, P08, "list_users", "list_users"),
])
def test_database_outage_is_unverified_everywhere(env, monkeypatch, who, server, tool, target):
    _, gw, cap = env
    monkeypatch.setattr(arepo, target, db_down)
    r = call(gw, who, server, tool)
    assert not r.success and not r.verified and r.data is None and r.error_code == ErrorCode.DATABASE_UNAVAILABLE
    assert "prod-db" not in r.model_dump_json() + repr(cap.events)


def test_write_failure_midway_does_not_claim_success(env, monkeypatch):
    _, gw, _ = env
    monkeypatch.setattr(arepo, "set_medicine_stock", db_down)
    assert call(gw, ADMIN, P07, "set_medicine_stock", {"medicine_id": 5, "stock_quantity": 3}).error_code == ErrorCode.DATABASE_UNAVAILABLE


def test_admin_writes_are_audited_without_values(env):
    _, gw, cap = env
    call(gw, ADMIN, P06, "add_doctor", {**ADD, "email": "secret.person@example.com"})
    assert "secret.person@example.com" not in repr(cap.events)
    event = [e for e in cap.events if e["tool"] == "add_doctor"][-1]
    assert event["permission_result"] == "ALLOWED" and "email" in event["argument_names"]
