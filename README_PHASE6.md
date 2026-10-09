# Phase 6 - admin MCP servers (MCP 06, 07, 08)

Unzip into your project root (the folder with `main.py`). Phase 5 must already be applied.

## New files
| File | Purpose |
|---|---|
| `src/database/mcp_admin_repository.py` | admin queries (doctors, appointments, medicines, orders, users). Existing repositories untouched |
| `mcp_layer/servers/mcp_06_doctor_admin.py` | 11 admin tools + 7 doctor-own tools |
| `mcp_layer/servers/mcp_07_pharmacy_admin.py` | 8 tools, one of them DESTRUCTIVE (`remove_medicine`) |
| `mcp_layer/servers/mcp_08_user_admin.py` | `list_users`, `get_user_details`, `set_user_role` (SUPER_ADMIN only) |
| `tests/test_mcp_phase6.py` | RBAC, validation, separation-of-powers, outage and audit tests with in-memory fakes |

## Existing files replaced (additions only - diffed against the zip you sent)
- `mcp_layer/servers/__init__.py` - `register_tools()` now also registers MCP 06-08
- `mcp_layer/servers/_common.py` - adds `opt_text`, `money`, `whole_number`, `clean_email` (nothing removed)
- `tests/test_mcp_phase5.py` - one test changed: admin servers 06-08 are no longer ping-only

## Tools
**MCP 06 admin (ADMIN, SUPER_ADMIN):** admin_list_doctors, add_doctor, update_doctor, set_doctor_enabled, list_doctor_applications, review_doctor_application, admin_list_appointments, admin_set_appointment_status, admin_get_doctor_availability, admin_add_availability, admin_remove_availability
**MCP 06 doctor (DOCTOR, own data only - the doctor id comes from the session, never an argument):** get_my_doctor_profile, get_my_assigned_appointments, update_my_appointment_status, get_my_availability, add_my_availability, remove_my_availability, set_my_accepting_appointments
**MCP 07 (ADMIN, SUPER_ADMIN):** admin_list_medicines, add_medicine, update_medicine, set_medicine_stock, set_medicine_availability, remove_medicine (needs confirm=true; refuses medicines that appear in any order), admin_list_pharmacy_orders, update_pharmacy_order_status
**MCP 08:** list_users and get_user_details (ADMIN, SUPER_ADMIN; admins never see SUPER_ADMIN accounts); set_user_role (SUPER_ADMIN only; targets PATIENT / DOCTOR / ADMIN; never yourself, never a SUPER_ADMIN, never creates one)

## Things to know
- Approving a doctor takes TWO things, same as your `sql/approve_doctor.sql`: `review_doctor_application` sets `users.doctor_status` to APPROVED, and a `doctors` row with the SAME email must exist (`add_doctor` creates it). Set `MCP_DOCTOR_APPROVED_STATUS` in `.env` only if you ever use a word other than APPROVED.
- `set_user_role` to DOCTOR leaves the account PENDING until it is approved (the MCP role resolver refuses a doctor with no status).
- `pharmacy_orders.order_status` must accept `'CANCELLED'` (same check as Phase 5): `SHOW COLUMNS FROM pharmacy_orders LIKE 'order_status';`
- Admin order statuses are limited to PLACED -> CONFIRMED (cash orders) and PLACED -> CANCELLED (unpaid; cash stock is restored). Online orders are confirmed by your Cashfree callback, and paid orders need a manual refund, so neither can be changed here.
- NOT built: activate/deactivate users. `users` has no active flag and login would have to check it, so it belongs with the Phase 12 route work. Admin-visible user activity belongs to MCP 09 (audit logs).
- Admin appointment lists include patient names but not phone numbers.

## Run
`pytest tests/` (all phases), then restart the app and run `python scripts/mcp_smoke.py <an admin user_id>` if you want to eyeball it.
