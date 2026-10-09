# Phase 5 - user MCP servers (MCP 01-05)

Unzip into your project root (the folder that contains `main.py`). Paths are project-relative.

## New files
| File | Purpose |
|---|---|
| `mcp_layer/servers/_common.py` | validation + result helpers shared by the tool modules |
| `mcp_layer/servers/mcp_01_patient.py` | `get_my_profile`, `update_my_profile` |
| `mcp_layer/servers/mcp_02_appointment.py` | `find_doctors`, `get_doctor_availability`, `book_appointment`, `get_my_appointments`, `cancel_appointment`, `reschedule_appointment` |
| `mcp_layer/servers/mcp_03_pharmacy.py` | `search_medicine`, `check_medicine_availability`, `find_nearby_pharmacies`, `get_my_cart`, `add_to_my_cart`, `create_medicine_order`, `get_my_orders`, `get_my_order_details`, `cancel_order` |
| `mcp_layer/servers/mcp_04_screening.py` | `run_screening` (labelled screening, never diagnosis) |
| `mcp_layer/servers/mcp_05_assistant.py` | `get_verified_context` (calls MCP 01-03 through the gateway, all-or-nothing) |
| `src/database/mcp_repository.py` | new queries: user profile, list/cancel pharmacy orders. Existing repositories are untouched |
| `tests/test_mcp_phase5.py` | tests using in-memory fake repositories (no MySQL needed) |
| `scripts/mcp_smoke.py` | read-only smoke test against your real database |

## Existing files that are REPLACED (small edits - diff them before overwriting)
- `mcp_layer/schemas.py` - adds `ErrorCode.NOT_FOUND` and `ErrorCode.CONFLICT`
- `mcp_layer/servers/__init__.py` - `build_registry(with_tools=False)` + `register_tools()`. The default stays ping-only so your Phase 1-4 tests are unchanged
- `mcp_layer/bootstrap.py` - `build_gateway()` now builds the registry with tools and binds the gateway to MCP 05

## Before you run
1. `SHOW COLUMNS FROM pharmacy_orders LIKE 'order_status';` - `cancel_order` sets `'CANCELLED'`. If your ENUM lacks it, add it (keep the existing values) or the cancel tool will fail.
2. `pytest tests/` (all phases), then `python scripts/mcp_smoke.py <a real user_id>`.

## Decisions you may want to change
- Appointments and medicine orders made through MCP are pay-at-clinic (CASH). Online payment stays in your existing Cashfree routes.
- `get_my_health_profile` is not built: there is no health-profile table in the repositories.
- `run_screening` does not store results; saving/reading history needs a new table (a later migration).
- Prescription-required medicines are not blocked by the order tool (your repository does not check it either).
- `book_appointment` / `get_doctor_availability` are `Ownership.NONE` because the RBAC layer denies any `doctor_id` argument on `current_user_only` tools. The patient is never an argument.
