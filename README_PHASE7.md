# Phase 7 - MCP 09 Security / RBAC / Audit

Unzip into your project root (the folder with `main.py`). Phases 1-6 must already be applied.

## New files
| File | Purpose |
|---|---|
| `mcp_layer/servers/mcp_09_security_audit.py` | 6 tools (below) |
| `src/database/mcp_security_repository.py` | read-only audit / security queries + one insert into `mcp_audit_logs` (no new table, no migration) |
| `tests/test_mcp_phase7.py` | RBAC, validation, outage and audit tests with in-memory fakes |

## Existing files replaced (additions only - diff them first)
- `mcp_layer/servers/__init__.py` - `register_tools()` also registers MCP 09
- `mcp_layer/bootstrap.py` - `build_gateway()` also gives MCP 09 the gateway (used by `check_permission`)
- `tests/test_mcp_phase5.py`, `tests/test_mcp_phase6.py` - one test each: MCP 09 is no longer ping-only

## Tools
**SUPER_ADMIN only (read):** `get_audit_logs`, `get_audit_summary`, `get_security_events`, `check_permission`, `check_role`
**SYSTEM only (write):** `write_audit_log` - internal callers use `Principal.system()`; no browser session can reach it.
Admins, doctors and users are refused at the server level.

- `check_permission(role_name, server_id, tool_name)` asks the gateway's own RBAC engine (including database revocations), so the answer matches a real call. It never runs the tool.
- `write_audit_log` accepts only a fixed list of event types and a short code as the reason, never free text.
- Audit rows hold argument NAMES only, never values.

## NOT built here
`security_check()` - it must wrap the Phase 10 safety layer, so it is built there. Nothing calls `write_audit_log` yet; Phases 9-11 (grounding gate, safety layer, chatbot) will.

## Run
`pytest tests/`, restart the app, then (as a SUPER_ADMIN) call `get_audit_summary` and `check_permission` through the gateway and look at `mcp_audit_logs`.
