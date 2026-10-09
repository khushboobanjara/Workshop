# Phase 10 - Safety layer

Unzip into your project root (the folder with `main.py`). Phases 1-9 must already be applied.

## Files
| File | Change |
|---|---|
| `mcp_layer/safety.py` | NEW. `SafetyLayer` (the gateway's `SafetyGate`), `RateLimiter`, public `scan_text()` |
| `mcp_layer/bootstrap.py` | 3 small edits: `build_gateway()` now uses `SafetyLayer.from_settings(settings)` unless you pass `safety=` |
| `tests/test_mcp_phase10.py` | NEW. 27 tests (24 unit, 3 run inside the real gateway) |

## RBAC vs Safety
RBAC (unchanged) = "may this role do this?" and already rejects `role, is_admin, token, password, sql, command, code`
and named owners (`user_id, patient_id, owner_id, doctor_id`). Safety = "should this call run?". It runs after RBAC
allows and before the tool runs. First problem blocks; the caller sees only "This request cannot be processed.";
the reason goes to `mcp_audit_logs.denial_reason` with `security_result=BLOCKED`.

| Order | Check | Reason code |
|---|---|---|
| 1 | Rate limit per principal: read = `MCP_RATE_LIMIT_PER_MINUTE` (60), write <= 20, destructive <= 5. Blocked calls count too. SYSTEM is exempt. | `rate_limited` |
| 2 | Arguments are a dict of simple values: <= 30 keys, clean key names, depth <= 3, strings <= 2000 chars, lists <= 50, finite numbers | `unsafe_arguments` |
| 3 | Escalation keys: `permissions, acting_as, impersonate, principal, run_as, sudo, admin, is_super_admin ...` | `role_escalation_attempt` |
| 4 | Sensitive fields: `password_hash, api_key, otp, cvv, card_number, cookie, authorization ...` | `sensitive_field` |
| 5 | Every `*_id` is a positive int <= 2,000,000,000 (`server_id`/`mcp_server_id` must be registry-style names) | `invalid_id` |
| 6 | Text scan (not for SYSTEM): SQL / direct-DB manipulation, prompt injection, `../`, `<script>`, `file://`, NUL | `sql_injection`, `prompt_injection`, `unsafe_text` |
| 7 | DESTRUCTIVE tools need `confirm is True` | `destructive_unconfirmed` |

The text scan normalises full-width letters, zero-width characters, case and spacing before matching.
Ordinary text such as `Dr. O'Brien`, `Amoxicillin -- 250 mg`, `select a doctor from the list` and `ignore the swelling` is tested to pass.

## Behaviour changes to know about
- `build_gateway()` now has safety ON. `remove_medicine` without `confirm=true` is blocked by Safety (generic message) before the tool's own helpful message. The UI/chatbot must ask the admin to confirm first.
- `scan_text(text)` is public: Phase 11 should call it on the chatbot message before intent routing / the LLM.

## Limits (be honest in your report)
- The text scan is a tripwire, not the security boundary. The boundary is RBAC + ownership + parameterised queries.
- Rate limiting is in memory per process; several workers each count separately. RBAC-denied calls never reach Safety, so they are not rate limited here.
- Invalid-ID checks only cover top-level arguments named `id` or `*_id`.

## Run
`python -m pytest tests -q`
