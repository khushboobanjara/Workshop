# Model Context Protocol (MCP) integration

OpsPilot AI can act as an MCP server so an AI assistant can answer questions about invoices **from the database, with the
caller's permissions, and with every action audited**. It is read-only, off by default, and adds no dependency.

## What was added (and what was not touched)

| File | Change |
| --- | --- |
| `app/mcp_server/*` | new package (all MCP code) |
| `app/main.py` | 3 lines: import, `startup_checks()` in the lifespan, `include_router` |
| `app/config.py` | 6 settings (`OPSPILOT_MCP_*`) |
| `.env.example`, `docs/MCP.md`, `README.md` | documentation |
| `tests/mcp_support.py`, `tests/test_mcp_*.py` | new tests |

No existing endpoint, model, table, test or dependency was changed. There is no schema change: the `users.role` CHECK
constraint is untouched, and MCP roles are a mapping on top of the existing ones.

## Architecture

```
 MCP client ──HTTP POST /mcp──▶ routes.py ─┐   (switch, Origin, auth, size)
 MCP client ──stdio──────────▶ stdio.py  ─┤   (re-authenticates every message)
                                           ▼
                                   protocol.py  JSON-RPC dispatcher
              rate limit → tool exists → role allowed → args valid → run → serialise → AUDIT ROW → reply
                                           │
                       tools.py ──▶ app.services.processing (same code as the REST API) ──▶ MySQL
                                           │
                       grounding.py (allow-listed fields, missing_fields, envelope) + sanitize.py (redaction)
```

The protocol core never decides who the caller is; the transport authenticates first. Both transports share one core, so
HTTP and stdio can never enforce different rules.

## Roles

| MCP role | Application role | Credential |
| --- | --- | --- |
| **User** | `viewer`, `reviewer` | the user's JWT (`POST /auth/login` → `access_token`) as `Authorization: Bearer …` |
| **Super Admin** | `admin` | an admin's JWT |
| **System** | `service` | `OPSPILOT_MCP_SYSTEM_KEY` as `Authorization: Bearer …` |

Each tool lists the application roles allowed to call it, so the reviewer/viewer split is exactly the REST API's.
`rev1` can read the review queue; `view1` cannot, over REST or over MCP.

| Tool | viewer | reviewer | admin | system |
| --- | :-: | :-: | :-: | :-: |
| `get_my_access`, `get_metrics`, `list_models` | ✓ | ✓ | ✓ | ✓ |
| `search_invoices`, `get_invoice`, `list_vendors` | ✓ | ✓ | ✓ | |
| `get_invoice_audit_trail`, `get_review_queue` | | ✓ | ✓ | |
| `get_system_status` | | | ✓ | ✓ |
| `get_audit_log`, `get_settings`, `list_users` | | | ✓ | |

The system role gets aggregates and health only: no invoice, vendor or e-mail content.

## Database-grounded answers

* Every result is built by an **allow-list** of columns. The e-mail body, `password_hash`, `token_version`, keys and
  connection strings are never selected.
* A value that is not stored comes back as `null` and is named in `missing_fields`; it is never defaulted.
* "Nothing found" is explicit: `found: false`, `data: null` (or an empty list) and a note saying not to invent one.
* Each result has an envelope: `source`, `tool`, `as_of`, `record_refs` (e.g. `invoice:12`), `tables`,
  `untrusted_text_fields`, `notes`, `data`. The server's `initialize` reply also carries rules for the client model.
* Aggregates are computed by the same service functions as the REST API (tests assert the two agree).

**Limit, stated plainly:** the server guarantees that *its* output is real, complete about absence, and traceable. It cannot
stop a client model from inventing things in its *own* reply. `record_refs` exist so answers can be checked.

## Security controls

| Risk | Control |
| --- | --- |
| Unauthenticated access | MCP refuses to start unless `OPSPILOT_AUTH_ENABLED=true`; every request needs a valid Bearer token |
| Revoked users still connected | stateless; JWT signature, expiry, active flag and `token_version` re-checked on **every** request (stdio: every message) |
| Ambient browser credentials / CSRF | session cookie and ingest `X-API-Key` are **not** accepted; `Origin` must be in `OPSPILOT_MCP_ALLOWED_ORIGINS` |
| Privilege escalation | permission checked on every `tools/call`, independent of `tools/list`; deny by default; `service` only counts if it came from the MCP system key |
| Injection / bad input | strict argument models (unknown fields, wrong types, ranges, patterns); parameterised SQL; LIKE wildcards escaped; rejected values never echoed or logged |
| Prompt injection via e-mail text | e-mail bodies never returned; remaining outside text is cleaned of hidden/bidi/control characters and listed in `untrusted_text_fields` |
| Sensitive data in free text | card numbers (Luhn), IBANs (mod-97), SSN patterns redacted; sender addresses masked. Best-effort defence in depth, the field allow-list is the primary control |
| Abuse | per-principal rate limit, message size limit (also for chunked bodies), result size limit, failed-sign-in throttle |
| Tampering with the record | MCP rows go into the append-only `audit_log` (database triggers refuse UPDATE/DELETE) |
| Silent actions | **fail closed**: if the audit row cannot be written, the data is not returned |
| Error leakage | unexpected errors return a reference id; details only in the server log |
| Writes by an AI | none: no approve/reject/correct/ingest tools; the only write is the audit row |
| Public demo | MCP cannot be enabled together with demo mode |

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPSPILOT_MCP_ENABLED` | `false` | master switch; with it off `/mcp` answers 404 |
| `OPSPILOT_MCP_SYSTEM_KEY` | empty | ≥ 32 chars; enables the System role; must differ from `OPSPILOT_API_KEY` |
| `OPSPILOT_MCP_STDIO_TOKEN` | empty | credential for `python -m app.mcp_server` (a user JWT or the system key) |
| `OPSPILOT_MCP_ALLOWED_ORIGINS` | empty | comma-separated browser origins allowed to call `/mcp` |
| `OPSPILOT_MCP_RATE_LIMIT_PER_MINUTE` | `60` | tool calls per principal per minute |
| `OPSPILOT_MCP_MAX_MESSAGE_BYTES` | `65536` | maximum request size |

Also required: `OPSPILOT_AUTH_ENABLED=true` and `OPSPILOT_JWT_SECRET` (≥ 32 chars). Generate keys with
`python -c "import secrets; print(secrets.token_urlsafe(48))"`. Secrets live in `.env` (git-ignored) or the environment.

## Trying it

HTTP (PowerShell):

```powershell
$login = Invoke-RestMethod -Method Post http://127.0.0.1:8000/auth/login -ContentType application/json `
         -Body '{"username":"admin","password":"<your password>"}'
$h = @{ Authorization = "Bearer $($login.access_token)"; Accept = "application/json, text/event-stream" }
Invoke-RestMethod -Method Post http://127.0.0.1:8000/mcp -Headers $h -ContentType application/json `
  -Body '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_invoices","arguments":{"status":"pending_review"}}}'
```

stdio (e.g. a desktop client's MCP config; put the other `OPSPILOT_*` values in your `.env`):

```json
{ "mcpServers": { "opspilot": {
    "command": "C:\\path\\to\\OPS-AI\\.venv\\Scripts\\python.exe",
    "args": ["-m", "app.mcp_server"],
    "env": { "PYTHONPATH": "C:\\path\\to\\OPS-AI", "OPSPILOT_ENV_FILE": "C:\\path\\to\\OPS-AI\\.env",
             "OPSPILOT_MCP_STDIO_TOKEN": "<system key or a user JWT>" } } } }
```

Review what assistants did: dashboard Audit page, `GET /audit?stage=mcp`, or the `get_audit_log` tool.

## Tests

```powershell
pytest tests/test_mcp_unit.py tests/test_mcp_protocol.py tests/test_mcp_security.py tests/test_mcp_grounding.py tests/test_mcp_stdio.py -q
pytest -q          # the whole suite: the 182 existing tests plus the MCP tests
```

## Limitations and next steps

* Implements the MCP subset needed for tools (initialize, ping, tools/list, tools/call) over plain JSON; no SSE streaming,
  sessions, resources or prompts. Swapping in the official `mcp` SDK later only touches the transport files: tools and RBAC are independent.
* User JWTs expire after `OPSPILOT_SESSION_MINUTES` (default 480). Long-running desktop clients should use the system key
  or, better, per-user API tokens / OAuth, which would be the next step.
* Rate limits and the sign-in throttle are in memory, per process.
* Write tools (approve/reject) are deliberately absent. If added later they should require a confirmation step, the
  reviewer role, and an audit row *before* the change.
* The `service` principal exists in the application, but a real `super_admin` database role was not added, to avoid a schema change.
