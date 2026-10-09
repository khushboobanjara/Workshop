# Phase 8 - MCP 10 LLM / Token / Cost / Monitoring

Unzip into your project root (the folder with `main.py`). Phases 1-7 must already be applied.

## New files
| File | Purpose |
|---|---|
| `mcp_layer/llm_cost.py` | exact Decimal cost maths (no database, no prices in code) |
| `mcp_layer/llm_tracking.py` | `usage_from_response()` and `track_llm_usage()` for the chatbot (used in Phase 11); never raises |
| `mcp_layer/servers/mcp_10_llm_monitoring.py` | 7 tools (below) |
| `src/database/mcp_llm_repository.py` | pricing and usage queries on the existing `llm_pricing` / `llm_usage` tables (no migration) |
| `tests/test_mcp_phase8.py` | cost maths, RBAC, validation, outage, idempotency and audit tests with in-memory fakes |

## Existing files replaced (additions only - diff them first)
- `mcp_layer/servers/__init__.py` - `register_tools()` also registers MCP 10
- `mcp_layer/bootstrap.py` - `build_gateway()` also gives MCP 10 the gateway (used by `get_mcp_health`)
- `tests/test_mcp_phase5.py`, `test_mcp_phase6.py`, `test_mcp_phase7.py` - one test each: MCP 10 is no longer ping-only

## Tools
**SYSTEM only:** `record_llm_usage` (write), `calculate_cost` (read) - internal code, never a browser session.
**SUPER_ADMIN only:** `get_llm_usage_summary`, `get_llm_usage_logs`, `get_llm_pricing` (read), `set_llm_pricing` (write).
**SYSTEM + SUPER_ADMIN:** `get_mcp_health` - pings all 10 servers and adds the last 24 hours of audit activity per server.

## Rules built in
- Prices come only from the `llm_pricing` table. Price in force = newest ACTIVE row whose `effective_from` has arrived. `set_llm_pricing` adds rows, so price history is kept.
- No active price for a model: the request is still stored (tokens count) but its costs are NULL and `priced=false`. Nothing is guessed. The summary counts these as `unpriced_requests`.
- `request_ref` (the gateway request id) makes `record_llm_usage` idempotent: a repeat gives CONFLICT and is never counted twice.
- Costs are exact text in per-request results (`"0.00045000"`), numbers in summaries, and never added across currencies.
- Audit rows hold argument NAMES only.

## You must do once
Add a price for the model you use, from your provider's current price page (I did not invent one). Either call `set_llm_pricing` as SUPER_ADMIN, or run SQL:
`INSERT INTO llm_pricing (model_name, input_price_per_million, output_price_per_million, effective_from) VALUES ('openai/gpt-oss-120b', <input>, <output>, CURDATE());`
Until then every request is stored as unpriced.

## NOT wired yet
Nothing calls `track_llm_usage` yet. `chatbot/llm.py` returns only the reply text and drops `response.usage`; changing that and calling the tracker belongs to Phase 11 (chatbot migration).

## Run
`python -m pytest tests -q`, restart the app, then as SUPER_ADMIN call `get_mcp_health`.
