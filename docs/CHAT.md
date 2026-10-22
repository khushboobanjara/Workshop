# AI assistant (Groq + MCP)

A chat page in the dashboard (**AI Assistant** in the sidebar) that answers invoice and operations questions by calling the
MCP tools. Groq's language model decides *which* tools to call and words the answer; **the database is the only source of facts**.
It is read-only, off by default, and adds no dependency (it uses `httpx`, already in `requirements.txt`).

## How a question is answered

```
browser ──POST /chat (your login)──▶ routes.py   401 if not signed in, role check, rate limit, input limits
                                        │
                                        ▼
                                   service.py ◀──────────────▶ Groq  (chat completions with tool definitions)
                                        │   model asks for a tool
                                        ▼
                                   bridge.py ──▶ MCP core (as YOU): role check → strict arguments → rate limit → run → AUDIT ROW
                                        │   tool results go back to the model
                                        ▼
                                   verifier.py + fact check: figures must be in the tool results, every statement must be supported by them
                                        │   ok → answer      not ok → one repair attempt → otherwise a fixed "could not verify" text
                                        ▼
                                   audit_log: one 'chat' row per question (next to the per-tool 'mcp' rows)
```

The browser never talks to Groq and never sees the API key. Groq's own "remote MCP" feature is **not** used: it would run
tools without your per-user permissions and audit trail. Instead the tools run locally, in-process, through the same MCP core
as `POST /mcp`.

## Turning it on

Add to `.env` (authentication must already be on, as for MCP):

```
OPSPILOT_AUTH_ENABLED=true            # already set
OPSPILOT_CHAT_ENABLED=true
OPSPILOT_GROQ_API_KEY=gsk_...         # create at https://console.groq.com/keys
# optional
OPSPILOT_GROQ_MODEL=openai/gpt-oss-120b
OPSPILOT_CHAT_RATE_LIMIT_PER_MINUTE=20
OPSPILOT_CHAT_FACT_CHECK=true        # second model call that checks every statement against the records
```

Restart uvicorn (`--reload` does not watch `.env`), sign in, and open **AI Assistant**. The app refuses to start if chat is
enabled without authentication, in demo mode, without a key, or with a non-https `OPSPILOT_GROQ_BASE_URL`.

`OPSPILOT_MCP_ENABLED` is independent: the assistant uses the MCP core in-process and works whether or not the public `/mcp`
endpoint is switched on. Any Groq model that supports tool calling works; check Groq's current list at
https://console.groq.com/docs/tool-use/overview. Groq retires models over time: if chat fails with
"model does not exist", run `python scripts/check_groq.py`, which lists the models your key can use. (`openai/gpt-oss-*`
models do not support parallel tool calls; this app handles either.)

## What is sent to Groq

The question, the recent conversation, the tool definitions your role may use, and the tool results needed to answer
(invoice numbers, vendor names, amounts, dates, reasons). The MCP layer's protections apply first: e-mail bodies are never
included, card numbers / IBANs / SSN patterns are masked, sender addresses are masked. Amounts and vendor names **do** leave
the server. Do not enable the assistant if that is not acceptable for your data; review Groq's data policy first.

## "Only verified database results": how it is enforced

| Control | What it does |
| --- | --- |
| **Scope is enforced by the server** | Unless the message is a plain greeting, an answer is accepted only after at least one tool lookup was attempted. A model that answers a general-knowledge question ("what is python") straight away is sent back once, then the user gets a fixed text saying the assistant only answers questions about OpsPilot data. (`tool_choice=required` is sent as a hint, but some models ignore it, so it is never relied on.) |
| Same permissions as the user | Tools run through the MCP core with the user's identity; the model is only offered the tools its user may call, and a call to anything else is refused and audited |
| Allow-listed, redacted data | Tool results come from MCP's allow-lists; `null` / `missing_fields` / `found: false` tell the model what is not in the records |
| **Figure check** | Every amount, percentage, date, identifier and whole number (including small counts such as 0) in the answer must appear in this turn's tool results |
| **Fact check** (`OPSPILOT_CHAT_FACT_CHECK`, on by default) | A second model call compares every statement in the answer with the tool results and flags anything the records do not state: numbers attached to the wrong field, invented detail, and conclusions such as "healthy" or "models are loaded". Flagged answers get one repair attempt, then are replaced by a fixed text and never shown |
| Untrusted text | Vendor names, subjects and comments are labelled `untrusted_text_fields`; the prompt says to treat them as data, never instructions |
| Forged history | The client may send earlier turns, but only `user`/`assistant` text, capped; earlier assistant text is never evidence |
| Bounded work | max 5 tool rounds, 4 calls per round, 1 repair, 90 s, 16 000 characters per tool result, 20 questions/min/user |
| Fail closed | If the audit row cannot be written, the answer is withheld |

The badge says exactly what ran: **"Checked against database records"** (figure check and fact check both passed),
**"Figures checked against database records"** (the fact check is switched off), **"Only partly checked"** (the fact check could
not run, e.g. a provider error; the answer is shown with a warning), **"Could not verify"** (answer withheld) or
**"No database lookup"**.

**Limits, stated plainly.** Both checks are automatic. The figure check is exact but only matches a number to *some* number in
the records; the fact check closes that gap but is itself a language model, so it can miss a subtle error or occasionally flag
something that is fine (the answer is then repaired or withheld). Neither replaces looking at the record: each answer shows the
tools that ran and the record ids (reviewers/admins can click an id to open its audit trail). The fact check costs one extra
model call per answer; set `OPSPILOT_CHAT_FACT_CHECK=false` to turn it off if speed or the provider's rate limit matters more.

## Fixed rules first, the model last

Many questions never reach the model: greetings, counts ("how many invoices are pending?"), lists ("invoices from Acme", "top vendors", "show the audit log"), role refusals, and document questions answered by quoting. The server calls the same read-only MCP tools and fills a template with their values. See [RAG.md](RAG.md#when-the-ai-model-is-and-is-not-used) for the full table; `OPSPILOT_CHAT_HARDCODE=false` restores the model-for-everything behaviour.

## Documents (RAG)

With `OPSPILOT_RAG_ENABLED=true` the model also gets a `search_documents` tool next to the MCP tools, so it can answer from uploaded
policies, contracts and procedures, cite the passages it used, and combine them with live database records in one answer. The
same loop, limits, figure verifier and audit apply; document text is untrusted input like e-mail bodies; and a question the
documents cannot answer gets a fixed "not in the documents" reply instead of the model's guess. Everything about it is in
[RAG.md](RAG.md). With RAG off, the assistant behaves exactly as described on this page.

## Audit

* Per tool call: the existing MCP rows (`stage mcp`, `transport chat`, actor = the user, same `request_id`).
* Per question: `entity_type chat`, `action ask:ok | ask:blocked | ask:failed | ask:rate_limited` with the cleaned question,
  tools used and outcomes, record count, verified/blocked flags, retries, model, token counts, duration and a SHA-256 of the answer.
  The answer text, returned data and the Groq key are **not** stored.
* View: dashboard → Audit Logs, `GET /audit?stage=chat`, or ask the assistant as an admin.

## Troubleshooting

**"HTTP 429 ... tokens per minute"**: Groq's free tier limits tokens per minute per model. The assistant already waits out short limits and retries; for the rest, start a **New chat** (history costs tokens), turn off `OPSPILOT_CHAT_FACT_CHECK`, or pick a model with higher limits. Details in [RAG.md](RAG.md#groqs-free-tier-token-limit-http-429--tokens-per-minute).

| You see | Meaning |
| --- | --- |
| "The AI assistant is not enabled on this server" | `OPSPILOT_CHAT_ENABLED` is not `true`, or the server was not restarted |
| `503 ... not configured` | `OPSPILOT_GROQ_API_KEY` is empty |
| `502 ... rejected this server's credentials` | the Groq key is wrong or revoked |
| `502 ... AI provider is unavailable` | a generic provider failure. **Admins see the reason in brackets** (`[admin details: ...]`); it is also in the uvicorn console (`Groq request failed`) and in the `ask:failed` audit row (`error_detail`). Run `python scripts/check_groq.py` to test the connection step by step |
| `429 ... rate-limiting` | Groq's limits for your plan; wait and retry |
| "Could not verify" badge | the model wrote something the records do not state; rephrase or ask about one invoice, vendor or the review queue |
| "I can only answer questions about your OpsPilot data" | the question was not about invoices, vendors, review, metrics, models, audit or system status |
| App won't start: "Refusing to start the AI assistant" | the message lists what to fix |

## Tests

```powershell
pytest tests/test_chat_logic.py tests/test_chat_api.py -q    # assistant tests (Groq is faked; no network, no key needed)
pytest -q                                                    # everything
```
