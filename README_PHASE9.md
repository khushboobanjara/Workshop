# Phase 9 - Grounding Gate (hallucination prevention)

Unzip into your project root (the folder with `main.py`). Phases 1-8 must already be applied.
**No existing file is changed**, and nothing imports the gate yet, so the running app behaves exactly as before.

## New files
| File | Purpose |
|---|---|
| `mcp_layer/grounding.py` | `GroundingGate`: decides whether the LLM may be called, builds the grounded prompt, audits every block |
| `tests/test_mcp_phase9.py` | 28 tests; the LLM is a fake and every blocked case asserts it was never called |

## The rule
The LLM is called only if, for EVERY result: `success is True` AND `verified is True` AND `source` is allowed
(`mysql`, `ml_model`, `google_places`) AND data is present. Otherwise: controlled reply, no LLM call, one audit row.

| Case | reason | Reply |
|---|---|---|
| `DATABASE_UNAVAILABLE` | `database_unavailable` | "I'm unable to provide verified information right now because the required system data is unavailable." |
| any other failed MCP | `mcp_failed` | generic "service unavailable" (user-safe MCP messages such as ACCESS_DENIED pass through) |
| `verified=False` | `unverified` | unavailable message |
| `[]`, `{}`, `None`, `row_count=0`, or only empty parts | `empty_result` | the tool's own message, else "I couldn't find any matching records..." |
| source missing or not allowed (`mcp_internal` is NOT allowed) | `invalid_source` | unavailable message |
| not a result / bad label / cycle / any gate error | `malformed_result` | unavailable message (fails closed) |
| data over 20,000 chars | `data_too_large` | unavailable message (never truncated, so no half context) |

`0` and `False` are real values, not "empty". Several results are all-or-nothing.

## Other guarantees
- Prompt = your base prompt + grounding rules + ONLY the verified `data` inside `<verified_data>` (fenced as data; `<` is escaped so stored text cannot close the fence).
- Screening: a result must have `is_diagnosis == False` and a disclaimer or it is blocked; the prompt forbids diagnosis wording; the stored disclaimer is always appended to the reply.
- History: only `user`/`assistant` turns are passed on (a `system` entry is dropped).
- Audit row per block: `tool=llm_call_blocked`, `security_result=BLOCKED`, `denial_reason=<reason>`, the original `error_code` and `request_id`, argument NAMES only. Never data, question or model text. Uses the same sink as the gateway (`GroundingGate.from_gateway(gateway)`), no migration needed.
- LLM raises or returns nothing: controlled reply + audit row, never an exception.

## Use (Phase 11 will do this in the chatbot)
```python
gate = GroundingGate.from_gateway(gateway)
result = await gateway.invoke(principal, "mcp_02_doctor_appointment", "get_my_appointments", {})
reply = await gate.answer(results={"appointments": result}, user_message=text,
                          system_prompt=SYSTEM_PROMPT, llm=generate_response,
                          history=history, principal=principal, server_id="mcp_02_doctor_appointment")
send(reply.text)     # reply.grounded is True only when the LLM answered over verified data
```

## Run
`python -m pytest tests -q`
