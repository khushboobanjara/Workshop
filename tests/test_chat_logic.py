"""AI assistant: pure logic tests (no database, no network). Verifier, one chat turn with a fake model and fake MCP
server, the Groq client against a mock transport, and start-up safety checks."""
import json
import types

import httpx
import pytest
from pydantic import SecretStr

from app.chat import prompts, routes as chat_routes, service
from app.chat.errors import GroqError, public_message
from app.chat.groq_client import GroqClient
from app.chat.verifier import MAX_REPORTED, Evidence, find_unsupported
from app.config import settings

INVOICE = {"source": "opspilot_database", "found": True, "tool": "get_invoice", "record_refs": ["invoice:12"],
           "untrusted_text_fields": ["vendor_name"],
           "data": {"invoice": {"id": 12, "vendor_name": "Acme Corp", "invoice_number": "INV-1001", "total_amount": 1250.5,
                                "currency": "USD", "due_date": "2026-10-31", "risk_score": 0.12, "po_number": "PO-7",
                                "decision_reasons": ["amount 12500.0 above limit of 10000"]}}}


# ---------------------------------------------------------------- verifier
def check(answer, user="", today="2026-10-18"):
    ev = Evidence()
    ev.add_allowed_text(today)
    ev.add_user_text(user)
    ev.add_tool_payload(INVOICE)
    return find_unsupported(answer, ev)


@pytest.mark.parametrize("answer,expected", [
    ("Invoice INV-1001 from Acme Corp (#12) is USD 1,250.50, due 2026-10-31, risk 0.12 (12%), PO-7.", []),
    ("It exceeds the limit of 10,000 (amount 12500).", []),                       # figures inside tool strings count
    ("The total is $9,999.99", ["$9,999.99"]),
    ("Total 1,251", ["1,251"]),
    ("Due 2026-11-30", ["2026-11-30"]),
    ("As of 2026-10-18 it is approved", []),
    ("Invoice INV-9999 is approved", ["INV-9999"]),
    ("Invoice #13 is pending", ["#13"]),
    ("risk is 13%", ["13%"]),
    ("risk is 12%", []),
    ("Risk is 0.5", ["0.5"]),
    ("You have 3 invoices on page 2, in 2026.", ["3", "2"]),                      # small counts are checked; years are not
    ("No invoices are pending review (0).", ["0"]),                               # 0 is not in these results
    ("1. First point\n2. Second point", []),                                      # list markers are not figures
    ("There are 250 invoices", ["250"]),
    ("hello, how can I help?", []),
    ("$7.77 and $7.77", ["$7.77"]),
])
def test_answer_figures_must_come_from_tool_results(answer, expected):
    assert check(answer) == expected


def test_the_users_own_text_vouches_for_identifiers_but_never_for_amounts():
    assert check("Invoice INV-9999 was not found", user="what about INV-9999?") == []
    assert check("Invoice #13 not found", user="show invoice 13") == []
    assert check("There are 250 invoices", user="do I have 250 invoices?") == []
    assert check("Spend is $5,000", user="anything above $5,000?") == ["$5,000"]


def test_non_finite_numbers_are_never_evidence_and_reports_are_capped():
    ev = Evidence()
    ev.add_tool_payload({"x": float("nan"), "y": float("inf"), "z": True})
    assert not ev.numbers
    many = check(" ".join(f"A {n}{n}{n}.{n}{n}" for n in range(1, 10)))
    assert len(many) == MAX_REPORTED


# ---------------------------------------------------------------- one chat turn, with fakes
class _Tool:
    def __init__(self, name, props=None):
        self.name, self.props = name, props or {}

    def descriptor(self):
        return {"name": self.name, "description": "d", "inputSchema": {"type": "object", "properties": self.props, "additionalProperties": False}}


class FakeServer:
    def __init__(self):
        self.calls = []

    def allowed_tools(self, principal):
        return [_Tool("get_invoice", {"invoice_id": {"type": "integer"}}), _Tool("get_metrics")]

    def handle_message(self, msg, principal, transport):
        name, args = msg["params"]["name"], msg["params"]["arguments"]
        self.calls.append((name, args, transport, principal))
        if name == "list_users":
            return {"result": {"isError": True, "content": [{"type": "text", "text": json.dumps({"ok": False, "error": {"code": "forbidden", "message": "no"}})}]}}
        if name == "nope":
            return {"error": {"code": -32602, "message": "Unknown tool"}}
        if name == "get_invoice" and args.get("invoice_id") == 999:
            payload = {"source": "opspilot_database", "found": False, "record_refs": [], "data": None, "notes": ["No invoice with id 999 exists in the database."]}
        elif name == "huge":
            payload = {"source": "opspilot_database", "found": True, "record_refs": [], "data": "x" * 50}
        else:
            payload = INVOICE
        return {"result": {"isError": False, "content": [{"type": "text", "text": json.dumps(payload)}]}}


class FakeLLM:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def complete(self, messages, tools=None, tool_choice="auto"):
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools, "tool_choice": tool_choice})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return {"content": step.get("content"), "tool_calls": step.get("tool_calls", []), "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def tc(name, args, call_id="c1"):
    return {"content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}]}


def say(text):
    return {"content": text}


PRINCIPAL = types.SimpleNamespace(username="view1", role="viewer", via="session")


def run(script, msg="what is invoice 12?", history=(), **kw):
    kw.setdefault("limits", service.Limits(fact_check=False))
    llm, srv = FakeLLM(script), FakeServer()
    return service.answer(PRINCIPAL, msg, list(history), llm=llm, server=srv, role_label="user", today="2026-10-18", **kw), llm, srv


def test_a_grounded_answer_goes_model_to_tool_to_model_and_is_verified():
    r, llm, srv = run([tc("get_invoice", {"invoice_id": 12}), say("Invoice INV-1001 from Acme Corp is USD 1,250.50, due 2026-10-31 (#12).")])
    assert r.verified and r.grounded and not r.blocked and r.sources == ["invoice:12"] and r.groq_calls == 2
    assert r.tools_used == [{"tool": "get_invoice", "arguments": {"invoice_id": 12}, "outcome": "ok", "records": 1}]
    assert [q["tool_choice"] for q in llm.requests] == ["required", "auto"]          # a data answer must start with a tool call
    system = llm.requests[0]["messages"][0]
    assert system["role"] == "system" and 'role is "user"' in system["content"] and "2026-10-18" in system["content"]
    tool_msg = llm.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1" and "INV-1001" in tool_msg["content"]
    assert srv.calls[0][2] == "chat" and srv.calls[0][3] is PRINCIPAL                 # runs as the signed-in user, transport 'chat'
    assert (r.prompt_tokens, r.completion_tokens) == (20, 10)


def test_models_numeric_strings_are_fixed_and_nulls_dropped_but_validation_stays_with_mcp():
    r, _, srv = run([tc("get_invoice", {"invoice_id": "12", "x": None}), say("ok INV-1001")])
    assert srv.calls[0][1] == {"invoice_id": 12} and r.verified


def test_an_invented_figure_is_repaired_once():
    r, llm, _ = run([tc("get_invoice", {"invoice_id": 12}), say("The total is $9,999.99."), say("The total is USD 1,250.50.")])
    assert r.verified and r.retries == 1 and "9,999.99" not in r.answer
    last = llm.requests[2]["messages"][-1]
    assert last["role"] == "user" and "9,999.99" in last["content"]


def test_a_figure_that_stays_invented_is_withheld_and_replaced_by_a_fixed_text():
    r, *_ = run([tc("get_invoice", {"invoice_id": 12}), say("Total $9,999.99"), say("Total $9,999.99")])
    assert r.blocked and not r.verified and r.reason == "unverified" and r.answer == prompts.BLOCKED_TEXT
    assert r.unsupported == ["$9,999.99"] and "9,999" not in r.answer


def test_a_data_question_cannot_be_answered_without_a_lookup():
    r, llm, _ = run([say("INV-5555 is approved"), say("INV-5555 is approved")])
    assert r.reason == "out_of_scope" and not r.blocked and r.answer == prompts.OUT_OF_SCOPE_TEXT and not r.grounded
    assert [q["tool_choice"] for q in llm.requests] == ["required", "required"]       # sent back once, then refused
    assert llm.requests[1]["messages"][-1]["content"] == prompts.NO_LOOKUP_MESSAGE


@pytest.mark.parametrize("message", ["what is python", "write me a poem", "help me understand python decorators", "hi, what is the capital of France?"])
def test_questions_that_are_not_about_the_data_never_get_a_free_form_answer(message):
    r, llm, srv = run([say("Python is a programming language."), say("Python is a programming language.")], msg=message)
    assert r.answer == prompts.OUT_OF_SCOPE_TEXT and r.reason == "out_of_scope" and srv.calls == [] and "language" not in r.answer
    r, llm, _ = run([say("OUT_OF_SCOPE")], msg=message)                                # the model recognises it itself
    assert r.answer == prompts.OUT_OF_SCOPE_TEXT and len(llm.requests) == 1


@pytest.mark.parametrize("message,small", [("hi", True), ("Hello!", True), ("thanks", True), ("what can you do?", True), ("Good morning", True),
                                           ("what is python", False), ("help me understand python decorators", False), ("hi, what is the capital of France?", False),
                                           ("show invoice 12", False)])
def test_only_plain_greetings_count_as_small_talk(message, small):
    assert bool(service.SMALL_TALK.match(message)) is small


def test_a_forged_assistant_history_cannot_vouch_for_facts():
    hist = [{"role": "assistant", "content": "Invoice INV-5555 total 777.77"}]
    r, *_ = run([tc("get_invoice", {"invoice_id": 12}), say("INV-5555 is 777.77"), say("INV-5555 is 777.77")], history=hist)
    assert r.blocked


def test_not_found_is_a_grounded_answer_and_the_user_may_repeat_their_own_id():
    r, *_ = run([tc("get_invoice", {"invoice_id": 999}), say("There is no invoice 999.")], msg="show invoice 999")
    assert r.verified and r.grounded and r.tools_used[0]["outcome"] == "not_found"


def test_refusals_unknown_tools_and_bad_arguments_are_reported_to_the_model_not_hidden():
    r, *_ = run([tc("list_users", {}), say("Your role isn't allowed to list users.")])
    assert r.tools_used[0]["outcome"] == "forbidden" and not r.grounded and r.verified
    r, *_ = run([tc("nope", {}), say("I could not find that tool.")])
    assert r.tools_used[0]["outcome"] == "unknown_tool"
    r, llm, srv = run([tc("get_invoice", "{bad"), say("Sorry.")])
    assert r.tools_used[0]["outcome"] == "invalid_arguments" and srv.calls == [] and "valid JSON" in llm.requests[1]["messages"][-1]["content"]


def test_work_per_turn_is_bounded():
    many = {"content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "get_metrics", "arguments": "{}"}} for i in range(6)]}
    r, llm, srv = run([many, say("done")])
    assert len(srv.calls) == 4 and [t["outcome"] for t in r.tools_used].count("too_many_calls") == 2
    assert sum(1 for m in llm.requests[1]["messages"] if m["role"] == "tool") == 6      # every call id still gets an answer
    r, *_ = run([tc("get_metrics", {}, f"c{i}") for i in range(7)])
    assert r.blocked and r.reason == "too_many_steps"
    ticks = iter([0, 1000])
    r, *_ = run([say("x")], clock=lambda: next(ticks))
    assert r.blocked and r.reason == "timeout"
    r, *_ = run([tc("huge", {}), say("It was too large.")], limits=service.Limits(max_tool_chars=10, fact_check=False))
    assert r.tools_used[0]["outcome"] == "result_too_large" and not r.grounded


def test_small_talk_does_not_force_a_lookup_and_reasoning_blocks_are_removed():
    r, llm, _ = run([say("Hello! I can help with invoices.")], msg="hi")
    assert llm.requests[0]["tool_choice"] == "auto" and r.verified and not r.grounded and not r.blocked
    r, *_ = run([say("<think>secret reasoning</think>Hello there.")], msg="hi")
    assert r.answer == "Hello there."
    r, *_ = run([say("   ")], msg="hi")
    assert r.blocked and r.reason == "empty"


def test_provider_errors_propagate_and_a_malformed_tool_call_is_retried_once():
    with pytest.raises(GroqError) as e:
        run([GroqError("auth", 502)])
    assert e.value.code == "auth"
    r, llm, _ = run([GroqError("tool_use_failed"), say("Hello")], msg="hi")
    assert r.verified and llm.requests[1]["tool_choice"] == "auto"
    with pytest.raises(GroqError):
        run([GroqError("tool_use_failed"), GroqError("tool_use_failed")], msg="hi")


def test_history_is_capped_cleaned_and_limited_to_user_and_assistant_turns():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": "m%d\u200b" % i + "x" * 3000} for i in range(30)]
    _, llm, _ = run([say("hello")], msg="hi", history=hist)
    sent = llm.requests[0]["messages"]
    assert len(sent) == 1 + 8 + 1 and all(len(m["content"]) <= 2020 for m in sent[1:-1]) and "\u200b" not in json.dumps(sent)
    assert {m["role"] for m in sent[1:]} <= {"user", "assistant"}


# ---------------------------------------------------------------- Groq client
def client(handler, **kw):
    return GroqClient("gsk_secret_key_123", "m-1", transport=httpx.MockTransport(handler), **kw)


def ok_body(content="hi", tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def test_client_sends_an_openai_style_request_with_the_key_only_in_the_header():
    seen = {}

    def handler(request):
        seen["url"], seen["headers"], seen["body"] = str(request.url), dict(request.headers), json.loads(request.content)
        return httpx.Response(200, json=ok_body("hello"))
    out = client(handler).complete([{"role": "user", "content": "q"}], [{"type": "function", "function": {"name": "t"}}], "required")
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer gsk_secret_key_123"
    b = seen["body"]
    assert b["model"] == "m-1" and b["tool_choice"] == "required" and b["tools"] and b["max_completion_tokens"] and "gsk_secret" not in json.dumps(b)
    assert out == {"content": "hello", "tool_calls": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}, "finish_reason": "stop"}
    assert "gsk_secret" not in repr(client(handler))


def test_client_omits_tool_fields_when_there_are_no_tools_and_returns_tool_calls():
    calls = [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]

    def handler(request):
        assert "tools" not in json.loads(request.content) and "tool_choice" not in json.loads(request.content)
        return httpx.Response(200, json=ok_body(None, calls))
    assert client(handler).complete([{"role": "user", "content": "q"}])["tool_calls"] == calls


@pytest.mark.parametrize("response,code,status,retry", [
    (httpx.Response(401, json={"error": {"message": "Invalid API Key gsk_secret_key_123"}}), "auth", 502, None),
    (httpx.Response(403), "auth", 502, None),
    (httpx.Response(429, headers={"retry-after": "7"}), "rate_limited", 429, 7),
    (httpx.Response(500), "upstream", 502, None),
    (httpx.Response(400, json={"error": {"code": "tool_use_failed", "message": "bad call"}}), "tool_use_failed", 502, None),
    (httpx.Response(400, json={"error": {"code": "other"}}), "bad_request", 502, None),
    (httpx.Response(200, json={"unexpected": True}), "bad_response", 502, None),
    (httpx.Response(200, content=b"not json"), "bad_response", 502, None),
])
def test_client_maps_provider_failures_to_safe_error_codes(response, code, status, retry):
    with pytest.raises(GroqError) as e:
        client(lambda request: response).complete([{"role": "user", "content": "q"}])
    assert (e.value.code, e.value.http_status, e.value.retry_after) == (code, status, retry)
    assert "gsk_secret" not in public_message(e.value.code) and "gsk_secret" not in repr(e.value)


def test_client_maps_timeouts_and_network_errors():
    def slow(request):
        raise httpx.ReadTimeout("slow")

    def down(request):
        raise httpx.ConnectError("no route")
    with pytest.raises(GroqError) as e:
        client(slow).complete([{"role": "user", "content": "q"}])
    assert (e.value.code, e.value.http_status) == ("timeout", 504)
    with pytest.raises(GroqError) as e:
        client(down).complete([{"role": "user", "content": "q"}])
    assert (e.value.code, e.value.http_status) == ("unreachable", 502)


# ---------------------------------------------------------------- start-up safety
def configure(monkeypatch, **changes):
    base = {"chat_enabled": True, "auth_enabled": True, "demo_mode": False, "groq_api_key": SecretStr("gsk_" + "k" * 30),
            "groq_model": "llama-3.3-70b-versatile", "groq_base_url": "https://api.groq.com/openai/v1"}
    for k, v in {**base, **changes}.items():
        monkeypatch.setattr(settings, k, v)


@pytest.mark.parametrize("changes,fragment", [
    ({"auth_enabled": False}, "AUTH_ENABLED"), ({"demo_mode": True}, "DEMO_MODE"),
    ({"groq_api_key": SecretStr("")}, "GROQ_API_KEY"), ({"groq_model": "  "}, "GROQ_MODEL"),
    ({"groq_base_url": "http://api.example.com/v1"}, "https"),
])
def test_the_assistant_refuses_to_start_in_an_unsafe_configuration(monkeypatch, changes, fragment):
    configure(monkeypatch, **changes)
    with pytest.raises(RuntimeError, match=fragment):
        chat_routes.startup_checks()


def test_safe_or_disabled_configurations_start(monkeypatch):
    configure(monkeypatch)
    chat_routes.startup_checks()
    configure(monkeypatch, groq_base_url="http://127.0.0.1:9999/v1")             # a local gateway is fine
    chat_routes.startup_checks()
    configure(monkeypatch, chat_enabled=False, auth_enabled=False, groq_api_key=SecretStr(""))
    chat_routes.startup_checks()


# ---------------------------------------------------------------- provider error details (for diagnosis)
def test_provider_error_details_carry_groqs_message_with_the_key_scrubbed():
    reply = httpx.Response(400, json={"error": {"message": "The model `nope` does not exist. key=gsk_secret_key_123", "code": "model_not_found"}})
    with pytest.raises(GroqError) as e:
        client(lambda request: reply).complete([{"role": "user", "content": "q"}])
    assert e.value.code == "bad_request" and e.value.detail.startswith("HTTP 400: The model `nope` does not exist.")
    assert "gsk_secret" not in e.value.detail and len(e.value.detail) <= 300


def test_network_and_shape_errors_have_details_too():
    def down(request):
        raise httpx.ConnectError("getaddrinfo failed")
    with pytest.raises(GroqError) as e:
        client(down).complete([{"role": "user", "content": "q"}])
    assert "ConnectError" in e.value.detail and "getaddrinfo failed" in e.value.detail
    with pytest.raises(GroqError) as e:
        client(lambda request: httpx.Response(200, json={"oops": 1})).complete([{"role": "user", "content": "q"}])
    assert "unexpected response shape" in e.value.detail


# ---------------------------------------------------------------- the statement-level fact checker
JUDGE_OK = {"content": '{"unsupported_claims": []}'}
JUDGE_FLAGS = {"content": '{"unsupported_claims": ["the system is healthy"]}'}


def judged(script, **kw):
    return run(script, limits=service.Limits(fact_check=True), **kw)


@pytest.mark.parametrize("text,expected", [
    ('{"unsupported_claims": []}', []),
    ('```json\n{"unsupported_claims": ["a", " b ", ""]}\n```', ["a", "b"]),
    ('<think>hmm</think>{"unsupported_claims": ["c"]}', ["c"]),
    ('{"unsupported_claims": ["1","2","3","4","5","6","7"]}', ["1", "2", "3", "4", "5"]),
    ('{"unsupported_claims": "none"}', None), ('{"unsupported_claims": [1]}', None), ('{"other": []}', None),
    ("I think the answer is fine", None), ("{not json}", None), ("", None), (None, None),
])
def test_the_fact_checkers_verdict_is_parsed_strictly(text, expected):
    assert service.parse_verdict(text) == expected


def test_the_fact_checker_is_shown_the_records_the_question_and_the_answer():
    r, llm, _ = judged([tc("get_invoice", {"invoice_id": 12}), say("Invoice INV-1001 is for USD 1,250.50."), JUDGE_OK])
    assert r.verified and r.fact_checked and not r.fact_check_unavailable and r.groq_calls == 3
    judge = llm.requests[2]
    assert judge["tools"] is None and judge["messages"][0]["content"] == prompts.JUDGE_SYSTEM
    body = judge["messages"][1]["content"]
    assert "1250.5" in body and "Invoice INV-1001 is for USD 1,250.50." in body and "what is invoice 12?" in body


def test_statements_the_records_do_not_support_are_repaired_once_then_withheld():
    r, llm, _ = judged([tc("get_invoice", {"invoice_id": 12}), say("Invoice INV-1001 is fine and healthy."), JUDGE_FLAGS,
                        say("Invoice INV-1001 is for USD 1,250.50."), JUDGE_OK])
    assert r.verified and r.retries == 1 and r.fact_checked
    assert "the system is healthy" in llm.requests[3]["messages"][-1]["content"]
    r, *_ = judged([tc("get_invoice", {"invoice_id": 12}), say("Everything is healthy."), JUDGE_FLAGS, say("Still healthy."), JUDGE_FLAGS])
    assert r.blocked and r.reason == "unverified" and r.answer == prompts.BLOCKED_TEXT and r.unsupported == ["the system is healthy"]


@pytest.mark.parametrize("verdict", [{"content": "I think it is fine"}, {"content": '{"unsupported_claims": "none"}'}, GroqError("rate_limited", 429)])
def test_a_checker_that_cannot_answer_downgrades_the_badge_but_does_not_hide_the_answer(verdict):
    r, *_ = judged([tc("get_invoice", {"invoice_id": 12}), say("Invoice INV-1001 is for USD 1,250.50."), verdict])
    assert r.verified is False and not r.blocked and r.fact_check_unavailable and not r.fact_checked
    assert r.answer == "Invoice INV-1001 is for USD 1,250.50." and any("fact check" in w for w in r.to_dict("m")["warnings"])


def test_the_fact_check_is_skipped_for_greetings_and_when_disabled():
    r, llm, _ = judged([say("Hello!")], msg="hi")
    assert len(llm.requests) == 1 and r.verified and not r.fact_checked
    r, llm, _ = run([tc("get_invoice", {"invoice_id": 12}), say("It is INV-1001.")])
    assert len(llm.requests) == 2 and r.verified and not r.fact_checked


def test_the_figure_check_runs_first_so_the_fact_checker_is_not_called_for_invented_numbers():
    r, llm, _ = judged([tc("get_invoice", {"invoice_id": 12}), say("Total $9,999.99"), say("Total USD 1,250.50"), JUDGE_OK])
    assert r.verified and [q["tools"] is None for q in llm.requests] == [False, False, False, True]


# ---------------------------------------------------------------- Groq per-minute limits: wait and retry
GROQ_429 = ("Rate limit reached for model `m` in organization `org_x` service tier `on_demand` on tokens per minute (TPM): "
            "Limit 8000, Used 5008, Requested 3420. Please try again in {wait}. Need more tokens? Upgrade to Dev Tier today")


def limited(wait: str, **headers):
    return httpx.Response(429, headers=headers, json={"error": {"message": GROQ_429.format(wait=wait), "code": "rate_limit_exceeded"}})


@pytest.mark.parametrize("response,expected", [
    (limited("3.21s"), 3.21), (limited("850ms"), 0.85), (limited("1m5.2s"), 65.2), (limited("2s", **{"retry-after": "7"}), 7.0),
    (httpx.Response(429, headers={"retry-after": "1.5"}), 1.5), (httpx.Response(429), None), (httpx.Response(429, json={"error": {"message": "slow down"}}), None)])
def test_the_wait_groq_asks_for_is_understood(response, expected):
    from app.chat.groq_client import retry_wait_seconds
    got = retry_wait_seconds(response)
    assert (got is None and expected is None) or abs(got - expected) < 1e-6


def test_a_short_rate_limit_is_waited_out_and_the_question_still_succeeds():
    replies, waits = iter([limited("3.21s"), limited("1.5s"), httpx.Response(200, json=ok_body("done"))]), []
    c = client(lambda req: next(replies), rate_limit_retries=2, sleep=waits.append)
    assert c.complete([{"role": "user", "content": "x"}])["content"] == "done"
    assert waits == [3.21 + 0.25, 1.5 + 0.25]


def test_a_rate_limit_that_lasts_too_long_is_reported_with_the_wait():
    calls, waits = [], []
    def handler(req):
        calls.append(1)
        return limited("45s")
    with pytest.raises(GroqError) as e:
        client(handler, rate_limit_retries=2, max_wait=12.0, sleep=waits.append).complete([{"role": "user", "content": "x"}])
    assert e.value.code == "rate_limited" and e.value.retry_after == 45 and len(calls) == 1 and waits == []
    from app.chat.errors import public_message
    assert "45 seconds" in public_message("rate_limited", 45) and "org_x" not in public_message("rate_limited", 45)


def test_retries_are_bounded_and_off_by_default():
    calls = []
    def handler(req):
        calls.append(1)
        return limited("1s")
    with pytest.raises(GroqError):
        client(handler, rate_limit_retries=2, sleep=lambda s: None).complete([{"role": "user", "content": "x"}])
    assert len(calls) == 3                                                     # one try + two retries, then the error
    calls.clear()
    with pytest.raises(GroqError):
        client(handler).complete([{"role": "user", "content": "x"}])           # default client (and every existing test): no waiting
    assert len(calls) == 1


def test_history_is_trimmed_per_message_when_the_route_asks_for_it():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 3000} for i in range(10)]
    _, llm, _ = run([say("hello")], msg="hi", history=hist, limits=service.Limits(fact_check=False, history_turns=4, history_chars=300))
    sent = llm.requests[0]["messages"]
    assert len(sent) == 1 + 4 + 1 and all(len(m["content"]) <= 320 for m in sent[1:-1])
