"""The hard-coded layer: questions answered with NO model call (or at most one, with no tools), so that Groq's per-minute token
limit is spent only where reasoning is needed. Routing tables, templates, and end-to-end checks that count the requests Groq saw."""
import json

import pytest

from app.chat import prompts, rules
from app.config import settings
from app.db import AuditLog, SessionLocal
from app.rag import direct, grounding
from tests.rag_support import POLICY, chat, rag, say, tc, world  # noqa: F401  (fixtures / helpers)
from tests.test_rag_chat import RULES, WHAT_IF, ref_of, rows


# ---------------------------------------------------------------- pure: what is answered by fixed rules
@pytest.mark.parametrize("text,tool,args,view", [
    ("how many invoices are pending review?", "get_metrics", {}, "count:pending_review"),
    ("How many vendors do we have?", "list_vendors", {"limit": 1}, "count:vendors"),
    ("how many emails have we received", "get_metrics", {}, "count:emails"),
    ("how many invoices were rejected", "get_metrics", {}, "count:rejected"),
    ("how many invoices in total", "get_metrics", {}, "count:invoices"),
    ("how many invoices were auto approved", "get_metrics", {}, "count:auto_approved"),
    ("show me the review queue", "get_review_queue", {"limit": 10}, ""),
    ("what's waiting for review?", "get_review_queue", {"limit": 10}, ""),
    ("invoices from Acme", "search_invoices", {"limit": 10, "vendor": "acme"}, ""),
    ("rejected invoices", "search_invoices", {"limit": 10, "status": "rejected"}, ""),
    ("show approved invoices from Globex", "search_invoices", {"limit": 10, "status": "approved", "vendor": "globex"}, ""),
    ("list auto approved invoices", "search_invoices", {"limit": 10, "decision": "auto_approved"}, ""),
    ("top 5 vendors", "list_vendors", {"limit": 5}, ""),
    ("metrics", "get_metrics", {}, ""),
    ("which models are active?", "list_models", {}, ""),
    ("system status", "get_system_status", {}, ""),
    ("what can I access?", "get_my_access", {}, ""),
    ("show the audit log", "get_audit_log", {"limit": 10}, ""),
    ("list users", "list_users", {"limit": 50}, ""),
    ("invoice 12", "get_invoice", {"invoice_id": 12}, ""),
    ("show invoice #12 details", "get_invoice", {"invoice_id": 12}, ""),
    ("audit trail for invoice 12", "get_invoice_audit_trail", {"invoice_id": 12}, "")])
def test_recognised_questions_are_routed_to_the_right_tool(text, tool, args, view):
    r = rules.route(text)
    assert r is not None and (r.tool, r.args, r.view) == (tool, args, view), r


@pytest.mark.parametrize("text", [
    "why was invoice 12 rejected", "invoices over 5000", "invoices from acme last month", "how many invoices did acme send last month",
    "approve invoice 12", "compare the vendors", "which vendor should we drop", "what is the capital of France", "invoice 12 and invoice 13",
    "show invoices from acme and globex", "tell me about our top vendors and why they matter", "x" * 300, "", "   "])
def test_anything_not_fully_understood_is_left_to_the_model(text):
    assert rules.route(text) is None


@pytest.mark.parametrize("text,kind", [("hi", "hello"), ("Hello there!", "hello"), ("thanks a lot", "welcome"), ("help", "Try")])
def test_canned_replies(text, kind):
    out = rules.canned(text, ["get_metrics", "get_system_status", "get_audit_log"], True)
    assert out and kind.lower() in out.lower()


def test_help_lists_only_what_this_role_and_setup_can_do():
    assert "policies" in rules.canned("help", [], True) and "policies" not in rules.canned("help", [], False)
    assert "system status" in rules.canned("help", ["get_system_status"], False) and "system status" not in rules.canned("help", [], False)


def test_questions_are_not_mistaken_for_greetings():
    for text in ["hi, how many invoices are pending?", "help me with invoice 12", "thanks, now show vendors", "what is a vendor"]:
        assert rules.canned(text, [], True) is None


def test_templates_use_only_the_tools_own_values():
    m = {"data": {"pending_review": 7, "invoices": 40, "emails_received": 55, "by_status": {"approved": 20, "rejected": 3}, "by_decision": {"auto_approved": 11}}}
    assert rules.render("get_metrics", m, "count:pending_review") == "7 invoice(s) are pending review."
    assert rules.render("get_metrics", m, "count:approved") == "20 invoice(s) are approved."
    assert rules.render("get_metrics", m, "count:auto_approved") == "11 invoice(s) are auto-approved."
    assert rules.render("get_metrics", m, "count:new") == "No invoices are new."
    assert rules.render("get_metrics", {"data": {}}, "count:pending_review") is None            # unrecognised shape: ask the model instead
    assert rules.render("list_vendors", {"data": {"total_vendors": 3, "vendors": []}}, "count:vendors") == "There are 3 vendor(s) in OpsPilot."
    users = {"data": {"users": [{"username": "ann", "role": "admin", "active": True}, {"username": "bob", "role": "viewer", "active": False}]}}
    assert rules.render("list_users", users) == "2 user account(s):\n- ann (admin, active)\n- bob (viewer, disabled)"
    assert rules.render("get_audit_log", {"data": {"entries": []}}) == "The database has no matching records for that."


def test_document_versus_live_data_classification():
    assert rules.is_doc_question("What is the standard payment term for invoices?")
    assert rules.is_doc_question("Can the AI Assistant approve or reject an invoice?")
    assert rules.is_doc_question("What does our policy say about how many approvers are needed?")
    assert not rules.is_doc_question("What is the status of invoice INV-1001?") and not rules.is_doc_question("show invoice 12")
    assert not rules.is_doc_question("which vendors have pending invoices today")
    assert rules.looks_like_data("any problems with Acme?") and not rules.looks_like_data("what is machine learning")
    assert rules.needs_llm(WHAT_IF) and rules.needs_llm("explain the approval process") and rules.needs_llm("why is net 45 restricted")
    assert not rules.needs_llm("What is the standard payment term for invoices?")


def test_a_bare_follow_up_borrows_the_previous_question_for_searching():
    hist = [{"role": "user", "content": "What are the payment terms?"}, {"role": "assistant", "content": "Net 30."}]
    assert rules.retrieval_query("explain that", hist) == "What are the payment terms? explain that"
    assert rules.retrieval_query("What are the approval limits for invoices?", hist) == "What are the approval limits for invoices?"
    assert rules.retrieval_query("explain that", []) == "explain that"


# ---------------------------------------------------------------- pure: quoting the documents
PASSAGES = [{"ref": "DOC-1-C1", "section": "8. Payment terms", "title": "Policy",
             "text": "Standard payment terms are net 30 days from the invoice date. Net 45 needs written approval from the Finance Director.\n- Early-payment discounts may be taken only when Finance has approved them."},
            {"ref": "DOC-1-C2", "section": "9. The AI assistant", "title": "Policy",
             "text": "The AI assistant answers questions in plain language. It is a read-only helper: it cannot approve, reject, edit or delete anything."}]


def test_quotes_are_verbatim_cited_and_cover_the_question():
    d = direct.pick("What is the standard payment term for invoices?", PASSAGES)
    assert d.refs == ["DOC-1-C1"] and "Standard payment terms are net 30 days from the invoice date. [DOC-1-C1]" in d.text and d.text.startswith(direct.INTRO)
    d = direct.pick("Can the AI Assistant approve or reject an invoice?", PASSAGES)
    assert d.refs == ["DOC-1-C2"] and "cannot approve, reject, edit or delete anything" in d.text
    for line in d.text.splitlines():
        if line.startswith("- "):
            assert line[2:].rsplit(" [", 1)[0] in PASSAGES[1]["text"]                 # every quoted line really is in the passage


def test_vague_or_unsupported_questions_are_not_answered_by_quoting():
    assert direct.pick("What is up?", PASSAGES) is None and direct.pick("tell me", PASSAGES) is None
    assert direct.pick("Is alcohol reimbursable on trips?", PASSAGES) is None
    assert direct.pick("What is the standard payment term?", [], 0.5) is None


# ---------------------------------------------------------------- end to end: how many model requests did Groq see?
@pytest.fixture
def hc(rag, monkeypatch):
    monkeypatch.setattr(settings, "chat_hardcode", True)
    rag.upload(name="policy.md", data=POLICY, min_role="viewer", title="AP Policy")
    rag.upload(name="rules.md", data=RULES, min_role="viewer", title="Decision rules")
    return rag


def ask(w, message, script=(), who="view1", **kw):
    return w.chat.ask(message, who=who, script=list(script), **kw)


def test_greetings_and_help_cost_no_model_call(hc):
    for text in ["hi", "thanks", "help"]:
        r = ask(hc, text)
        assert r.status_code == 200 and r.json()["route"] == "canned" and r.json()["verified"] and r.json()["ai_calls"] == 0
    assert hc.fake.requests == []


def test_data_questions_are_answered_from_the_tool_with_no_model_call(hc):
    from app.services import processing
    m = processing.metrics()
    r = ask(hc, "how many invoices are pending review?")
    assert r.json()["answer"] == f"{m['pending_review']:,} invoice(s) are pending review." and r.json()["route"] == "rules" and r.json()["grounded"]
    r = ask(hc, "how many invoices in total")
    assert r.json()["answer"] == f"OpsPilot holds {m['invoices']:,} invoice(s) in total."
    r = ask(hc, "invoices from acme")
    assert r.json()["route"] == "rules" and "Acme" in r.json()["answer"] and r.json()["tools_used"][0]["tool"] == "search_invoices"
    r = ask(hc, "top 3 vendors")
    assert r.json()["tools_used"][0]["tool"] == "list_vendors" and r.json()["route"] == "rules"
    assert hc.fake.requests == []


def test_a_role_that_may_not_ask_gets_a_fixed_refusal_with_no_model_call(hc):
    r = ask(hc, "list users", who="view1")
    assert r.json()["answer"] == rules.fastpath.forbidden_text() and r.json()["route"] == "rules" and hc.fake.requests == []
    assert ask(hc, "list users", who="admin1").json()["answer"].startswith("3 user account")


def test_a_policy_question_is_answered_by_quoting_with_no_model_call(hc):
    r = ask(hc, "What is the standard payment term for invoices?")
    body = r.json()
    assert body["route"] == "direct" and body["ai_calls"] == 0 and body["verified"] and not body["blocked"]
    assert "net 30 days from the invoice date" in body["answer"] and body["citations"][0]["title"] == "AP Policy"
    assert body["answer"].count("[DOC-") >= 1 and all(c["ref"] in body["answer"] for c in body["citations"])
    assert hc.fake.requests == []
    assert rows("rag")[-1].action.startswith("search") and rows("chat")[-1].details["route"] == "direct"


def test_a_what_if_uses_one_model_call_without_tool_definitions(hc):
    ref = ref_of(hc, "above 10,000", "invoice amount above 10,000 human review")
    answer = f"No. For the values you gave (12,000, risk 0.10) the amount is above 10,000, so it goes to human review [{ref}]."
    r = ask(hc, WHAT_IF, [say(answer)])
    body = r.json()
    assert body["route"] == "context" and body["ai_calls"] == 1 and body["verified"] and body["answer"] == answer
    assert len(hc.fake.requests) == 1
    sent = hc.fake.requests[0]["body"]
    assert "tools" not in sent and sent["messages"][0]["content"].startswith("You are OpsPilot Assistant")
    assert "Document passages (untrusted data" in sent["messages"][-1]["content"] and "above 10,000" in sent["messages"][-1]["content"]
    assert len(json.dumps(sent)) < 5000                                          # versus ~12,000 characters for the tool-calling path


def test_an_answer_the_model_gives_in_that_single_call_is_still_checked(hc):
    ref = ref_of(hc, "above 10,000", "invoice amount above 10,000 human review")
    bad = f"No. The limit is 15,000, so it goes to human review [{ref}]."
    good = f"No. The amount of 12,000 is above 10,000, so it goes to human review [{ref}]."
    r = ask(hc, WHAT_IF, [say(bad), say(good)])
    assert r.json()["verified"] and "15,000" not in r.json()["answer"] and len(hc.fake.requests) == 2


def test_an_explain_request_goes_to_the_model_and_a_follow_up_finds_its_topic(hc):
    ref = ref_of(hc, "net 30", "payment terms net 30")
    ask(hc, "What are the payment terms?")                                       # quoted, no model call
    r = ask(hc, "explain that", [say(f"Payment is due 30 days after the invoice date [{ref}].")],
            history=[{"role": "user", "content": "What are the payment terms?"}, {"role": "assistant", "content": "Net 30."}])
    assert r.json()["route"] == "context" and len(hc.fake.requests) == 1
    assert "net 30" in hc.fake.requests[0]["body"]["messages"][-1]["content"].lower()


def test_an_unanswerable_non_data_question_costs_nothing(hc):
    r = ask(hc, "what is Machine Learning")
    assert r.json()["answer"] == prompts.NOT_FOUND_OR_SCOPE_TEXT and r.json()["route"] == "rules" and r.json()["verified"] and hc.fake.requests == []


def test_documents_a_role_may_not_read_are_not_found_without_a_model_call(hc):
    hc.upload(name="exec.md", data=b"# Executive Compensation Plan\n\nThe executive bonus pool for the year is 4,500,000 USD paid out in March.\n", min_role="admin", title="Exec")
    r = ask(hc, "What is the executive bonus pool?", who="view1")
    assert r.json()["answer"] == prompts.NOT_FOUND_OR_SCOPE_TEXT and hc.fake.requests == [] and "4,500,000" not in r.text
    assert "4,500,000" in ask(hc, "What is the executive bonus pool?", who="admin1").json()["answer"]


def test_live_data_questions_still_use_the_model_and_its_tools(hc):
    i1 = hc.chat.ids["i1"]
    r = ask(hc, "What is the status of invoice INV-1001 compared with our limits?",
            [tc("get_invoice", {"invoice_id": i1}), say("Invoice INV-1001 from Acme Corp is USD 1,250.50.")], who="rev1")
    assert r.json()["route"] == "llm" and r.json()["ai_calls"] == 2 and len(hc.fake.requests) == 2
    assert "tools" in hc.fake.requests[0]["body"]


def test_a_data_question_with_no_matching_document_falls_through_to_the_model(hc):
    r = ask(hc, "Any problems with Acme lately?", [tc("search_invoices", {"vendor": "Acme"}), say("There is one Acme invoice in the queue.")], who="rev1")
    assert r.json()["route"] == "llm" and len(hc.fake.requests) == 2 and r.json()["tools_used"][0]["tool"] == "search_invoices"


def test_the_document_search_failing_is_reported_not_guessed(hc, monkeypatch):
    from app.rag.embeddings import EmbedderUnavailable
    monkeypatch.setattr(hc.svc.embedder, "embed_query", lambda t: (_ for _ in ()).throw(EmbedderUnavailable("no model")))
    r = ask(hc, "What is the standard payment term for invoices?")
    assert r.json()["answer"] == prompts.DOCS_UNAVAILABLE_TEXT and hc.fake.requests == []


def test_llm_mode_always_asks_the_model_about_documents(hc, monkeypatch):
    monkeypatch.setattr(settings, "rag_answer_mode", "llm")
    ref = ref_of(hc, "net 30", "payment terms net 30")
    r = ask(hc, "What are the payment terms?", [say(f"Standard payment terms are net 30 days from the invoice date [{ref}].")])
    assert r.json()["route"] == "context" and len(hc.fake.requests) == 1


def test_switching_the_layer_off_restores_the_original_behaviour(hc, monkeypatch):
    monkeypatch.setattr(settings, "chat_hardcode", False)
    r = ask(hc, "hello", [say("Hello! How can I help?")])
    assert r.json()["answer"] == "Hello! How can I help?" and r.json()["route"] == "llm" and len(hc.fake.requests) == 1


def test_without_documents_a_clearly_unrelated_question_is_declined_for_free(chat, monkeypatch):
    monkeypatch.setattr(settings, "chat_hardcode", True)
    r = chat.ask("what is Machine Learning", who="view1", script=[])
    assert r.json()["answer"] == prompts.OUT_OF_SCOPE_TEXT and r.json()["route"] == "rules" and chat.fake.requests == []


# ---------------------------------------------------------------- the assistant is read-only: say so for free; never a 502
@pytest.mark.parametrize("text", ["Can you have access to approve any invoice", "Can the AI Assistant approve or reject an invoice?", "approve invoice 12",
                                  "please delete invoice 5", "Are you able to pay vendors?", "Do you have permission to edit invoices", "reject it"])
def test_requests_for_the_assistant_to_change_things_get_a_fixed_read_only_answer(text):
    assert rules.canned(text, [], True) == rules.READ_ONLY_TEXT


@pytest.mark.parametrize("text", ["update me on pending invoices", "Can you list invoices to approve", "what can you approve", "pay attention to vendors",
                                  "how does the AI approve invoices automatically", "Can a reviewer approve invoices above 10,000?", "show approved invoices",
                                  "which invoices need approval", "what is the approval limit"])
def test_nearby_questions_are_not_mistaken_for_requests_to_act(text):
    assert rules.canned(text, [], True) is None


def test_the_read_only_reply_costs_no_model_call(hc):
    r = ask(hc, "Can you have access to approve any invoice", who="admin1")
    assert r.status_code == 200 and r.json()["answer"] == rules.READ_ONLY_TEXT and r.json()["route"] == "canned" and hc.fake.requests == []


def test_a_model_that_will_neither_call_a_tool_nor_answer_is_not_a_502(rag, monkeypatch):
    """The reported failure: tool_choice=required, the model answers in text, Groq answers 400 tool_use_failed, twice."""
    monkeypatch.setattr(settings, "chat_hardcode", False)
    import httpx
    refused = httpx.Response(400, json={"error": {"code": "tool_use_failed", "message": "Tool choice is required, but model did not call a tool"}})
    r = rag.chat.ask("Tell me something about yourself and your abilities", who="admin1", script=[refused, say("I am an assistant."), refused])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == prompts.NO_LOOKUP_TEXT and body["verified"] and not body["blocked"] and body["tools_used"] == []


def test_a_genuinely_malformed_tool_call_is_still_an_upstream_error(rag, monkeypatch):
    import httpx
    monkeypatch.setattr(settings, "chat_hardcode", False)
    bad = httpx.Response(400, json={"error": {"code": "tool_use_failed", "message": "Failed to parse tool call arguments"}})
    assert rag.chat.ask("tell me about invoices please", who="admin1", script=[bad, bad]).status_code == 502


def test_a_flattened_pdf_table_is_quoted_in_short_verbatim_windows_not_as_one_wall_of_text():
    table = ("Quick reference Item Value Enforced by Automatic approval limit 10,000 System New-vendor review amount 2,000 System "
             "Automatic approval risk score 0.30 or lower System Automatic rejection risk score 0.85 or higher System Classifier minimum confidence 70% System "
             "Duplicate amount tolerance 0.5% System Required invoice fields Vendor name, invoice number, total amount, due date System "
             "Supported currencies USD, EUR, GBP, INR, CAD, AUD System Longest payment terms before a flag 120 days System Standard payment terms Net 30 days Procedure")
    windows = direct.sentences(table)
    assert len(windows) > 1 and max(map(len, windows)) <= direct.MAX_QUOTE_CHARS
    assert all(w.strip("\u2026 ").strip() in table for w in windows)                              # still exactly what the document says
    d = direct.pick("Which currencies are supported?", [{"ref": "R1", "section": "Appendix A. Quick reference", "text": table}])
    assert d and "Supported currencies USD, EUR, GBP, INR, CAD, AUD" in d.text and len(d.text.splitlines()[1]) < 320
