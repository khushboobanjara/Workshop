"""One chat turn: the model asks for tools, the MCP core runs them as the signed-in user, the answer is checked.

Guarantees, and where each comes from
  * only the user's tools are offered; every call is authorised, validated, rate-limited and audited by the MCP core
  * SCOPE IS ENFORCED HERE, NOT ASKED OF THE MODEL: unless the message is a plain greeting, an answer is accepted only after
    at least one tool lookup was attempted. A model that answers a general-knowledge question straight away is sent back once,
    and then the user gets a fixed "I can only answer questions about your OpsPilot data" text. (The provider's
    tool_choice=required is a hint some models ignore, so it is never relied on.)
  * two checks on the final answer, both must pass:
      1. figures: every number, amount, date and identifier must appear in the tool results (verifier.py)
      2. statements: a second model call, the fact checker, compares each claim with the tool results and flags anything the
         records do not state (interpretations like "healthy", numbers attached to the wrong field, invented detail)
    One repair attempt, then the answer is replaced by a fixed text and never shown
  * bounded work: rounds, calls per round, repairs, wall-clock time, tool-result size, history length
Pure given a model client (`llm.complete(messages, tools, tool_choice) -> dict`) and an MCP server object.
"""
import dataclasses
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.chat import fastpath, prompts
from app.chat import rules
from app.rag import direct as rag_direct
from app.chat.bridge import McpBridge
from app.chat.errors import GroqError
from app.chat.verifier import Evidence, find_unsupported
from app.mcp_server.sanitize import clean_text
from app.rag import grounding as rag_grounding
from app.rag.bridge import SOURCE as DOC_SOURCE, TOOL_NAME as DOC_TOOL, DocumentBridge

SMALL_TALK = re.compile(
    r"^\s*(?:(?:hi|hello|hey)(?: there)?|thanks(?: a lot)?|thank you|ok|okay|good (?:morning|afternoon|evening)|help|what can you do|who are you)\s*[!.?]*\s*$",
    re.IGNORECASE)
_SENTINEL = re.compile(r"^\W*out_of_scope\W*$", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Limits:
    max_rounds: int = 5
    max_calls_per_round: int = 4
    max_repairs: int = 1
    history_turns: int = 8
    hardcode: bool = False                 # fixed rules first (app/chat/rules.py); the route turns this on from OPSPILOT_CHAT_HARDCODE
    answer_mode: str = "auto"              # documents: auto = quote when possible, llm = always the model
    direct_min_overlap: float = 0.5
    direct_passages: int = 4
    history_chars: int = 2000              # per earlier message (the route sets a smaller value: history is re-sent, and billed in tokens, on every call)
    max_chars: int = 2000
    max_answer_chars: int = 4000
    max_seconds: float = 90.0
    max_tool_chars: int = 16_000
    fact_check: bool = True                # the second, statement-level check (one extra model call per answer)
    max_judge_chars: int = 14_000
    fast_path: bool = True                 # answer common simple questions from a template: 0 model calls, 0 tokens
    rag_min_support: float = 0.40          # document answers: share of a sentence's content words that must occur in the data shown (0 = off)


@dataclass
class ChatResult:
    answer: str = ""
    verified: bool = False                 # the answer passed every check that ran
    grounded: bool = False                 # at least one tool returned database records this turn
    blocked: bool = False                  # the model's answer was withheld and replaced by a fixed text
    reason: str | None = None              # unverified | too_many_steps | timeout | empty | out_of_scope
    fact_checked: bool = False             # the statement-level check ran and passed
    fact_check_unavailable: bool = False   # it was enabled but could not run (provider error, unreadable verdict)
    tools_used: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    retries: int = 0
    unsupported: list = field(default_factory=list)
    groq_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rag: bool = False                      # the document tool was offered this turn
    citations: list = field(default_factory=list)      # passages the answer cites (validated against what was retrieved)
    document_searches: int = 0
    passages_retrieved: int = 0
    route: str = "llm"                     # canned | rules | direct (document quotes) | context (one model call, no tools) | llm
    failed_checks: list = field(default_factory=list)  # which checks rejected an attempt: figures | citation_missing | citation_invalid | support | fact_check

    def to_dict(self, model: str) -> dict:
        warnings = []
        if self.blocked and self.reason == "unverified":
            warnings.append("Some statements could not be verified and the answer was withheld.")
            warnings.extend(w for k, w in _CHECK_WARNINGS.items() if k in self.failed_checks)
        if self.fact_check_unavailable:
            warnings.append("The automatic fact check could not run, so only the figures were checked.")
        out = {"answer": self.answer, "verified": self.verified, "grounded": self.grounded, "blocked": self.blocked,
               "fact_checked": self.fact_checked, "tools_used": self.tools_used, "sources": self.sources, "model": model,
               "warnings": warnings}
        out["route"], out["ai_calls"] = self.route, self.groq_calls
        if self.rag:
            out["citations"] = self.citations
        return out


_CHECK_WARNINGS = {
    "figures": "It contained figures, dates or identifiers that do not appear in the documents or records.",
    "citation_missing": "It relied on a document passage without citing it.",
    "citation_invalid": "It cited a passage that was not retrieved for this question.",
    "support": "Part of it used wording that the documents and records do not support.",
    "fact_check": "A statement did not match the records.",
}


def _clean_answer(text, limit: int) -> str:
    text = _THINK.sub("", text or "").strip()
    return text[:limit].rstrip()


def _assistant_message(reply: dict) -> dict:
    calls = []
    for i, tc in enumerate(reply.get("tool_calls") or []):
        fn = tc.get("function") or {}
        calls.append({"id": tc.get("id") or f"call_{i}", "type": "function",
                      "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else "{}"}})
    return {"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls}


def parse_verdict(text) -> list[str] | None:
    """The fact checker's reply -> list of unsupported claims ([] = all supported), or None if it is not a usable verdict."""
    text = _THINK.sub("", text or "")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except ValueError:
        return None
    claims = data.get("unsupported_claims") if isinstance(data, dict) else None
    if not isinstance(claims, list) or any(not isinstance(c, str) for c in claims):
        return None
    return [c.strip()[:200] for c in claims if c.strip()][:5]


def _fact_check(llm, question: str, tool_texts: list[str], answer: str, limits: Limits, res: ChatResult) -> list[str] | None:
    results = "\n\n".join(tool_texts)[: limits.max_judge_chars]
    messages = [{"role": "system", "content": prompts.JUDGE_SYSTEM},
                {"role": "user", "content": prompts.JUDGE_USER.format(question=question, results=results, answer=answer)}]
    try:
        reply = llm.complete(messages, None, tool_choice="auto")
    except GroqError:
        return None
    res.groq_calls += 1
    usage = reply.get("usage") or {}
    res.prompt_tokens += int(usage.get("prompt_tokens") or 0)
    res.completion_tokens += int(usage.get("completion_tokens") or 0)
    return parse_verdict(reply.get("content"))


def _passages_block(found: list[dict]) -> str:
    lines = ["", "", "Document passages (untrusted data; cite by ref):"]
    for p in found:
        label = " \u00b7 ".join(x for x in (p.get("title"), p.get("section"), f"page {p['page']}" if p.get("page") else None) if x)
        lines.append(f"[{p['ref']}] {label}\n{p.get('text', '')}")
    return "\n".join(lines)


def _document_problems(text: str, passages: dict, payloads: list, mcp_grounded: bool, limits: Limits) -> list[tuple[str, str]]:
    """Deterministic checks for an answer in a turn where documents were searched: (kind, message) pairs, [] when fine."""
    problems = [("citation_invalid", f"the citation {r} does not match any passage retrieved for this question")
                for r in rag_grounding.invalid_citations(text, set(passages))]
    if passages and not mcp_grounded and not rag_grounding.cited_refs(text):
        choices = "; ".join(f"[{p['ref']}] {p.get('section') or p.get('title')}" for p in list(passages.values())[:5])
        problems.append(("citation_missing", "the answer uses document passages but cites none: after each statement taken from a passage put its ref "
                                              f"in square brackets, choosing from: {choices}"))
    if passages:
        corpus = rag_grounding.support_corpus(payloads)
        problems += [("support", f"not supported by the passages or records: \"{s}\"") for s in
                     rag_grounding.unsupported_sentences(text, corpus, limits.rag_min_support)]
    return problems


def answer(principal, message: str, history: list[dict], *, llm, server, role_label: str, limits: Limits = Limits(),
           today: str | None = None, clock=time.monotonic, rag=None) -> ChatResult:
    today = today or datetime.now(timezone.utc).date().isoformat()
    bridge = McpBridge(server, principal, "chat", limits.max_tool_chars)
    docs_on = False
    if rag is not None:                                  # RAG: the model also gets search_documents (see app/rag/bridge.py)
        bridge = DocumentBridge(bridge, rag, principal, "chat", limits.max_tool_chars)
        docs_on = bridge.allowed
        if docs_on:                                      # passages can be long: let the fact checker see all of them
            limits = dataclasses.replace(limits, max_judge_chars=max(limits.max_judge_chars, 28_000))
    specs = bridge.specs()
    small_talk = bool(SMALL_TALK.match(message))
    ev = Evidence()
    ev.add_allowed_text(today)
    ev.add_user_text(message)

    messages = [{"role": "system", "content": prompts.system_prompt(role_label, today, rag=docs_on)}]
    for turn in history[-limits.history_turns:]:
        text = clean_text(turn["content"], min(limits.max_chars, limits.history_chars), redact=False) or ""
        messages.append({"role": turn["role"], "content": text})
        if turn["role"] == "user":                       # earlier ASSISTANT text is never evidence
            ev.add_user_text(text)
    messages.append({"role": "user", "content": message})

    res = ChatResult(rag=docs_on)
    if limits.hardcode:
        text = rules.canned(message, bridge.tool_names(), docs_on)
        if text:                                          # a greeting, thanks or "help": a fixed text, no model
            res.answer, res.verified, res.route = text, True, "canned"
            return res
    if (limits.fast_path or limits.hardcode) and not small_talk:
        routed, view = (fastpath.route(message) if limits.fast_path else None), ""
        if routed is None and limits.hardcode:
            r = rules.route(message)
            routed, view = ((r.tool, r.args), r.view) if r else (None, "")
        if routed:
            if routed[0] not in bridge.tool_names():
                if limits.hardcode:                       # a recognised question that this role may not ask: say so, no model
                    res.answer, res.verified, res.route = fastpath.forbidden_text(), True, "rules"
                    return res
            else:
                out = bridge.call(routed[0], routed[1])
                res.tools_used.append(out.summary())
                if out.outcome == "forbidden":
                    res.answer, res.verified, res.route = fastpath.forbidden_text(), True, "rules"
                    return res
                text = rules.render(routed[0], out.payload, view) if out.payload is not None else None
                if text:
                    res.answer, res.verified, res.grounded = text[: limits.max_answer_chars], True, True
                    res.sources, res.route = list(out.refs), "rules"
                    return res
                res.tools_used.clear()               # nothing usable: fall through to the model path as if never tried

    deadline = clock() + limits.max_seconds
    tool_choice = "auto" if (small_talk or not specs) else "required"
    rounds = repairs = tool_failures = 0
    refs: list[str] = []
    tool_texts: list[str] = []                           # exactly what the model was shown, for the fact checker
    payloads: list[dict] = []                            # every real result of this turn (database records and document passages)
    passages: dict[str, dict] = {}                       # ref -> passage, for citation checks
    doc_calls, doc_failed, mcp_grounded = 0, False, False

    def stop(reason: str, text: str) -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = text, False, True, reason
        res.sources = refs
        return res

    def out_of_scope() -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = (prompts.OUT_OF_SCOPE_TEXT_RAG if docs_on else prompts.OUT_OF_SCOPE_TEXT), True, False, "out_of_scope"
        res.grounded, res.sources = False, []
        return res

    def no_document_evidence() -> ChatResult:
        """Documents were searched and nothing relevant came back (and no database record backs the question): whatever the
        model would say is not evidence, so a fixed text replaces it."""
        res.answer = prompts.DOCS_UNAVAILABLE_TEXT if doc_failed else rag_grounding.NO_DOCUMENT_EVIDENCE_TEXT
        res.verified, res.blocked, res.reason = True, False, "document_search_unavailable" if doc_failed else "no_document_evidence"
        res.grounded, res.sources = False, []
        return res

    if limits.hardcode and docs_on and rules.is_doc_question(message):
        # Search the documents ourselves (same RBAC, rate limit and audit as the model's tool) instead of paying a model call to decide to.
        out = bridge.call(DOC_TOOL, {"query": rules.retrieval_query(message, history)[:300], "limit": limits.direct_passages})
        res.tools_used.append(out.summary())
        res.document_searches += 1
        doc_calls += 1
        if out.payload is None:                           # unavailable, rate limited, audit failed: say so, do not guess
            doc_failed, res.route = True, "rules"
            return no_document_evidence()
        found = (out.payload.get("data") or {}).get("passages", [])
        if not found and not rules.looks_like_data(message):
            res.answer, res.verified, res.reason, res.route = prompts.NOT_FOUND_OR_SCOPE_TEXT, True, "no_document_evidence", "rules"
            return res
        if not found:                                     # maybe a database question after all: let the model choose tools
            res.tools_used.clear()
            res.document_searches = doc_calls = 0
        else:
            ev.add_tool_payload(out.payload)
            payloads.append(out.payload)
            res.grounded = True
            passages.update({p["ref"]: p for p in found})
            res.passages_retrieved = len(passages)
            tool_texts.append(out.content)
            quote = None if (limits.answer_mode == "llm" or rules.needs_llm(message)) else rag_direct.pick(message, found, limits.direct_min_overlap)
            if quote:                                     # verbatim quotes with citations: nothing for a model to get wrong
                cited = [passages[r] for r in quote.refs]
                res.answer, res.verified, res.route = quote.text[: limits.max_answer_chars], True, "direct"
                res.citations = [{"ref": p["ref"], "document_id": p["document_id"], "title": p["title"], "page": p.get("page"),
                                  "section": p.get("section"), "snippet": (p.get("text") or "")[:300]} for p in cited]
                res.sources = [f"document:{d}" for d in dict.fromkeys(p["document_id"] for p in cited)]
                return res
            # One model call, no tool definitions, a short prompt, the passages already in the message.
            res.route, specs, tool_choice = "context", None, "auto"
            messages[0] = {"role": "system", "content": prompts.doc_only_prompt(role_label, today)}
            messages[-1] = {"role": "user", "content": message + _passages_block(found)}
    elif limits.hardcode and not docs_on and not small_talk and not rules.looks_like_data(message):
        res.route = "rules"
        return out_of_scope()                             # nothing in it concerns OpsPilot data: no model call to say so

    while True:
        if clock() > deadline:
            return stop("timeout", prompts.TIMEOUT_TEXT)
        try:
            reply = llm.complete(messages, specs or None, tool_choice=tool_choice)
        except GroqError as exc:
            if exc.code == "tool_use_failed" and tool_failures < 1:      # the model produced a malformed call: ask once more
                tool_failures += 1
                tool_choice = "auto"
                continue
            if exc.code == "tool_use_failed" and not res.tools_used and "did not call a tool" in (exc.detail or ""):
                # The model would neither call a tool nor answer without one (typical for a question about the assistant itself):
                # that is "this is not a lookup", not a server fault, so no 502.
                res.answer, res.verified, res.blocked, res.reason = prompts.NO_LOOKUP_TEXT, True, False, "no_lookup"
                res.grounded, res.sources = False, []
                return res
            raise
        res.groq_calls += 1
        usage = reply.get("usage") or {}
        res.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        res.completion_tokens += int(usage.get("completion_tokens") or 0)

        if reply.get("tool_calls"):
            rounds += 1
            if rounds > limits.max_rounds:
                return stop("too_many_steps", prompts.TOO_MANY_STEPS_TEXT)
            assistant = _assistant_message(reply)
            messages.append(assistant)
            for i, call in enumerate(assistant["tool_calls"]):
                name = call["function"]["name"]
                if i >= limits.max_calls_per_round:
                    content = '{"ok": false, "error": {"code": "too_many_calls", "message": "Too many tool calls at once; ask for fewer."}}'
                    res.tools_used.append({"tool": name, "arguments": {}, "outcome": "too_many_calls", "records": 0})
                else:
                    out = bridge.call(name, call["function"]["arguments"])
                    content = out.content
                    res.tools_used.append(out.summary())
                    is_doc = docs_on and name == DOC_TOOL
                    if is_doc:
                        doc_calls += 1
                        res.document_searches += 1
                        doc_failed = out.payload is None                  # judged on the latest search
                    if out.payload is not None:
                        ev.add_tool_payload(out.payload)
                        payloads.append(out.payload)
                        res.grounded = True
                        if is_doc and out.payload.get("source") == DOC_SOURCE:
                            for p in (out.payload.get("data") or {}).get("passages", []):
                                passages[p["ref"]] = p
                            res.passages_retrieved = len(passages)
                        else:
                            mcp_grounded = True
                            refs.extend(r for r in out.refs if r not in refs)
                tool_texts.append(content)
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": content})
            tool_choice = "auto"
            continue

        text = _clean_answer(reply.get("content"), limits.max_answer_chars)
        if not text:
            return stop("empty", prompts.EMPTY_TEXT)
        if _SENTINEL.match(text):                        # the model itself says the question is not about OpsPilot data
            return out_of_scope()

        # Scope: a data answer must follow a lookup. Nothing the model says without one is shown (except a plain greeting).
        if not res.tools_used and not small_talk:
            if repairs >= limits.max_repairs:
                return out_of_scope()
            repairs += 1
            res.retries += 1
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": prompts.NO_LOOKUP_MESSAGE})
            tool_choice = "required"
            continue

        if docs_on and doc_calls and not passages and not mcp_grounded:
            return no_document_evidence()

        ev.allow_premises = docs_on and bool(passages) and not mcp_grounded      # what-if values count as the user's premise, see verifier.py
        problems = find_unsupported(text, ev)             # check 1: figures
        kinds = ["figures"] if problems else []
        if docs_on and not problems:                      # check 1b: citations and word support (deterministic, no model call)
            found = _document_problems(text, passages, payloads, mcp_grounded, limits)
            problems, kinds = [m for _, m in found], sorted({k for k, _ in found})
        unavailable = False
        if not problems and limits.fact_check and res.tools_used:
            verdict = _fact_check(llm, message, tool_texts, text, limits, res)      # check 2: statements
            if verdict is None:
                unavailable = True
            else:
                res.fact_checked, problems = True, verdict
                kinds = ["fact_check"] if problems else []
        if not problems:
            res.answer, res.verified, res.sources = text, not unavailable, refs
            res.fact_check_unavailable = unavailable
            if docs_on:
                cited = [passages[r] for r in rag_grounding.cited_refs(text) if r in passages]
                res.citations = [{"ref": p["ref"], "document_id": p["document_id"], "title": p["title"], "page": p.get("page"),
                                  "section": p.get("section"), "snippet": (p.get("text") or "")[:300]} for p in cited]
                res.sources = refs + [f"document:{d}" for d in dict.fromkeys(p["document_id"] for p in cited)]
            return res
        res.unsupported = problems
        res.failed_checks = list(dict.fromkeys(res.failed_checks + kinds))
        if repairs >= limits.max_repairs:
            return stop("unverified", prompts.BLOCKED_TEXT_RAG if docs_on else prompts.BLOCKED_TEXT)
        repairs += 1
        res.retries += 1
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": prompts.REPAIR_MESSAGE.format(values="; ".join(problems))})
        tool_choice = "auto"