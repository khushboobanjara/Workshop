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
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.chat import fastpath, prompts
from app.chat.bridge import McpBridge
from app.chat.errors import GroqError
from app.chat.verifier import Evidence, find_unsupported
from app.mcp_server.sanitize import clean_text

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
    max_chars: int = 2000
    max_answer_chars: int = 4000
    max_seconds: float = 90.0
    max_tool_chars: int = 16_000
    fact_check: bool = True                # the second, statement-level check (one extra model call per answer)
    max_judge_chars: int = 14_000
    fast_path: bool = True                 # answer common simple questions from a template: 0 model calls, 0 tokens


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

    def to_dict(self, model: str) -> dict:
        warnings = []
        if self.blocked and self.reason == "unverified":
            warnings.append("Some statements could not be verified and the answer was withheld.")
        if self.fact_check_unavailable:
            warnings.append("The automatic fact check could not run, so only the figures were checked.")
        return {"answer": self.answer, "verified": self.verified, "grounded": self.grounded, "blocked": self.blocked,
                "fact_checked": self.fact_checked, "tools_used": self.tools_used, "sources": self.sources, "model": model,
                "warnings": warnings}


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


def answer(principal, message: str, history: list[dict], *, llm, server, role_label: str, limits: Limits = Limits(),
           today: str | None = None, clock=time.monotonic) -> ChatResult:
    today = today or datetime.now(timezone.utc).date().isoformat()
    bridge = McpBridge(server, principal, "chat", limits.max_tool_chars)
    specs = bridge.specs()
    small_talk = bool(SMALL_TALK.match(message))
    ev = Evidence()
    ev.add_allowed_text(today)
    ev.add_user_text(message)

    messages = [{"role": "system", "content": prompts.system_prompt(role_label, today)}]
    for turn in history[-limits.history_turns:]:
        text = clean_text(turn["content"], limits.max_chars, redact=False) or ""
        messages.append({"role": turn["role"], "content": text})
        if turn["role"] == "user":                       # earlier ASSISTANT text is never evidence
            ev.add_user_text(text)
    messages.append({"role": "user", "content": message})

    res = ChatResult()
    if limits.fast_path and not small_talk:
        routed = fastpath.route(message)
        if routed and routed[0] in bridge.tool_names():
            out = bridge.call(routed[0], routed[1])
            res.tools_used.append(out.summary())
            if out.outcome == "forbidden":
                res.answer, res.verified = fastpath.forbidden_text(), True
                return res
            text = fastpath.render(routed[0], out.payload) if out.payload is not None else None
            if text:
                res.answer, res.verified, res.grounded = text[: limits.max_answer_chars], True, True
                res.sources = list(out.refs)
                return res
            res.tools_used.clear()               # nothing usable: fall through to the model path as if never tried

    deadline = clock() + limits.max_seconds
    tool_choice = "auto" if (small_talk or not specs) else "required"
    rounds = repairs = tool_failures = 0
    refs: list[str] = []
    tool_texts: list[str] = []                           # exactly what the model was shown, for the fact checker

    def stop(reason: str, text: str) -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = text, False, True, reason
        res.sources = refs
        return res

    def out_of_scope() -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = prompts.OUT_OF_SCOPE_TEXT, True, False, "out_of_scope"
        res.grounded, res.sources = False, []
        return res

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
                    if out.payload is not None:
                        ev.add_tool_payload(out.payload)
                        res.grounded = True
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

        problems = find_unsupported(text, ev)             # check 1: figures
        unavailable = False
        if not problems and limits.fact_check and res.tools_used:
            verdict = _fact_check(llm, message, tool_texts, text, limits, res)      # check 2: statements
            if verdict is None:
                unavailable = True
            else:
                res.fact_checked, problems = True, verdict
        if not problems:
            res.answer, res.verified, res.sources = text, not unavailable, refs
            res.fact_check_unavailable = unavailable
            return res
        res.unsupported = problems
        if repairs >= limits.max_repairs:
            return stop("unverified", prompts.BLOCKED_TEXT)
        repairs += 1
        res.retries += 1
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": prompts.REPAIR_MESSAGE.format(values="; ".join(problems))})
        tool_choice = "auto""""One chat turn: the model asks for tools, the MCP core runs them as the signed-in user, the answer is checked.

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
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.chat import fastpath, prompts
from app.chat.bridge import McpBridge
from app.chat.errors import GroqError
from app.chat.verifier import Evidence, find_unsupported
from app.mcp_server.sanitize import clean_text

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
    max_chars: int = 2000
    max_answer_chars: int = 4000
    max_seconds: float = 90.0
    max_tool_chars: int = 16_000
    fact_check: bool = True                # the second, statement-level check (one extra model call per answer)
    max_judge_chars: int = 14_000
    fast_path: bool = True                 # answer common simple questions from a template: 0 model calls, 0 tokens


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

    def to_dict(self, model: str) -> dict:
        warnings = []
        if self.blocked and self.reason == "unverified":
            warnings.append("Some statements could not be verified and the answer was withheld.")
        if self.fact_check_unavailable:
            warnings.append("The automatic fact check could not run, so only the figures were checked.")
        return {"answer": self.answer, "verified": self.verified, "grounded": self.grounded, "blocked": self.blocked,
                "fact_checked": self.fact_checked, "tools_used": self.tools_used, "sources": self.sources, "model": model,
                "warnings": warnings}


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


def answer(principal, message: str, history: list[dict], *, llm, server, role_label: str, limits: Limits = Limits(),
           today: str | None = None, clock=time.monotonic) -> ChatResult:
    today = today or datetime.now(timezone.utc).date().isoformat()
    bridge = McpBridge(server, principal, "chat", limits.max_tool_chars)
    specs = bridge.specs()
    small_talk = bool(SMALL_TALK.match(message))
    ev = Evidence()
    ev.add_allowed_text(today)
    ev.add_user_text(message)

    messages = [{"role": "system", "content": prompts.system_prompt(role_label, today)}]
    for turn in history[-limits.history_turns:]:
        text = clean_text(turn["content"], limits.max_chars, redact=False) or ""
        messages.append({"role": turn["role"], "content": text})
        if turn["role"] == "user":                       # earlier ASSISTANT text is never evidence
            ev.add_user_text(text)
    messages.append({"role": "user", "content": message})

    res = ChatResult()
    if limits.fast_path and not small_talk:
        routed = fastpath.route(message)
        if routed and routed[0] in bridge.tool_names():
            out = bridge.call(routed[0], routed[1])
            res.tools_used.append(out.summary())
            if out.outcome == "forbidden":
                res.answer, res.verified = fastpath.forbidden_text(), True
                return res
            text = fastpath.render(routed[0], out.payload) if out.payload is not None else None
            if text:
                res.answer, res.verified, res.grounded = text[: limits.max_answer_chars], True, True
                res.sources = list(out.refs)
                return res
            res.tools_used.clear()               # nothing usable: fall through to the model path as if never tried

    deadline = clock() + limits.max_seconds
    tool_choice = "auto" if (small_talk or not specs) else "required"
    rounds = repairs = tool_failures = 0
    refs: list[str] = []
    tool_texts: list[str] = []                           # exactly what the model was shown, for the fact checker

    def stop(reason: str, text: str) -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = text, False, True, reason
        res.sources = refs
        return res

    def out_of_scope() -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = prompts.OUT_OF_SCOPE_TEXT, True, False, "out_of_scope"
        res.grounded, res.sources = False, []
        return res

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
                    if out.payload is not None:
                        ev.add_tool_payload(out.payload)
                        res.grounded = True
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

        problems = find_unsupported(text, ev)             # check 1: figures
        unavailable = False
        if not problems and limits.fact_check and res.tools_used:
            verdict = _fact_check(llm, message, tool_texts, text, limits, res)      # check 2: statements
            if verdict is None:
                unavailable = True
            else:
                res.fact_checked, problems = True, verdict
        if not problems:
            res.answer, res.verified, res.sources = text, not unavailable, refs
            res.fact_check_unavailable = unavailable
            return res
        res.unsupported = problems
        if repairs >= limits.max_repairs:
            return stop("unverified", prompts.BLOCKED_TEXT)
        repairs += 1
        res.retries += 1
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": prompts.REPAIR_MESSAGE.format(values="; ".join(problems))})
        tool_choice = "auto""""One chat turn: the model asks for tools, the MCP core runs them as the signed-in user, the answer is checked.

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
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.chat import fastpath, prompts
from app.chat.bridge import McpBridge
from app.chat.errors import GroqError
from app.chat.verifier import Evidence, find_unsupported
from app.mcp_server.sanitize import clean_text

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
    max_chars: int = 2000
    max_answer_chars: int = 4000
    max_seconds: float = 90.0
    max_tool_chars: int = 16_000
    fact_check: bool = True                # the second, statement-level check (one extra model call per answer)
    max_judge_chars: int = 14_000
    fast_path: bool = True                 # answer common simple questions from a template: 0 model calls, 0 tokens


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

    def to_dict(self, model: str) -> dict:
        warnings = []
        if self.blocked and self.reason == "unverified":
            warnings.append("Some statements could not be verified and the answer was withheld.")
        if self.fact_check_unavailable:
            warnings.append("The automatic fact check could not run, so only the figures were checked.")
        return {"answer": self.answer, "verified": self.verified, "grounded": self.grounded, "blocked": self.blocked,
                "fact_checked": self.fact_checked, "tools_used": self.tools_used, "sources": self.sources, "model": model,
                "warnings": warnings}


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


def answer(principal, message: str, history: list[dict], *, llm, server, role_label: str, limits: Limits = Limits(),
           today: str | None = None, clock=time.monotonic) -> ChatResult:
    today = today or datetime.now(timezone.utc).date().isoformat()
    bridge = McpBridge(server, principal, "chat", limits.max_tool_chars)
    specs = bridge.specs()
    small_talk = bool(SMALL_TALK.match(message))
    ev = Evidence()
    ev.add_allowed_text(today)
    ev.add_user_text(message)

    messages = [{"role": "system", "content": prompts.system_prompt(role_label, today)}]
    for turn in history[-limits.history_turns:]:
        text = clean_text(turn["content"], limits.max_chars, redact=False) or ""
        messages.append({"role": turn["role"], "content": text})
        if turn["role"] == "user":                       # earlier ASSISTANT text is never evidence
            ev.add_user_text(text)
    messages.append({"role": "user", "content": message})

    res = ChatResult()
    if limits.fast_path and not small_talk:
        routed = fastpath.route(message)
        if routed and routed[0] in bridge.tool_names():
            out = bridge.call(routed[0], routed[1])
            res.tools_used.append(out.summary())
            if out.outcome == "forbidden":
                res.answer, res.verified = fastpath.forbidden_text(), True
                return res
            text = fastpath.render(routed[0], out.payload) if out.payload is not None else None
            if text:
                res.answer, res.verified, res.grounded = text[: limits.max_answer_chars], True, True
                res.sources = list(out.refs)
                return res
            res.tools_used.clear()               # nothing usable: fall through to the model path as if never tried

    deadline = clock() + limits.max_seconds
    tool_choice = "auto" if (small_talk or not specs) else "required"
    rounds = repairs = tool_failures = 0
    refs: list[str] = []
    tool_texts: list[str] = []                           # exactly what the model was shown, for the fact checker

    def stop(reason: str, text: str) -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = text, False, True, reason
        res.sources = refs
        return res

    def out_of_scope() -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = prompts.OUT_OF_SCOPE_TEXT, True, False, "out_of_scope"
        res.grounded, res.sources = False, []
        return res

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
                    if out.payload is not None:
                        ev.add_tool_payload(out.payload)
                        res.grounded = True
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

        problems = find_unsupported(text, ev)             # check 1: figures
        unavailable = False
        if not problems and limits.fact_check and res.tools_used:
            verdict = _fact_check(llm, message, tool_texts, text, limits, res)      # check 2: statements
            if verdict is None:
                unavailable = True
            else:
                res.fact_checked, problems = True, verdict
        if not problems:
            res.answer, res.verified, res.sources = text, not unavailable, refs
            res.fact_check_unavailable = unavailable
            return res
        res.unsupported = problems
        if repairs >= limits.max_repairs:
            return stop("unverified", prompts.BLOCKED_TEXT)
        repairs += 1
        res.retries += 1
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": prompts.REPAIR_MESSAGE.format(values="; ".join(problems))})
        tool_choice = "auto""""One chat turn: the model asks for tools, the MCP core runs them as the signed-in user, the answer is checked.

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
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.chat import fastpath, prompts
from app.chat.bridge import McpBridge
from app.chat.errors import GroqError
from app.chat.verifier import Evidence, find_unsupported
from app.mcp_server.sanitize import clean_text

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
    max_chars: int = 2000
    max_answer_chars: int = 4000
    max_seconds: float = 90.0
    max_tool_chars: int = 16_000
    fact_check: bool = True                # the second, statement-level check (one extra model call per answer)
    max_judge_chars: int = 14_000
    fast_path: bool = True                 # answer common simple questions from a template: 0 model calls, 0 tokens


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

    def to_dict(self, model: str) -> dict:
        warnings = []
        if self.blocked and self.reason == "unverified":
            warnings.append("Some statements could not be verified and the answer was withheld.")
        if self.fact_check_unavailable:
            warnings.append("The automatic fact check could not run, so only the figures were checked.")
        return {"answer": self.answer, "verified": self.verified, "grounded": self.grounded, "blocked": self.blocked,
                "fact_checked": self.fact_checked, "tools_used": self.tools_used, "sources": self.sources, "model": model,
                "warnings": warnings}


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


def answer(principal, message: str, history: list[dict], *, llm, server, role_label: str, limits: Limits = Limits(),
           today: str | None = None, clock=time.monotonic) -> ChatResult:
    today = today or datetime.now(timezone.utc).date().isoformat()
    bridge = McpBridge(server, principal, "chat", limits.max_tool_chars)
    specs = bridge.specs()
    small_talk = bool(SMALL_TALK.match(message))
    ev = Evidence()
    ev.add_allowed_text(today)
    ev.add_user_text(message)

    messages = [{"role": "system", "content": prompts.system_prompt(role_label, today)}]
    for turn in history[-limits.history_turns:]:
        text = clean_text(turn["content"], limits.max_chars, redact=False) or ""
        messages.append({"role": turn["role"], "content": text})
        if turn["role"] == "user":                       # earlier ASSISTANT text is never evidence
            ev.add_user_text(text)
    messages.append({"role": "user", "content": message})

    res = ChatResult()
    if limits.fast_path and not small_talk:
        routed = fastpath.route(message)
        if routed and routed[0] in bridge.tool_names():
            out = bridge.call(routed[0], routed[1])
            res.tools_used.append(out.summary())
            if out.outcome == "forbidden":
                res.answer, res.verified = fastpath.forbidden_text(), True
                return res
            text = fastpath.render(routed[0], out.payload) if out.payload is not None else None
            if text:
                res.answer, res.verified, res.grounded = text[: limits.max_answer_chars], True, True
                res.sources = list(out.refs)
                return res
            res.tools_used.clear()               # nothing usable: fall through to the model path as if never tried

    deadline = clock() + limits.max_seconds
    tool_choice = "auto" if (small_talk or not specs) else "required"
    rounds = repairs = tool_failures = 0
    refs: list[str] = []
    tool_texts: list[str] = []                           # exactly what the model was shown, for the fact checker

    def stop(reason: str, text: str) -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = text, False, True, reason
        res.sources = refs
        return res

    def out_of_scope() -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = prompts.OUT_OF_SCOPE_TEXT, True, False, "out_of_scope"
        res.grounded, res.sources = False, []
        return res

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
                    if out.payload is not None:
                        ev.add_tool_payload(out.payload)
                        res.grounded = True
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

        problems = find_unsupported(text, ev)             # check 1: figures
        unavailable = False
        if not problems and limits.fact_check and res.tools_used:
            verdict = _fact_check(llm, message, tool_texts, text, limits, res)      # check 2: statements
            if verdict is None:
                unavailable = True
            else:
                res.fact_checked, problems = True, verdict
        if not problems:
            res.answer, res.verified, res.sources = text, not unavailable, refs
            res.fact_check_unavailable = unavailable
            return res
        res.unsupported = problems
        if repairs >= limits.max_repairs:
            return stop("unverified", prompts.BLOCKED_TEXT)
        repairs += 1
        res.retries += 1
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": prompts.REPAIR_MESSAGE.format(values="; ".join(problems))})
        tool_choice = "auto""""One chat turn: the model asks for tools, the MCP core runs them as the signed-in user, the answer is checked.

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
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.chat import fastpath, prompts
from app.chat.bridge import McpBridge
from app.chat.errors import GroqError
from app.chat.verifier import Evidence, find_unsupported
from app.mcp_server.sanitize import clean_text

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
    max_chars: int = 2000
    max_answer_chars: int = 4000
    max_seconds: float = 90.0
    max_tool_chars: int = 16_000
    fact_check: bool = True                # the second, statement-level check (one extra model call per answer)
    max_judge_chars: int = 14_000
    fast_path: bool = True                 # answer common simple questions from a template: 0 model calls, 0 tokens


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

    def to_dict(self, model: str) -> dict:
        warnings = []
        if self.blocked and self.reason == "unverified":
            warnings.append("Some statements could not be verified and the answer was withheld.")
        if self.fact_check_unavailable:
            warnings.append("The automatic fact check could not run, so only the figures were checked.")
        return {"answer": self.answer, "verified": self.verified, "grounded": self.grounded, "blocked": self.blocked,
                "fact_checked": self.fact_checked, "tools_used": self.tools_used, "sources": self.sources, "model": model,
                "warnings": warnings}


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


def answer(principal, message: str, history: list[dict], *, llm, server, role_label: str, limits: Limits = Limits(),
           today: str | None = None, clock=time.monotonic) -> ChatResult:
    today = today or datetime.now(timezone.utc).date().isoformat()
    bridge = McpBridge(server, principal, "chat", limits.max_tool_chars)
    specs = bridge.specs()
    small_talk = bool(SMALL_TALK.match(message))
    ev = Evidence()
    ev.add_allowed_text(today)
    ev.add_user_text(message)

    messages = [{"role": "system", "content": prompts.system_prompt(role_label, today)}]
    for turn in history[-limits.history_turns:]:
        text = clean_text(turn["content"], limits.max_chars, redact=False) or ""
        messages.append({"role": turn["role"], "content": text})
        if turn["role"] == "user":                       # earlier ASSISTANT text is never evidence
            ev.add_user_text(text)
    messages.append({"role": "user", "content": message})

    res = ChatResult()
    if limits.fast_path and not small_talk:
        routed = fastpath.route(message)
        if routed and routed[0] in bridge.tool_names():
            out = bridge.call(routed[0], routed[1])
            res.tools_used.append(out.summary())
            if out.outcome == "forbidden":
                res.answer, res.verified = fastpath.forbidden_text(), True
                return res
            text = fastpath.render(routed[0], out.payload) if out.payload is not None else None
            if text:
                res.answer, res.verified, res.grounded = text[: limits.max_answer_chars], True, True
                res.sources = list(out.refs)
                return res
            res.tools_used.clear()               # nothing usable: fall through to the model path as if never tried

    deadline = clock() + limits.max_seconds
    tool_choice = "auto" if (small_talk or not specs) else "required"
    rounds = repairs = tool_failures = 0
    refs: list[str] = []
    tool_texts: list[str] = []                           # exactly what the model was shown, for the fact checker

    def stop(reason: str, text: str) -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = text, False, True, reason
        res.sources = refs
        return res

    def out_of_scope() -> ChatResult:
        res.answer, res.verified, res.blocked, res.reason = prompts.OUT_OF_SCOPE_TEXT, True, False, "out_of_scope"
        res.grounded, res.sources = False, []
        return res

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
                    if out.payload is not None:
                        ev.add_tool_payload(out.payload)
                        res.grounded = True
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

        problems = find_unsupported(text, ev)             # check 1: figures
        unavailable = False
        if not problems and limits.fact_check and res.tools_used:
            verdict = _fact_check(llm, message, tool_texts, text, limits, res)      # check 2: statements
            if verdict is None:
                unavailable = True
            else:
                res.fact_checked, problems = True, verdict
        if not problems:
            res.answer, res.verified, res.sources = text, not unavailable, refs
            res.fact_check_unavailable = unavailable
            return res
        res.unsupported = problems
        if repairs >= limits.max_repairs:
            return stop("unverified", prompts.BLOCKED_TEXT)
        repairs += 1
        res.retries += 1
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": prompts.REPAIR_MESSAGE.format(values="; ".join(problems))})
        tool_choice = "auto"