"""Hard-coded answers: the questions that do not need a language model. Pure Python (no I/O, no model).

The assistant spends AI tokens only on what rules cannot do. In order, a question is:

  1. canned      a greeting, thanks or "what can you do" -> a fixed text
  2. routed      a recognisable data question ("how many invoices are pending?", "invoices from Acme", "show vendors") -> the
                 right MCP tool is called by the server and the answer is a TEMPLATE filled with the tool's own values
  3. document    a question about policies/contracts/procedures -> server-side search, then verbatim quotes (app/rag/direct.py),
                 or ONE model call with the passages already in the prompt when reasoning is needed (a what-if, "explain")
  4. everything else goes to the model with its tools (app/chat/service.py), exactly as before

Routing is deliberately conservative: a message is routed only when EVERY word of it is understood (known filler, a known keyword,
a number or a name in a known slot). Anything with an extra clause, a comparison, a "why" or a "should" falls through to the model,
so a wrongly routed question is rarer than an unrouted one. Switch the whole layer off with OPSPILOT_CHAT_HARDCODE=false.
"""
import re
from dataclasses import dataclass

from app.chat import fastpath
from app.chat.fastpath import _invoice_line, _money, _NOTHING_TEXT  # noqa: F401  (shared formatting)

MAX_CHARS = 200


@dataclass(frozen=True)
class Route:
    tool: str
    args: dict
    view: str = ""                       # which template: "" = the tool's standard one, else e.g. "count:pending"


def _norm(message: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[?!.,;:]+", " ", (message or "").lower())).strip()


FILLER = frozenset("""a an the me my our us we i you please can could would will do does did is are was were be to of for on in at by from with and or
that which who what whats what's that's it's i'd i'll let's show list give get find display see view open tell about all any every some there here it its this these those
currently now right today just also again let lets want need like know have has had""".split())


def _only(tokens: list[str], allowed: set | frozenset) -> bool:
    return all(t in FILLER or t in allowed or t.isdigit() for t in tokens)


# ------------------------------------------------------------------ 1. canned replies
_GREET = re.compile(r"^(hi|hello|hey|hiya|yo|good (morning|afternoon|evening)|hi there|hello there|hey there)( opspilot| assistant)?$")
_THANKS = re.compile(r"^(thanks|thank you|thx|ty|cheers|ok|okay|great|cool|nice|got it|perfect|awesome)( a lot| so much| very much)?( thanks)?$")
_HELP = re.compile(r"^(help|help me|what can you do|what can you help( me)? with|who are you|what are you|how do you work|how can you help( me)?|what do you do)$")


# "Can you approve invoices?", "Do you have access to delete…", "please approve invoice 12": the assistant is read-only, so the answer
# is fixed. The subject must be the assistant itself (you / the assistant / the bot): "does the AI approve invoices automatically"
# is a question about the pipeline, which the documents answer.
_ACTION = r"(approve|reject|edit|delete|remove|modify|override|pay)"
_ASK = re.compile(rf"\b(can|could|will|would|may|do|does|are|is|able|allowed|permission|access|authority|ability)\b[^.]{{0,40}}\b(you|assistant|chatbot|bot)\b[^.]{{0,40}}\b{_ACTION}\b"
                  rf"|^(please )?(just )?(approve|reject|delete|remove|edit|override|modify)\b")
_NOT_ASK = re.compile(r"\b(list|show|which|what|how|why|who|when|queue|waiting|pending|need|needs|many|automatically)\b")
READ_ONLY_TEXT = ("No. I'm a read-only assistant: I can look things up, but I can't approve, reject, edit, delete or pay anything. "
                  "Invoices are approved or rejected by reviewers and admins in the Review Queue (or automatically by the pipeline's rules). "
                  "I can show you what is waiting: try \"show the review queue\".")


def canned(message: str, tools: list[str], docs_on: bool) -> str | None:
    m = _norm(message)
    if _ASK.search(m) and len(m) <= 160 and not _NOT_ASK.search(m):
        return READ_ONLY_TEXT
    if _GREET.match(m):
        return "Hello! Ask me about your invoices, vendors, the review queue or metrics" + (", or what your company documents say" if docs_on else "") + ". Type \"help\" to see examples."
    if _THANKS.match(m):
        return "You're welcome! Ask me anything else about your OpsPilot data" + (" or documents." if docs_on else ".")
    if _HELP.match(m):
        lines = ["I'm the read-only OpsPilot assistant. I only see what your role can see, and every lookup is audited. Try:",
                 "- \"how many invoices are pending review?\", \"show the review queue\", \"invoice 12\"",
                 "- \"invoices from Acme\", \"rejected invoices\", \"top vendors\", \"metrics\", \"which models are active?\", \"what can I access?\""]
        if "get_system_status" in tools:
            lines.append("- \"system status\"")
        if "get_audit_log" in tools:
            lines.append("- \"show the audit log\", \"list users\"")
        if docs_on:
            lines.append("- questions about your policies, contracts and procedures, answered with citations, e.g. \"what are our payment terms?\"")
        lines.append("Simple questions are answered instantly from fixed rules; harder ones use the AI model.")
        return "\n".join(lines)
    return None


# ------------------------------------------------------------------ 2. routed data questions
_STATUS = (("auto_approved", None, r"auto[- ]?approved|automatically approved"), (None, "rejected", r"rejected|declined"),
           (None, "approved", r"approved"), (None, "pending_review", r"pending( review)?|awaiting( review)?|in review|unreviewed"),
           (None, "new", r"new|unprocessed"))
_STATUS_WORDS = {"auto", "approved", "automatically", "rejected", "declined", "pending", "review", "awaiting", "in", "unreviewed", "new", "unprocessed", "invoice", "invoices", "bills"}
_BAD_VENDOR = re.compile(r"\b(last|this|today|yesterday|week|month|year|since|before|after|over|above|below|more|less|than|between|and|or|with|that|which|where|when)\b|\d")
_TOP = re.compile(r"\btop (\d{1,2})\b")


def _status_of(m: str) -> tuple[str | None, str | None]:
    for decision, status, pattern in _STATUS:
        if re.search(rf"\b({pattern})\b", m):
            return decision, status
    return None, None


def _limit(m: str, default: int = 10) -> int:
    t = _TOP.search(m)
    return max(1, min(int(t.group(1)), 25)) if t else default


def route(message: str) -> Route | None:
    if not message or len(message) > MAX_CHARS:
        return None
    m = _norm(message)
    tokens = m.split()
    if not tokens or fastpath._NEEDS_REASONING.search(message) or re.search(r"\b(why|how come|compare|versus|vs|trend|best|worst|average|should|explain|recommend)\b", m):
        return None
    ids = re.findall(r"\d+", m)

    # --- one invoice (by database id) and its audit trail
    mi = re.search(r"\binvoice\s*(?:id\s*)?#?\s*(\d{1,9})\b", m) or re.search(r"(?<!\w)#(\d{1,9})\b", m)
    if mi and len(ids) == 1:
        trail = re.search(r"\b(audit|history|trail|timeline|log)\b", m)
        rest = [t for t in tokens if t not in {"invoice", "id", "audit", "history", "trail", "timeline", "log", "details", "detail", "status", "info", "information", "for", "of"} and not t.isdigit() and not t.startswith("#")]
        if _only(rest, set()):
            args = {"invoice_id": int(mi.group(1))}
            return Route("get_invoice_audit_trail" if trail else "get_invoice", args)
        return None
    if ids and not _TOP.search(m):
        return None                          # numbers we do not understand (an amount, a date): leave it to the model

    # --- counts: "how many invoices are pending review?"
    if "how many" in m or m.startswith("number of") or m.startswith("count "):
        words = set(tokens)
        decision, status = _status_of(m)
        allowed = _STATUS_WORDS | {"how", "many", "number", "count", "total", "processed", "received", "invoices", "emails", "email", "vendors", "suppliers", "users", "models", "accounts", "in", "system", "we", "have", "were", "are"}
        if not _only(tokens, allowed):
            return None
        if words & {"vendors", "suppliers"}:
            return Route("list_vendors", {"limit": 1}, "count:vendors")
        if words & {"users", "accounts"}:
            return Route("list_users", {"limit": 200}, "count:users")
        if words & {"models"}:
            return Route("list_models", {}, "count:models")
        if words & {"emails", "email"}:
            return Route("get_metrics", {}, "count:emails")
        if words & {"invoices", "bills"} or status or decision:
            key = decision or status
            return Route("get_metrics", {}, f"count:{key}" if key else "count:invoices")
        return None

    # --- the review queue
    if re.search(r"\b(review queue|queue|pending review|waiting for review|awaiting review|needs? (human )?review|to review|human review|for review)\b", m) \
            and _only(tokens, {"review", "queue", "pending", "waiting", "awaiting", "needs", "need", "human", "invoices", "invoice", "items", "what", "is", "in", "top", "next", "oldest"}):
        return Route("get_review_queue", {"limit": _limit(m)})

    # --- invoices filtered by status / decision / vendor
    if re.search(r"\b(invoices?|bills)\b", m):
        left, vendor = m, None
        mv = re.search(r"\b(?:from|by|for|of)\s+(.+)$", m)
        if mv:
            left, vendor = m[:mv.start()].strip(), mv.group(1).strip()
        decision, status = _status_of(left)
        if vendor is not None and (not vendor or _BAD_VENDOR.search(vendor) or len(vendor) > 60 or not re.fullmatch(r"[a-z0-9&.' \-]+", vendor)):
            return None
        vendor = re.sub(r"^(the|vendor|supplier)\s+", "", vendor).strip() if vendor else None
        if (decision or status or vendor) and _only(left.split(), _STATUS_WORDS | {"latest", "recent", "newest", "top"}):
            if status == "pending_review" and not vendor:
                return Route("get_review_queue", {"limit": _limit(m)})
            args = {"limit": _limit(m)}
            if decision:
                args["decision"] = decision
            if status:
                args["status"] = status
            if vendor:
                args["vendor"] = vendor
            return Route("search_invoices", args)

    # --- vendors
    if re.search(r"\b(vendors?|suppliers?)\b", m) and _only(tokens, {"vendors", "vendor", "suppliers", "supplier", "top", "biggest", "largest", "highest", "spend", "spending", "most", "by", "summary", "overview", "list", "invoices", "our", "main", "major", "key"}):
        return Route("list_vendors", {"limit": _limit(m)})

    # --- metrics, models, status, access, audit, users
    if re.search(r"\b(metrics?|stats|statistics|kpis?|automation rate|performance summary|dashboard summary|savings|time saved|money saved|overview)\b", m) \
            and _only(tokens, {"metrics", "metric", "stats", "statistics", "kpi", "kpis", "automation", "rate", "performance", "summary", "dashboard", "savings", "time", "money", "saved", "overview", "key", "the", "numbers"}):
        return Route("get_metrics", {})
    if re.search(r"\bmodels?\b", m) and _only(tokens, {"models", "model", "active", "versions", "version", "ml", "classifier", "extractor", "detector", "which", "are", "loaded", "registry", "installed"}):
        return Route("list_models", {})
    if re.search(r"\b(system|health|healthy|database|db)\b", m) and _only(tokens, {"system", "status", "health", "healthy", "database", "db", "ok", "okay", "up", "running", "working", "overall", "of", "queue", "sizes"}):
        return Route("get_system_status", {})
    if re.search(r"\b(my (role|access|permissions?)|who am i|what can i (do|see|access)|which tools)\b", m) and _only(tokens, {"my", "role", "access", "permissions", "permission", "who", "am", "what", "can", "i", "do", "see", "which", "tools", "tool", "available", "use"}):
        return Route("get_my_access", {})
    if re.search(r"\b(audit|activity)\b", m) and _only(tokens, {"audit", "log", "logs", "trail", "recent", "latest", "activity", "last", "entries", "events", "top"}):
        return Route("get_audit_log", {"limit": _limit(m)})
    if re.search(r"\b(users?|accounts?)\b", m) and _only(tokens, {"users", "user", "accounts", "account", "all", "who", "has", "access", "registered", "active", "system", "the"}):
        return Route("list_users", {"limit": 50})
    return None


# ------------------------------------------------------------------ templates for the routed answers
def _count_sentence(view: str, data: dict) -> str | None:
    kind = view.split(":", 1)[1]
    if kind == "vendors":
        return f"There are {data.get('total_vendors', 0):,} vendor(s) in OpsPilot." if "total_vendors" in data else None
    if kind == "users":
        n = len(data.get("users") or [])
        return f"There are {n:,} user account(s)."
    if kind == "models":
        return f"There are {len(data.get('models') or []):,} model version(s) registered." if "models" in data else None
    if kind == "emails":
        return f"OpsPilot has received {data['emails_received']:,} e-mail(s)." if "emails_received" in data else None
    if kind == "invoices":
        return f"OpsPilot holds {data['invoices']:,} invoice(s) in total." if "invoices" in data else None
    if kind in ("pending_review",):
        return f"{data['pending_review']:,} invoice(s) are pending review." if "pending_review" in data else None
    names = {"approved": "approved", "rejected": "rejected", "new": "new", "auto_approved": "auto-approved"}
    if kind in names:
        pool = data.get("by_decision" if kind == "auto_approved" else "by_status") or {}
        if kind in pool:
            return f"{pool[kind]:,} invoice(s) are {names[kind]}."
        return f"No invoices are {names[kind]}." if pool else None
    return None


def render(tool: str, payload: dict, view: str = "") -> str | None:
    """The answer for a routed question, built only from the tool's own result. None = not renderable: ask the model instead."""
    data = payload.get("data") or {}
    if view.startswith("count:"):
        return _count_sentence(view, data)
    if tool == "search_invoices":
        rows = data.get("invoices") or []
        if not rows:
            return _NOTHING_TEXT
        filt = ", ".join(f"{k} {v}" for k, v in (data.get("filters") or {}).items() if v and v != "any") or "all"
        head = f"{data.get('total_matching', len(rows)):,} invoice(s) match ({filt}); showing the newest {len(rows)}:"
        return "\n".join([head, *[_invoice_line(i) for i in rows]])
    if tool == "get_audit_log":
        rows = data.get("entries") or data.get("audit") or data.get("rows") or []
        if not rows:
            return _NOTHING_TEXT
        return "\n".join([f"Latest {len(rows)} audit entr{'y' if len(rows) == 1 else 'ies'} (newest first):",
                          *[f"- #{r.get('id')} {str(r.get('timestamp') or r.get('created_at') or '')[:19]} · {r.get('stage')} · {r.get('action')} · {r.get('actor')}" for r in rows]])
    if tool == "get_invoice_audit_trail":
        rows = data.get("entries") or data.get("trail") or data.get("audit") or []
        if not rows:
            return _NOTHING_TEXT
        return "\n".join([f"Audit trail ({len(rows)} entr{'y' if len(rows) == 1 else 'ies'}):",
                          *[f"- {str(r.get('timestamp') or r.get('created_at') or '')[:19]} · {r.get('stage')} · {r.get('action')} · {r.get('actor')}" for r in rows]])
    if tool == "list_users":
        rows = data.get("users") or []
        if not rows:
            return _NOTHING_TEXT
        return "\n".join([f"{len(rows)} user account(s):", *[f"- {u.get('username')} ({u.get('role')}, {'active' if u.get('active', True) else 'disabled'})" for u in rows]])
    return fastpath.render(tool, payload)


# ------------------------------------------------------------------ 3. is it a document question?
_IDENT = re.compile(r"\b[a-z]{2,6}-\d+\b|(?<!\w)#\d+|\binvoice\s*(?:id\s*|number\s*|no\.?\s*)?#?\d+", re.IGNORECASE)
_DOC_TOPIC = re.compile(r"\b(policy|policies|procedure|procedures|sop|contract|agreement|document|documents|handbook|guideline|guidelines|rule|rules|"
                        r"terms|according to|per our|our policy|what does|what do|allowed|permitted|required|requirement|requirements|limit|limits|"
                        r"threshold|approval|retention|retain|onboarding|escalat\w+)\b", re.IGNORECASE)
_LIVE = re.compile(r"\b(how many|count|list|latest|recent|newest|oldest|currently|right now|today|yesterday|this (week|month|year)|last (week|month|year)|"
                   r"pending|in the queue|waiting|my invoices|status of|top \d+|total (spend|amount|value))\b", re.IGNORECASE)
_DATA = re.compile(r"\b(invoices?|bills?|billed|vendors?|suppliers?|payments?|paid|pay|due|amounts?|spend|spent|cost|price|currency|queue|review|reviewers?|"
                   r"approve[sd]?|approval|reject(ed|ion)?|pending|anomal\w+|duplicates?|risk\w*|metrics?|stats|statistics|automation|savings?|models?|classifier|"
                   r"extractor|detector|audit|logs?|users?|accounts?|roles?|permissions?|access|status|health|system|e-?mails?|thresholds?|settings?|decisions?|"
                   r"problems?|issues?|errors?|fail\w*|alerts?|flag\w*|suspicious|fraud|late|overdue|unpaid|total|report|dashboard|data|records?|database|"
                   r"opspilot|assistant|vendor|inbox|mailbox|pipeline|policy|policies|contracts?)\b", re.IGNORECASE)
_NEEDS_LLM = re.compile(r"\b(why|how come|compare|comparison|versus|vs|difference|differences|should|recommend\w*|explain|summari[sz]e|summary|overview|"
                        r"what if|if|would|could i|tell me about|describe|pros|cons|best|worst|list all|everything)\b|\d", re.IGNORECASE)


def is_doc_question(message: str) -> bool:
    """Plausibly answered from documents alone (no live data needed)? A record identifier means live data: the model decides."""
    if _IDENT.search(message or ""):
        return False
    if _DOC_TOPIC.search(message or ""):
        return True
    return not _LIVE.search(message or "")


def looks_like_data(message: str) -> bool:
    return bool(_DATA.search(message or ""))


def needs_llm(message: str) -> bool:
    """Reasoning, a what-if or an explanation is wanted: quoting will not do."""
    return bool(_NEEDS_LLM.search(message or ""))


def retrieval_query(message: str, history: list[dict]) -> str:
    """A bare follow-up ("explain that") has no words to search on: borrow the previous question."""
    from app.rag.direct import terms
    if len(terms(message)) >= 2:
        return message
    for turn in reversed(history or []):
        if turn.get("role") == "user":
            return f"{turn.get('content', '')} {message}".strip()
    return message
