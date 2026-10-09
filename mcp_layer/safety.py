"""Safety layer (Phase 10). Plugs into the gateway's SafetyGate hook.

RBAC asks "is this role allowed to do this?" and already rejects identity smuggling (role, is_admin,
token, password, sql, command, code) and named owners (user_id, patient_id, ...). Safety asks a different
question: "should this particular call be executed?". It runs AFTER RBAC allows and BEFORE the tool
runs, and checks, cheapest first (the first problem blocks):

  1. rate limit            per principal, separate buckets for read / write / destructive
  2. argument structure    dict of simple values, bounded size, depth and string length
  3. escalation keys       permissions, acting_as, impersonate, ... (RBAC covers role / is_admin)
  4. sensitive fields      password_hash, api_key, otp, card numbers, cookies ...
  5. id validity           every *_id is a positive int (registry names for server ids)
  6. text scan             SQL / direct-database manipulation, prompt injection, unsafe markup
  7. destructive tools     need an explicit confirm=True

Honest limits: the text scan is a tripwire for obvious attacks, not the security boundary. The
boundary is RBAC + ownership checks + parameterised queries in the repositories. Rate limiting is
per process (in memory); with several workers each has its own counters.

Duck-typed on purpose: no pydantic, fastmcp or database import, so it can be tested alone. The
gateway only reads `.allowed` and `.reason` from what check() returns. Reasons go to the audit log;
the caller always sees the same generic message. Logs hold the reason and names, never values.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Tuple

log = logging.getLogger("mcp.safety")

# ------------------------------------------------------------------ limits
MAX_KEYS = 30
MAX_DEPTH = 3
MAX_STRING_CHARS = 2000
MAX_LIST_ITEMS = 50
MAX_ID = 2_000_000_000
MAX_TRACKED_PRINCIPALS = 10_000
WINDOW_SECONDS = 60.0
WRITE_LIMIT_CAP = 20            # writes per minute, never above the general limit
DESTRUCTIVE_LIMIT_CAP = 5       # destructive calls per minute

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_STRING_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# Registry names, not database ids: they are text by design.
STRING_ID_KEYS = frozenset({"server_id", "mcp_server_id"})

ESCALATION_KEYS = frozenset({
    "is_super_admin", "is_superadmin", "superuser", "admin", "permission", "permissions", "privilege",
    "privileges", "scope", "scopes", "acting_as", "act_as", "impersonate", "as_user", "run_as", "sudo",
    "principal", "user_role", "effective_role",
})
SENSITIVE_KEYS = frozenset({
    "password_hash", "passwd", "pwd", "api_key", "apikey", "secret", "client_secret", "otp", "cvv",
    "card_number", "authorization", "cookie", "session_id", "session_secret",
})

# ------------------------------------------------------------- text rules
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_FILLER = r"(?:all\s+|any\s+|the\s+|your\s+|my\s+|these\s+|those\s+|previous\s+|prior\s+|above\s+|earlier\s+|safety\s+){0,4}"
_RULE_WORDS = r"(?:instructions?|rules?|prompts?|guidelines?|directions?|restrictions?|policies|policy)"

SQL_PATTERNS = tuple(re.compile(p) for p in (
    r"\bunion\s+(?:all\s+)?select\b",
    r"\b(?:drop|truncate|alter)\s+(?:table|database|schema|user|view|index)\b",
    r"\b(?:insert\s+into|delete\s+from|update)\s+[\w`.]+\s+(?:set|values|where)\b",
    r";\s*(?:drop|delete|update|insert|alter|truncate|exec|execute|select)\b",
    r"\binformation_schema\b|\bmysql\.user\b|\bperformance_schema\b",
    r"\b(?:sleep|benchmark|load_file|extractvalue|updatexml)\s*\(",
    r"\binto\s+(?:out|dump)file\b",
    r"\bxp_cmdshell\b|\bexec(?:ute)?\s*\(",
    r"['\"`]\s*\)?\s*or\s*['\"`(]?\s*\w+\s*['\"`)]?\s*=\s*['\"`(]?\s*\w+",      # ' OR '1'='1
    r"\bor\s+\d+\s*=\s*\d+\b",                                                  # OR 1=1
    r"['\"`]\s*;?\s*(?:--|#|/\*)",                                              # quote then comment
))
INJECTION_PATTERNS = tuple(re.compile(p) for p in (
    rf"\bignore\s+{_FILLER}{_RULE_WORDS}",
    rf"\bdisregard\s+{_FILLER}{_RULE_WORDS}",
    rf"\bforget\s+{_FILLER}{_RULE_WORDS}",
    r"\boverride\s+(?:the\s+|all\s+|your\s+)?(?:safety|security|rbac|permissions?|restrictions?|rules?)\b",
    r"\b(?:reveal|show|print|repeat|leak|display|tell)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+|hidden\s+|initial\s+|original\s+)?(?:prompt|instructions)\b",
    r"\byou\s+are\s+now\s+(?:a|an|the|in)\b",
    r"\b(?:developer|dan|jailbreak|god|admin|debug)\s+mode\b",
    r"\bact\s+as\s+(?:an?\s+|the\s+)?(?:admin|administrator|super\s*admin|root|system|developer)\b",
    r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you)\b",
    r"\bbypass\s+(?:the\s+)?(?:rbac|security|safety|permissions?|authentication|guard|filter)s?\b",
    r"\bgrant\s+(?:me|myself)\s+(?:admin|super\s*admin|all\s+permissions)\b",
    r"</?\s*(?:system|assistant|verified_data)\s*>|<\|\s*(?:im_start|im_end|system)\s*\|>",
))
UNSAFE_PATTERNS = tuple(re.compile(p) for p in (
    r"\.\.[/\\]",
    r"<\s*script\b|\bjavascript\s*:|\bfile\s*://|\bdata\s*:\s*text/html",
    r"\x00",
))

_CATEGORIES = (("sql_injection", SQL_PATTERNS), ("prompt_injection", INJECTION_PATTERNS),
               ("unsafe_text", UNSAFE_PATTERNS))


def _normalise(text: str) -> str:
    """Defeat cheap evasion: full-width letters, zero-width characters, case and spacing tricks."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return " ".join(text.lower().split())


def scan_text(text: Any) -> Optional[str]:
    """Category of the first suspicious pattern ('sql_injection', 'prompt_injection',
    'unsafe_text'), or None. Public so the chatbot (Phase 11) can screen a message before it
    reaches any intent router or the LLM."""
    if not isinstance(text, str) or not text:
        return None
    original = text[:MAX_STRING_CHARS]
    if "\x00" in original:
        return "unsafe_text"
    cleaned = _normalise(original)
    for category, patterns in _CATEGORIES:
        for pattern in patterns:
            if pattern.search(cleaned):
                return category
    return None


@dataclass(frozen=True)
class SafetyVerdict:
    """Same two fields the gateway reads from its own SafetyDecision."""
    allowed: bool
    reason: str = ""


ALLOW = SafetyVerdict(True)


def _enum_value(obj: Any) -> str:
    return str(getattr(obj, "value", obj))


class RateLimiter:
    """Sliding window per (principal, bucket). In memory, thread safe, clock injectable for tests."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, window: float = WINDOW_SECONDS,
                 max_tracked: int = MAX_TRACKED_PRINCIPALS):
        self._clock, self._window, self._max = clock, window, max_tracked
        self._hits: Dict[Tuple[str, str], Deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, bucket: str, limit: int) -> bool:
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault((key, bucket), deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            if len(self._hits) > self._max:
                self._evict(now)
            return True

    def _evict(self, now: float) -> None:
        for k in [k for k, h in self._hits.items() if not h or now - h[-1] >= self._window]:
            del self._hits[k]
        while len(self._hits) > self._max:               # still too many: drop the least recently active
            oldest = min(self._hits, key=lambda k: self._hits[k][-1] if self._hits[k] else 0.0)
            del self._hits[oldest]


class SafetyLayer:
    def __init__(self, limit_per_minute: int = 60, limiter: Optional[RateLimiter] = None):
        limit = max(1, int(limit_per_minute))
        self.limits = {"read": limit, "write": min(limit, WRITE_LIMIT_CAP),
                       "destructive": min(limit, DESTRUCTIVE_LIMIT_CAP)}
        self.limiter = limiter or RateLimiter()

    @classmethod
    def from_settings(cls, settings: Any, **kwargs) -> "SafetyLayer":
        return cls(limit_per_minute=settings.rate_limit_per_minute, **kwargs)

    # ------------------------------------------------------------ the hook
    def check(self, principal: Any, server: Any, tool: Any, arguments: Any) -> SafetyVerdict:
        role = _enum_value(getattr(principal, "role", None))
        kind = _enum_value(getattr(tool, "kind", None)).upper()
        reason = (
            self._rate(principal, role, kind)
            or self._structure(arguments)
            or self._escalation(arguments)
            or self._sensitive(arguments)
            or self._ids(arguments)
            or (None if role == "SYSTEM" else self._text(arguments))   # SYSTEM args are internal code, not input
            or (self._destructive(arguments) if kind == "DESTRUCTIVE" else None)
        )
        if reason is None:
            return ALLOW
        log.warning("request blocked by safety layer", extra={"extra_data": {
            "reason": reason, "role": role, "user_id": getattr(principal, "user_id", None),
            "server": getattr(server, "server_id", None), "tool": getattr(tool, "name", None)}})
        return SafetyVerdict(False, reason)

    # -------------------------------------------------------------- checks
    def _rate(self, principal: Any, role: str, kind: str) -> Optional[str]:
        if role == "SYSTEM":                              # internal usage recording must never be throttled
            return None
        bucket = kind.lower() if kind.lower() in self.limits else "write"   # unknown kind: stricter bucket
        key = f"{role}:{getattr(principal, 'user_id', None)}"
        return None if self.limiter.allow(key, bucket, self.limits[bucket]) else "rate_limited"

    @staticmethod
    def _structure(arguments: Any) -> Optional[str]:
        if not isinstance(arguments, Mapping) or len(arguments) > MAX_KEYS:
            return "unsafe_arguments"
        for key, value in arguments.items():
            if not isinstance(key, str) or not _KEY.match(key) or key.startswith("__"):
                return "unsafe_arguments"
            if not SafetyLayer._value_ok(value, 1):
                return "unsafe_arguments"
        return None

    @staticmethod
    def _value_ok(value: Any, depth: int) -> bool:
        if value is None or isinstance(value, (bool, int)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, str):
            return len(value) <= MAX_STRING_CHARS
        if depth >= MAX_DEPTH:
            return False
        if isinstance(value, (list, tuple)):
            return len(value) <= MAX_LIST_ITEMS and all(SafetyLayer._value_ok(v, depth + 1) for v in value)
        if isinstance(value, Mapping):
            return (len(value) <= MAX_KEYS
                    and all(isinstance(k, str) and _KEY.match(k) and SafetyLayer._value_ok(v, depth + 1)
                            for k, v in value.items()))
        return False

    @staticmethod
    def _escalation(arguments: Mapping) -> Optional[str]:
        return "role_escalation_attempt" if any(k.strip().lower() in ESCALATION_KEYS for k in arguments) else None

    @staticmethod
    def _sensitive(arguments: Mapping) -> Optional[str]:
        return "sensitive_field" if any(k.strip().lower() in SENSITIVE_KEYS for k in arguments) else None

    @staticmethod
    def _ids(arguments: Mapping) -> Optional[str]:
        for key, value in arguments.items():
            name = key.lower()
            if value is None or not (name == "id" or name.endswith("_id")):
                continue
            if name in STRING_ID_KEYS:
                if not isinstance(value, str) or not _STRING_ID.match(value):
                    return "invalid_id"
            elif isinstance(value, bool) or not isinstance(value, int) or not (0 < value <= MAX_ID):
                return "invalid_id"
        return None

    @staticmethod
    def _text(arguments: Mapping) -> Optional[str]:
        def walk(value: Any) -> Optional[str]:
            if isinstance(value, str):
                return scan_text(value)
            if isinstance(value, (list, tuple)):
                return next((hit for hit in map(walk, value) if hit), None)
            if isinstance(value, Mapping):
                return next((hit for hit in map(walk, value.values()) if hit), None)
            return None
        return next((hit for hit in map(walk, arguments.values()) if hit), None)

    @staticmethod
    def _destructive(arguments: Mapping) -> Optional[str]:
        return None if arguments.get("confirm") is True else "destructive_unconfirmed"
