"""Minimal client for Groq's OpenAI-compatible chat completions API (https://api.groq.com/openai/v1).

Local tool calling: we send `tools`, Groq answers with `tool_calls`, our server runs them (bridge.py) and sends the
results back as role=tool messages. The API key lives only in this object and in the Authorization header of the outgoing
request: it is never logged, never returned to the browser and never part of an error message.
"""
import math
import re
import time

import httpx
from app.chat.errors import GroqError


def _int(value) -> int | None:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return None


_WAIT = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|[smh])(?![a-z])")
_UNIT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def retry_wait_seconds(r: httpx.Response) -> float | None:
    """How long Groq asks us to wait: the Retry-After header, else the 'Please try again in 3.21s / 1m5.2s / 850ms' in its message."""
    try:
        return max(0.0, float(r.headers.get("retry-after")))
    except (TypeError, ValueError):
        pass
    try:
        message = str((r.json().get("error") or {}).get("message") or "")
    except (ValueError, AttributeError):
        return None
    m = re.search(r"try again in ([0-9a-z. ]+)", message, re.IGNORECASE)
    parts = _WAIT.findall(m.group(1).lower()) if m else []
    return sum(float(n) * _UNIT[u] for n, u in parts) if parts else None


class GroqClient:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.groq.com/openai/v1", timeout: float = 30.0,
                 temperature: float = 0.1, max_tokens: int = 2048, transport: httpx.BaseTransport | None = None,
                 rate_limit_retries: int = 0, max_wait: float = 12.0, sleep=time.sleep):
        self.rate_limit_retries, self.max_wait, self._sleep = rate_limit_retries, max_wait, sleep
        self._key = api_key
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.temperature, self.max_tokens, self._transport = temperature, max_tokens, transport

    def __repr__(self) -> str:                                   # never leak the key through logging or tracebacks
        return f"GroqClient(model={self.model!r})"

    def _scrub(self, text: str) -> str:
        text = text.replace(self._key, "***") if self._key else text
        return text[:300]

    def _detail(self, r: httpx.Response) -> str:
        message = None
        try:
            err = r.json().get("error")
            if isinstance(err, dict):
                message = err.get("message") or err.get("code")
        except (ValueError, AttributeError):
            pass
        return self._scrub(f"HTTP {r.status_code}" + (f": {message}" if message else ""))

    def complete(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> dict:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens}
        if tools:
            body["tools"], body["tool_choice"] = tools, tool_choice
        attempt = 0
        while True:
            try:
                with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                    r = client.post(f"{self.base_url}/chat/completions", json=body,
                                    headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
            except httpx.TimeoutException as exc:
                raise GroqError("timeout", 504, detail=self._scrub(f"{type(exc).__name__}: {exc}")) from None
            except httpx.HTTPError as exc:
                raise GroqError("unreachable", 502, detail=self._scrub(f"{type(exc).__name__}: {exc}")) from None
            if r.status_code != 429:
                break
            wait = retry_wait_seconds(r)
            if attempt < self.rate_limit_retries and wait is not None and wait <= self.max_wait:
                attempt += 1                                     # a per-minute limit that frees up shortly: wait for it, then ask again
                self._sleep(wait + 0.25)
                continue
            raise GroqError("rate_limited", 429, retry_after=math.ceil(wait) if wait is not None else _int(r.headers.get("retry-after")),
                            detail=self._detail(r))
        if r.status_code in (401, 403):
            raise GroqError("auth", 502, detail=self._detail(r))
        if r.status_code >= 500:
            raise GroqError("upstream", 502, detail=self._detail(r))
        if r.status_code >= 400:
            try:
                code = (r.json().get("error") or {}).get("code")
            except ValueError:
                code = None
            raise GroqError("tool_use_failed" if code == "tool_use_failed" else "bad_request", 502, detail=self._detail(r))
        try:
            data = r.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise GroqError("bad_response", 502, detail=self._scrub(f"HTTP {r.status_code}: unexpected response shape")) from None
        return {"content": message.get("content"), "tool_calls": message.get("tool_calls") or [],
                "usage": data.get("usage") or {}, "finish_reason": choice.get("finish_reason")}
