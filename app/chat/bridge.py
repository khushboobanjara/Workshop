"""Runs the model's tool calls through the MCP core as the SIGNED-IN USER.

The model never gets its own identity or permissions: every call goes through McpServer.handle_message with the user's
Principal, so the same role check, strict argument validation, rate limit, redaction and audit row apply as for a direct
MCP call (the audit row says transport='chat'). The model only SEES the tools its user may call, but a call to anything
else still reaches the server and is refused and logged there: hiding tools is a convenience, not the control.
"""
import json
import re
from dataclasses import dataclass, field

from app.mcp_server.sanitize import sanitize_json

MAX_TOOL_CHARS = 16_000
MAX_ARGUMENT_CHARS = 4_000


@dataclass
class ToolOutcome:
    tool: str
    arguments: dict
    outcome: str                                  # ok | not_found | forbidden | invalid_arguments | unknown_tool | rate_limited | ...
    content: str                                  # exactly what the model is shown (a JSON string)
    payload: dict | None = None                   # parsed envelope; only set for real database results
    refs: list = field(default_factory=list)

    @property
    def records(self) -> int:
        return len(self.refs)

    def summary(self) -> dict:
        return {"tool": self.tool, "arguments": sanitize_json(self.arguments, max_depth=3, max_items=20, max_str=120),
                "outcome": self.outcome, "records": self.records}


def _error_content(code: str, message: str) -> str:
    return json.dumps({"ok": False, "error": {"code": code, "message": message}}, ensure_ascii=False)


class McpBridge:
    def __init__(self, server, principal, transport: str = "chat", max_chars: int = MAX_TOOL_CHARS):
        self.server, self.principal, self.transport, self.max_chars = server, principal, transport, max_chars
        self._descriptors = [t.descriptor() for t in server.allowed_tools(principal)]
        self._props = {d["name"]: d["inputSchema"].get("properties", {}) for d in self._descriptors}
        self._seq = 0

    def tool_names(self) -> list[str]:
        return [d["name"] for d in self._descriptors]

    def specs(self) -> list[dict]:
        return [{"type": "function", "function": {"name": d["name"], "description": d["description"], "parameters": d["inputSchema"]}}
                for d in self._descriptors]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _parse(raw) -> tuple[dict | None, str | None]:
        if isinstance(raw, dict):
            return raw, None
        text = (raw or "").strip() if isinstance(raw, str) else ""
        if text in ("", "null"):
            return {}, None
        if len(text) > MAX_ARGUMENT_CHARS:
            return None, "The arguments are too long."
        try:
            value = json.loads(text)
        except ValueError:
            return None, "The arguments were not valid JSON."
        return (value, None) if isinstance(value, dict) else (None, "The arguments must be a JSON object.")

    def _coerce(self, name: str, args: dict) -> dict:
        """Models often send "12" for an integer or null for 'not set'. Fix only those two shapes; the MCP layer still
        validates everything strictly, so this never lets an invalid value through."""
        props, out = self._props.get(name, {}), {}
        for k, v in args.items():
            if v is None:
                continue
            if props.get(k, {}).get("type") == "integer" and isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
                v = int(v.strip())
            out[k] = v
        return out

    def _failed(self, name: str, args: dict, code: str, message: str) -> ToolOutcome:
        return ToolOutcome(name, args, code, _error_content(code, message))

    # ------------------------------------------------------------------ the call
    def call(self, name: str, raw_arguments) -> ToolOutcome:
        args, problem = self._parse(raw_arguments)
        if problem:
            return self._failed(name, {}, "invalid_arguments", problem)
        args = self._coerce(name, args)
        self._seq += 1
        reply = self.server.handle_message({"jsonrpc": "2.0", "id": self._seq, "method": "tools/call",
                                            "params": {"name": name, "arguments": args}}, self.principal, self.transport)
        if reply is None or "error" in reply:
            return self._failed(name, args, "unknown_tool", "That tool does not exist.")
        result = reply["result"]
        text = result["content"][0]["text"]
        try:
            payload = json.loads(text)
        except ValueError:
            return self._failed(name, args, "error", "The tool returned an unreadable result.")
        if result.get("isError"):
            code = ((payload or {}).get("error") or {}).get("code", "error")
            return ToolOutcome(name, args, code, text)                       # forbidden / invalid_arguments / rate_limited ...
        if len(text) > self.max_chars:
            return self._failed(name, args, "result_too_large",
                                f"The result is too large ({len(text)} characters). Ask for fewer rows (a smaller limit) or add filters.")
        return ToolOutcome(name, args, "ok" if payload.get("found", True) else "not_found", text, payload,
                           list(payload.get("record_refs", [])))
