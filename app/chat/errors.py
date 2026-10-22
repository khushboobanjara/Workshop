"""Errors from the language-model provider. `code` is stable and safe to show; nothing here carries keys or request data."""


class GroqError(Exception):
    def __init__(self, code: str, http_status: int = 502, retry_after: int | None = None, detail: str | None = None):
        super().__init__(code)
        # `detail` is the provider's own error text (key scrubbed, shortened). It goes to the server log, the audit row and
        # to ADMINS in the HTTP error, never to other users.
        self.code, self.http_status, self.retry_after, self.detail = code, http_status, retry_after, detail


PUBLIC_MESSAGES = {
    "rate_limited": "The AI provider is rate-limiting requests. Try again in a moment.",
    "timeout": "The AI provider took too long to answer. Try again.",
    "auth": "The AI provider rejected this server's credentials. Ask an administrator to check OPSPILOT_GROQ_API_KEY.",
    "tool_use_failed": "The AI model could not complete that request. Try rephrasing it.",
}
DEFAULT_PUBLIC_MESSAGE = "The AI provider is unavailable right now. Try again shortly."


def public_message(code: str, retry_after: int | None = None) -> str:
    if code == "rate_limited" and retry_after:
        return f"The AI provider is rate-limiting requests (its per-minute limit was reached). Try again in about {retry_after} seconds."
    return PUBLIC_MESSAGES.get(code, DEFAULT_PUBLIC_MESSAGE)
