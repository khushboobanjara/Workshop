"""Per-request context shared by the web layer and the pipeline (set by middleware, read when audit rows are written)."""
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
