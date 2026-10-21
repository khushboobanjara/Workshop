"""Access control and abuse limits, kept in one place.

- API key (optional): if OPSPILOT_API_KEY is set, every API call must send header X-API-Key.
  Leave it unset for local development.
- Write rate limit (demo mode only): per client IP, in memory. Behind a reverse proxy, run uvicorn
  with --proxy-headers so the real client IP is used; otherwise all visitors share one bucket,
  which is stricter, never looser.
- Request size cap (middleware): bodies over OPSPILOT_MAX_REQUEST_BYTES get 413. This relies on the
  Content-Length header, so keep a proxy size limit in front of anything public."""
import secrets
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings

OPEN_PATHS = {"/", "/health", "/auth/login"}      # the dashboard page itself, and health checks, never need a key


class WriteRateLimiter:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str, per_minute: int) -> bool:
        now = self.clock()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= per_minute:
            return False
        q.append(now)
        if len(self.hits) > 10_000:                      # bound memory under a flood of distinct IPs
            for k in [k for k, v in self.hits.items() if not v or now - v[-1] > 60]:
                del self.hits[k]
        return True


def make_guard(limiter: WriteRateLimiter):
    def guard(request: Request) -> None:
        if request.url.path in OPEN_PATHS:
            return
        expected = settings.api_key.get_secret_value()
        if expected and not settings.auth_enabled:      # with auth on, the key only authorises ingestion (see auth.py)
            supplied = request.headers.get("x-api-key", "")
            if not secrets.compare_digest(supplied.encode(), expected.encode()):
                raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})
        if settings.demo_mode and request.method == "POST":
            ip = request.client.host if request.client else "unknown"
            if not limiter.allow(ip, settings.demo_writes_per_minute):
                raise HTTPException(429, "Demo rate limit reached, please try again in a minute")
    return guard


UPLOAD_PATH = "/rag/documents"
UPLOAD_OVERHEAD_BYTES = 1_000_000        # multipart framing and form fields on top of the file itself


def body_limit(request: Request) -> int:
    """Every route keeps the small global limit except the document upload, which may carry a file (its own, larger limit)."""
    if request.method == "POST" and request.url.path == UPLOAD_PATH:
        return settings.rag_max_upload_bytes + UPLOAD_OVERHEAD_BYTES
    return settings.max_request_bytes


async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > body_limit(request):
        return JSONResponse({"detail": "Request body too large"}, status_code=413)
    return await call_next(request)
