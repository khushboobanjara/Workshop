"""Streamable HTTP transport: POST /mcp (JSON in, JSON out). Off unless OPSPILOT_MCP_ENABLED=true.

Checks before a message is even parsed: feature switch -> Origin allow-list -> protocol header -> authentication
(401 + audit on failure, throttled per IP) -> message size. A browser sends an Origin header; scripts and desktop MCP
clients usually do not. An Origin that is not in OPSPILOT_MCP_ALLOWED_ORIGINS is refused (DNS-rebinding defence).
Serve it over HTTPS in anything shared: the bearer token is a credential.
"""
import json

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import settings
from app.mcp_server import audit
from app.mcp_server.auth import MIN_KEY_LENGTH, bearer_token, principal_from_token
from app.mcp_server.protocol import INVALID_REQUEST, PARSE_ERROR, SUPPORTED_VERSIONS, rpc_error
from app.mcp_server.server import server
from app.mcp_server.status import build_status, render_html
from app.security import WriteRateLimiter

router = APIRouter()
auth_failures = WriteRateLimiter()
MAX_AUTH_FAILURES_PER_MINUTE = 20


def startup_checks() -> None:
    """Fail closed: with MCP on, refuse to start rather than run it unauthenticated or in a public demo."""
    if not settings.mcp_enabled:
        return
    problems = []
    if not settings.auth_enabled:
        problems.append("OPSPILOT_AUTH_ENABLED must be true (MCP never runs unauthenticated)")
    if settings.demo_mode:
        problems.append("MCP cannot be enabled together with OPSPILOT_DEMO_MODE")
    key = settings.mcp_system_key.get_secret_value()
    if key and len(key) < MIN_KEY_LENGTH:
        problems.append(f"OPSPILOT_MCP_SYSTEM_KEY must be at least {MIN_KEY_LENGTH} characters")
    if key and key == settings.api_key.get_secret_value():
        problems.append("OPSPILOT_MCP_SYSTEM_KEY must differ from OPSPILOT_API_KEY")
    if problems:
        raise RuntimeError("Refusing to start MCP: " + "; ".join(problems))


def _origin_allowed(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True
    allowed = {o.strip().rstrip("/") for o in settings.mcp_allowed_origins.split(",") if o.strip()}
    return origin.rstrip("/") in allowed


def _authenticate(request: Request):
    ip = request.client.host if request.client else "unknown"
    token = bearer_token(request.headers.get("authorization", ""))
    principal = principal_from_token(token)
    if principal is not None:
        return principal
    if not auth_failures.allow(ip, MAX_AUTH_FAILURES_PER_MINUTE):
        raise HTTPException(429, "Too many failed attempts, wait a minute", headers={"Retry-After": "60"})
    try:
        audit.record_auth_failure(reason="invalid_token" if token else "missing_token", ip=ip, transport="http")
    except Exception:
        pass                                                   # the refusal does not depend on the log
    raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": 'Bearer realm="opspilot-mcp"'})


async def _read_limited(request: Request) -> bytes:
    limit = settings.mcp_max_message_bytes
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, "MCP message too large")
    buf = bytearray()
    async for chunk in request.stream():                       # also bounds chunked uploads that send no Content-Length
        buf += chunk
        if len(buf) > limit:
            raise HTTPException(413, "MCP message too large")
    return bytes(buf)


@router.post("/mcp", include_in_schema=False)
async def mcp_post(request: Request):
    if not settings.mcp_enabled:
        raise HTTPException(404, "Not Found")
    if not _origin_allowed(request):
        return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
    version = request.headers.get("mcp-protocol-version")
    if version and version not in SUPPORTED_VERSIONS:
        return JSONResponse(rpc_error(None, INVALID_REQUEST, f"Unsupported MCP-Protocol-Version; use one of {list(SUPPORTED_VERSIONS)}"),
                            status_code=400)
    principal = await run_in_threadpool(_authenticate, request)
    raw = await _read_limited(request)
    try:
        message = json.loads(raw)
    except ValueError:
        return JSONResponse(rpc_error(None, PARSE_ERROR, "Parse error"), status_code=400)
    if isinstance(message, list):
        return JSONResponse(rpc_error(None, INVALID_REQUEST, "Batch requests are not supported"), status_code=400)
    response = await run_in_threadpool(server.handle_message, message, principal, "http")
    if response is None:
        return Response(status_code=202)
    status = 400 if response.get("error", {}).get("code") in (PARSE_ERROR, INVALID_REQUEST) else 200
    return JSONResponse(response, status_code=status)


@router.get("/mcp", include_in_schema=False)
async def mcp_status(request: Request):
    """A read-only status page for humans (browser) or scripts (JSON). It never runs a tool."""
    if not settings.mcp_enabled:
        raise HTTPException(404, "Not Found")
    data = await run_in_threadpool(build_status, request)
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(render_html(data))
    return JSONResponse(data)


@router.delete("/mcp", include_in_schema=False)
def mcp_unsupported():
    if not settings.mcp_enabled:
        raise HTTPException(404, "Not Found")
    raise HTTPException(405, "Use POST for MCP messages or GET for the status page", headers={"Allow": "GET, POST"})
