"""MCP 10 - LLM / Token / Cost / Monitoring (SYSTEM server).

Who can call what:
  * SYSTEM only:        record_llm_usage (write), calculate_cost (read) - internal code only.
  * SUPER_ADMIN only:   get_llm_usage_summary, get_llm_usage_logs, get_llm_pricing (read),
                        set_llm_pricing (write).
  * SYSTEM + SUPER_ADMIN: get_mcp_health (read).
Admins, doctors and users are refused at the server level.

Pricing is never in Python: every price comes from the llm_pricing table. A model with no active
price is stored with NULL costs ("unpriced") - a price is never guessed, and unpriced requests are
counted separately in the summary so they are visible.
"""
import re
import uuid
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from src.database import mcp_admin_repository as admin_repo
from src.database import mcp_llm_repository as repo
from src.database import mcp_security_repository as audit_repo

from ..llm_cost import MAX_TOKENS, calculate_cost, money_text
from ..registry import MCPServerSpec
from ..schemas import ErrorCode, Ownership, Role, ToolKind, ToolMeta
from ._common import BadInput, clean_text, fail, ok, ok_rows, opt_text, parse_date, positive_int, rules, whole_number

SUPER_ONLY = frozenset({Role.SUPER_ADMIN})
SYSTEM_ONLY = frozenset({Role.SYSTEM})
SYSTEM_AND_SUPER = frozenset({Role.SYSTEM, Role.SUPER_ADMIN})
STATUSES = {"SUCCESS", "FAILED", "BLOCKED"}
LOG_LIMIT = 100
MAX_HOURS = 720
MAX_PRICE = Decimal("100000")
MIN_PRICE_DATE = date(2020, 1, 1)

_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{1,99}$")
_IDENT = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_REQUEST_REF = re.compile(r"^[0-9a-f]{32}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
# users.role -> the role name stored beside usage rows
_STORED_ROLE = {"PATIENT": "USER", "DOCTOR": "DOCTOR", "ADMIN": "ADMIN", "SUPER_ADMIN": "SUPER_ADMIN"}

_gateway: Optional[Any] = None


def bind_gateway(gateway: Any) -> None:
    """bootstrap.build_gateway() calls this so get_mcp_health can ping the servers."""
    global _gateway
    _gateway = gateway


# ------------------------------------------------------------------ validation
def _model(value: Any) -> str:
    text = clean_text(value, "Model", 2, 100)
    if not _MODEL.match(text):
        raise BadInput("Model name is invalid.")
    return text


def _ident(value: Any, label: str) -> str:
    text = clean_text(value, label, 2, 64).lower()
    if not _IDENT.match(text):
        raise BadInput(f"{label} is invalid.")
    return text


def _price(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise BadInput(f"{label} must be a number.")
    try:
        price = Decimal(str(value).strip())
    except InvalidOperation:
        raise BadInput(f"{label} must be a number.") from None
    if not price.is_finite() or not (0 <= price <= MAX_PRICE):
        raise BadInput(f"{label} must be between 0 and {MAX_PRICE:f}.")
    if -price.as_tuple().exponent > 6:
        raise BadInput(f"{label} can have at most 6 decimal places.")
    return price


def _tokens(value: Any, label: str) -> int:
    return whole_number(value, label, 0, MAX_TOKENS)


def _no_price_message(model: str) -> str:
    return "No active price is configured for this model, so the cost is not calculated."


def register(spec: MCPServerSpec) -> None:
    # ------------------------------------------------------------- SYSTEM only
    @spec.tool(ToolMeta(
        name="calculate_cost", allowed_roles=SYSTEM_ONLY, permission="llm.cost.calculate",
        ownership=Ownership.NONE, kind=ToolKind.READ, requires_verified_data=True,
        description="INTERNAL. Cost of input_tokens + output_tokens for a model, from the llm_pricing table. "
                    "Costs are returned as exact text. Unpriced models return priced=false."))
    @rules
    def calculate_cost_tool(model: str, input_tokens: int, output_tokens: int) -> dict:
        name = _model(model)
        tin, tout = _tokens(input_tokens, "Input tokens"), _tokens(output_tokens, "Output tokens")
        price = repo.get_active_price(name)
        if not price:
            return ok({"model": name, "priced": False, "input_cost": None, "output_cost": None, "total_cost": None},
                      message=_no_price_message(name))
        cost = calculate_cost(tin, tout, price["input_price_per_million"], price["output_price_per_million"])
        return ok({"model": name, "priced": True, "currency": price["currency"],
                   "input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout,
                   "input_cost": money_text(cost.input_cost), "output_cost": money_text(cost.output_cost),
                   "total_cost": money_text(cost.total_cost)})

    @spec.tool(ToolMeta(
        name="record_llm_usage", allowed_roles=SYSTEM_ONLY, permission="llm.usage.record",
        ownership=Ownership.NONE, kind=ToolKind.WRITE,
        description="INTERNAL. Store one LLM request: model, input/output tokens and status, plus who it was for "
                    "and which MCP tool it served. Cost is computed here from llm_pricing. request_ref (the "
                    "gateway request id, 32 hex chars) makes the call idempotent."))
    @rules
    def record_llm_usage(model: str, input_tokens: int, output_tokens: int, status: str = "SUCCESS",
                         request_ref: Optional[str] = None, subject_user_id: Optional[int] = None,
                         mcp_server_id: Optional[str] = None, mcp_tool_name: Optional[str] = None) -> dict:
        name = _model(model)
        tin, tout = _tokens(input_tokens, "Input tokens"), _tokens(output_tokens, "Output tokens")
        state = clean_text(status, "Status", 6, 7).upper()
        if state not in STATUSES:
            raise BadInput("Status must be SUCCESS, FAILED or BLOCKED.")
        ref = opt_text(request_ref, "Request reference", 32, 32)
        if ref is not None and not _REQUEST_REF.match(ref):
            raise BadInput("Request reference must be 32 lowercase hex characters.")
        ref = ref or uuid.uuid4().hex
        server = _ident(mcp_server_id, "Server") if opt_text(mcp_server_id, "Server", 2, 64) else None
        tool = _ident(mcp_tool_name, "Tool") if opt_text(mcp_tool_name, "Tool", 2, 64) else None

        who, stored_role = None, None
        if subject_user_id is not None:
            who = positive_int(subject_user_id, "User")
            user = admin_repo.get_user(who)
            if not user:
                return fail(ErrorCode.NOT_FOUND, "That user could not be found.")
            stored_role = _STORED_ROLE.get(user["role"])

        price = repo.get_active_price(name)
        if price:
            cost = calculate_cost(tin, tout, price["input_price_per_million"], price["output_price_per_million"])
            parts, currency = (cost.input_cost, cost.output_cost, cost.total_cost), price["currency"]
        else:
            parts, currency = (None, None, None), "USD"      # unpriced: NULL, never guessed

        if not repo.insert_usage(ref, who, stored_role, server, tool, name, tin, tout, *parts, currency, state):
            return fail(ErrorCode.CONFLICT, "That request was already recorded.")
        return ok({"recorded": True, "request_id": ref, "model": name, "total_tokens": tin + tout,
                   "priced": price is not None, "currency": currency, "total_cost": money_text(parts[2])})

    # ----------------------------------------------------------- SUPER_ADMIN reads
    @spec.tool(ToolMeta(
        name="get_llm_usage_summary", allowed_roles=SUPER_ONLY, permission="llm.usage.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Requests, tokens and cost for the last N hours (1-720): totals, by model, by MCP tool, top users. "
                    "Costs are never added across currencies; unpriced requests are counted separately."))
    @rules
    def get_llm_usage_summary(hours: int = 24) -> dict:
        window = whole_number(hours, "Hours", 1, MAX_HOURS)
        return ok({
            "window_hours": window,
            "totals": repo.usage_totals(window),
            "cost_by_currency": repo.cost_by_currency(window),
            "by_model": repo.usage_by_model(window),
            "by_tool": repo.usage_by_tool(window),
            "top_users": repo.usage_top_users(window),
        })

    @spec.tool(ToolMeta(
        name="get_llm_usage_logs", allowed_roles=SUPER_ONLY, permission="llm.usage.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Recent LLM requests (newest first, max 100). Filters: hours (1-720), model_filter, target_user_id."))
    @rules
    def get_llm_usage_logs(hours: int = 24, model_filter: Optional[str] = None,
                           target_user_id: Optional[int] = None) -> dict:
        window = whole_number(hours, "Hours", 1, MAX_HOURS)
        name = _model(model_filter) if opt_text(model_filter, "Model", 2, 100) else None
        who = positive_int(target_user_id, "User") if target_user_id is not None else None
        return ok_rows(repo.query_usage(window, name, who, LOG_LIMIT), "No LLM usage matches.")

    @spec.tool(ToolMeta(
        name="get_llm_pricing", allowed_roles=SUPER_ONLY, permission="llm.usage.read",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Every configured model price, newest first per model (price history is kept)."))
    @rules
    def get_llm_pricing() -> dict:
        return ok_rows(repo.list_pricing(), "No model prices are configured yet.")

    # ---------------------------------------------------------- SUPER_ADMIN write
    @spec.tool(ToolMeta(
        name="set_llm_pricing", allowed_roles=SUPER_ONLY, permission="llm.pricing.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Add a price for a model: USD (or other 3-letter currency) per 1,000,000 tokens, from "
                    "effective_from (YYYY-MM-DD). History is kept; the newest price in force is used."))
    @rules
    def set_llm_pricing(model_name: str, input_price_per_million: float, output_price_per_million: float,
                        effective_from: str, currency: str = "USD") -> dict:
        name = _model(model_name)
        price_in = _price(input_price_per_million, "Input price")
        price_out = _price(output_price_per_million, "Output price")
        day = parse_date(effective_from)
        if not (MIN_PRICE_DATE <= day <= date.today() + timedelta(days=366)):
            raise BadInput("Effective date is out of range.")
        code = clean_text(currency, "Currency", 3, 3).upper()
        if not _CURRENCY.match(code):
            raise BadInput("Currency must be a 3-letter code such as USD.")
        if not repo.insert_pricing(name, price_in, price_out, code, day):
            return fail(ErrorCode.CONFLICT, "That model already has a price for that date.")
        return ok({"model_name": name, "input_price_per_million": money_text(price_in),
                   "output_price_per_million": money_text(price_out), "currency": code,
                   "effective_from": day.isoformat()}, message="Price saved.")

    # ------------------------------------------------------------- monitoring
    @spec.tool(ToolMeta(
        name="get_mcp_health", allowed_roles=SYSTEM_AND_SUPER, permission="system.monitor",
        ownership=Ownership.NONE, kind=ToolKind.READ,
        description="Liveness of all 10 MCP servers (pings each) with tool counts and the last 24 hours of "
                    "activity from the audit log."))
    @rules
    async def get_mcp_health() -> dict:
        if _gateway is None:
            return fail(ErrorCode.MCP_UNAVAILABLE, "The monitoring service is currently unavailable.", source="mcp_internal")
        status = await _gateway.health_check()
        servers = [{
            "server_id": s.server_id, "server_name": s.server_name, "server_type": s.server_type.value,
            "status": s.status, "health": status.get(s.server_id, "unknown"), "tool_count": len(s.tools),
        } for s in _gateway.registry.servers()]
        activity, activity_available = {}, True
        try:   # the liveness part is internal and stays valid even if the audit database is down
            activity = {r["server_id"]: r for r in audit_repo.audit_by_server(24)}
        except Exception:
            activity_available = False
        for row in servers:
            seen = activity.get(row["server_id"], {})
            row.update(calls_24h=seen.get("calls", 0), denied_24h=seen.get("denied", 0),
                       failed_24h=seen.get("failed", 0), last_activity=seen.get("last_activity"))
        return ok({"servers": servers, "activity_available": activity_available},
                  source="mcp_internal", message=None if activity_available else "Activity data is unavailable.")
