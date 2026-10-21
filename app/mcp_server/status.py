"""The page you see when you open GET /mcp in a browser. Read-only: it never runs a tool.

  everyone      : is the server running, how to connect, protocol versions (nothing sensitive)
  signed in     : + who you are, your MCP role and the tools that role may call
  admin         : + the latest MCP activity from audit_log (who called what, outcome, how long)

"Signed in" accepts the dashboard's session cookie or a Bearer token. That is safe here only because this page cannot
execute anything and shows an admin nothing the dashboard's Audit page does not; the POST endpoint still ignores cookies.
"""
import html

from fastapi import Request
from sqlalchemy import func

from app import auth as app_auth
from app.db import AuditLog, SessionLocal
from app.mcp_server import rbac
from app.mcp_server.auth import bearer_token, principal_from_token
from app.mcp_server.protocol import SERVER_VERSION, SUPPORTED_VERSIONS
from app.mcp_server.server import server

RECENT = 25
SAMPLE = 500


def _viewer(request: Request):
    token = bearer_token(request.headers.get("authorization", "")) or request.cookies.get(app_auth.COOKIE, "")
    return principal_from_token(token) if token else None


def _activity() -> dict:
    with SessionLocal() as s:
        base = s.query(AuditLog).filter(AuditLog.entity_type == "mcp")
        total = s.query(func.count(AuditLog.id)).filter(AuditLog.entity_type == "mcp").scalar() or 0
        rows = base.order_by(AuditLog.id.desc()).limit(SAMPLE).all()
    by_outcome: dict[str, int] = {}
    for r in rows:
        outcome = r.action.rsplit(":", 1)[-1]
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
    recent = []
    for r in rows[:RECENT]:
        d = r.details or {}
        recent.append({"time": r.timestamp.isoformat() if r.timestamp else None, "who": r.actor, "action": r.action,
                       "role": d.get("mcp_role"), "via": d.get("via"), "transport": d.get("transport"),
                       "records": d.get("result_count"), "ms": r.duration_ms})
    return {"total_logged_events": total, "outcomes_in_last_%d" % SAMPLE: by_outcome, "latest": recent}


def build_status(request: Request) -> dict:
    principal = _viewer(request)
    label = rbac.mcp_role(principal.role, principal.via) if principal else None
    data = {
        "server": {"name": "OpsPilot AI MCP server", "version": SERVER_VERSION, "status": "running", "read_only": True,
                   "protocol_versions": list(SUPPORTED_VERSIONS), "tools_available": len(server.tools)},
        "how_to_connect": {
            "http": "POST /mcp with 'Authorization: Bearer <token>' and a JSON-RPC body (this page is only a status view)",
            "stdio": "python -m app.mcp_server (set OPSPILOT_MCP_STDIO_TOKEN)",
            "token": "a user's access_token from POST /auth/login, or OPSPILOT_MCP_SYSTEM_KEY for the system role"},
        "signed_in": principal is not None and label is not None,
    }
    if not data["signed_in"]:
        data["note"] = "Log in to the dashboard in this browser to also see your tools" + " (admins also see live activity)."
        return data
    data["you"] = {"username": principal.username, "role": principal.role, "mcp_role": label}
    data["tools_you_may_call"] = [{"name": t.name, "title": t.title, "description": t.description}
                                  for t in server.allowed_tools(principal)]
    if principal.role == "admin":
        data["activity"] = _activity()
    return data


# ---------------------------------------------------------------- HTML
def _e(v) -> str:
    return html.escape("" if v is None else str(v))


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{_e(label)}</th>" for _, label in cols)
    body = "".join("<tr>" + "".join(f"<td>{_e(r.get(k))}</td>" for k, _ in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def render_html(d: dict) -> str:
    s = d["server"]
    parts = [f"<h1>{_e(s['name'])} <span class='ok'>● {_e(s['status'])}</span></h1>",
             f"<p>Version {_e(s['version'])} · read-only · {_e(s['tools_available'])} tools · protocol "
             f"{_e(', '.join(s['protocol_versions']))}</p>",
             "<h2>How to connect</h2><ul>" + "".join(f"<li><b>{_e(k)}</b>: {_e(v)}</li>" for k, v in d["how_to_connect"].items()) + "</ul>"]
    if not d["signed_in"]:
        parts.append(f"<p class='note'>{_e(d.get('note'))}</p>")
    else:
        y = d["you"]
        parts.append(f"<h2>You</h2><p>{_e(y['username'])} · role <b>{_e(y['role'])}</b> · MCP role <b>{_e(y['mcp_role'])}</b></p>")
        parts.append("<h2>Tools you may call</h2>" + _table(d["tools_you_may_call"], [("name", "Tool"), ("description", "What it does")]))
        if "activity" in d:
            a = d["activity"]
            outcomes = next((v for k, v in a.items() if k.startswith("outcomes")), {})
            parts.append(f"<h2>Activity</h2><p>{_e(a['total_logged_events'])} logged events · recent outcomes: "
                         f"{_e(', '.join(f'{k}={v}' for k, v in outcomes.items()) or 'none yet')}</p>")
            parts.append(_table(a["latest"], [("time", "Time"), ("who", "Who"), ("action", "Tool : outcome"), ("role", "MCP role"),
                                              ("transport", "Transport"), ("records", "Records"), ("ms", "ms")]) if a["latest"]
                         else "<p>No MCP calls yet.</p>")
    css = ("body{font:15px system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fff}"
           "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.35rem .5rem;text-align:left;font-size:13px}"
           "th{background:#f0f0f0}.ok{color:#188038;font-size:.6em}.note{background:#fff7e0;padding:.6rem;border-radius:6px}"
           "@media(prefers-color-scheme:dark){body{background:#16181c;color:#e6e6e6}th{background:#262a30}th,td{border-color:#444}"
           ".note{background:#3a3220}}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>OpsPilot MCP</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"
