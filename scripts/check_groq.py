"""Checks the Groq connection exactly the way the AI assistant uses it, and says which step fails and why.
Run from the project root (it reads your .env):   python scripts/check_groq.py
Prints Groq's own error messages; never prints the API key."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx  # noqa: E402

from app.config import settings  # noqa: E402
from app.mcp_server.server import server  # noqa: E402

KEY = settings.groq_api_key.get_secret_value()
BASE = settings.groq_base_url.rstrip("/")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def show(label, r):
    ok = r.status_code == 200
    msg = ""
    if not ok:
        try:
            msg = (r.json().get("error") or {}).get("message", "")[:400]
        except Exception:
            msg = r.text[:200]
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: HTTP {r.status_code} {msg}".replace(KEY, "***") if KEY else f"  {label}: HTTP {r.status_code}")
    return ok


def post(body):
    return httpx.post(f"{BASE}/chat/completions", json=body, headers=H, timeout=40)


def main() -> int:
    print(f"model: {settings.groq_model}\nbase url: {BASE}\nkey set: {'yes (%d chars)' % len(KEY) if KEY else 'NO - set OPSPILOT_GROQ_API_KEY'}")
    if not KEY:
        return 1
    print("\n1. Can this machine reach Groq and is the key accepted?")
    try:
        r = httpx.get(f"{BASE}/models", headers=H, timeout=30)
    except httpx.HTTPError as exc:
        print(f"  [FAIL] network: {type(exc).__name__}: {exc}\n  -> a firewall, proxy, VPN or DNS problem on this machine; Groq was never reached.")
        return 1
    if not show("list models", r):
        print("  -> 401/403: wrong or revoked key. Create a new one at https://console.groq.com/keys")
        return 1
    ids = [m.get("id") for m in r.json().get("data", [])]
    print(f"  model '{settings.groq_model}' available to your key: {'yes' if settings.groq_model in ids else 'NO'}")
    if settings.groq_model not in ids:
        print("  -> set OPSPILOT_GROQ_MODEL to one of:", ", ".join(sorted(i for i in ids if i and "whisper" not in i and "tts" not in i)[:20]))
        return 1

    print("\n2. Plain request")
    if not show("chat completion", post({"model": settings.groq_model, "messages": [{"role": "user", "content": "Reply with the word OK"}], "max_completion_tokens": 10})):
        return 1

    print("\n3. Request with the assistant's tools (all 12 MCP tools, tool_choice=required)")
    specs = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema()}} for t in server.tools.values()]
    body = {"model": settings.groq_model, "messages": [{"role": "user", "content": "What is in the review queue?"}], "tools": specs,
            "tool_choice": "required", "max_completion_tokens": 256}
    r = post(body)
    if show("with tools", r):
        calls = r.json()["choices"][0]["message"].get("tool_calls") or []
        print("  model asked for:", [c["function"]["name"] for c in calls] or "nothing")
        print("\nEverything works. If the dashboard still fails, restart uvicorn and read its console for 'Groq request failed'.")
        return 0
    print("\n4. Finding which tool definition Groq rejects")
    for spec in specs:
        one = dict(body, tools=[spec])
        print(f"  {spec['function']['name']}:", end="")
        show("", post(one))
    return 1


if __name__ == "__main__":
    sys.exit(main())
