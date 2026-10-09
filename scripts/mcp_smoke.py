"""Smoke test of the user MCP tools against your REAL database.

Run from the project root:   python scripts/mcp_smoke.py <user_id>

Read-only: it only calls GET-style tools and prints success / verified / error code / row count,
never the data itself. Check logs/ and the mcp_audit_logs table afterwards.
"""
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from mcp_layer.auth import RepositoryIdentityStore, resolve_principal   # noqa: E402
from mcp_layer.bootstrap import build_gateway, initialize_mcp           # noqa: E402

CALLS = [
    ("mcp_01_patient_profile", "get_my_profile", {}),
    ("mcp_02_doctor_appointment", "find_doctors", {}),
    ("mcp_02_doctor_appointment", "get_my_appointments", {}),
    ("mcp_03_pharmacy", "search_medicine", {"query": "para"}),
    ("mcp_03_pharmacy", "get_my_cart", {}),
    ("mcp_03_pharmacy", "get_my_orders", {}),
    ("mcp_05_ai_assistant", "get_verified_context", {"topics": ["profile", "appointments"]}),
]


async def main(user_id: int) -> None:
    gateway = build_gateway()
    print("startup:", await initialize_mcp(gateway))
    outcome = resolve_principal(user_id, RepositoryIdentityStore())
    if not outcome.ok:
        print("no principal:", outcome.error_code, outcome.reason)
        return
    print("principal:", outcome.principal.role.value)
    for server, tool, args in CALLS:
        r = await gateway.invoke(outcome.principal, server, tool, args)
        print(f"{tool:24} success={r.success} verified={r.verified} error={r.error_code} "
              f"rows={r.metadata.get('row_count')}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1])))
