"""FastMCP server exposing ten role-protected Sanjeevani Clinic tools."""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_layer.security import MCPBearerAuthMiddleware, current_mcp_role
from mcp_layer.tools import TOOL_CATALOG, invoke_tool

mcp = FastMCP(
    "Sanjeevani Clinic MCP",
    instructions=(
        "Sanjeevani Clinic MCP gateway. Tool access is checked server-side using a role-specific "
        "bearer token. Clinic facts come from the configured database. If the database is unavailable, "
        "data tools fail closed and do not generate replacement answers."
    ),
)


@mcp.tool()
def user_search_doctors(specialization: str = "") -> dict:
    """Search currently available clinic doctors by optional specialization (USER tool 1/5)."""
    return invoke_tool("user_search_doctors", {"specialization": specialization})


@mcp.tool()
def user_list_specializations() -> dict:
    """List specialties present in the clinic database (USER tool 2/5)."""
    return invoke_tool("user_list_specializations", {})


@mcp.tool()
def user_get_doctor_availability(doctor_id: int) -> dict:
    """Get upcoming recorded availability for a doctor ID (USER tool 3/5)."""
    return invoke_tool("user_get_doctor_availability", {"doctor_id": doctor_id})


@mcp.tool()
def user_search_medicines(query: str) -> dict:
    """Search available medicines by name, generic name, or category (USER tool 4/5)."""
    return invoke_tool("user_search_medicines", {"query": query})


@mcp.tool()
def user_get_medicine_details(medicine_id: int) -> dict:
    """Get a medicine record by its database ID (USER tool 5/5)."""
    return invoke_tool("user_get_medicine_details", {"medicine_id": medicine_id})


@mcp.tool()
def admin_list_pending_doctors() -> dict:
    """List doctor registrations awaiting approval (ADMIN tool 1/3)."""
    return invoke_tool("admin_list_pending_doctors", {})


@mcp.tool()
def admin_approve_doctor(email: str) -> dict:
    """Approve a pending doctor registration by email (ADMIN tool 2/3)."""
    return invoke_tool("admin_approve_doctor", {"email": email})


@mcp.tool()
def admin_clinic_metrics() -> dict:
    """Return clinic-wide aggregate counts without exposing patient records (ADMIN tool 3/3)."""
    return invoke_tool("admin_clinic_metrics", {})


@mcp.tool()
def system_database_health() -> dict:
    """Check database reachability using a simple query (SYSTEM tool 1/2)."""
    return invoke_tool("system_database_health", {})


@mcp.tool()
def system_mcp_status() -> dict:
    """Return tool counts and configured role names without secrets (SYSTEM tool 2/2)."""
    return invoke_tool("system_mcp_status", {})


def create_mcp_asgi_app():
    """Create a protected FastMCP HTTP app to mount beneath FastAPI `/mcp`."""
    # Root path avoids a duplicated /mcp/mcp prefix when mounted in FastAPI.
    http_app = mcp.http_app(path="/")
    return http_app, MCPBearerAuthMiddleware(http_app)


def tool_count() -> int:
    return len(TOOL_CATALOG)
