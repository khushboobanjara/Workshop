"""Role mapping and the single authorisation decision. Pure functions: no database, no web framework.

The application already has roles (admin / reviewer / viewer, plus an ingest-only 'service' principal). MCP exposes
three labels on top of them WITHOUT changing the users table or its CHECK constraint:

    MCP role      application role(s)                 how the caller authenticates
    ------------  ----------------------------------  ----------------------------------------------------
    user          viewer, reviewer                    a user's JWT (from POST /auth/login) as a Bearer token
    super_admin   admin                               an admin's JWT
    system        service                             OPSPILOT_MCP_SYSTEM_KEY as a Bearer token

Each tool lists the application roles that may call it. The reviewer/viewer split therefore stays exactly as it is in
the REST API, and MCP never grants a human more than the REST API does. The ingest API key (X-API-Key) is NOT an MCP
credential: a leaked ingest key must not become read access to invoices.
"""

ROLE_LABELS = {"viewer": "user", "reviewer": "user", "admin": "super_admin", "service": "system"}
SYSTEM_VIA = "mcp_system_key"            # Principal.via of the system principal; nothing else may act as 'service' here


def mcp_role(role: str, via: str) -> str | None:
    """The MCP label for a principal, or None when the principal has no business using MCP at all."""
    label = ROLE_LABELS.get(role)
    if role == "service" and via != SYSTEM_VIA:
        return None
    return label


def is_allowed(role: str, via: str, allowed_roles) -> bool:
    """Deny by default: unknown roles, a 'service' principal that did not come from the MCP system key, and roles the
    tool does not list are all refused."""
    return mcp_role(role, via) is not None and role in allowed_roles
