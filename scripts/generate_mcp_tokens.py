"""Print strong role-specific MCP tokens for adding to a local .env file.

Usage: python scripts/generate_mcp_tokens.py
The script prints secrets to the terminal only; it never writes or overwrites .env.
"""

import secrets

ROLES = (
    "MCP_USER_TOKEN",
    "MCP_ADMIN_TOKEN",
    "MCP_SYSTEM_TOKEN",
    "MCP_SUPER_ADMIN_TOKEN",
)

if __name__ == "__main__":
    for name in ROLES:
        print(f"{name}={secrets.token_urlsafe(48)}")
