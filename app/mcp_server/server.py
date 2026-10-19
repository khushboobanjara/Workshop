"""The process-wide MCP server instance (tools + rate limiter). Tests reset `server.limiter.hits`."""
from app.mcp_server.protocol import McpServer
from app.mcp_server.tools import TOOLS

server = McpServer(TOOLS)
