"""Model Context Protocol (MCP) server for OpsPilot AI.

A read-only, database-grounded tool surface for AI assistants. Layers (each file has one job):

  contracts.py   Tool / ToolResult / ToolContext dataclasses
  sanitize.py   redaction and cleaning of free text before it leaves the server (pure, no I/O)
  rbac.py       maps the application's roles to MCP roles and decides who may call what (pure)
  grounding.py  allow-listed output shaping + the envelope that marks every answer as database-backed
  audit.py      one append-only audit_log row per tool call, denial, failure and failed sign-in
  tools.py      the tools themselves (argument schemas + handlers on top of app.services.processing)
  protocol.py   transport-independent JSON-RPC / MCP dispatcher (auth is already done by the transport)
  auth.py       bearer token -> Principal (user JWT or the dedicated MCP system key)
  routes.py     Streamable HTTP transport: POST /mcp, plus start-up safety checks
  stdio.py      stdio transport:  python -m app.mcp_server

Nothing here is imported by the existing API except the router in app/main.py, and everything is off unless
OPSPILOT_MCP_ENABLED=true.
"""
