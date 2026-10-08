-- Rollback: removes ONLY the MCP tables. Your existing tables are untouched.
DROP TABLE IF EXISTS llm_usage;
DROP TABLE IF EXISTS llm_pricing;
DROP TABLE IF EXISTS mcp_audit_logs;
DROP TABLE IF EXISTS mcp_permissions;
DROP TABLE IF EXISTS mcp_tools;
DROP TABLE IF EXISTS mcp_servers;
