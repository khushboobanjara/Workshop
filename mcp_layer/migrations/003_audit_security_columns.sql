-- Sanjeevani MCP Phase 4: record the safety-layer result and the internal denial reason.
-- Touches ONLY mcp_audit_logs. Idempotent: re-running does nothing.
-- Run:  mysql -u <user> -p sanjeevani_clinic < 003_audit_security_columns.sql
-- Prerequisite: 001_mcp_foundation.sql

SET @has_cols := (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'mcp_audit_logs'
      AND COLUMN_NAME IN ('security_result', 'denial_reason')
);

SET @ddl := IF(@has_cols = 0,
    'ALTER TABLE mcp_audit_logs
        ADD COLUMN security_result ENUM(''PASSED'',''BLOCKED'',''NOT_CHECKED'') NOT NULL DEFAULT ''NOT_CHECKED'' AFTER permission_result,
        ADD COLUMN denial_reason VARCHAR(60) NULL AFTER security_result',
    'SELECT 1');

PREPARE mcp_stmt FROM @ddl;
EXECUTE mcp_stmt;
DEALLOCATE PREPARE mcp_stmt;
