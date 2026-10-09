-- Sanjeevani MCP Phase 2: add SUPER_ADMIN to the MCP role column.
-- Touches ONLY the MCP table mcp_permissions. Safe to re-run (MODIFY is idempotent).
-- Run:  mysql -u <user> -p sanjeevani_clinic < 002_super_admin_role.sql
-- Prerequisite: 001_mcp_foundation.sql has been applied.

ALTER TABLE mcp_permissions
    MODIFY role ENUM('USER','DOCTOR','ADMIN','SUPER_ADMIN','SYSTEM') NOT NULL;
