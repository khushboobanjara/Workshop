-- Rollback for 002. Fails (by design) if any SUPER_ADMIN permission rows exist; delete them first:
--   DELETE FROM mcp_permissions WHERE role = 'SUPER_ADMIN';
ALTER TABLE mcp_permissions
    MODIFY role ENUM('USER','DOCTOR','ADMIN','SYSTEM') NOT NULL;
