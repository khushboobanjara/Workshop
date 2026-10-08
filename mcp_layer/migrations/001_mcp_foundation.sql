-- Sanjeevani MCP foundation. Creates NEW tables only; existing tables are not altered.
-- Run once:  mysql -u <user> -p sanjeevani_clinic < 001_mcp_foundation.sql
-- Requires MySQL 8.0.16+ (CHECK constraints are enforced) and InnoDB.

CREATE TABLE IF NOT EXISTS mcp_servers (
    server_id        VARCHAR(64)  NOT NULL,
    server_name      VARCHAR(120) NOT NULL,
    server_type      ENUM('USER','ADMIN','SYSTEM') NOT NULL,
    description      VARCHAR(255) NULL,
    status           ENUM('active','disabled') NOT NULL DEFAULT 'active',
    health_status    ENUM('healthy','degraded','down','unknown') NOT NULL DEFAULT 'unknown',
    last_health_check DATETIME NULL,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (server_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS mcp_tools (
    tool_id          INT NOT NULL AUTO_INCREMENT,
    server_id        VARCHAR(64) NOT NULL,
    tool_name        VARCHAR(64) NOT NULL,
    permission       VARCHAR(100) NOT NULL,
    ownership        ENUM('none','current_user_only','doctor_own','administrative') NOT NULL DEFAULT 'none',
    kind             ENUM('READ','WRITE','DESTRUCTIVE') NOT NULL DEFAULT 'READ',
    audit_required   TINYINT(1) NOT NULL DEFAULT 1,
    requires_verified_data TINYINT(1) NOT NULL DEFAULT 0,
    is_enabled       TINYINT(1) NOT NULL DEFAULT 1,
    PRIMARY KEY (tool_id),
    UNIQUE KEY uq_tool (server_id, tool_name),
    CONSTRAINT fk_tool_server FOREIGN KEY (server_id) REFERENCES mcp_servers (server_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Role grants can only RESTRICT what the code-defined tool metadata allows (never widen it).
CREATE TABLE IF NOT EXISTS mcp_permissions (
    permission_id    INT NOT NULL AUTO_INCREMENT,
    role             ENUM('USER','DOCTOR','ADMIN','SYSTEM') NOT NULL,
    tool_id          INT NOT NULL,
    granted          TINYINT(1) NOT NULL DEFAULT 1,
    PRIMARY KEY (permission_id),
    UNIQUE KEY uq_role_tool (role, tool_id),
    CONSTRAINT fk_perm_tool FOREIGN KEY (tool_id) REFERENCES mcp_tools (tool_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS mcp_audit_logs (
    audit_id         BIGINT NOT NULL AUTO_INCREMENT,
    request_id       CHAR(32) NOT NULL,
    user_id          INT NULL,
    role             VARCHAR(20) NULL,
    server_id        VARCHAR(64) NULL,           -- no FK: attempts on unknown servers are logged too
    tool_name        VARCHAR(64) NULL,
    argument_names   VARCHAR(255) NULL,          -- names only, never values
    permission_result ENUM('ALLOWED','DENIED','NOT_CHECKED') NOT NULL,
    success          TINYINT(1) NOT NULL,
    error_code       VARCHAR(64) NULL,
    source           VARCHAR(64) NULL,
    verified         TINYINT(1) NOT NULL DEFAULT 0,
    llm_model        VARCHAR(100) NULL,
    total_tokens     INT NULL,
    estimated_cost   DECIMAL(14,8) NULL,
    duration_ms      DECIMAL(10,1) NULL,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (audit_id),
    KEY idx_audit_time (created_at),
    KEY idx_audit_user (user_id),
    KEY idx_audit_tool (server_id, tool_name),
    CONSTRAINT fk_audit_user FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Prices are DATA, not code. Intentionally EMPTY: add rows yourself, e.g.
--   INSERT INTO llm_pricing (model_name, input_price_per_million, output_price_per_million, effective_from)
--   VALUES ('<model>', <usd per 1M input tokens>, <usd per 1M output tokens>, CURDATE());
CREATE TABLE IF NOT EXISTS llm_pricing (
    pricing_id       INT NOT NULL AUTO_INCREMENT,
    model_name       VARCHAR(100) NOT NULL,
    input_price_per_million  DECIMAL(12,6) NOT NULL,
    output_price_per_million DECIMAL(12,6) NOT NULL,
    currency         CHAR(3) NOT NULL DEFAULT 'USD',
    effective_from   DATE NOT NULL,
    is_active        TINYINT(1) NOT NULL DEFAULT 1,
    PRIMARY KEY (pricing_id),
    UNIQUE KEY uq_model_from (model_name, effective_from),
    CONSTRAINT chk_prices CHECK (input_price_per_million >= 0 AND output_price_per_million >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS llm_usage (
    usage_id         BIGINT NOT NULL AUTO_INCREMENT,
    request_id       CHAR(32) NOT NULL,
    user_id          INT NULL,
    role             VARCHAR(20) NULL,
    mcp_server_id    VARCHAR(64) NULL,
    tool_name        VARCHAR(64) NULL,
    model            VARCHAR(100) NOT NULL,
    input_tokens     INT NOT NULL DEFAULT 0,
    output_tokens    INT NOT NULL DEFAULT 0,
    total_tokens     INT NOT NULL DEFAULT 0,
    input_cost       DECIMAL(14,8) NULL,         -- NULL = no active price for this model (not guessed)
    output_cost      DECIMAL(14,8) NULL,
    total_cost       DECIMAL(14,8) NULL,
    currency         CHAR(3) NOT NULL DEFAULT 'USD',
    status           ENUM('SUCCESS','FAILED','BLOCKED') NOT NULL,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (usage_id),
    UNIQUE KEY uq_usage_request (request_id),
    KEY idx_usage_time (created_at),
    KEY idx_usage_user (user_id),
    CONSTRAINT chk_total_tokens CHECK (total_tokens = input_tokens + output_tokens),
    CONSTRAINT fk_usage_user FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
