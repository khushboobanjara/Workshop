-- Approved knowledge source for database-grounded chatbot answers.
-- The table is intentionally created empty: only verified clinic information
-- should be inserted and published by an authorized administrator.

CREATE TABLE IF NOT EXISTS clinic_knowledge (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    answer MEDIUMTEXT NOT NULL,
    keywords VARCHAR(1000) NULL,
    category VARCHAR(100) NOT NULL DEFAULT 'CLINIC',
    is_published TINYINT(1) NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_clinic_knowledge_published (is_published),
    INDEX idx_clinic_knowledge_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Example template only. Replace with reviewed, accurate clinic-approved content
-- before inserting. Draft rows remain invisible until is_published = 1.
-- INSERT INTO clinic_knowledge (title, answer, keywords, category, is_published)
-- VALUES ('Example title', 'Reviewed answer text from an approved source.',
--         'keyword one, keyword two', 'FAQ', 1);
