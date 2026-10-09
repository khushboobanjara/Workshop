-- OPTIONAL and MANUAL. This is the only script that touches the existing `users` table.
-- Read it, then run the statements you need by hand. Take a backup first (mysqldump).

-- Step 1: see how users.role is defined.
SELECT COLUMN_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'users' AND COLUMN_NAME = 'role';

-- Step 2a: if COLUMN_TYPE is an ENUM, it must list SUPER_ADMIN. Copy the existing values from
--          Step 1 and add 'SUPER_ADMIN'. Example (EDIT the list to match YOUR output):
-- ALTER TABLE users MODIFY role ENUM('PATIENT','DOCTOR','ADMIN','SUPER_ADMIN') NOT NULL;

-- Step 2b: if it is VARCHAR, no change is needed.

-- Step 3: promote ONE existing account (replace the e-mail). Never expose this through the app.
-- UPDATE users SET role = 'SUPER_ADMIN' WHERE email = 'owner@example.com' LIMIT 1;
