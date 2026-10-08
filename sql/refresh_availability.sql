-- Re-create bookable slots for the NEXT 14 DAYS for every available doctor.
-- Why: the app only shows slots where available_date >= CURDATE(), and the
-- existing rows are all dated 2026-09-23..28 (already in the past), so the
-- chatbot says "no available appointment dates".
-- Safe to re-run: it skips slots that already exist. Needs MySQL 8+ or MariaDB.
-- Skips Sundays; two slots per day (10:00-13:00 and 16:00-19:00). Edit to taste.

INSERT INTO doctor_availability (doctor_id, available_date, start_time, end_time, is_available)
SELECT d.doctor_id,
       DATE_ADD(CURDATE(), INTERVAL n.i DAY),
       s.st,
       s.et,
       1
FROM doctors d
CROSS JOIN (
    SELECT 0 AS i UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3
    UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7
    UNION ALL SELECT 8 UNION ALL SELECT 9 UNION ALL SELECT 10 UNION ALL SELECT 11
    UNION ALL SELECT 12 UNION ALL SELECT 13
) n
CROSS JOIN (
    SELECT '10:00:00' AS st, '13:00:00' AS et
    UNION ALL
    SELECT '16:00:00', '19:00:00'
) s
WHERE d.available = TRUE
  AND DAYOFWEEK(DATE_ADD(CURDATE(), INTERVAL n.i DAY)) <> 1
  AND NOT EXISTS (
      SELECT 1 FROM doctor_availability a
      WHERE a.doctor_id = d.doctor_id
        AND a.available_date = DATE_ADD(CURDATE(), INTERVAL n.i DAY)
        AND a.start_time = s.st
  );

-- Check:
SELECT doctor_id, COUNT(*) AS upcoming_slots
FROM doctor_availability
WHERE available_date >= CURDATE() AND is_available = 1
GROUP BY doctor_id;
