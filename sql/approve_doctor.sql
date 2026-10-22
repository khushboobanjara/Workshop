-- Approve a doctor account so the Doctor Dashboard opens.
-- Registration creates doctors with doctor_status = 'PENDING' and there is no
-- admin screen yet, so approval is done here.
-- Replace the e-mail with the doctor's login e-mail.

UPDATE users
SET doctor_status = 'APPROVED'
WHERE role = 'DOCTOR' AND email = 'doctor@example.com';

-- The dashboard ALSO needs a row in `doctors` with the SAME e-mail (that is how
-- a login is linked to a doctor). If this returns nothing, the dashboard shows
-- "unlinked" until you add one:
SELECT doctor_id, doctor_name, email FROM doctors WHERE LOWER(email) = LOWER('doctor@example.com');

-- Example if it is missing (adjust values):
-- INSERT INTO doctors (doctor_name, specialization, experience, phone, email, consultation_fee, available)
-- VALUES ('Dr. Dhruv', 'General Physician', 5, '9999999999', 'doctor@example.com', 500, TRUE);
