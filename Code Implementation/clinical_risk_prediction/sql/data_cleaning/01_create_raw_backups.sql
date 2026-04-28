-- Asclena clinical cleaning pipeline
-- Step 1: create safety backups of full raw tables.
--
-- CREATE TABLE IF NOT EXISTS keeps existing backups unchanged. If you need a
-- fresh backup snapshot, rename or drop the existing *_raw_backup table first.

BEGIN;

CREATE TABLE IF NOT EXISTS asclena.ed_stays_raw_backup AS
SELECT * FROM asclena.ed_stays;

CREATE TABLE IF NOT EXISTS asclena.triage_raw_backup AS
SELECT * FROM asclena.triage;

CREATE TABLE IF NOT EXISTS asclena.vital_sign_raw_backup AS
SELECT * FROM asclena.vital_sign;

CREATE TABLE IF NOT EXISTS asclena.diagnosis_raw_backup AS
SELECT * FROM asclena.diagnosis;

CREATE TABLE IF NOT EXISTS asclena.pyxis_raw_backup AS
SELECT * FROM asclena.pyxis;

CREATE TABLE IF NOT EXISTS asclena.med_recon_raw_backup AS
SELECT * FROM asclena.med_recon;

ANALYZE asclena.ed_stays_raw_backup;
ANALYZE asclena.triage_raw_backup;
ANALYZE asclena.vital_sign_raw_backup;
ANALYZE asclena.diagnosis_raw_backup;
ANALYZE asclena.pyxis_raw_backup;
ANALYZE asclena.med_recon_raw_backup;

COMMIT;
