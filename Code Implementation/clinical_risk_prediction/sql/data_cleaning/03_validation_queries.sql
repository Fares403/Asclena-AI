-- Asclena clinical cleaning pipeline
-- Step 5-6: validation queries for cleaned tables and feature engineering readiness.

SELECT COUNT(*) AS raw_ed_stays_count
FROM asclena.ed_stays;

SELECT COUNT(*) AS cleaned_ed_stays_count
FROM asclena.cleaned_ed_stays;

SELECT COUNT(DISTINCT stay_id) AS cleaned_ed_stays_distinct_stay_id
FROM asclena.cleaned_ed_stays;

SELECT COUNT(*) AS cleaned_triage_count
FROM asclena.cleaned_triage;

SELECT COUNT(DISTINCT stay_id) AS cleaned_triage_distinct_stay_id
FROM asclena.cleaned_triage;

SELECT COUNT(*) AS cleaned_vital_sign_count
FROM asclena.cleaned_vital_sign;

SELECT COUNT(DISTINCT stay_id) AS cleaned_vital_sign_distinct_stay_id
FROM asclena.cleaned_vital_sign;

SELECT COUNT(*) AS cleaned_diagnosis_count
FROM asclena.cleaned_diagnosis;

SELECT COUNT(*) AS cleaned_pyxis_count
FROM asclena.cleaned_pyxis;

SELECT COUNT(*) AS cleaned_med_recon_count
FROM asclena.cleaned_med_recon;

SELECT *
FROM asclena.feature_engineering_exclusions
ORDER BY source_table, column_name;

SELECT 'cleaned_ed_stays' AS table_name, COUNT(*) AS null_identifier_rows
FROM asclena.cleaned_ed_stays
WHERE stay_id IS NULL OR subject_id IS NULL
UNION ALL
SELECT 'cleaned_triage', COUNT(*)
FROM asclena.cleaned_triage
WHERE stay_id IS NULL OR subject_id IS NULL
UNION ALL
SELECT 'cleaned_vital_sign', COUNT(*)
FROM asclena.cleaned_vital_sign
WHERE stay_id IS NULL OR subject_id IS NULL
UNION ALL
SELECT 'cleaned_diagnosis', COUNT(*)
FROM asclena.cleaned_diagnosis
WHERE stay_id IS NULL OR subject_id IS NULL
UNION ALL
SELECT 'cleaned_pyxis', COUNT(*)
FROM asclena.cleaned_pyxis
WHERE stay_id IS NULL OR subject_id IS NULL
UNION ALL
SELECT 'cleaned_med_recon', COUNT(*)
FROM asclena.cleaned_med_recon
WHERE stay_id IS NULL OR subject_id IS NULL;

SELECT COUNT(*) AS cleaned_ed_stays_duplicate_stay_id_rows
FROM (
  SELECT stay_id
  FROM asclena.cleaned_ed_stays
  GROUP BY stay_id
  HAVING COUNT(*) > 1
) d;

SELECT COUNT(*) AS cleaned_triage_duplicate_stay_id_rows
FROM (
  SELECT stay_id
  FROM asclena.cleaned_triage
  GROUP BY stay_id
  HAVING COUNT(*) > 1
) d;

SELECT 'cleaned_ed_stays' AS table_name, COALESCE(SUM(row_count - 1), 0)::bigint AS exact_duplicate_rows
FROM (
  SELECT COUNT(*) AS row_count
  FROM asclena.cleaned_ed_stays
  GROUP BY stay_id, subject_id, hadm_id, intime, outtime, gender, race, arrival_transport, disposition, created_at
  HAVING COUNT(*) > 1
) d
UNION ALL
SELECT 'cleaned_triage', COALESCE(SUM(row_count - 1), 0)::bigint
FROM (
  SELECT COUNT(*) AS row_count
  FROM asclena.cleaned_triage
  GROUP BY stay_id, subject_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint, temperature_missing, heartrate_missing, resprate_missing, o2sat_missing, sbp_missing, dbp_missing, created_at
  HAVING COUNT(*) > 1
) d
UNION ALL
SELECT 'cleaned_vital_sign', COALESCE(SUM(row_count - 1), 0)::bigint
FROM (
  SELECT COUNT(*) AS row_count
  FROM asclena.cleaned_vital_sign
  GROUP BY stay_id, subject_id, charttime, temperature, heartrate, resprate, o2sat, sbp, dbp, rhythm, pain, created_at
  HAVING COUNT(*) > 1
) d
UNION ALL
SELECT 'cleaned_diagnosis', COALESCE(SUM(row_count - 1), 0)::bigint
FROM (
  SELECT COUNT(*) AS row_count
  FROM asclena.cleaned_diagnosis
  GROUP BY stay_id, subject_id, seq_num, icd_code, icd_version, icd_title, created_at
  HAVING COUNT(*) > 1
) d
UNION ALL
SELECT 'cleaned_pyxis', COALESCE(SUM(row_count - 1), 0)::bigint
FROM (
  SELECT COUNT(*) AS row_count
  FROM asclena.cleaned_pyxis
  GROUP BY stay_id, subject_id, charttime, med_rn, name, gsn_rn, gsn, created_at
  HAVING COUNT(*) > 1
) d
UNION ALL
SELECT 'cleaned_med_recon', COALESCE(SUM(row_count - 1), 0)::bigint
FROM (
  SELECT COUNT(*) AS row_count
  FROM asclena.cleaned_med_recon
  GROUP BY stay_id, subject_id, charttime, name, gsn, ndc, etc_rn, etccode, etcdescription, created_at
  HAVING COUNT(*) > 1
) d;

SELECT 'cleaned_triage' AS table_name, COUNT(*) AS invalid_clinical_rows
FROM asclena.cleaned_triage
WHERE (temperature IS NOT NULL AND temperature NOT BETWEEN 90 AND 110)
   OR (heartrate IS NOT NULL AND heartrate NOT BETWEEN 20 AND 250)
   OR (resprate IS NOT NULL AND resprate NOT BETWEEN 5 AND 80)
   OR (o2sat IS NOT NULL AND o2sat NOT BETWEEN 50 AND 100)
   OR (sbp IS NOT NULL AND sbp NOT BETWEEN 50 AND 300)
   OR (dbp IS NOT NULL AND dbp NOT BETWEEN 20 AND 200)
   OR (acuity IS NOT NULL AND acuity NOT BETWEEN 1 AND 5)
UNION ALL
SELECT 'cleaned_vital_sign', COUNT(*)
FROM asclena.cleaned_vital_sign
WHERE (temperature IS NOT NULL AND temperature NOT BETWEEN 90 AND 110)
   OR (heartrate IS NOT NULL AND heartrate NOT BETWEEN 20 AND 250)
   OR (resprate IS NOT NULL AND resprate NOT BETWEEN 5 AND 80)
   OR (o2sat IS NOT NULL AND o2sat NOT BETWEEN 50 AND 100)
   OR (sbp IS NOT NULL AND sbp NOT BETWEEN 50 AND 300)
   OR (dbp IS NOT NULL AND dbp NOT BETWEEN 20 AND 200);

SELECT 'cleaned_triage' AS child_table, COUNT(*) AS orphan_stay_id_rows
FROM asclena.cleaned_triage c
LEFT JOIN asclena.cleaned_ed_stays e
  ON c.stay_id = e.stay_id
WHERE e.stay_id IS NULL
UNION ALL
SELECT 'cleaned_vital_sign', COUNT(*)
FROM asclena.cleaned_vital_sign c
LEFT JOIN asclena.cleaned_ed_stays e
  ON c.stay_id = e.stay_id
WHERE e.stay_id IS NULL
UNION ALL
SELECT 'cleaned_diagnosis', COUNT(*)
FROM asclena.cleaned_diagnosis c
LEFT JOIN asclena.cleaned_ed_stays e
  ON c.stay_id = e.stay_id
WHERE e.stay_id IS NULL
UNION ALL
SELECT 'cleaned_pyxis', COUNT(*)
FROM asclena.cleaned_pyxis c
LEFT JOIN asclena.cleaned_ed_stays e
  ON c.stay_id = e.stay_id
WHERE e.stay_id IS NULL
UNION ALL
SELECT 'cleaned_med_recon', COUNT(*)
FROM asclena.cleaned_med_recon c
LEFT JOIN asclena.cleaned_ed_stays e
  ON c.stay_id = e.stay_id
WHERE e.stay_id IS NULL;
