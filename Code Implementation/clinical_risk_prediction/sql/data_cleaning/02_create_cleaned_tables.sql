-- Asclena clinical cleaning pipeline
-- Step 3-4: clean full raw tables and save cleaned staging tables.
--
-- This script reads asclena raw tables and writes only asclena.cleaned_* tables
-- plus the helper text-cleaning function and downstream exclusion manifest.

BEGIN;

CREATE OR REPLACE FUNCTION asclena.clean_text_basic(value TEXT)
RETURNS TEXT
LANGUAGE SQL
IMMUTABLE
AS $$
  SELECT NULLIF(regexp_replace(btrim(value), '[[:space:]]+', ' ', 'g'), '')
$$;

DROP TABLE IF EXISTS asclena.cleaned_med_recon;
DROP TABLE IF EXISTS asclena.cleaned_pyxis;
DROP TABLE IF EXISTS asclena.cleaned_diagnosis;
DROP TABLE IF EXISTS asclena.cleaned_vital_sign;
DROP TABLE IF EXISTS asclena.cleaned_triage;
DROP TABLE IF EXISTS asclena.cleaned_ed_stays;
DROP TABLE IF EXISTS asclena.feature_engineering_exclusions;

CREATE TABLE asclena.cleaned_ed_stays AS
WITH ranked AS (
  SELECT
    e.*,
    ROW_NUMBER() OVER (
      PARTITION BY e.stay_id
      ORDER BY
        (
          (e.stay_id IS NOT NULL)::int
          + (e.subject_id IS NOT NULL)::int
          + (e.hadm_id IS NOT NULL)::int
          + (e.intime IS NOT NULL)::int
          + (e.outtime IS NOT NULL)::int
          + (e.gender IS NOT NULL)::int
          + (e.race IS NOT NULL)::int
          + (e.arrival_transport IS NOT NULL)::int
          + (e.disposition IS NOT NULL)::int
          + (e.created_at IS NOT NULL)::int
        ) DESC,
        e.created_at DESC NULLS LAST
    ) AS rn
  FROM asclena.ed_stays e
  WHERE e.stay_id IS NOT NULL
    AND e.subject_id IS NOT NULL
    AND e.intime IS NOT NULL
)
SELECT
  stay_id,
  subject_id,
  hadm_id,
  intime,
  outtime,
  CASE
    WHEN upper(asclena.clean_text_basic(gender)) IN ('M', 'MALE') THEN 'Male'
    WHEN upper(asclena.clean_text_basic(gender)) IN ('F', 'FEMALE') THEN 'Female'
    ELSE 'Unknown'
  END::varchar(20) AS gender,
  asclena.clean_text_basic(race) AS race,
  asclena.clean_text_basic(arrival_transport) AS arrival_transport,
  asclena.clean_text_basic(disposition) AS disposition,
  created_at
FROM ranked
WHERE rn = 1;

ALTER TABLE asclena.cleaned_ed_stays
ADD PRIMARY KEY (stay_id);

CREATE TABLE asclena.cleaned_triage AS
WITH normalized AS (
  SELECT
    t.triage_id,
    t.stay_id,
    t.subject_id,
    CASE WHEN t.temperature BETWEEN 90 AND 110 THEN t.temperature ELSE NULL END AS temperature,
    CASE WHEN t.heartrate BETWEEN 20 AND 250 THEN t.heartrate ELSE NULL END AS heartrate,
    CASE WHEN t.resprate BETWEEN 5 AND 80 THEN t.resprate ELSE NULL END AS resprate,
    CASE WHEN t.o2sat BETWEEN 50 AND 100 THEN t.o2sat ELSE NULL END AS o2sat,
    CASE WHEN t.sbp BETWEEN 50 AND 300 THEN t.sbp ELSE NULL END AS sbp,
    CASE WHEN t.dbp BETWEEN 20 AND 200 THEN t.dbp ELSE NULL END AS dbp,
    COALESCE(asclena.clean_text_basic(t.pain), 'Unknown') AS pain,
    CASE WHEN t.acuity BETWEEN 1 AND 5 THEN t.acuity ELSE NULL END AS acuity,
    COALESCE(asclena.clean_text_basic(t.chiefcomplaint), 'Unknown') AS chiefcomplaint,
    t.created_at
  FROM asclena.triage t
  JOIN asclena.cleaned_ed_stays e
    ON t.stay_id = e.stay_id
  WHERE t.stay_id IS NOT NULL
    AND t.subject_id IS NOT NULL
),
ranked AS (
  SELECT
    n.*,
    ROW_NUMBER() OVER (
      PARTITION BY n.stay_id
      ORDER BY n.created_at ASC NULLS LAST, n.triage_id ASC NULLS LAST
    ) AS rn
  FROM normalized n
)
SELECT
  stay_id,
  subject_id,
  temperature,
  heartrate,
  resprate,
  o2sat,
  sbp,
  dbp,
  pain,
  acuity,
  chiefcomplaint,
  (temperature IS NULL) AS temperature_missing,
  (heartrate IS NULL) AS heartrate_missing,
  (resprate IS NULL) AS resprate_missing,
  (o2sat IS NULL) AS o2sat_missing,
  (sbp IS NULL) AS sbp_missing,
  (dbp IS NULL) AS dbp_missing,
  created_at
FROM ranked
WHERE rn = 1;

CREATE UNIQUE INDEX idx_cleaned_triage_stay_id
ON asclena.cleaned_triage(stay_id);

CREATE TABLE asclena.cleaned_vital_sign AS
WITH normalized AS (
  SELECT
    v.vital_id,
    v.stay_id,
    v.subject_id,
    v.charttime,
    CASE WHEN v.temperature BETWEEN 90 AND 110 THEN v.temperature ELSE NULL END AS temperature,
    CASE WHEN v.heartrate BETWEEN 20 AND 250 THEN v.heartrate ELSE NULL END AS heartrate,
    CASE WHEN v.resprate BETWEEN 5 AND 80 THEN v.resprate ELSE NULL END AS resprate,
    CASE WHEN v.o2sat BETWEEN 50 AND 100 THEN v.o2sat ELSE NULL END AS o2sat,
    CASE WHEN v.sbp BETWEEN 50 AND 300 THEN v.sbp ELSE NULL END AS sbp,
    CASE WHEN v.dbp BETWEEN 20 AND 200 THEN v.dbp ELSE NULL END AS dbp,
    asclena.clean_text_basic(v.rhythm) AS rhythm,
    asclena.clean_text_basic(v.pain) AS pain,
    v.created_at
  FROM asclena.vital_sign v
  JOIN asclena.cleaned_ed_stays e
    ON v.stay_id = e.stay_id
  WHERE v.stay_id IS NOT NULL
    AND v.subject_id IS NOT NULL
),
measurement_rows AS (
  SELECT *
  FROM normalized
  WHERE temperature IS NOT NULL
     OR heartrate IS NOT NULL
     OR resprate IS NOT NULL
     OR o2sat IS NOT NULL
     OR sbp IS NOT NULL
     OR dbp IS NOT NULL
),
ranked AS (
  SELECT
    m.*,
    ROW_NUMBER() OVER (
      PARTITION BY
        m.stay_id,
        m.subject_id,
        m.charttime,
        m.temperature,
        m.heartrate,
        m.resprate,
        m.o2sat,
        m.sbp,
        m.dbp,
        m.rhythm,
        m.pain
      ORDER BY m.created_at ASC NULLS LAST, m.vital_id ASC NULLS LAST
    ) AS rn
  FROM measurement_rows m
)
SELECT
  stay_id,
  subject_id,
  charttime,
  temperature,
  heartrate,
  resprate,
  o2sat,
  sbp,
  dbp,
  rhythm,
  pain,
  created_at
FROM ranked
WHERE rn = 1;

CREATE INDEX idx_cleaned_vital_sign_stay_id
ON asclena.cleaned_vital_sign(stay_id);

CREATE INDEX idx_cleaned_vital_sign_charttime
ON asclena.cleaned_vital_sign(charttime);

CREATE TABLE asclena.cleaned_diagnosis AS
WITH normalized AS (
  SELECT
    d.diagnosis_id,
    d.stay_id,
    d.subject_id,
    d.seq_num,
    upper(asclena.clean_text_basic(d.icd_code)) AS icd_code,
    d.icd_version,
    asclena.clean_text_basic(d.icd_title) AS icd_title,
    d.created_at
  FROM asclena.diagnosis d
  JOIN asclena.cleaned_ed_stays e
    ON d.stay_id = e.stay_id
  WHERE d.stay_id IS NOT NULL
    AND d.subject_id IS NOT NULL
),
valid_rows AS (
  SELECT *
  FROM normalized
  WHERE icd_code IS NOT NULL
     OR icd_title IS NOT NULL
),
ranked AS (
  SELECT
    v.*,
    ROW_NUMBER() OVER (
      PARTITION BY
        v.stay_id,
        v.subject_id,
        v.seq_num,
        v.icd_code,
        v.icd_version,
        COALESCE(v.icd_title, 'Unknown diagnosis')
      ORDER BY v.created_at ASC NULLS LAST, v.diagnosis_id ASC NULLS LAST
    ) AS rn
  FROM valid_rows v
)
SELECT
  stay_id,
  subject_id,
  seq_num,
  icd_code,
  icd_version,
  COALESCE(icd_title, 'Unknown diagnosis') AS icd_title,
  created_at
FROM ranked
WHERE rn = 1;

CREATE INDEX idx_cleaned_diagnosis_stay_id
ON asclena.cleaned_diagnosis(stay_id);

CREATE INDEX idx_cleaned_diagnosis_icd_code
ON asclena.cleaned_diagnosis(icd_code);

CREATE TABLE asclena.cleaned_pyxis AS
WITH normalized AS (
  SELECT
    p.pyxis_id,
    p.stay_id,
    p.subject_id,
    p.charttime,
    asclena.clean_text_basic(p.med_rn) AS med_rn,
    asclena.clean_text_basic(p.name) AS name,
    asclena.clean_text_basic(p.gsn_rn) AS gsn_rn,
    asclena.clean_text_basic(p.gsn) AS gsn,
    p.created_at
  FROM asclena.pyxis p
  JOIN asclena.cleaned_ed_stays e
    ON p.stay_id = e.stay_id
  WHERE p.stay_id IS NOT NULL
    AND p.subject_id IS NOT NULL
),
valid_rows AS (
  SELECT *
  FROM normalized
  WHERE name IS NOT NULL
     OR gsn IS NOT NULL
     OR med_rn IS NOT NULL
),
ranked AS (
  SELECT
    v.*,
    ROW_NUMBER() OVER (
      PARTITION BY
        v.stay_id,
        v.subject_id,
        v.charttime,
        v.med_rn,
        v.name,
        v.gsn_rn,
        v.gsn
      ORDER BY v.created_at ASC NULLS LAST, v.pyxis_id ASC NULLS LAST
    ) AS rn
  FROM valid_rows v
)
SELECT
  stay_id,
  subject_id,
  charttime,
  med_rn,
  name,
  gsn_rn,
  gsn,
  created_at
FROM ranked
WHERE rn = 1;

CREATE INDEX idx_cleaned_pyxis_stay_id
ON asclena.cleaned_pyxis(stay_id);

CREATE INDEX idx_cleaned_pyxis_charttime
ON asclena.cleaned_pyxis(charttime);

CREATE TABLE asclena.cleaned_med_recon AS
WITH normalized AS (
  SELECT
    m.med_recon_id,
    m.stay_id,
    m.subject_id,
    m.charttime,
    asclena.clean_text_basic(m.name) AS name,
    asclena.clean_text_basic(m.gsn) AS gsn,
    asclena.clean_text_basic(m.ndc) AS ndc,
    asclena.clean_text_basic(m.etc_rn) AS etc_rn,
    asclena.clean_text_basic(m.etccode) AS etccode,
    asclena.clean_text_basic(m.etcdescription) AS etcdescription,
    m.created_at
  FROM asclena.med_recon m
  JOIN asclena.cleaned_ed_stays e
    ON m.stay_id = e.stay_id
  WHERE m.stay_id IS NOT NULL
    AND m.subject_id IS NOT NULL
),
valid_rows AS (
  SELECT *
  FROM normalized
  WHERE name IS NOT NULL
     OR gsn IS NOT NULL
     OR ndc IS NOT NULL
     OR etcdescription IS NOT NULL
),
ranked AS (
  SELECT
    v.*,
    ROW_NUMBER() OVER (
      PARTITION BY
        v.stay_id,
        v.subject_id,
        v.charttime,
        v.name,
        v.gsn,
        v.ndc,
        v.etc_rn,
        v.etccode,
        v.etcdescription
      ORDER BY v.created_at ASC NULLS LAST, v.med_recon_id ASC NULLS LAST
    ) AS rn
  FROM valid_rows v
)
SELECT
  stay_id,
  subject_id,
  charttime,
  name,
  gsn,
  ndc,
  etc_rn,
  etccode,
  etcdescription,
  created_at
FROM ranked
WHERE rn = 1;

CREATE INDEX idx_cleaned_med_recon_stay_id
ON asclena.cleaned_med_recon(stay_id);

CREATE INDEX idx_cleaned_med_recon_ndc
ON asclena.cleaned_med_recon(ndc);

CREATE TABLE asclena.feature_engineering_exclusions (
  source_table TEXT NOT NULL,
  column_name TEXT NOT NULL,
  exclude_from_feature_engineering BOOLEAN NOT NULL,
  exclude_from_model_training BOOLEAN NOT NULL,
  reason TEXT NOT NULL,
  handling_policy TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (source_table, column_name)
);

INSERT INTO asclena.feature_engineering_exclusions (
  source_table,
  column_name,
  exclude_from_feature_engineering,
  exclude_from_model_training,
  reason,
  handling_policy
)
VALUES (
  'asclena.cleaned_vital_sign',
  'rhythm',
  TRUE,
  TRUE,
  'High missingness and absence of a reliable replacement; imputing Normal would create fake clinical information.',
  'Keep the column in cleaned_vital_sign for schema consistency and auditability. Do not impute, encode, aggregate, or train on rhythm unless a later clinically validated feature policy is approved.'
);

ANALYZE asclena.cleaned_ed_stays;
ANALYZE asclena.cleaned_triage;
ANALYZE asclena.cleaned_vital_sign;
ANALYZE asclena.cleaned_diagnosis;
ANALYZE asclena.cleaned_pyxis;
ANALYZE asclena.cleaned_med_recon;
ANALYZE asclena.feature_engineering_exclusions;

COMMIT;
