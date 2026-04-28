-- Asclena feature engineering pipeline
-- Build one ML-ready feature row per ED stay from cleaned full-dataset tables.
--
-- This script does not train a model.
-- risk_target is the MVP outcome label:
-- high risk when the ED stay led to admission/transfer/ICU/death signal.

BEGIN;

DROP TABLE IF EXISTS asclena.patient_feature_store;

CREATE TABLE asclena.patient_feature_store AS
WITH ed AS (
  SELECT
    stay_id,
    subject_id,
    hadm_id,
    intime,
    outtime,
    gender,
    race,
    arrival_transport,
    disposition,
    CASE WHEN hadm_id IS NOT NULL THEN 1 ELSE 0 END AS has_hadm_id,
    EXTRACT(EPOCH FROM (outtime - intime)) / 3600.0 AS length_of_stay_hours,
    CASE WHEN gender = 'Male' THEN 1 ELSE 0 END AS gender_male,
    CASE WHEN gender = 'Female' THEN 1 ELSE 0 END AS gender_female,
    CASE WHEN gender = 'Unknown' THEN 1 ELSE 0 END AS gender_unknown
  FROM asclena.cleaned_ed_stays
),
triage AS (
  SELECT
    stay_id,
    temperature AS triage_temperature,
    heartrate AS triage_heartrate,
    resprate AS triage_resprate,
    o2sat AS triage_o2sat,
    sbp AS triage_sbp,
    dbp AS triage_dbp,
    acuity,
    temperature_missing::int AS triage_temperature_missing,
    heartrate_missing::int AS triage_heartrate_missing,
    resprate_missing::int AS triage_resprate_missing,
    o2sat_missing::int AS triage_o2sat_missing,
    sbp_missing::int AS triage_sbp_missing,
    dbp_missing::int AS triage_dbp_missing,
    CASE WHEN acuity IS NULL THEN 1 ELSE 0 END AS acuity_missing,
    CASE WHEN pain = 'Unknown' THEN 1 ELSE 0 END AS triage_pain_unknown,
    CASE WHEN chiefcomplaint = 'Unknown' THEN 1 ELSE 0 END AS chiefcomplaint_unknown,
    CASE
      WHEN heartrate IS NOT NULL AND sbp IS NOT NULL AND sbp > 0
        THEN heartrate / sbp
      ELSE NULL
    END AS triage_shock_index
  FROM asclena.cleaned_triage
),
vital_base AS (
  SELECT
    stay_id,
    charttime,
    temperature,
    heartrate,
    resprate,
    o2sat,
    sbp,
    dbp,
    pain
  FROM asclena.cleaned_vital_sign
),
vital_agg AS (
  SELECT
    stay_id,
    COUNT(*) AS vital_row_count,
    COUNT(temperature) AS temperature_count,
    COUNT(heartrate) AS heartrate_count,
    COUNT(resprate) AS resprate_count,
    COUNT(o2sat) AS o2sat_count,
    COUNT(sbp) AS sbp_count,
    COUNT(dbp) AS dbp_count,
    AVG(temperature) AS temperature_mean,
    MIN(temperature) AS temperature_min,
    MAX(temperature) AS temperature_max,
    AVG(heartrate) AS hr_mean,
    MIN(heartrate) AS hr_min,
    MAX(heartrate) AS hr_max,
    AVG(resprate) AS rr_mean,
    MIN(resprate) AS rr_min,
    MAX(resprate) AS rr_max,
    AVG(o2sat) AS spo2_mean,
    MIN(o2sat) AS spo2_min,
    MAX(o2sat) AS spo2_max,
    AVG(sbp) AS sbp_mean,
    MIN(sbp) AS sbp_min,
    MAX(sbp) AS sbp_max,
    AVG(dbp) AS dbp_mean,
    MIN(dbp) AS dbp_min,
    MAX(dbp) AS dbp_max,
    AVG(CASE WHEN heartrate IS NOT NULL AND sbp IS NOT NULL AND sbp > 0 THEN heartrate / sbp END) AS shock_index_mean,
    MAX(CASE WHEN heartrate IS NOT NULL AND sbp IS NOT NULL AND sbp > 0 THEN heartrate / sbp END) AS shock_index_max,
    SUM(CASE WHEN heartrate >= 120 THEN 1 ELSE 0 END) AS tachycardia_count,
    SUM(CASE WHEN sbp < 90 THEN 1 ELSE 0 END) AS hypotension_count,
    SUM(CASE WHEN o2sat < 92 THEN 1 ELSE 0 END) AS hypoxia_count,
    SUM(CASE WHEN temperature >= 100.4 THEN 1 ELSE 0 END) AS fever_count,
    SUM(CASE WHEN pain IS NULL THEN 1 ELSE 0 END) AS vital_pain_missing_count,
    MIN(charttime) AS first_vital_time,
    MAX(charttime) AS last_vital_time
  FROM vital_base
  GROUP BY stay_id
),
vital_first_last AS (
  SELECT
    stay_id,
    (ARRAY_AGG(heartrate ORDER BY charttime ASC) FILTER (WHERE heartrate IS NOT NULL))[1] AS hr_first,
    (ARRAY_AGG(heartrate ORDER BY charttime DESC) FILTER (WHERE heartrate IS NOT NULL))[1] AS hr_last,
    (ARRAY_AGG(charttime ORDER BY charttime ASC) FILTER (WHERE heartrate IS NOT NULL))[1] AS hr_first_time,
    (ARRAY_AGG(charttime ORDER BY charttime DESC) FILTER (WHERE heartrate IS NOT NULL))[1] AS hr_last_time,
    (ARRAY_AGG(sbp ORDER BY charttime ASC) FILTER (WHERE sbp IS NOT NULL))[1] AS sbp_first,
    (ARRAY_AGG(sbp ORDER BY charttime DESC) FILTER (WHERE sbp IS NOT NULL))[1] AS sbp_last,
    (ARRAY_AGG(charttime ORDER BY charttime ASC) FILTER (WHERE sbp IS NOT NULL))[1] AS sbp_first_time,
    (ARRAY_AGG(charttime ORDER BY charttime DESC) FILTER (WHERE sbp IS NOT NULL))[1] AS sbp_last_time
  FROM vital_base
  GROUP BY stay_id
),
vital_features AS (
  SELECT
    a.*,
    f.hr_first,
    f.hr_last,
    f.sbp_first,
    f.sbp_last,
    CASE
      WHEN f.hr_first IS NOT NULL
       AND f.hr_last IS NOT NULL
       AND EXTRACT(EPOCH FROM (f.hr_last_time - f.hr_first_time)) <> 0
        THEN (f.hr_last - f.hr_first) / (EXTRACT(EPOCH FROM (f.hr_last_time - f.hr_first_time)) / 3600.0)
      ELSE NULL
    END AS hr_slope,
    CASE
      WHEN f.sbp_first IS NOT NULL
       AND f.sbp_last IS NOT NULL
       AND EXTRACT(EPOCH FROM (f.sbp_last_time - f.sbp_first_time)) <> 0
        THEN (f.sbp_last - f.sbp_first) / (EXTRACT(EPOCH FROM (f.sbp_last_time - f.sbp_first_time)) / 3600.0)
      ELSE NULL
    END AS bp_slope,
    CASE WHEN a.vital_row_count > 0 THEN 1.0 - (a.temperature_count::numeric / a.vital_row_count) ELSE NULL END AS temperature_missing_rate,
    CASE WHEN a.vital_row_count > 0 THEN 1.0 - (a.heartrate_count::numeric / a.vital_row_count) ELSE NULL END AS heartrate_missing_rate,
    CASE WHEN a.vital_row_count > 0 THEN 1.0 - (a.resprate_count::numeric / a.vital_row_count) ELSE NULL END AS resprate_missing_rate,
    CASE WHEN a.vital_row_count > 0 THEN 1.0 - (a.o2sat_count::numeric / a.vital_row_count) ELSE NULL END AS o2sat_missing_rate,
    CASE WHEN a.vital_row_count > 0 THEN 1.0 - (a.sbp_count::numeric / a.vital_row_count) ELSE NULL END AS sbp_missing_rate,
    CASE WHEN a.vital_row_count > 0 THEN 1.0 - (a.dbp_count::numeric / a.vital_row_count) ELSE NULL END AS dbp_missing_rate
  FROM vital_agg a
  LEFT JOIN vital_first_last f
    ON a.stay_id = f.stay_id
),
diagnosis_features AS (
  SELECT
    stay_id,
    COUNT(*) AS diagnosis_count,
    COUNT(DISTINCT icd_code) AS distinct_diagnosis_count,
    MAX(CASE WHEN icd_version = 9 THEN 1 ELSE 0 END) AS has_icd9,
    MAX(CASE WHEN icd_version = 10 THEN 1 ELSE 0 END) AS has_icd10,
    MAX(CASE WHEN LEFT(icd_code, 1) = 'I' OR icd_code ~ '^(39|40|41|42|43|44|45)' THEN 1 ELSE 0 END) AS has_cardiovascular_dx,
    MAX(CASE WHEN LEFT(icd_code, 1) = 'J' OR icd_code ~ '^(46|47|48|49|50|51)' THEN 1 ELSE 0 END) AS has_respiratory_dx,
    MAX(CASE WHEN LEFT(icd_code, 1) = 'E' OR icd_code ~ '^(24|25|26|27)' THEN 1 ELSE 0 END) AS has_endocrine_dx,
    MAX(CASE WHEN LEFT(icd_code, 1) = 'N' OR icd_code ~ '^(58|59|60|61|62)' THEN 1 ELSE 0 END) AS has_renal_dx,
    COUNT(*) AS comorbidity_score
  FROM asclena.cleaned_diagnosis
  GROUP BY stay_id
),
pyxis_features AS (
  SELECT
    stay_id,
    COUNT(*) AS pyxis_med_count,
    COUNT(DISTINCT name) AS pyxis_distinct_med_count,
    COUNT(DISTINCT gsn) FILTER (WHERE gsn IS NOT NULL) AS pyxis_distinct_gsn_count,
    SUM(CASE WHEN gsn IS NULL THEN 1 ELSE 0 END) AS pyxis_missing_gsn_count
  FROM asclena.cleaned_pyxis
  GROUP BY stay_id
),
med_recon_features AS (
  SELECT
    stay_id,
    COUNT(*) AS med_recon_count,
    COUNT(DISTINCT name) AS med_recon_distinct_med_count,
    COUNT(DISTINCT ndc) FILTER (WHERE ndc IS NOT NULL) AS med_recon_distinct_ndc_count,
    SUM(CASE WHEN etccode IS NULL THEN 1 ELSE 0 END) AS med_recon_missing_etccode_count
  FROM asclena.cleaned_med_recon
  GROUP BY stay_id
)
SELECT
  ed.stay_id,
  ed.subject_id,
  ed.gender,
  ed.gender_male,
  ed.gender_female,
  ed.gender_unknown,
  ed.race,
  ed.arrival_transport,
  ed.disposition,
  ed.has_hadm_id,
  ed.length_of_stay_hours::numeric(10,2) AS length_of_stay_hours,
  triage.triage_temperature,
  triage.triage_heartrate,
  triage.triage_resprate,
  triage.triage_o2sat,
  triage.triage_sbp,
  triage.triage_dbp,
  triage.acuity,
  triage.triage_temperature_missing,
  triage.triage_heartrate_missing,
  triage.triage_resprate_missing,
  triage.triage_o2sat_missing,
  triage.triage_sbp_missing,
  triage.triage_dbp_missing,
  triage.acuity_missing,
  triage.triage_pain_unknown,
  triage.chiefcomplaint_unknown,
  triage.triage_shock_index::numeric(8,4) AS triage_shock_index,
  COALESCE(vital.vital_row_count, 0) AS vital_row_count,
  COALESCE(vital.temperature_count, 0) AS temperature_count,
  COALESCE(vital.heartrate_count, 0) AS heartrate_count,
  COALESCE(vital.resprate_count, 0) AS resprate_count,
  COALESCE(vital.o2sat_count, 0) AS o2sat_count,
  COALESCE(vital.sbp_count, 0) AS sbp_count,
  COALESCE(vital.dbp_count, 0) AS dbp_count,
  vital.temperature_mean::numeric(8,3) AS temperature_mean,
  vital.temperature_min,
  vital.temperature_max,
  vital.hr_mean::numeric(8,3) AS hr_mean,
  vital.hr_min,
  vital.hr_max,
  vital.rr_mean::numeric(8,3) AS rr_mean,
  vital.rr_min,
  vital.rr_max,
  vital.spo2_mean::numeric(8,3) AS spo2_mean,
  vital.spo2_min,
  vital.spo2_max,
  vital.sbp_mean::numeric(8,3) AS sbp_mean,
  vital.sbp_min,
  vital.sbp_max,
  vital.dbp_mean::numeric(8,3) AS dbp_mean,
  vital.dbp_min,
  vital.dbp_max,
  vital.shock_index_mean::numeric(8,4) AS shock_index,
  vital.shock_index_max::numeric(8,4) AS shock_index_max,
  vital.hr_first,
  vital.hr_last,
  vital.hr_slope::numeric(8,4) AS hr_slope,
  vital.sbp_first,
  vital.sbp_last,
  vital.bp_slope::numeric(8,4) AS bp_slope,
  vital.temperature_missing_rate::numeric(8,4) AS temperature_missing_rate,
  vital.heartrate_missing_rate::numeric(8,4) AS heartrate_missing_rate,
  vital.resprate_missing_rate::numeric(8,4) AS resprate_missing_rate,
  vital.o2sat_missing_rate::numeric(8,4) AS o2sat_missing_rate,
  vital.sbp_missing_rate::numeric(8,4) AS sbp_missing_rate,
  vital.dbp_missing_rate::numeric(8,4) AS dbp_missing_rate,
  COALESCE(vital.tachycardia_count, 0) AS tachycardia_count,
  COALESCE(vital.hypotension_count, 0) AS hypotension_count,
  COALESCE(vital.hypoxia_count, 0) AS hypoxia_count,
  COALESCE(vital.fever_count, 0) AS fever_count,
  COALESCE(vital.vital_pain_missing_count, 0) AS vital_pain_missing_count,
  COALESCE(dx.diagnosis_count, 0) AS diagnosis_count,
  COALESCE(dx.distinct_diagnosis_count, 0) AS distinct_diagnosis_count,
  COALESCE(dx.has_icd9, 0) AS has_icd9,
  COALESCE(dx.has_icd10, 0) AS has_icd10,
  COALESCE(dx.has_cardiovascular_dx, 0) AS has_cardiovascular_dx,
  COALESCE(dx.has_respiratory_dx, 0) AS has_respiratory_dx,
  COALESCE(dx.has_endocrine_dx, 0) AS has_endocrine_dx,
  COALESCE(dx.has_renal_dx, 0) AS has_renal_dx,
  COALESCE(dx.comorbidity_score, 0) AS comorbidity_score,
  COALESCE(pyxis.pyxis_med_count, 0) AS pyxis_med_count,
  COALESCE(pyxis.pyxis_distinct_med_count, 0) AS pyxis_distinct_med_count,
  COALESCE(pyxis.pyxis_distinct_gsn_count, 0) AS pyxis_distinct_gsn_count,
  COALESCE(pyxis.pyxis_missing_gsn_count, 0) AS pyxis_missing_gsn_count,
  COALESCE(recon.med_recon_count, 0) AS med_recon_count,
  COALESCE(recon.med_recon_distinct_med_count, 0) AS med_recon_distinct_med_count,
  COALESCE(recon.med_recon_distinct_ndc_count, 0) AS med_recon_distinct_ndc_count,
  COALESCE(recon.med_recon_missing_etccode_count, 0) AS med_recon_missing_etccode_count,
  (
    COALESCE(pyxis.pyxis_med_count, 0)
    + COALESCE(recon.med_recon_count, 0)
  ) AS medication_intensity_score,
  CASE
    WHEN ed.hadm_id IS NOT NULL THEN 1
    WHEN POSITION('ADMIT' IN UPPER(COALESCE(ed.disposition, ''))) > 0 THEN 1
    WHEN POSITION('TRANSFER' IN UPPER(COALESCE(ed.disposition, ''))) > 0 THEN 1
    WHEN POSITION('ICU' IN UPPER(COALESCE(ed.disposition, ''))) > 0 THEN 1
    WHEN POSITION('DECEASED' IN UPPER(COALESCE(ed.disposition, ''))) > 0 THEN 1
    WHEN POSITION('EXPIRED' IN UPPER(COALESCE(ed.disposition, ''))) > 0 THEN 1
    WHEN POSITION('DIED' IN UPPER(COALESCE(ed.disposition, ''))) > 0 THEN 1
    ELSE 0
  END AS risk_target,
  CURRENT_TIMESTAMP AS updated_at
FROM ed
LEFT JOIN triage
  ON ed.stay_id = triage.stay_id
LEFT JOIN vital_features vital
  ON ed.stay_id = vital.stay_id
LEFT JOIN diagnosis_features dx
  ON ed.stay_id = dx.stay_id
LEFT JOIN pyxis_features pyxis
  ON ed.stay_id = pyxis.stay_id
LEFT JOIN med_recon_features recon
  ON ed.stay_id = recon.stay_id;

ALTER TABLE asclena.patient_feature_store
ADD PRIMARY KEY (stay_id);

ALTER TABLE asclena.patient_feature_store
ADD CONSTRAINT fk_feature_store_cleaned_stay
FOREIGN KEY (stay_id)
REFERENCES asclena.cleaned_ed_stays(stay_id)
ON DELETE CASCADE;

CREATE INDEX idx_patient_feature_store_subject_id
ON asclena.patient_feature_store(subject_id);

CREATE INDEX idx_patient_feature_store_risk_target
ON asclena.patient_feature_store(risk_target);

DROP TABLE IF EXISTS asclena.model_feature_manifest;

CREATE TABLE asclena.model_feature_manifest (
  feature_name TEXT PRIMARY KEY,
  include_in_current_xgboost BOOLEAN NOT NULL,
  feature_role TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO asclena.model_feature_manifest (feature_name, include_in_current_xgboost, feature_role, reason)
VALUES
  ('stay_id', FALSE, 'identifier', 'Tracking only; identifiers are excluded from model training.'),
  ('subject_id', FALSE, 'identifier', 'Tracking only; identifiers are excluded from model training.'),
  ('race', FALSE, 'analysis_only', 'Raw categorical text not included in initial numeric feature matrix.'),
  ('arrival_transport', FALSE, 'analysis_only', 'Raw categorical text not included in initial numeric feature matrix.'),
  ('disposition', FALSE, 'target_leakage', 'Disposition is used to define risk_target and must not be used as a predictor.'),
  ('has_hadm_id', FALSE, 'target_leakage', 'Admission indicator is used to define risk_target and must not be used as a predictor.'),
  ('length_of_stay_hours', FALSE, 'target_leakage', 'Length of stay is an outcome/post-arrival feature and may leak severity/outcome.'),
  ('diagnosis_count', FALSE, 'future_feature', 'Diagnosis-derived features may not be available at early triage time.'),
  ('distinct_diagnosis_count', FALSE, 'future_feature', 'Diagnosis-derived features may not be available at early triage time.'),
  ('comorbidity_score', FALSE, 'future_feature', 'Diagnosis-derived feature held out from initial MVP model.'),
  ('pyxis_med_count', FALSE, 'future_feature', 'Medication administration may occur after triage and can leak care intensity.'),
  ('pyxis_distinct_med_count', FALSE, 'future_feature', 'Medication administration may occur after triage and can leak care intensity.'),
  ('med_recon_count', FALSE, 'future_feature', 'Medication reconciliation availability may vary by workflow.'),
  ('med_recon_distinct_med_count', FALSE, 'future_feature', 'Medication reconciliation availability may vary by workflow.'),
  ('medication_intensity_score', FALSE, 'future_feature', 'Medication intensity can leak downstream care decisions.'),
  ('risk_target', FALSE, 'target', 'Binary training label, never a predictor.');

ANALYZE asclena.patient_feature_store;
ANALYZE asclena.model_feature_manifest;

COMMIT;
