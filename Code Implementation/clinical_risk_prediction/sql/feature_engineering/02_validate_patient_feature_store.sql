-- Validation queries for asclena.patient_feature_store.

SELECT COUNT(*) AS cleaned_ed_stays_count
FROM asclena.cleaned_ed_stays;

SELECT COUNT(*) AS patient_feature_store_count
FROM asclena.patient_feature_store;

SELECT COUNT(DISTINCT stay_id) AS patient_feature_store_distinct_stay_id
FROM asclena.patient_feature_store;

SELECT COUNT(*) AS null_identifier_rows
FROM asclena.patient_feature_store
WHERE stay_id IS NULL OR subject_id IS NULL;

SELECT COUNT(*) AS duplicate_stay_id_groups
FROM (
  SELECT stay_id
  FROM asclena.patient_feature_store
  GROUP BY stay_id
  HAVING COUNT(*) > 1
) d;

SELECT COUNT(*) AS orphan_feature_rows
FROM asclena.patient_feature_store f
LEFT JOIN asclena.cleaned_ed_stays e
  ON f.stay_id = e.stay_id
WHERE e.stay_id IS NULL;

SELECT COUNT(*) AS null_risk_target_rows
FROM asclena.patient_feature_store
WHERE risk_target IS NULL;

SELECT risk_target, COUNT(*) AS row_count
FROM asclena.patient_feature_store
GROUP BY risk_target
ORDER BY risk_target;

SELECT *
FROM asclena.model_feature_manifest
ORDER BY include_in_current_xgboost DESC, feature_name;

SELECT COUNT(*) AS negative_history_count_rows
FROM asclena.patient_feature_store
WHERE prior_ed_visit_count < 0
   OR prior_ed_visit_count_30d < 0
   OR prior_ed_visit_count_90d < 0
   OR prior_admission_count < 0
   OR prior_admission_count_1y < 0
   OR prior_icu_or_death_count < 0
   OR prior_cardiovascular_dx_count < 0
   OR prior_respiratory_dx_count < 0
   OR prior_endocrine_dx_count < 0
   OR prior_renal_dx_count < 0
   OR prior_distinct_diagnosis_count < 0
   OR prior_high_risk_prediction_count < 0;

SELECT COUNT(*) AS negative_time_since_last_visit_rows
FROM asclena.patient_feature_store
WHERE time_since_last_ed_visit_days IS NOT NULL
  AND time_since_last_ed_visit_days < 0;

SELECT COUNT(*) AS first_visit_history_mismatch_rows
FROM asclena.patient_feature_store
WHERE time_since_last_ed_visit_days IS NULL
  AND prior_ed_visit_count <> 0;

SELECT COUNT(*) AS risk_prediction_invalid_severity_rows
FROM asclena.risk_predictions
WHERE severity_index IS NOT NULL
  AND (
    severity_index < 1
    OR severity_index > 6
    OR severity_label IS NULL
    OR severity_description IS NULL
    OR severity_scale_name IS NULL
  );

SELECT
  COUNT(*) AS rows_with_any_vital_data,
  COUNT(*) FILTER (WHERE vital_row_count = 0) AS rows_without_vital_data,
  COUNT(*) FILTER (WHERE diagnosis_count = 0) AS rows_without_diagnosis,
  COUNT(*) FILTER (WHERE pyxis_med_count = 0) AS rows_without_pyxis,
  COUNT(*) FILTER (WHERE med_recon_count = 0) AS rows_without_med_recon
FROM asclena.patient_feature_store;
