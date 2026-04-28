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

SELECT
  COUNT(*) AS rows_with_any_vital_data,
  COUNT(*) FILTER (WHERE vital_row_count = 0) AS rows_without_vital_data,
  COUNT(*) FILTER (WHERE diagnosis_count = 0) AS rows_without_diagnosis,
  COUNT(*) FILTER (WHERE pyxis_med_count = 0) AS rows_without_pyxis,
  COUNT(*) FILTER (WHERE med_recon_count = 0) AS rows_without_med_recon
FROM asclena.patient_feature_store;
