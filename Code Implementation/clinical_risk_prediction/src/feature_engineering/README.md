# Asclena Feature Engineering Pipeline

This pipeline builds `asclena.patient_feature_store` from the cleaned full-data
tables. It does not train a model.

Input tables:

- `asclena.cleaned_ed_stays`
- `asclena.cleaned_triage`
- `asclena.cleaned_vital_sign`
- `asclena.cleaned_diagnosis`
- `asclena.cleaned_pyxis`
- `asclena.cleaned_med_recon`

Output table:

- `asclena.patient_feature_store`

Reports are written to `reports/feature_engineering/<run_id>/`:

- `00_feature_store_validation.csv`
- `01_feature_store_summary.csv`
- `02_feature_store_null_report.csv`

Run from `Code Implementation/clinical_risk_prediction`:

```bash
python src/feature_engineering/run_feature_engineering.py
```

`risk_target` is populated with the MVP outcome definition:

- `1` when `hadm_id` exists or disposition contains an admission, transfer, ICU,
  deceased, expired, or died signal.
- `0` otherwise.

Leakage columns such as `disposition`, `has_hadm_id`, and
`length_of_stay_hours` are kept for audit/reporting but excluded from the
current XGBoost feature matrix. `rhythm` is not engineered or used in this
feature store.
