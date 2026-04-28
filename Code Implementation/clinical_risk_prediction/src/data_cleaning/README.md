# Asclena PostgreSQL Data Cleaning Pipeline

This pipeline prepares the full raw clinical PostgreSQL dataset for later feature engineering.
It does **not** train a model and does **not** modify the original raw tables.

Default execution order:

1. Create safety backups of the full raw tables.
2. Profile the full raw tables.
3. Clean the full raw tables.
4. Save `asclena.cleaned_*` tables.
5. Validate cleaned tables.
6. Write readiness reports for feature engineering.

## Files

- `sql/data_cleaning/01_create_raw_backups.sql`
- `sql/data_cleaning/02_create_cleaned_tables.sql`
- `sql/data_cleaning/03_validation_queries.sql`
- `src/data_cleaning/run_cleaning_pipeline.py`

There is no sampling step. The pipeline does not create or read `sampled_*`
tables, and it does not use hash-based or random sampling.

## Python Dependencies

Install these in the project environment:

```bash
pip install -r requirements-data-cleaning.txt
```

## Run

From `Code Implementation/clinical_risk_prediction`:

```bash
python src/data_cleaning/run_cleaning_pipeline.py
```

Connection settings are read from `configs/db_config.yaml`. You can override
with either `ASCLENA_DATABASE_URL` or these environment variables:

- `ASCLENA_DB_HOST`
- `ASCLENA_DB_PORT`
- `ASCLENA_DB_NAME`
- `ASCLENA_DB_USER`
- `ASCLENA_DB_PASSWORD`

## Outputs

Database tables:

- `asclena.ed_stays_raw_backup`
- `asclena.triage_raw_backup`
- `asclena.vital_sign_raw_backup`
- `asclena.diagnosis_raw_backup`
- `asclena.pyxis_raw_backup`
- `asclena.med_recon_raw_backup`
- `asclena.cleaned_ed_stays`
- `asclena.cleaned_triage`
- `asclena.cleaned_vital_sign`
- `asclena.cleaned_diagnosis`
- `asclena.cleaned_pyxis`
- `asclena.cleaned_med_recon`
- `asclena.feature_engineering_exclusions`

CSV reports under `reports/data_cleaning/<run_id>/`:

- `00_raw_backup_validation.csv`
- `01_data_quality_before_cleaning.csv`
- `02_data_quality_after_cleaning.csv`
- `03_rows_removed_summary.csv`
- `04_null_handling_summary.csv`
- `05_invalid_values_fixed_summary.csv`
- `06_cleaned_validation.csv`
- `07_feature_exclusion_manifest.csv`

The cleaned tables preserve clinically meaningful NULLs and avoid categorical
encoding. They are intended as the input layer for feature engineering.

`asclena.cleaned_vital_sign.rhythm` is intentionally kept in the cleaned table
for schema consistency and auditability, with missing values preserved as NULL.
It must be excluded from feature engineering and model training unless a later
clinically validated policy is approved. Do not impute missing rhythm as
`Normal`, because that would create fake clinical information.
