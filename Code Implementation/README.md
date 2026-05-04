# Asclena Clinical Risk Prediction

## Project Overview

Asclena Clinical Risk Prediction is a production-grade medical AI pipeline that predicts near-term clinical escalation risk for emergency department (ED) stays. The system uses cleaned EHR-like encounter data and a calibrated XGBoost classifier to estimate the likelihood that an ED stay will result in admission, transfer, ICU-level care, or a death-related disposition.

### Problem Definition

The project addresses the clinical need to identify ED patients at high risk for escalation of care during or immediately after the current ED visit. By predicting this risk early, Asclena can prioritize workflow, allocate clinician review, and reduce delays for patients likely to require admission or critical care.

### Objective of the Model

The objective is to produce a calibrated probability score (`risk_score`) indicating the likelihood of severe ED outcome, and to classify each encounter into scaled risk labels and severity categories that support operational decision-making.

### Business / Clinical Value

- Early identification of patients likely to require admission or ICU transfer.
- Support clinical triage and patient disposition decisions.
- Improve care coordination by signaling high-risk cases before final disposition.
- Provide explainable risk evidence through top contributor analysis and clinical interpretation.

## Data Pipeline

### Data Sources

The pipeline consumes the following PostgreSQL tables from the `asclena` schema:

- `asclena.ed_stays`
- `asclena.triage`
- `asclena.vital_sign`
- `asclena.diagnosis`
- `asclena.pyxis`
- `asclena.med_recon`
- `asclena.risk_predictions` (for prior model history features)

### Cleaning and Preprocessing Steps

Data cleaning is performed in `sql/data_cleaning/02_create_cleaned_tables.sql` and executed by `src/data_cleaning/run_cleaning_pipeline.py`.

Key cleaning rules:

- `asclena.ed_stays` is deduplicated on `stay_id` and preserves the best row by completeness and timestamp.
- Free-text fields are whitespace-normalized and blank strings are converted to `NULL`.
- Gender values are normalized into `Male`, `Female`, or `Unknown`.
- Clinical observations outside plausible ranges are set to `NULL`.
- `triage` is reduced to the earliest valid triage record per stay.
- `vital_sign` retains only rows with at least one valid vital value and deduplicates exact measurement rows.
- Diagnosis, medication administration, and medication reconciliation rows are normalized, deduplicated, and filtered to retain clinically meaningful records.

The cleaned staging tables are:

- `asclena.cleaned_ed_stays`
- `asclena.cleaned_triage`
- `asclena.cleaned_vital_sign`
- `asclena.cleaned_diagnosis`
- `asclena.cleaned_pyxis`
- `asclena.cleaned_med_recon`

### Missing Data Handling

The pipeline preserves missingness rather than imputing early. Missing indicators are created at the feature-engineering layer for triage fields.

- `triage_*_missing` binary flags are created for missing triage measurements.
- Vital event count and missing-rate features preserve the completeness of time-series availability.
- Invalid or out-of-range values are converted to `NULL` during cleaning, allowing the model imputer to handle them at training time.

### Filtering Rules

- Only records with non-null `stay_id` and `subject_id` are retained.
- Triage and vital rows are only kept when they belong to cleaned ED stays.
- Diagnosis entries are kept only when at least one of `icd_code` or `icd_title` is present.
- Medication records are kept only when there is at least one meaningful medication identifier.

## Feature Engineering

Feature engineering is implemented in `sql/feature_engineering/01_create_patient_feature_store.sql` and materializes the training-ready table `asclena.patient_feature_store`.

The pipeline produces one row per ED stay and a binary training label `risk_target`.

### Feature Groups

The model features are grouped as follows:

- Demographic features
- Triage features
- Vital summary features
- Derived clinical features
- Temporal / trend features
- Event count features
- Missingness features
- Patient history features

> Note: There are no laboratory-derived features in the current XGBoost feature set. Lab-related raw tables may exist in the schema but are not included in this version.

### Demographic Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `gender_male` | One-hot indicator for male sex. | `cleaned_ed_stays.gender` | `1` if gender = 'Male', else `0` | Static |
| `gender_female` | One-hot indicator for female sex. | `cleaned_ed_stays.gender` | `1` if gender = 'Female', else `0` | Static |
| `gender_unknown` | One-hot indicator for unknown sex. | `cleaned_ed_stays.gender` | `1` if gender is neither Male nor Female | Static |

### Triage Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `triage_temperature` | Initial triage temperature measurement. | `cleaned_triage.temperature` | Direct value from earliest triage row | Static |
| `triage_heartrate` | Initial triage heart rate measurement. | `cleaned_triage.heartrate` | Direct value from earliest triage row | Static |
| `triage_resprate` | Initial triage respiratory rate measurement. | `cleaned_triage.resprate` | Direct value from earliest triage row | Static |
| `triage_o2sat` | Initial triage oxygen saturation. | `cleaned_triage.o2sat` | Direct value from earliest triage row | Static |
| `triage_sbp` | Initial systolic blood pressure. | `cleaned_triage.sbp` | Direct value from earliest triage row | Static |
| `triage_dbp` | Initial diastolic blood pressure. | `cleaned_triage.dbp` | Direct value from earliest triage row | Static |
| `acuity` | Emergency severity or triage acuity score. | `cleaned_triage.acuity` | Direct value from earliest triage row | Static |
| `triage_shock_index` | Shock index from triage vitals. | `cleaned_triage.heartrate`, `cleaned_triage.sbp` | `heartrate / sbp` when both present and `sbp > 0` | Static derived |

### Triage Missingness Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `triage_temperature_missing` | Triage temperature missing indicator. | `cleaned_triage.temperature` | `1` when temperature is null, else `0` | Static |
| `triage_heartrate_missing` | Triage heart rate missing indicator. | `cleaned_triage.heartrate` | `1` when heartrate is null, else `0` | Static |
| `triage_resprate_missing` | Triage respiratory rate missing indicator. | `cleaned_triage.resprate` | `1` when resprate is null, else `0` | Static |
| `triage_o2sat_missing` | Triage oxygen saturation missing indicator. | `cleaned_triage.o2sat` | `1` when o2sat is null, else `0` | Static |
| `triage_sbp_missing` | Triage systolic blood pressure missing indicator. | `cleaned_triage.sbp` | `1` when sbp is null, else `0` | Static |
| `triage_dbp_missing` | Triage diastolic blood pressure missing indicator. | `cleaned_triage.dbp` | `1` when dbp is null, else `0` | Static |
| `acuity_missing` | Triage acuity missing indicator. | `cleaned_triage.acuity` | `1` when acuity is null, else `0` | Static |

### Vital Summary Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `vital_row_count` | Number of valid vital sign rows for the stay. | `cleaned_vital_sign` | Count of rows after cleaning | Aggregated time-series |
| `temperature_mean` | Mean temperature across the encounter. | `cleaned_vital_sign.temperature` | AVG(temperature) | Aggregated time-series |
| `temperature_min` | Minimum temperature across the encounter. | `cleaned_vital_sign.temperature` | MIN(temperature) | Aggregated time-series |
| `temperature_max` | Maximum temperature across the encounter. | `cleaned_vital_sign.temperature` | MAX(temperature) | Aggregated time-series |
| `hr_mean` | Mean heart rate across the encounter. | `cleaned_vital_sign.heartrate` | AVG(heartrate) | Aggregated time-series |
| `hr_min` | Minimum heart rate across the encounter. | `cleaned_vital_sign.heartrate` | MIN(heartrate) | Aggregated time-series |
| `hr_max` | Maximum heart rate across the encounter. | `cleaned_vital_sign.heartrate` | MAX(heartrate) | Aggregated time-series |
| `rr_mean` | Mean respiratory rate across the encounter. | `cleaned_vital_sign.resprate` | AVG(resprate) | Aggregated time-series |
| `rr_min` | Minimum respiratory rate across the encounter. | `cleaned_vital_sign.resprate` | MIN(resprate) | Aggregated time-series |
| `rr_max` | Maximum respiratory rate across the encounter. | `cleaned_vital_sign.resprate` | MAX(resprate) | Aggregated time-series |
| `spo2_mean` | Mean oxygen saturation across the encounter. | `cleaned_vital_sign.o2sat` | AVG(o2sat) | Aggregated time-series |
| `spo2_min` | Minimum oxygen saturation across the encounter. | `cleaned_vital_sign.o2sat` | MIN(o2sat) | Aggregated time-series |
| `spo2_max` | Maximum oxygen saturation across the encounter. | `cleaned_vital_sign.o2sat` | MAX(o2sat) | Aggregated time-series |
| `sbp_mean` | Mean systolic blood pressure across the encounter. | `cleaned_vital_sign.sbp` | AVG(sbp) | Aggregated time-series |
| `sbp_min` | Minimum systolic blood pressure across the encounter. | `cleaned_vital_sign.sbp` | MIN(sbp) | Aggregated time-series |
| `sbp_max` | Maximum systolic blood pressure across the encounter. | `cleaned_vital_sign.sbp` | MAX(sbp) | Aggregated time-series |
| `dbp_mean` | Mean diastolic blood pressure across the encounter. | `cleaned_vital_sign.dbp` | AVG(dbp) | Aggregated time-series |
| `dbp_min` | Minimum diastolic blood pressure across the encounter. | `cleaned_vital_sign.dbp` | MIN(dbp) | Aggregated time-series |
| `dbp_max` | Maximum diastolic blood pressure across the encounter. | `cleaned_vital_sign.dbp` | MAX(dbp) | Aggregated time-series |

### Derived Clinical Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `shock_index` | Mean shock index across valid vital rows. | `cleaned_vital_sign.heartrate`, `cleaned_vital_sign.sbp` | AVG(heartrate / sbp) when both values exist and `sbp > 0` | Aggregated derived |
| `shock_index_max` | Maximum shock index across valid vital rows. | `cleaned_vital_sign.heartrate`, `cleaned_vital_sign.sbp` | MAX(heartrate / sbp) when both values exist and `sbp > 0` | Aggregated derived |

### Temporal / Trend Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `hr_slope` | Hourly change in heart rate over the encounter. | `cleaned_vital_sign.heartrate`, `cleaned_vital_sign.charttime` | `(last_hr - first_hr) / elapsed_hours` | Time-series derived |
| `bp_slope` | Hourly change in systolic blood pressure over the encounter. | `cleaned_vital_sign.sbp`, `cleaned_vital_sign.charttime` | `(last_sbp - first_sbp) / elapsed_hours` | Time-series derived |

### Event Count Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `tachycardia_count` | Count of vital rows with HR ≥ 120 bpm. | `cleaned_vital_sign.heartrate` | SUM(heartrate >= 120) | Aggregated event count |
| `hypotension_count` | Count of vital rows with SBP < 90 mmHg. | `cleaned_vital_sign.sbp` | SUM(sbp < 90) | Aggregated event count |
| `hypoxia_count` | Count of vital rows with SpO₂ < 92%. | `cleaned_vital_sign.o2sat` | SUM(o2sat < 92) | Aggregated event count |
| `fever_count` | Count of vital rows with temperature ≥ 100.4°F. | `cleaned_vital_sign.temperature` | SUM(temperature >= 100.4) | Aggregated event count |

### Vital Missingness Rate Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `temperature_missing_rate` | Fraction of vital rows missing temperature. | `cleaned_vital_sign.temperature` | `1 - temperature_count / vital_row_count` | Aggregated missingness |
| `heartrate_missing_rate` | Fraction of vital rows missing heart rate. | `cleaned_vital_sign.heartrate` | `1 - heartrate_count / vital_row_count` | Aggregated missingness |
| `resprate_missing_rate` | Fraction of vital rows missing respiratory rate. | `cleaned_vital_sign.resprate` | `1 - resprate_count / vital_row_count` | Aggregated missingness |
| `o2sat_missing_rate` | Fraction of vital rows missing oxygen saturation. | `cleaned_vital_sign.o2sat` | `1 - o2sat_count / vital_row_count` | Aggregated missingness |
| `sbp_missing_rate` | Fraction of vital rows missing systolic blood pressure. | `cleaned_vital_sign.sbp` | `1 - sbp_count / vital_row_count` | Aggregated missingness |
| `dbp_missing_rate` | Fraction of vital rows missing diastolic blood pressure. | `cleaned_vital_sign.dbp` | `1 - dbp_count / vital_row_count` | Aggregated missingness |

### Patient History Features

| Feature | Definition | Source | Computation | Static / Time-series |
|---|---|---|---|---|
| `prior_ed_visit_count` | Prior ED visits before current stay. | `cleaned_ed_stays.intime` | Count of prior stays for the same patient | Patient history |
| `prior_ed_visit_count_30d` | Prior ED visits within 30 days. | `cleaned_ed_stays.intime` | Count of prior stays with `intime >= current_intime - 30d` | Patient history |
| `prior_ed_visit_count_90d` | Prior ED visits within 90 days. | `cleaned_ed_stays.intime` | Count of prior stays with `intime >= current_intime - 90d` | Patient history |
| `time_since_last_ed_visit_days` | Days since the prior ED visit. | `cleaned_ed_stays.intime` | Interval from last prior `intime` to current `intime` | Patient history |
| `prior_admission_count` | Prior admissions/transfers before current stay. | `cleaned_ed_stays.hadm_id`, `cleaned_ed_stays.disposition` | Count of prior stays with admission/transfer/ICU/deceased signals | Patient history |
| `prior_admission_count_1y` | Prior admissions/transfers within 1 year. | `cleaned_ed_stays.hadm_id`, `cleaned_ed_stays.disposition` | Count of prior stays in the preceding 365 days with admission/transfer/ICU/deceased signals | Patient history |
| `prior_icu_or_death_count` | Prior ICU or death-related episodes. | `cleaned_ed_stays.disposition` | Count of prior stays where disposition contains ICU/DECEASED/EXPIRED/DIED | Patient history |
| `prior_cardiovascular_dx_count` | Prior cardiovascular diagnosis count. | `cleaned_diagnosis.icd_code` | Count of prior diagnosis rows matching ICD-9/10 cardiovascular prefixes | Patient history |
| `prior_respiratory_dx_count` | Prior respiratory diagnosis count. | `cleaned_diagnosis.icd_code` | Count of prior diagnosis rows matching ICD-9/10 respiratory prefixes | Patient history |
| `prior_endocrine_dx_count` | Prior endocrine diagnosis count. | `cleaned_diagnosis.icd_code` | Count of prior diagnosis rows matching ICD-9/10 endocrine prefixes | Patient history |
| `prior_renal_dx_count` | Prior renal diagnosis count. | `cleaned_diagnosis.icd_code` | Count of prior diagnosis rows matching ICD-9/10 renal prefixes | Patient history |
| `prior_distinct_diagnosis_count` | Distinct prior diagnostic codes count. | `cleaned_diagnosis.icd_code` | Count of unique prior ICD codes | Patient history |
| `prior_high_risk_prediction_count` | Prior high-risk model prediction count. | `risk_predictions.risk_label`, `risk_predictions.created_at` | Count of prior predictions labeled HIGH before current encounter | Patient history |
| `last_risk_score` | Most recent prior model risk score. | `risk_predictions.risk_score`, `risk_predictions.created_at` | Latest prior risk score before current encounter | Patient history |
| `avg_prior_risk_score` | Average prior model risk score. | `risk_predictions.risk_score` | Mean of prior risk scores before current encounter | Patient history |
| `max_prior_risk_score` | Maximum prior model risk score. | `risk_predictions.risk_score` | Max of prior risk scores before current encounter | Patient history |

## Model Details

The training pipeline is implemented in `src/modeling/train_xgboost_risk_model.py`.

### Model Type

- XGBoost classifier (`xgboost.XGBClassifier`) with binary logistic objective.
- Calibrated with scikit-learn `CalibratedClassifierCV`.

### Hyperparameters

Default training hyperparameters:

- `objective`: `binary:logistic`
- `eval_metric`: `auc`
- `n_estimators`: `300`
- `max_depth`: `4`
- `learning_rate`: `0.05`
- `subsample`: `0.8`
- `colsample_bytree`: `0.8`
- `scale_pos_weight`: negative_count / positive_count (computed from training set)
- `random_state`: `42`
- `early_stopping_rounds`: `50`

Calibration settings:

- default calibration method: `isotonic`
- alternative option: `sigmoid`

### Training Strategy

- Load the feature store from `asclena.patient_feature_store`.
- Select `MODEL_FEATURES` only, excluding identifiers and leakage columns.
- Impute numeric features using median imputation via `sklearn.impute.SimpleImputer`.
- Train on a stratified split of the data.
- Calibrate predicted probabilities on a held-out validation fold.
- Save the calibrated model artifact and evaluation reports.

### Split Strategy

- Default test split: `20%` of the dataset.
- The remaining `80%` is split again into training and validation with `20%` of that set reserved for calibration.
- Both splits are stratified by `risk_target`.

### Imbalance Handling

- `scale_pos_weight` is computed from the positive/negative balance in the training set.
- The calibration step also helps produce well-scaled probabilities.

### Model Artifact

The saved artifact includes:

- calibrated model
- raw XGBoost model for explanation
- imputer
- `feature_names`
- `classification_threshold`
- `risk_label_thresholds`
- `calibration_method`
- `model_name` and `model_version`

## Target Definition

### `risk_target`

`risk_target` is the binary outcome label used for training.

It is constructed in `sql/feature_engineering/01_create_patient_feature_store.sql` and is defined as `1` when any of the following are true for the current ED stay:

- `hadm_id` is not null (the patient was admitted)
- `disposition` contains `ADMIT`
- `disposition` contains `TRANSFER`
- `disposition` contains `ICU`
- `disposition` contains `DECEASED`, `EXPIRED`, or `DIED`

Otherwise, `risk_target` is `0`.

### Time Window

The label is an outcome of the current ED stay. The model is trained using features available at or shortly after triage and those derived from prior patient history before the current stay. No future clinical events from the current encounter are used as features.

## Evaluation

The pipeline computes and persists these evaluation metrics:

- `roc_auc` (ROC-AUC)
- `pr_auc` (precision-recall AUC)
- `brier_score`
- `accuracy`
- `precision`
- `recall`
- `f1_score`
- Confusion matrix counts (`tn`, `fp`, `fn`, `tp`)

A calibration curve image is also generated as `05_calibration_curve.png`.

### Results Summary

The repository does not hardcode final benchmark values, but the training process writes quantitative model performance reports to `reports/modeling/<model_version>/metrics.json` and `00_evaluation_metrics.csv`.

## Output Format

The model service is implemented in `src/serving/app.py` and exposes a FastAPI contract.

### Returned Prediction Fields

- `risk_score`: calibrated probability in `[0.0, 1.0]`
- `display_risk_score`: formatted probability string, with `>0.99` for very high risk
- `predicted_target`: binary decision using the classification threshold
- `risk_label`: `LOW`, `MODERATE`, or `HIGH`
- `severity_index`: Asclena severity bucket `1..6`
- `severity_label`: descriptive severity bucket label
- `severity_description`: narrative guidance for the predicted risk
- `severity_scale_name`: fixed string `Asclena Severity Index`
- `threshold_used`: numeric classification threshold used for binary prediction

### Probability Interpretation

- `risk_score` is the calibrated probability of the positive `risk_target` class.
- `predicted_target` is computed as `risk_score >= classification_threshold`.
- Default risk label thresholds are:
  - `LOW`: `[0.0, 0.4)`
  - `MODERATE`: `[0.4, 0.7)`
  - `HIGH`: `[0.7, 1.0]`

### Prediction API Envelope

A prediction response contains:

- request metadata (`request_id`, optional subject context)
- model metadata
- prediction result
- explanation payload (`top_contributors`, optional `feature_snapshot`)
- clinical interpretation
- contract version

The request schema requires a complete dictionary of model features, matching the current serving contract.

## Schema Mapping Safety Layer

This section enumerates training features and maps them to logical clinical entities for safe integration with the Asclena ERD.

### Logical Entity Mapping

- Patient / demographic: `gender_male`, `gender_female`, `gender_unknown`
- Encounter / triage: `triage_temperature`, `triage_heartrate`, `triage_resprate`, `triage_o2sat`, `triage_sbp`, `triage_dbp`, `acuity`, `triage_shock_index`
- Observation / vital summary: `temperature_*`, `hr_*`, `rr_*`, `spo2_*`, `sbp_*`, `dbp_*`, `shock_index*`, `hr_slope`, `bp_slope`, `tachycardia_count`, `hypotension_count`, `hypoxia_count`, `fever_count`
- Missingness / data quality: `triage_*_missing`, `acuity_missing`, `*_missing_rate`
- Patient history: `prior_*` counts and prior risk score aggregations

### Ambiguous or Derived Features

- `triage_shock_index`: derived from heart rate and systolic blood pressure; requires both observations to be mapped in the same encounter.
- `hr_slope` and `bp_slope`: depend on first and last valid timestamped observations and require consistent charttime ordering.
- `last_risk_score`, `avg_prior_risk_score`, `max_prior_risk_score`, `prior_high_risk_prediction_count`: derived from the internal `risk_predictions` table and not from external clinical data.
- `prior_*_diagnosis_count`: derived from prior diagnosis history using ICD prefix matching; mapping must align with the destination ERD’s diagnosis classification rules.

### Potential Mismatch Points

- Disposition-based label logic may not align exactly with future ERD disposition semantics. The current label uses substring matching against `ADMIT`, `TRANSFER`, `ICU`, `DECEASED`, `EXPIRED`, and `DIED`.
- The model excludes raw categorical fields such as `race` and `arrival_transport` from training, even though they are present in cleaned data. Their availability should not be presumed for model inference.
- Vital sign row aggregation assumes the current stay’s full vital series is available. If an integration source provides only early triage vitals, the feature distribution may differ dramatically.
- History features depend on prior encounters from the same patient. If the ERD does not reliably link `subject_id` across stays, prior-history features will be incomplete.

## How to Run

### Full Pipeline

```bash
python -m src.pipeline.run_clinical_risk_pipeline --stage full
```

### Individual Stages

```bash
python -m src.pipeline.run_clinical_risk_pipeline --stage clean
python -m src.pipeline.run_clinical_risk_pipeline --stage features
python -m src.pipeline.run_clinical_risk_pipeline --stage train
```

### API Server

Start the FastAPI server through the existing application entrypoint or a compatible ASGI runner. The model artifact path is loaded from configuration in `src/serving/config.py`.

## Review Notes

- The current production-ready model contract is implemented in `src/serving/feature_contract.py` and enforced by the API validator.
- Feature selection is explicitly captured in `sql/feature_engineering/01_create_patient_feature_store.sql` and `sql/modeling/01_create_risk_predictions.sql`.
- The pipeline is structured to separate cleaning, feature engineering, and model training into distinct stages.
