# Asclena Clinical Risk Model API

## Purpose

This FastAPI service exposes the trained XGBoost ED risk model as a stateless inference API.

For the full offline-to-online lifecycle, see:

[clinical_risk_pipeline.md](</home/fares-ashraf/Fares Ashraf/Asclena/Asclena-AI/Code Implementation/clinical_risk_prediction/docs/clinical_risk_pipeline.md>)

The service is designed for:

- separate deployment from the main Asclena AI application
- safe integration through a strict input and output contract
- future alignment with EHR and FHIR adapters without requiring FHIR today

The service does **not** read PostgreSQL during inference.
All required model features must be sent in the request payload.

V2 note:

- the model remains visit-based at inference time
- the payload now includes additive patient-history features engineered offline before the current ED stay
- the response also includes the internal `Asclena Severity Index (ASI)` derived from `risk_score`

## Deployment Structure

```text
clinical_risk_prediction/
  models/
    asclena_xgboost_risk_v2_<model_version>.joblib
  src/
    serving/
      __init__.py
      app.py
      config.py
      feature_contract.py
      predictor.py
      schemas.py
  docs/
    risk_model_fastapi.md
  requirements-api.txt
```

## Run

From `Code Implementation/clinical_risk_prediction`:

```bash
pip install -r requirements-api.txt
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Optional environment variable:

```bash
export ASCLENA_RISK_MODEL_PATH=/absolute/path/to/model.joblib
```

If `ASCLENA_RISK_MODEL_PATH` is not set, the service loads the latest `.joblib` file from `models/`.

## Stateless Contract Design

The API is intentionally normalized around:

- `subject`: pass-through identifiers for integration tracing
- `features`: complete model-ready feature map

This keeps the service stateless and independent from any database session or EHR connector.

In the future, a FHIR adapter can map:

- `Patient`
- `Encounter`
- `Observation`
- `Condition`
- `MedicationStatement`
- `MedicationAdministration`

into the same normalized `features` object before calling this API.

## Endpoints

### `GET /health`

Simple liveness probe.

Response:

```json
{
  "status": "ok"
}
```

### `GET /v1/health`

Versioned health endpoint.

Response:

```json
{
  "status": "ok",
  "contract_version": "2026-04-29"
}
```

### `GET /v1/model`

Returns loaded model metadata.

Response fields:

- `model_name`
- `model_version`
- `feature_count`
- `classification_threshold`
- `risk_label_thresholds`
- `calibration_method`
- `contract_version`

### `GET /v1/contract`

Returns the full machine-readable input contract.

Response includes:

- all required feature names
- category of each feature
- expected value type
- description
- future FHIR source hint
- excluded columns that must never be sent as predictors

This endpoint should be treated as the source of truth for integration.

### `POST /v1/predict`

Single stateless prediction request.

Request:

```json
{
  "request_id": "req-001",
  "subject": {
    "patient_id": "P-1001",
    "encounter_id": "E-9001",
    "stay_id": "S-3321",
    "source_system": "asclena-ai"
  },
  "features": {
    "gender_male": 1,
    "gender_female": 0,
    "gender_unknown": 0,
    "triage_temperature": 99.1,
    "triage_heartrate": 118,
    "triage_resprate": 24,
    "triage_o2sat": 90,
    "triage_sbp": 96,
    "triage_dbp": 58,
    "acuity": 2,
    "triage_shock_index": 1.2292,
    "triage_temperature_missing": 0,
    "triage_heartrate_missing": 0,
    "triage_resprate_missing": 0,
    "triage_o2sat_missing": 0,
    "triage_sbp_missing": 0,
    "triage_dbp_missing": 0,
    "acuity_missing": 0,
    "vital_row_count": 4,
    "temperature_mean": 99.4,
    "temperature_min": 99.1,
    "temperature_max": 99.8,
    "hr_mean": 121,
    "hr_min": 118,
    "hr_max": 125,
    "rr_mean": 25,
    "rr_min": 24,
    "rr_max": 28,
    "spo2_mean": 89,
    "spo2_min": 87,
    "spo2_max": 90,
    "sbp_mean": 94,
    "sbp_min": 90,
    "sbp_max": 96,
    "dbp_mean": 57,
    "dbp_min": 54,
    "dbp_max": 58,
    "shock_index": 1.2872,
    "shock_index_max": 1.3889,
    "hr_slope": 3.2,
    "bp_slope": -1.8,
    "tachycardia_count": 4,
    "hypotension_count": 3,
    "hypoxia_count": 4,
    "fever_count": 0,
    "temperature_missing_rate": 0.0,
    "heartrate_missing_rate": 0.0,
    "resprate_missing_rate": 0.0,
    "o2sat_missing_rate": 0.0,
    "sbp_missing_rate": 0.0,
    "dbp_missing_rate": 0.0,
    "prior_ed_visit_count": 3,
    "prior_ed_visit_count_30d": 1,
    "prior_ed_visit_count_90d": 2,
    "time_since_last_ed_visit_days": 19.0,
    "prior_admission_count": 1,
    "prior_admission_count_1y": 1,
    "prior_icu_or_death_count": 0,
    "prior_cardiovascular_dx_count": 1,
    "prior_respiratory_dx_count": 2,
    "prior_endocrine_dx_count": 1,
    "prior_renal_dx_count": 0,
    "prior_distinct_diagnosis_count": 5,
    "prior_high_risk_prediction_count": 1,
    "last_risk_score": 0.73125,
    "avg_prior_risk_score": 0.58422,
    "max_prior_risk_score": 0.73125
  }
}
```

Rules:

- all required feature keys must be present
- values may be `null`
- unknown feature names are rejected
- excluded or leakage columns are not allowed in the feature contract
- V2 payloads include both current-visit features and additive history features for prior utilization, prior diagnoses, prior admissions, and prior model scores

Success response:

```json
{
  "request_id": "req-001",
  "subject": {
    "patient_id": "P-1001",
    "encounter_id": "E-9001",
    "stay_id": "S-3321",
    "source_system": "asclena-ai"
  },
  "model": {
    "model_name": "asclena_xgboost_risk_v2",
    "model_version": "patient_aware_v2_20260430T145844Z",
    "feature_count": 67,
    "classification_threshold": 0.4,
    "risk_label_thresholds": {
      "LOW": [0.0, 0.4],
      "MODERATE": [0.4, 0.7],
      "HIGH": [0.7, 1.0]
    },
    "calibration_method": "isotonic",
    "contract_version": "2026-04-29"
  },
  "prediction": {
    "risk_score": 1.0,
    "predicted_target": 1,
    "risk_label": "HIGH",
    "severity_index": 1,
    "severity_label": "ASI-1 Critical",
    "severity_description": "Immediate clinician review recommended.",
    "severity_scale_name": "Asclena Severity Index",
    "threshold_used": 0.4
  },
  "explanation": {
    "top_contributors": [
      {
        "feature_name": "shock_index_max",
        "feature_value": 1.3889,
        "contribution": 0.842115,
        "contribution_direction": "increases_risk"
      }
    ]
  },
  "contract_version": "2026-04-29"
}
```

Optional debug query parameter:

- `include_feature_snapshot=true`

When provided, the `explanation` object also includes the full ordered `feature_snapshot`.

### `POST /v1/predict/batch`

Batch version of the same contract.

Request:

```json
{
  "instances": [
    {
      "request_id": "req-001",
      "subject": {
        "patient_id": "P-1001"
      },
      "features": {}
    }
  ]
}
```

Rules:

- maximum batch size is `128`
- one invalid instance rejects the full batch

Response:

- `model`
- `predictions`
- `batch_size`
- `contract_version`

## ASI Severity Output

Every successful prediction now includes:

- `severity_index`
- `severity_label`
- `severity_description`
- `severity_scale_name`

Important:

- `severity_scale_name` is always `Asclena Severity Index`
- ASI is an internal Asclena severity scale derived from model probability
- ASI is not an official ESI score and does not replace clinician triage judgment

## Error Cases

### `400 invalid_feature_contract`

Returned when:

- a required feature is missing
- an unknown feature is provided
- the caller sends a feature set that does not match the model contract

### `400 batch_too_large`

Returned when `instances` exceeds the configured batch limit.

### `422 Unprocessable Entity`

Returned automatically by FastAPI when:

- the JSON body shape is malformed
- field types are invalid
- required top-level fields such as `features` are missing

### `503 prediction_failed`

Returned when the service loads but inference fails at runtime.

## Input Contract Guidance

The caller should treat the service like a model engine, not like a feature-engineering engine.

That means:

- upstream systems must construct the feature map before calling the API
- feature naming and semantics must remain versioned
- if the model retrains with a changed feature set, the contract version must be updated

## Output Contract Guidance

Use the output as:

- `risk_score`: primary ranking signal
- `risk_score` is served from the calibrated classifier, while local feature contributions still come from the raw XGBoost explanation model
- `predicted_target`: thresholded binary flag
- `risk_label`: human-readable band for UI
- `top_contributors`: local explanation for audit and doctor-facing review

Important:

- `top_contributors` are model contribution signals, not final medical diagnosis explanations
- final clinical interpretation must remain in the Asclena AI application layer

## Future FHIR Readiness

This API is prepared for a future adapter that converts FHIR resources into the same contract.

Suggested future mapping:

- `Patient.gender` -> gender one-hot features
- `Observation` -> triage and longitudinal vital features
- `Encounter` -> acuity or encounter-level metadata
- `Condition` -> diagnosis-derived features if later enabled
- `MedicationStatement` and `MedicationAdministration` -> medication-derived features if later enabled

When the FHIR adapter is built, it should call this service without changing the model-serving endpoints.
