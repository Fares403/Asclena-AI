# Asclena Clinical Risk Pipeline

## Purpose

This document defines the full pipeline for the Asclena clinical risk model from raw clinical data to deployable stateless inference.

The pipeline is intentionally split into:

- `offline pipeline`: data cleaning, feature engineering, model training
- `online pipeline`: stateless FastAPI inference service

This separation is important for deployment:

- offline stages are data- and database-dependent
- online inference is model-dependent and should stay stateless

## End-to-End Flow

```text
Raw PostgreSQL clinical tables
  ->
Data cleaning pipeline
  ->
Cleaned PostgreSQL tables
  ->
Feature engineering pipeline
  ->
patient_feature_store
  ->
XGBoost training pipeline
  ->
model artifact + evaluation reports
  ->
FastAPI serving layer
  ->
Asclena AI integration
```

## Pipeline Stages

### Stage 0: Raw Clinical Data

Primary raw sources in PostgreSQL:

- `asclena.ed_stays`
- `asclena.triage`
- `asclena.vital_sign`
- `asclena.diagnosis`
- `asclena.pyxis`
- `asclena.med_recon`

These tables are the source of truth for offline model preparation.

### Stage 1: Data Cleaning

Entry point:

[run_cleaning_pipeline.py](</home/fares-ashraf/Fares Ashraf/Asclena/Asclena-AI/Code Implementation/clinical_risk_prediction/src/data_cleaning/run_cleaning_pipeline.py>)

Purpose:

- create raw backups
- profile source tables
- apply cleaning rules
- validate cleaned tables
- produce readiness reports

Inputs:

- raw PostgreSQL tables
- SQL cleaning scripts in `sql/data_cleaning/`

Outputs:

- `asclena.cleaned_ed_stays`
- `asclena.cleaned_triage`
- `asclena.cleaned_vital_sign`
- `asclena.cleaned_diagnosis`
- `asclena.cleaned_pyxis`
- `asclena.cleaned_med_recon`
- reports in `reports/data_cleaning/<run_id>/`

### Stage 2: Feature Engineering

Entry point:

[run_feature_engineering.py](</home/fares-ashraf/Fares Ashraf/Asclena/Asclena-AI/Code Implementation/clinical_risk_prediction/src/feature_engineering/run_feature_engineering.py>)

Purpose:

- aggregate encounter features
- derive missingness indicators
- derive temporal and clinical summary features
- construct one row per ED stay

Inputs:

- cleaned PostgreSQL tables from Stage 1
- SQL feature scripts in `sql/feature_engineering/`

Outputs:

- `asclena.patient_feature_store`
- reports in `reports/feature_engineering/<run_id>/`

### Stage 3: Model Training

Entry point:

[train_xgboost_risk_model.py](</home/fares-ashraf/Fares Ashraf/Asclena/Asclena-AI/Code Implementation/clinical_risk_prediction/src/modeling/train_xgboost_risk_model.py>)

Purpose:

- train XGBoost classifier on `patient_feature_store`
- impute missing numeric values
- split train, validation, and test data
- evaluate model quality
- save artifact and prediction outputs

Inputs:

- `asclena.patient_feature_store`
- model feature manifest embedded in the training script

Outputs:

- `.joblib` model artifact in `models/`
- reports in `reports/modeling/<model_version>/`
- optional predictions in `asclena.risk_predictions`

### Stage 4: Stateless Inference Serving

Entry point:

[app.py](</home/fares-ashraf/Fares Ashraf/Asclena/Asclena-AI/Code Implementation/clinical_risk_prediction/src/serving/app.py>)

Purpose:

- load trained artifact
- expose prediction endpoints
- validate strict input contract
- return model output and explanation safely

Important boundary:

- the serving layer does not clean data
- the serving layer does not engineer features
- the serving layer does not query PostgreSQL during inference

It only accepts model-ready engineered features.

## Online vs Offline Responsibility

### Offline responsibility

Handled inside the clinical risk project:

- cleaning raw data
- engineering model features
- training and validating the model
- exporting the final model artifact

### Online responsibility

Handled at serving and integration time:

- receive already normalized feature payload
- validate feature contract
- run inference
- return risk score, label, and top contributors

### Upstream Asclena AI responsibility

Handled by the main product application later:

- collect live EHR or workflow data
- map product data to this model contract
- optionally build a future FHIR adapter
- decide UI, orchestration, and human review workflow

## Unified Orchestrator

New orchestration entry point:

[run_clinical_risk_pipeline.py](</home/fares-ashraf/Fares Ashraf/Asclena/Asclena-AI/Code Implementation/clinical_risk_prediction/src/pipeline/run_clinical_risk_pipeline.py>)

This file provides one clear pipeline command surface.

### Run full offline pipeline

```bash
python src/pipeline/run_clinical_risk_pipeline.py --stage full
```

### Run only data cleaning

```bash
python src/pipeline/run_clinical_risk_pipeline.py --stage clean
```

### Run only feature engineering

```bash
python src/pipeline/run_clinical_risk_pipeline.py --stage features
```

### Run only model training

```bash
python src/pipeline/run_clinical_risk_pipeline.py --stage train
```

The orchestrator keeps one fixed path through the offline lifecycle while still allowing stage-by-stage execution.

## Deployment Path

The deployment path should be treated as:

1. Run offline stages to produce a validated model artifact.
2. Freeze the chosen `.joblib` model version.
3. Deploy the FastAPI service using that artifact.
4. Let Asclena AI call the service through its stateless API.

This means the deployable unit is:

- model artifact
- serving code
- serving documentation
- strict data contract

Not the full training pipeline.

## Recommended File Structure

```text
clinical_risk_prediction/
  configs/
  docs/
    clinical_risk_pipeline.md
    risk_model_fastapi.md
  models/
  reports/
  requirements-data-cleaning.txt
  requirements-api.txt
  requirements-full.txt
  sql/
    data_cleaning/
    feature_engineering/
    modeling/
  src/
    data_cleaning/
    feature_engineering/
    modeling/
    pipeline/
      run_clinical_risk_pipeline.py
    serving/
      app.py
      config.py
      feature_contract.py
      predictor.py
      schemas.py
```

## Dependency Files

- `requirements-data-cleaning.txt`: offline pipeline dependencies for cleaning, feature engineering, and training
- `requirements-api.txt`: serving dependencies for the stateless FastAPI model service
- `requirements-full.txt`: one pinned environment for the whole clinical risk project

## Data Contract Boundary

The serving API accepts only engineered features, not raw clinical tables.

That means:

- raw table schema changes affect Stage 1 or Stage 2
- model feature changes affect Stage 3 and Stage 4
- application integration changes should usually stay outside the model service

This keeps deployment safer and more stable.

## Future FHIR Readiness

This architecture is already positioned for future FHIR integration:

- raw EHR or FHIR resources should be mapped upstream
- mapping output should match the strict serving contract
- the FastAPI model service itself can remain unchanged

That is the cleanest long-term path because the model service stays stateless and versioned even when EHR integration evolves.
