"""Pydantic contracts for the Asclena clinical risk model API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SubjectContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = Field(default=None, description="External patient identifier from the calling system.")
    encounter_id: str | None = Field(default=None, description="Encounter identifier from the calling system.")
    stay_id: str | None = Field(default=None, description="Optional ED stay identifier when available.")
    source_system: str | None = Field(default=None, description="Originating application or EHR name.")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(default=None, description="Caller-generated correlation identifier.")
    subject: SubjectContext | None = Field(
        default=None,
        description="Optional context identifiers passed through unchanged for integration tracing.",
    )
    features: dict[str, float | int | None] = Field(
        ...,
        description="Complete feature map for one encounter. All required feature keys must be present; values may be null.",
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[PredictionRequest] = Field(..., min_length=1, description="Batch of independent stateless prediction requests.")


class LocalExplanationItem(BaseModel):
    feature_name: str
    feature_value: float | int | None
    contribution: float
    contribution_direction: str


class FeatureSnapshotItem(BaseModel):
    feature_name: str
    feature_value: float | int | None
    category: str | None = None


class ExplanationPayload(BaseModel):
    top_contributors: list[LocalExplanationItem]
    feature_snapshot: list[FeatureSnapshotItem] | None = None


class RiskSummary(BaseModel):
    risk_label: str
    risk_score: float
    display_risk_score: str
    risk_score_note: str
    clinical_priority: str
    interpretation: str


class ClinicalIndicator(BaseModel):
    indicator: str
    severity: str
    evidence: list[str]
    doctor_message: str


class ClinicalPattern(BaseModel):
    pattern: str
    severity: str
    evidence: list[str]
    doctor_message: str


class DominantClinicalDriver(BaseModel):
    feature_name: str
    clinical_meaning: str
    feature_value: float | int | None
    driver_type: str
    contribution_direction: str
    doctor_message: str


class DataQualityPayload(BaseModel):
    vital_observation_count: int | None
    trend_interpretability: str
    missingness_concern: bool
    missing_features: list[str]
    data_quality_note: str


class ClinicalInterpretationPayload(BaseModel):
    risk_summary: RiskSummary
    clinical_indicators: list[ClinicalIndicator]
    clinical_patterns: list[ClinicalPattern]
    dominant_clinical_drivers: list[DominantClinicalDriver]
    data_quality: DataQualityPayload
    recommended_review_focus: list[str]
    safety_note: str


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    feature_count: int
    classification_threshold: float
    risk_label_thresholds: dict[str, list[float]]
    calibration_method: str
    contract_version: str


class PredictionResult(BaseModel):
    risk_score: float
    display_risk_score: str
    predicted_target: int
    risk_label: str
    severity_index: int
    severity_label: str
    severity_description: str
    severity_scale_name: str
    threshold_used: float


class PredictionResponse(BaseModel):
    request_id: str | None
    subject: SubjectContext | None
    model: ModelMetadata
    prediction: PredictionResult
    explanation: ExplanationPayload
    clinical_interpretation: ClinicalInterpretationPayload
    contract_version: str


class BatchPredictionResponse(BaseModel):
    model: ModelMetadata
    predictions: list[PredictionResponse]
    batch_size: int
    contract_version: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] | None = None
