"""Model loading and stateless prediction logic for the Asclena risk API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb

from .clinical_interpretation import build_clinical_interpretation
from .feature_contract import FEATURE_GROUPS, FEATURE_SPECS


CONTRACT_VERSION = "2026-04-29"
EXPECTED_V2_FEATURE_COUNT = 67
DEFAULT_RISK_LABEL_THRESHOLDS = {
    "LOW": [0.0, 0.4],
    "MODERATE": [0.4, 0.7],
    "HIGH": [0.7, 1.0],
}


@dataclass
class PredictionArtifacts:
    model: Any
    explanation_model: Any
    imputer: Any
    feature_names: list[str]
    transformed_feature_names: list[str]
    leakage_or_excluded_columns: list[str]
    model_name: str
    model_version: str
    classification_threshold: float
    risk_label_thresholds: dict[str, list[float]]
    calibration_method: str
    model_path: Path


class ContractValidationError(ValueError):
    """Raised when the request feature contract is invalid."""

    def __init__(
        self,
        message: str,
        *,
        missing_features: list[str] | None = None,
        unknown_features: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing_features = missing_features or []
        self.unknown_features = unknown_features or []


def load_prediction_artifacts(model_path: Path) -> PredictionArtifacts:
    artifact = joblib.load(model_path)
    explanation_model = artifact.get("explanation_model", artifact["model"])
    feature_names = list(artifact["feature_names"])
    if not {"model", "imputer", "feature_names", "classification_threshold", "model_name", "model_version"} <= set(artifact):
        missing_keys = sorted(
            {"model", "imputer", "feature_names", "classification_threshold", "model_name", "model_version"} - set(artifact)
        )
        raise ValueError(f"Loaded model artifact is missing required serving fields: {missing_keys}")
    unknown_features = [feature_name for feature_name in feature_names if feature_name not in FEATURE_SPECS]
    if unknown_features:
        raise ValueError(
            "Loaded model artifact contains feature names that are missing from the serving contract: "
            f"{unknown_features}"
        )
    model_name = str(artifact["model_name"])
    transformed_feature_names = list(artifact.get("transformed_feature_names", []))
    if not transformed_feature_names:
        statistics = getattr(artifact["imputer"], "statistics_", None)
        if statistics is not None and len(statistics) == len(feature_names):
            transformed_feature_names = [
                feature_name
                for feature_name, statistic in zip(feature_names, statistics, strict=True)
                if not (isinstance(statistic, float) and np.isnan(statistic))
            ]
        else:
            transformed_feature_names = list(feature_names)
    if model_name.endswith("_v2") and len(feature_names) != EXPECTED_V2_FEATURE_COUNT:
        raise ValueError(
            f"Loaded V2 model artifact must expose {EXPECTED_V2_FEATURE_COUNT} features, got {len(feature_names)}."
        )
    return PredictionArtifacts(
        model=artifact["model"],
        explanation_model=explanation_model,
        imputer=artifact["imputer"],
        feature_names=feature_names,
        transformed_feature_names=transformed_feature_names,
        leakage_or_excluded_columns=list(artifact.get("leakage_or_excluded_columns", [])),
        model_name=model_name,
        model_version=str(artifact["model_version"]),
        classification_threshold=float(artifact["classification_threshold"]),
        risk_label_thresholds=dict(artifact.get("risk_label_thresholds", DEFAULT_RISK_LABEL_THRESHOLDS)),
        calibration_method=str(artifact.get("calibration_method", "none")),
        model_path=model_path,
    )


def risk_label(score: float, thresholds: dict[str, list[float]]) -> str:
    high_floor = float(thresholds["HIGH"][0])
    moderate_floor = float(thresholds["MODERATE"][0])
    if score >= high_floor:
        return "HIGH"
    if score >= moderate_floor:
        return "MODERATE"
    return "LOW"


def display_risk_score(score: float) -> str:
    if score >= 0.99:
        return ">0.99"
    return f"{score:.2f}"


def asclena_severity(score: float) -> dict[str, Any]:
    if score >= 0.85:
        return {
            "severity_index": 1,
            "severity_label": "ASI-1 Critical",
            "severity_description": "Immediate clinician review recommended.",
            "severity_scale_name": "Asclena Severity Index",
        }
    if score >= 0.70:
        return {
            "severity_index": 2,
            "severity_label": "ASI-2 Very High Risk",
            "severity_description": "Very high risk; urgent clinician review recommended.",
            "severity_scale_name": "Asclena Severity Index",
        }
    if score >= 0.55:
        return {
            "severity_index": 3,
            "severity_label": "ASI-3 High Risk",
            "severity_description": "High risk; close monitoring recommended.",
            "severity_scale_name": "Asclena Severity Index",
        }
    if score >= 0.40:
        return {
            "severity_index": 4,
            "severity_label": "ASI-4 Moderate Risk",
            "severity_description": "Moderate risk; reassessment may be appropriate.",
            "severity_scale_name": "Asclena Severity Index",
        }
    if score >= 0.20:
        return {
            "severity_index": 5,
            "severity_label": "ASI-5 Low Risk",
            "severity_description": "Low predicted risk based on available data.",
            "severity_scale_name": "Asclena Severity Index",
        }
    return {
        "severity_index": 6,
        "severity_label": "ASI-6 Minimal Risk",
        "severity_description": "Minimal predicted risk based on available data.",
        "severity_scale_name": "Asclena Severity Index",
    }


def validate_feature_payload(feature_names: list[str], features: dict[str, float | int | None]) -> None:
    expected = set(feature_names)
    provided = set(features)
    missing = sorted(expected - provided)
    unknown = sorted(provided - expected)
    if missing or unknown:
        raise ContractValidationError(
            "Prediction request does not match the feature contract.",
            missing_features=missing,
            unknown_features=unknown,
        )


def _ordered_feature_values(feature_names: list[str], features: dict[str, float | int | None]) -> np.ndarray:
    values = []
    for feature_name in feature_names:
        value = features[feature_name]
        values.append(np.nan if value is None else float(value))
    return np.array([values], dtype=float)


def _local_explanations(
    artifacts: PredictionArtifacts,
    transformed_matrix: np.ndarray,
    raw_features: dict[str, float | int | None],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if not hasattr(artifacts.explanation_model, "get_booster"):
        return []
    effective_feature_names = list(artifacts.transformed_feature_names)
    if transformed_matrix.shape[1] != len(effective_feature_names):
        return []
    booster = artifacts.explanation_model.get_booster()
    dmatrix = xgb.DMatrix(transformed_matrix, feature_names=effective_feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    ranked = []
    for feature_name, contribution in zip(effective_feature_names, contribs[:-1], strict=True):
        ranked.append(
            {
                "feature_name": feature_name,
                "feature_value": raw_features[feature_name],
                "contribution": round(float(contribution), 6),
                "contribution_direction": "increases_risk" if contribution >= 0 else "decreases_risk",
            }
        )
    ranked.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return ranked[:top_n]


def _feature_snapshot(feature_names: list[str], features: dict[str, float | int | None]) -> list[dict[str, Any]]:
    snapshot = []
    for feature_name in feature_names:
        spec = FEATURE_SPECS.get(feature_name, {})
        snapshot.append(
            {
                "feature_name": feature_name,
                "feature_value": features[feature_name],
                "category": spec.get("category"),
            }
        )
    return snapshot


def predict_one(
    artifacts: PredictionArtifacts,
    features: dict[str, float | int | None],
    *,
    include_feature_snapshot: bool = False,
) -> dict[str, Any]:
    validate_feature_payload(artifacts.feature_names, features)
    raw_matrix = _ordered_feature_values(artifacts.feature_names, features)
    transformed = artifacts.imputer.transform(raw_matrix)
    raw_risk_score = float(artifacts.model.predict_proba(transformed)[0, 1])
    risk_score = min(raw_risk_score, 0.999)
    severity = asclena_severity(risk_score)
    top_contributors = _local_explanations(artifacts, transformed, features)
    feature_snapshot = _feature_snapshot(artifacts.feature_names, features)
    response = {
        "risk_score": round(risk_score, 6),
        "display_risk_score": display_risk_score(risk_score),
        "predicted_target": int(risk_score >= artifacts.classification_threshold),
        "risk_label": risk_label(risk_score, artifacts.risk_label_thresholds),
        "severity_index": severity["severity_index"],
        "severity_label": severity["severity_label"],
        "severity_description": severity["severity_description"],
        "severity_scale_name": severity["severity_scale_name"],
        "threshold_used": artifacts.classification_threshold,
        "top_contributors": top_contributors,
        "clinical_interpretation": build_clinical_interpretation(
            risk_score=risk_score,
            risk_label=risk_label(risk_score, artifacts.risk_label_thresholds),
            feature_snapshot=feature_snapshot,
            top_contributors=top_contributors,
        ),
    }
    if include_feature_snapshot:
        response["feature_snapshot"] = feature_snapshot
    return response


def feature_contract(artifacts: PredictionArtifacts) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "required_feature_count": len(artifacts.feature_names),
        "required_features": [
            {
                "feature_name": feature_name,
                **FEATURE_SPECS[feature_name],
            }
            for feature_name in artifacts.feature_names
        ],
        "excluded_columns": artifacts.leakage_or_excluded_columns,
        "feature_groups": FEATURE_GROUPS,
        "future_fhir_alignment": {
            "status": "prepared_not_implemented",
            "guidance": "The contract is normalized for future mapping from Patient, Encounter, Observation, Condition, and Medication resources.",
        },
    }


def model_metadata(artifacts: PredictionArtifacts) -> dict[str, Any]:
    return {
        "model_name": artifacts.model_name,
        "model_version": artifacts.model_version,
        "feature_count": len(artifacts.feature_names),
        "classification_threshold": artifacts.classification_threshold,
        "risk_label_thresholds": artifacts.risk_label_thresholds,
        "calibration_method": artifacts.calibration_method,
        "contract_version": CONTRACT_VERSION,
    }
