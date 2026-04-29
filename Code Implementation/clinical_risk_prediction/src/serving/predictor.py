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
    unknown_features = [feature_name for feature_name in feature_names if feature_name not in FEATURE_SPECS]
    if unknown_features:
        raise ValueError(
            "Loaded model artifact contains feature names that are missing from the serving contract: "
            f"{unknown_features}"
        )
    return PredictionArtifacts(
        model=artifact["model"],
        explanation_model=explanation_model,
        imputer=artifact["imputer"],
        feature_names=feature_names,
        leakage_or_excluded_columns=list(artifact.get("leakage_or_excluded_columns", [])),
        model_name=str(artifact["model_name"]),
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
    booster = artifacts.explanation_model.get_booster()
    dmatrix = xgb.DMatrix(transformed_matrix, feature_names=artifacts.feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    ranked = []
    for feature_name, contribution in zip(artifacts.feature_names, contribs[:-1], strict=True):
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
    risk_score = float(artifacts.model.predict_proba(transformed)[0, 1])
    top_contributors = _local_explanations(artifacts, transformed, features)
    feature_snapshot = _feature_snapshot(artifacts.feature_names, features)
    response = {
        "risk_score": round(risk_score, 6),
        "predicted_target": int(risk_score >= artifacts.classification_threshold),
        "risk_label": risk_label(risk_score, artifacts.risk_label_thresholds),
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
