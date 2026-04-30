from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving import app as serving_app_module
from src.serving.app import batch_predict, get_contract, get_model_metadata, predict
from src.serving.predictor import PredictionArtifacts, asclena_severity, load_prediction_artifacts
from src.serving.schemas import BatchPredictionRequest, PredictionRequest


class FakeModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, transformed: np.ndarray) -> np.ndarray:
        return np.array([[1.0 - self.score, self.score]], dtype=float)


class FakeImputer:
    def transform(self, raw_matrix: np.ndarray) -> np.ndarray:
        return raw_matrix


FEATURES_67 = [
    "gender_male",
    "gender_female",
    "gender_unknown",
    "triage_temperature",
    "triage_heartrate",
    "triage_resprate",
    "triage_o2sat",
    "triage_sbp",
    "triage_dbp",
    "acuity",
    "triage_shock_index",
    "triage_temperature_missing",
    "triage_heartrate_missing",
    "triage_resprate_missing",
    "triage_o2sat_missing",
    "triage_sbp_missing",
    "triage_dbp_missing",
    "acuity_missing",
    "vital_row_count",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "hr_mean",
    "hr_min",
    "hr_max",
    "rr_mean",
    "rr_min",
    "rr_max",
    "spo2_mean",
    "spo2_min",
    "spo2_max",
    "sbp_mean",
    "sbp_min",
    "sbp_max",
    "dbp_mean",
    "dbp_min",
    "dbp_max",
    "shock_index",
    "shock_index_max",
    "hr_slope",
    "bp_slope",
    "tachycardia_count",
    "hypotension_count",
    "hypoxia_count",
    "fever_count",
    "temperature_missing_rate",
    "heartrate_missing_rate",
    "resprate_missing_rate",
    "o2sat_missing_rate",
    "sbp_missing_rate",
    "dbp_missing_rate",
    "prior_ed_visit_count",
    "prior_ed_visit_count_30d",
    "prior_ed_visit_count_90d",
    "time_since_last_ed_visit_days",
    "prior_admission_count",
    "prior_admission_count_1y",
    "prior_icu_or_death_count",
    "prior_cardiovascular_dx_count",
    "prior_respiratory_dx_count",
    "prior_endocrine_dx_count",
    "prior_renal_dx_count",
    "prior_distinct_diagnosis_count",
    "prior_high_risk_prediction_count",
    "last_risk_score",
    "avg_prior_risk_score",
    "max_prior_risk_score",
]


def full_payload() -> dict[str, object]:
    features = {name: 0.0 for name in FEATURES_67}
    features.update(
        {
            "gender_female": 1.0,
            "triage_temperature": 98.4,
            "triage_heartrate": 78.0,
            "triage_resprate": 16.0,
            "triage_o2sat": 98.0,
            "triage_sbp": 122.0,
            "triage_dbp": 76.0,
            "acuity": 5.0,
            "triage_shock_index": 0.6393,
            "vital_row_count": 2.0,
            "temperature_mean": 98.5,
            "temperature_min": 98.4,
            "temperature_max": 98.6,
            "hr_mean": 79.0,
            "hr_min": 78.0,
            "hr_max": 80.0,
            "rr_mean": 16.0,
            "rr_min": 16.0,
            "rr_max": 16.0,
            "spo2_mean": 98.0,
            "spo2_min": 98.0,
            "spo2_max": 99.0,
            "sbp_mean": 121.0,
            "sbp_min": 120.0,
            "sbp_max": 122.0,
            "dbp_mean": 76.0,
            "dbp_min": 75.0,
            "dbp_max": 76.0,
            "shock_index": 0.6529,
            "shock_index_max": 0.6667,
            "hr_slope": 0.2,
            "bp_slope": 0.1,
        }
    )
    return {
        "request_id": "req-001",
        "subject": {"patient_id": "P-1001"},
        "features": features,
    }


class ServingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_artifacts = getattr(serving_app_module.app.state, "artifacts", None)
        self.original_settings = getattr(serving_app_module.app.state, "settings", None)
        serving_app_module.app.state.artifacts = PredictionArtifacts(
            model=FakeModel(0.82),
            explanation_model=SimpleNamespace(),
            imputer=FakeImputer(),
            feature_names=FEATURES_67,
            transformed_feature_names=FEATURES_67,
            leakage_or_excluded_columns=[],
            model_name="asclena_xgboost_risk_v2",
            model_version="patient_aware_v2_test",
            classification_threshold=0.4,
            risk_label_thresholds={"LOW": [0.0, 0.4], "MODERATE": [0.4, 0.7], "HIGH": [0.7, 1.0]},
            calibration_method="isotonic",
            model_path=Path("dummy.joblib"),
        )
        serving_app_module.app.state.settings = SimpleNamespace(max_batch_size=128)

    def tearDown(self) -> None:
        serving_app_module.app.state.artifacts = self.original_artifacts
        serving_app_module.app.state.settings = self.original_settings

    def test_model_metadata_reports_67_features(self) -> None:
        response = get_model_metadata()
        self.assertEqual(response["feature_count"], 67)

    def test_contract_reports_67_required_features(self) -> None:
        body = get_contract()
        self.assertEqual(body["required_feature_count"], 67)
        self.assertEqual(len(body["required_features"]), 67)

    def test_single_prediction_success(self) -> None:
        response = predict(PredictionRequest(**full_payload()))
        prediction = response.prediction.model_dump()
        self.assertEqual(prediction["severity_index"], 2)
        self.assertEqual(prediction["severity_label"], "ASI-2 Very High Risk")
        self.assertTrue(0.0 <= prediction["risk_score"] <= 1.0)
        self.assertEqual(prediction["display_risk_score"], "0.82")

    def test_missing_feature_returns_400(self) -> None:
        payload = full_payload()
        del payload["features"]["prior_ed_visit_count"]
        with self.assertRaises(HTTPException) as exc:
            predict(PredictionRequest(**payload))
        self.assertEqual(exc.exception.status_code, 400)
        self.assertEqual(exc.exception.detail["error_code"], "invalid_feature_contract")

    def test_unknown_feature_returns_400(self) -> None:
        payload = full_payload()
        payload["features"]["unknown_feature"] = 1
        with self.assertRaises(HTTPException) as exc:
            predict(PredictionRequest(**payload))
        self.assertEqual(exc.exception.status_code, 400)
        self.assertEqual(exc.exception.detail["error_code"], "invalid_feature_contract")

    def test_batch_prediction_success(self) -> None:
        payload = BatchPredictionRequest(instances=[PredictionRequest(**full_payload()), PredictionRequest(**full_payload())])
        response = batch_predict(payload)
        self.assertEqual(response.batch_size, 2)
        self.assertEqual(response.predictions[0].prediction.severity_scale_name, "Asclena Severity Index")

    def test_batch_prediction_rejects_invalid_instance(self) -> None:
        valid = full_payload()
        invalid = full_payload()
        del invalid["features"]["avg_prior_risk_score"]
        payload = BatchPredictionRequest(instances=[PredictionRequest(**valid), PredictionRequest(**invalid)])
        with self.assertRaises(HTTPException) as exc:
            batch_predict(payload)
        self.assertEqual(exc.exception.status_code, 400)
        self.assertEqual(exc.exception.detail["error_code"], "invalid_feature_contract")

    def test_severity_boundaries(self) -> None:
        self.assertEqual(asclena_severity(0.85)["severity_index"], 1)
        self.assertEqual(asclena_severity(0.70)["severity_index"], 2)
        self.assertEqual(asclena_severity(0.55)["severity_index"], 3)
        self.assertEqual(asclena_severity(0.40)["severity_index"], 4)
        self.assertEqual(asclena_severity(0.20)["severity_index"], 5)
        self.assertEqual(asclena_severity(0.19)["severity_index"], 6)

    def test_display_risk_score_caps_certainty_style(self) -> None:
        serving_app_module.app.state.artifacts = PredictionArtifacts(
            model=FakeModel(1.0),
            explanation_model=SimpleNamespace(),
            imputer=FakeImputer(),
            feature_names=FEATURES_67,
            transformed_feature_names=FEATURES_67,
            leakage_or_excluded_columns=[],
            model_name="asclena_xgboost_risk_v2",
            model_version="patient_aware_v2_test",
            classification_threshold=0.4,
            risk_label_thresholds={"LOW": [0.0, 0.4], "MODERATE": [0.4, 0.7], "HIGH": [0.7, 1.0]},
            calibration_method="isotonic",
            model_path=Path("dummy.joblib"),
        )
        response = predict(PredictionRequest(**full_payload()))
        prediction = response.prediction.model_dump()
        self.assertEqual(prediction["risk_score"], 0.999)
        self.assertEqual(prediction["display_risk_score"], ">0.99")

    def test_v2_artifact_loader_rejects_wrong_feature_count(self) -> None:
        artifact = {
            "model": FakeModel(0.5),
            "explanation_model": SimpleNamespace(),
            "imputer": FakeImputer(),
            "feature_names": FEATURES_67[:-1],
            "model_name": "asclena_xgboost_risk_v2",
            "model_version": "bad_test",
            "classification_threshold": 0.4,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "bad.joblib"
            joblib.dump(artifact, artifact_path)
            with self.assertRaises(ValueError):
                load_prediction_artifacts(artifact_path)
