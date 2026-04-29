from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.clinical_interpretation import build_clinical_interpretation
from src.serving.predictor import PredictionArtifacts, predict_one


def make_snapshot(values: dict[str, float | int | None]) -> list[dict[str, float | int | None]]:
    return [
        {"feature_name": feature_name, "feature_value": feature_value}
        for feature_name, feature_value in values.items()
    ]


class FakeModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, transformed: np.ndarray) -> np.ndarray:
        return np.array([[1.0 - self.score, self.score]], dtype=float)


class FakeImputer:
    def transform(self, raw_matrix: np.ndarray) -> np.ndarray:
        return raw_matrix


class ClinicalInterpretationTests(unittest.TestCase):
    def test_high_risk_respiratory_case(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.89136,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": 94,
                    "spo2_mean": 94,
                    "spo2_min": 93,
                    "triage_resprate": 22,
                    "rr_mean": 21,
                    "triage_heartrate": 104,
                    "hr_mean": 106,
                    "shock_index": 0.991,
                    "shock_index_max": 1.028,
                    "fever_count": 1,
                    "acuity": 3,
                }
            ),
            top_contributors=[
                {
                    "feature_name": "spo2_mean",
                    "feature_value": 94,
                    "contribution": 0.346997,
                    "contribution_direction": "increases_risk",
                }
            ],
        )
        names = [item["indicator"] for item in interpretation["clinical_indicators"]]
        self.assertIn("Respiratory compromise signal", names)
        self.assertEqual(
            interpretation["risk_summary"]["clinical_priority"],
            "High concern - clinician review recommended",
        )

    def test_tachycardia_case(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.55,
            risk_label="MODERATE",
            feature_snapshot=make_snapshot(
                {
                    "triage_heartrate": 122,
                    "hr_mean": 118,
                    "hr_min": 110,
                    "hr_slope": 1.6,
                    "tachycardia_count": 2,
                    "bp_slope": 0.0,
                }
            ),
            top_contributors=[],
        )
        indicator = next(
            item for item in interpretation["clinical_indicators"] if item["indicator"] == "Tachycardia / physiologic stress"
        )
        self.assertEqual(indicator["severity"], "high")

    def test_hypotension_case(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.72,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_sbp": 88,
                    "sbp_min": 86,
                    "hypotension_count": 1,
                    "bp_slope": -1.1,
                    "hr_slope": 0.9,
                }
            ),
            top_contributors=[],
        )
        names = [item["indicator"] for item in interpretation["clinical_indicators"]]
        self.assertIn("Hemodynamic instability signal", names)
        self.assertIn("Hemodynamic instability pattern", names)

    def test_shock_index_high_case(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.81,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_shock_index": 1.02,
                    "shock_index": 0.99,
                    "shock_index_max": 1.08,
                }
            ),
            top_contributors=[],
        )
        indicator = next(
            item
            for item in interpretation["clinical_indicators"]
            if item["indicator"] == "Elevated shock index / hemodynamic strain"
        )
        self.assertEqual(indicator["severity"], "high")

    def test_sepsis_like_pattern_case(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.88,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "temperature_max": 101.3,
                    "fever_count": 1,
                    "triage_heartrate": 112,
                    "triage_resprate": 24,
                    "shock_index_max": 0.95,
                }
            ),
            top_contributors=[],
        )
        names = [item["indicator"] for item in interpretation["clinical_indicators"]]
        self.assertIn("Sepsis-like physiology pattern", names)

    def test_sparse_data_case(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.44,
            risk_label="MODERATE",
            feature_snapshot=make_snapshot(
                {
                    "vital_row_count": 1,
                    "triage_temperature_missing": 1,
                    "triage_heartrate_missing": 1,
                    "triage_resprate_missing": 1,
                    "acuity_missing": 1,
                    "o2sat_missing_rate": 0.75,
                }
            ),
            top_contributors=[],
        )
        indicator = next(
            item
            for item in interpretation["clinical_indicators"]
            if item["indicator"] == "Sparse data / low confidence signal"
        )
        self.assertIn(indicator["severity"], {"moderate", "high"})

    def test_low_risk_case_with_no_strong_indicators(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.05,
            risk_label="LOW",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": 98,
                    "triage_heartrate": 78,
                    "triage_resprate": 16,
                    "triage_sbp": 122,
                    "shock_index": 0.65,
                    "acuity": 5,
                    "vital_row_count": 2,
                }
            ),
            top_contributors=[],
        )
        self.assertEqual(
            interpretation["risk_summary"]["interpretation"],
            "No strong model signal of escalation risk.",
        )

    def test_missing_values_do_not_crash(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.31,
            risk_label="LOW",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": None,
                    "triage_heartrate": None,
                    "triage_resprate": None,
                    "sbp_min": None,
                    "shock_index": None,
                }
            ),
            top_contributors=[],
        )
        self.assertIn("risk_summary", interpretation)
        self.assertIn("clinical_indicators", interpretation)

    def test_predict_one_keeps_existing_fields_and_adds_clinical_layer(self) -> None:
        artifacts = PredictionArtifacts(
            model=FakeModel(0.8),
            explanation_model=SimpleNamespace(),
            imputer=FakeImputer(),
            feature_names=["triage_o2sat", "triage_heartrate", "triage_resprate", "sbp_min", "shock_index"],
            leakage_or_excluded_columns=[],
            model_name="asclena_xgboost_risk",
            model_version="test",
            classification_threshold=0.4,
            risk_label_thresholds={"LOW": [0.0, 0.4], "MODERATE": [0.4, 0.7], "HIGH": [0.7, 1.0]},
            calibration_method="isotonic",
            model_path=Path("dummy.joblib"),
        )
        result = predict_one(
            artifacts,
            {
                "triage_o2sat": 91,
                "triage_heartrate": 118,
                "triage_resprate": 24,
                "sbp_min": 92,
                "shock_index": 0.98,
            },
        )
        self.assertIn("risk_score", result)
        self.assertIn("predicted_target", result)
        self.assertIn("risk_label", result)
        self.assertIn("threshold_used", result)
        self.assertIn("top_contributors", result)
        self.assertIn("clinical_interpretation", result)


if __name__ == "__main__":
    unittest.main()
