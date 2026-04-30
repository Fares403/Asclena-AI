from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.clinical_interpretation import build_clinical_interpretation
from src.serving.predictor import PredictionArtifacts, asclena_severity, predict_one


def make_snapshot(values: dict[str, float | int | None]) -> list[dict[str, float | int | None]]:
    return [{"feature_name": name, "feature_value": value} for name, value in values.items()]


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
            risk_score=1.0,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": 90,
                    "spo2_mean": 89,
                    "spo2_min": 87,
                    "triage_resprate": 24,
                    "rr_mean": 25,
                    "rr_max": 28,
                    "triage_heartrate": 118,
                    "hr_mean": 121,
                    "hr_max": 125,
                    "shock_index": 1.287,
                    "shock_index_max": 1.389,
                    "acuity": 2,
                    "vital_row_count": 4,
                    "hypoxia_count": 4,
                }
            ),
            top_contributors=[
                {
                    "feature_name": "vital_row_count",
                    "feature_value": 4,
                    "contribution": 0.417219,
                    "contribution_direction": "increases_risk",
                }
            ],
        )
        indicator_names = [item["indicator"] for item in interpretation["clinical_indicators"]]
        self.assertIn("Respiratory compromise signal", indicator_names)
        self.assertEqual(interpretation["risk_summary"]["display_risk_score"], ">0.99")
        self.assertEqual(
            interpretation["risk_summary"]["clinical_priority"],
            "High concern - clinician review recommended",
        )
        respiratory = next(item for item in interpretation["clinical_indicators"] if item["indicator"] == "Respiratory compromise signal")
        self.assertTrue(any("breaths/min" in entry for entry in respiratory["evidence"]))
        self.assertEqual(interpretation["dominant_clinical_drivers"][0]["driver_type"], "data_context")
        self.assertIn("data_quality", interpretation)

    def test_tachycardia_case(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.55,
            risk_label="MODERATE",
            feature_snapshot=make_snapshot(
                {
                    "triage_heartrate": 122,
                    "hr_mean": 118,
                    "hr_min": 110,
                    "hr_max": 122,
                    "hr_slope": 1.6,
                    "tachycardia_count": 2,
                    "bp_slope": 0.0,
                }
            ),
            top_contributors=[],
        )
        indicator = next(item for item in interpretation["clinical_indicators"] if item["indicator"] == "Tachycardia / physiologic stress")
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
        pattern_names = [item["pattern"] for item in interpretation["clinical_patterns"]]
        self.assertIn("Hemodynamic instability pattern", pattern_names)

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
            item for item in interpretation["clinical_indicators"] if item["indicator"] == "Elevated shock index / hemodynamic strain"
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
                    "rr_mean": 24,
                    "shock_index_max": 0.95,
                    "vital_row_count": 3,
                }
            ),
            top_contributors=[],
        )
        names = [item["pattern"] for item in interpretation["clinical_patterns"]]
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
        indicator = next(item for item in interpretation["clinical_indicators"] if item["indicator"] == "Sparse data / low confidence signal")
        self.assertIn(indicator["severity"], {"moderate", "high"})
        self.assertTrue(interpretation["data_quality"]["missingness_concern"])

    def test_low_risk_case_with_no_strong_indicators(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.049485,
            risk_label="LOW",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": 98,
                    "triage_heartrate": 78,
                    "triage_resprate": 16,
                    "triage_sbp": 122,
                    "hr_mean": 79,
                    "hr_min": 78,
                    "hr_slope": 0.2,
                    "bp_slope": 0.1,
                    "shock_index": 0.65,
                    "acuity": 5,
                    "vital_row_count": 2,
                }
            ),
            top_contributors=[],
        )
        self.assertEqual(
            interpretation["risk_summary"]["interpretation"],
            "Lower predicted risk overall, with no major abnormal vital-sign indicators detected. Trend interpretation is limited by sparse repeated measurements.",
        )
        indicator_names = [item["indicator"] for item in interpretation["clinical_indicators"]]
        self.assertNotIn("Tachycardia / physiologic stress", indicator_names)
        self.assertIn("Triage acuity context", indicator_names)
        self.assertIn("Limited trend data context", indicator_names)
        self.assertEqual(
            interpretation["recommended_review_focus"],
            ["Review limited repeated vital-sign trend data in clinical context"],
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
        self.assertIn("clinical_patterns", interpretation)

    def test_predict_one_keeps_existing_fields_and_adds_clinical_layer(self) -> None:
        artifacts = PredictionArtifacts(
            model=FakeModel(0.8),
            explanation_model=SimpleNamespace(),
            imputer=FakeImputer(),
            feature_names=["triage_o2sat", "triage_heartrate", "triage_resprate", "sbp_min", "shock_index"],
            transformed_feature_names=["triage_o2sat", "triage_heartrate", "triage_resprate", "sbp_min", "shock_index"],
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
        self.assertIn("severity_index", result)
        self.assertIn("severity_label", result)
        self.assertIn("severity_description", result)
        self.assertIn("severity_scale_name", result)
        self.assertIn("threshold_used", result)
        self.assertIn("top_contributors", result)
        self.assertIn("clinical_interpretation", result)
        self.assertIn("data_quality", result["clinical_interpretation"])
        self.assertIn("clinical_patterns", result["clinical_interpretation"])
        self.assertEqual(result["severity_index"], 2)
        self.assertEqual(result["severity_label"], "ASI-2 Very High Risk")

    def test_review_focus_items_are_deduplicated(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.84,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": 91,
                    "spo2_min": 89,
                    "triage_resprate": 24,
                    "rr_mean": 24,
                    "triage_heartrate": 122,
                    "hr_mean": 118,
                    "sbp_min": 88,
                    "bp_slope": -1.0,
                    "hr_slope": 1.2,
                    "triage_shock_index": 1.05,
                    "vital_row_count": 4,
                }
            ),
            top_contributors=[],
        )
        review_focus = interpretation["recommended_review_focus"]
        self.assertEqual(len(review_focus), len(set(review_focus)))
        self.assertLessEqual(len(review_focus), 6)

    def test_indicators_and_patterns_are_separated(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.9,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": 90,
                    "spo2_min": 88,
                    "triage_resprate": 25,
                    "rr_mean": 24,
                    "triage_heartrate": 120,
                    "shock_index_max": 1.1,
                }
            ),
            top_contributors=[],
        )
        self.assertTrue(all("indicator" in item for item in interpretation["clinical_indicators"]))
        self.assertTrue(all("pattern" in item for item in interpretation["clinical_patterns"]))

    def test_low_risk_normal_hr_does_not_trigger_tachycardia(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.08,
            risk_label="LOW",
            feature_snapshot=make_snapshot(
                {
                    "triage_heartrate": 78,
                    "hr_mean": 79,
                    "hr_min": 78,
                    "hr_max": 80,
                    "hr_slope": 0.2,
                    "tachycardia_count": 0,
                    "vital_row_count": 2,
                    "acuity": 5,
                }
            ),
            top_contributors=[
                {
                    "feature_name": "hr_slope",
                    "feature_value": 0.2,
                    "contribution": 0.1,
                    "contribution_direction": "increases_risk",
                }
            ],
        )
        self.assertNotIn("Tachycardia / physiologic stress", [item["indicator"] for item in interpretation["clinical_indicators"]])
        self.assertIn("within expected range", interpretation["dominant_clinical_drivers"][0]["doctor_message"])

    def test_low_risk_two_observations_does_not_trigger_moderate_deterioration(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.12,
            risk_label="LOW",
            feature_snapshot=make_snapshot(
                {
                    "vital_row_count": 2,
                    "hr_slope": 0.2,
                    "bp_slope": 0.1,
                }
            ),
            top_contributors=[],
        )
        trend = next(item for item in interpretation["clinical_indicators"] if item["indicator"] == "Limited trend data context")
        self.assertEqual(trend["severity"], "low")
        self.assertEqual(interpretation["clinical_patterns"], [])

    def test_high_risk_normal_oxygen_does_not_trigger_respiratory_indicator(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.92,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": 98,
                    "spo2_mean": 98,
                    "spo2_min": 97,
                    "triage_resprate": 18,
                    "rr_mean": 18,
                    "rr_max": 19,
                    "hypoxia_count": 0,
                }
            ),
            top_contributors=[],
        )
        self.assertNotIn("Respiratory compromise signal", [item["indicator"] for item in interpretation["clinical_indicators"]])

    def test_high_risk_normal_sbp_does_not_trigger_hemodynamic_indicator(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.91,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_sbp": 122,
                    "sbp_mean": 121,
                    "sbp_min": 120,
                    "hypotension_count": 0,
                    "triage_shock_index": 0.65,
                    "shock_index": 0.66,
                }
            ),
            top_contributors=[],
        )
        self.assertNotIn("Hemodynamic instability signal", [item["indicator"] for item in interpretation["clinical_indicators"]])

    def test_moderate_risk_normal_temperature_does_not_trigger_fever_signal(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.55,
            risk_label="MODERATE",
            feature_snapshot=make_snapshot(
                {
                    "triage_temperature": 98.4,
                    "temperature_mean": 98.5,
                    "temperature_min": 98.1,
                    "temperature_max": 99.1,
                    "fever_count": 0,
                }
            ),
            top_contributors=[],
        )
        self.assertNotIn("Fever / infection-context signal", [item["indicator"] for item in interpretation["clinical_indicators"]])

    def test_acuity_five_uses_context_not_concern(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.05,
            risk_label="LOW",
            feature_snapshot=make_snapshot({"acuity": 5}),
            top_contributors=[],
        )
        indicator = interpretation["clinical_indicators"][0]
        self.assertEqual(indicator["indicator"], "Triage acuity context")

    def test_review_focus_does_not_mention_tachycardia_without_indicator(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.2,
            risk_label="LOW",
            feature_snapshot=make_snapshot(
                {
                    "triage_heartrate": 78,
                    "hr_mean": 79,
                    "hr_slope": 0.3,
                    "vital_row_count": 2,
                    "acuity": 5,
                }
            ),
            top_contributors=[],
        )
        self.assertTrue(all("tachycardia" not in item.lower() for item in interpretation["recommended_review_focus"]))

    def test_high_risk_normal_vitals_uses_contextual_summary(self) -> None:
        interpretation = build_clinical_interpretation(
            risk_score=0.88,
            risk_label="HIGH",
            feature_snapshot=make_snapshot(
                {
                    "triage_o2sat": 98,
                    "spo2_mean": 98,
                    "spo2_min": 97,
                    "triage_resprate": 18,
                    "rr_mean": 18,
                    "triage_heartrate": 82,
                    "hr_mean": 80,
                    "hr_max": 84,
                    "triage_sbp": 124,
                    "sbp_mean": 122,
                    "sbp_min": 120,
                    "triage_temperature": 98.5,
                    "temperature_max": 99.1,
                    "triage_shock_index": 0.65,
                    "shock_index": 0.66,
                }
            ),
            top_contributors=[],
        )
        self.assertEqual(
            interpretation["risk_summary"]["interpretation"],
            "High model-estimated risk, but no major threshold-based vital-sign abnormality is detected in the provided features. Review model drivers, data completeness, and clinical context.",
        )
        self.assertEqual(interpretation["clinical_patterns"], [])

    def test_severity_boundaries(self) -> None:
        self.assertEqual(asclena_severity(0.85)["severity_index"], 1)
        self.assertEqual(asclena_severity(0.70)["severity_index"], 2)
        self.assertEqual(asclena_severity(0.55)["severity_index"], 3)
        self.assertEqual(asclena_severity(0.40)["severity_index"], 4)
        self.assertEqual(asclena_severity(0.20)["severity_index"], 5)
        self.assertEqual(asclena_severity(0.19)["severity_index"], 6)


if __name__ == "__main__":
    unittest.main()
