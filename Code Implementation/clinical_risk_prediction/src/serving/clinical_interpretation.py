"""Deterministic clinical interpretation layer for model outputs.

This module converts risk scores and feature-level model outputs into
clinician-friendly review signals. It is intentionally rule-based and
conservative: the output is clinical decision support, not diagnosis.
"""

from __future__ import annotations

from typing import Any


FEATURE_TO_CLINICAL_MEANING = {
    "triage_temperature": "Initial body temperature",
    "temperature_mean": "Average body temperature",
    "temperature_min": "Lowest body temperature",
    "temperature_max": "Peak body temperature",
    "triage_heartrate": "Initial heart rate",
    "hr_mean": "Average heart-rate stress",
    "hr_min": "Lowest recorded heart rate",
    "hr_max": "Peak heart rate",
    "triage_resprate": "Initial respiratory rate",
    "rr_mean": "Average respiratory rate",
    "rr_min": "Lowest respiratory rate",
    "rr_max": "Peak respiratory rate",
    "triage_o2sat": "Initial oxygen saturation",
    "spo2_mean": "Overall oxygenation",
    "spo2_min": "Lowest oxygen saturation episode",
    "spo2_max": "Highest oxygen saturation",
    "triage_sbp": "Initial systolic blood pressure",
    "triage_dbp": "Initial diastolic blood pressure",
    "sbp_mean": "Average systolic blood pressure",
    "sbp_min": "Lowest systolic blood pressure",
    "sbp_max": "Peak systolic blood pressure",
    "dbp_mean": "Average diastolic blood pressure",
    "dbp_min": "Lowest diastolic blood pressure",
    "dbp_max": "Peak diastolic blood pressure",
    "triage_shock_index": "Initial shock index",
    "shock_index": "Average hemodynamic strain",
    "shock_index_max": "Peak hemodynamic strain",
    "hr_slope": "Heart-rate trend",
    "bp_slope": "Blood-pressure trend",
    "tachycardia_count": "Tachycardia episodes",
    "hypotension_count": "Hypotension episodes",
    "hypoxia_count": "Hypoxia episodes",
    "fever_count": "Fever episodes",
    "acuity": "Triage severity level",
    "acuity_missing": "Missing triage acuity",
    "vital_row_count": "Number of vital-sign observations",
    "temperature_missing_rate": "Temperature missingness",
    "heartrate_missing_rate": "Heart-rate missingness",
    "resprate_missing_rate": "Respiratory-rate missingness",
    "o2sat_missing_rate": "Oxygen-saturation missingness",
    "sbp_missing_rate": "Systolic-BP missingness",
    "dbp_missing_rate": "Diastolic-BP missingness",
    "gender_male": "Recorded male sex",
    "gender_female": "Recorded female sex",
    "gender_unknown": "Unknown recorded sex",
}

REVIEW_FOCUS_MAP = {
    "Respiratory compromise signal": [
        "Review respiratory status",
        "Review oxygen requirement and work of breathing",
    ],
    "Tachycardia / physiologic stress": [
        "Review causes of tachycardia such as pain, fever, dehydration, hypoxia, or infection",
    ],
    "Hemodynamic instability signal": [
        "Review hemodynamic stability and perfusion status",
    ],
    "Fever / infection-context signal": [
        "Review infection or fever context if clinically relevant",
    ],
    "Elevated shock index / hemodynamic strain": [
        "Review hemodynamic stability",
    ],
    "Deterioration trend signal": [
        "Review repeated vital-sign trends for deterioration",
    ],
    "Triage acuity concern": [
        "Review triage urgency in context of current physiology",
    ],
    "Sparse data / low confidence signal": [
        "Review missing or sparse vital-sign data before relying on model output",
    ],
    "Sepsis-like physiology pattern": [
        "Review sepsis context and available labs if clinically relevant",
    ],
    "Respiratory deterioration pattern": [
        "Review respiratory status",
        "Review oxygen requirement and work of breathing",
    ],
    "Hemodynamic instability pattern": [
        "Review hemodynamic stability and perfusion status",
    ],
    "Cardiopulmonary stress pattern": [
        "Review cardiopulmonary stress drivers",
    ],
    "Fever with physiologic instability": [
        "Review infection or fever context if clinically relevant",
        "Review hemodynamic stability",
    ],
}

SEVERITY_RANK = {"unknown": 0, "low": 1, "moderate": 2, "high": 3}
SAFETY_NOTE = "This output is clinical decision support only and does not replace clinician judgment."


def get_feature_value(feature_snapshot: list[dict[str, Any]], feature_name: str) -> float | int | None:
    for item in feature_snapshot:
        if item.get("feature_name") == feature_name:
            return item.get("feature_value")
    return None


def _feature_map(feature_snapshot: list[dict[str, Any]]) -> dict[str, float | int | None]:
    return {
        str(item.get("feature_name")): item.get("feature_value")
        for item in feature_snapshot
        if item.get("feature_name") is not None
    }


def _as_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _add_evidence(evidence: list[str], label: str, value: float | int | None, unit: str = "") -> None:
    if value is None:
        return
    if isinstance(value, float) and unit == "":
        evidence.append(f"{label}: {value:.3f}".rstrip("0").rstrip("."))
        return
    formatted = f"{value:g}" if isinstance(value, float) else str(value)
    evidence.append(f"{label}: {formatted}{unit}")


def _severity_max(*levels: str) -> str:
    return max(levels, key=lambda item: SEVERITY_RANK[item])


def _risk_priority(risk_score: float) -> tuple[str, str]:
    if risk_score >= 0.70:
        return (
            "High concern - clinician review recommended",
            "High predicted risk; clinician review recommended.",
        )
    if risk_score >= 0.40:
        return (
            "Intermediate concern - review contributing clinical signals",
            "Intermediate risk; review contributing clinical signals.",
        )
    return (
        "Lower concern - continue routine clinical review",
        "No strong model signal of escalation risk.",
    )


def detect_respiratory_signal(features: dict[str, float | int | None]) -> dict[str, Any] | None:
    triage_o2sat = _as_float(features.get("triage_o2sat"))
    spo2_mean = _as_float(features.get("spo2_mean"))
    spo2_min = _as_float(features.get("spo2_min"))
    rr_mean = _as_float(features.get("rr_mean"))
    triage_rr = _as_float(features.get("triage_resprate"))
    hypoxia_count = _as_float(features.get("hypoxia_count")) or 0.0

    if all(value is None for value in (triage_o2sat, spo2_mean, spo2_min, rr_mean, triage_rr)) and hypoxia_count == 0:
        return None

    severity = "unknown"
    if any(value is not None and value < 90 for value in (triage_o2sat, spo2_mean, spo2_min)) or rr_mean is not None and rr_mean >= 30:
        severity = "high"
    elif hypoxia_count > 0 or any(value is not None and value < 92 for value in (triage_o2sat, spo2_mean, spo2_min)) or rr_mean is not None and rr_mean >= 24 or triage_rr is not None and triage_rr >= 24:
        severity = "moderate"
    elif any(value is not None and value < 95 for value in (triage_o2sat, spo2_mean, spo2_min)) or rr_mean is not None and rr_mean > 20 or triage_rr is not None and triage_rr > 20:
        severity = "low"

    if severity == "unknown":
        return None

    evidence: list[str] = []
    _add_evidence(evidence, "Triage SpO2", triage_o2sat, "%")
    _add_evidence(evidence, "Mean SpO2", spo2_mean, "%")
    _add_evidence(evidence, "Minimum SpO2", spo2_min, "%")
    _add_evidence(evidence, "Triage respiratory rate", triage_rr, " bpm")
    _add_evidence(evidence, "Mean respiratory rate", rr_mean, " bpm")
    if hypoxia_count > 0:
        evidence.append(f"Hypoxia episodes: {hypoxia_count:g}")

    if any(value is not None and value < 92 for value in (triage_o2sat, spo2_mean, spo2_min)) and (rr_mean is not None and rr_mean > 20 or triage_rr is not None and triage_rr > 20):
        message = "Reduced oxygen saturation with elevated respiratory rate suggests a respiratory compromise pattern."
    elif any(value is not None and value < 95 for value in (triage_o2sat, spo2_mean, spo2_min)):
        message = "Oxygen saturation is below ideal range and contributed to elevated respiratory risk."
    else:
        message = "Respiratory status should be reviewed in clinical context."

    return {
        "indicator": "Respiratory compromise signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
    }


def detect_tachycardia_signal(features: dict[str, float | int | None]) -> dict[str, Any] | None:
    triage_hr = _as_float(features.get("triage_heartrate"))
    hr_mean = _as_float(features.get("hr_mean"))
    hr_min = _as_float(features.get("hr_min"))
    hr_max = _as_float(features.get("hr_max"))
    hr_slope = _as_float(features.get("hr_slope"))
    bp_slope = _as_float(features.get("bp_slope"))
    tachycardia_count = _as_float(features.get("tachycardia_count")) or 0.0

    if all(value is None for value in (triage_hr, hr_mean, hr_min, hr_max, hr_slope)) and tachycardia_count == 0:
        return None

    persistent_tachy = sum(
        1 for value in (triage_hr, hr_mean, hr_min) if value is not None and value > 100
    ) >= 2
    severe_tachy = any(value is not None and value >= 120 for value in (triage_hr, hr_mean, hr_max))
    mild_tachy = any(value is not None and value > 100 for value in (triage_hr, hr_mean, hr_max))
    bradycardia = any(value is not None and value < 60 for value in (triage_hr, hr_mean, hr_min))

    severity = "unknown"
    if severe_tachy or (hr_slope is not None and hr_slope > 0 and bp_slope is not None and bp_slope < 0):
        severity = "high"
    elif persistent_tachy or any(value is not None and 110 <= value < 120 for value in (triage_hr, hr_mean, hr_max)) or tachycardia_count > 0:
        severity = "moderate"
    elif mild_tachy or bradycardia:
        severity = "low"

    if severity == "unknown":
        return None

    evidence: list[str] = []
    _add_evidence(evidence, "Triage heart rate", triage_hr, " bpm")
    _add_evidence(evidence, "Mean heart rate", hr_mean, " bpm")
    _add_evidence(evidence, "Minimum heart rate", hr_min, " bpm")
    _add_evidence(evidence, "Maximum heart rate", hr_max, " bpm")
    _add_evidence(evidence, "Heart-rate slope", hr_slope)
    if tachycardia_count > 0:
        evidence.append(f"Tachycardia episodes: {tachycardia_count:g}")

    if bradycardia and not mild_tachy:
        message = "Heart rate is below the typical adult range and should be reviewed in context."
    elif hr_slope is not None and hr_slope > 0:
        message = "Heart rate is elevated with a rising trend, suggesting physiologic stress and possible deterioration."
    else:
        message = "Heart rate is elevated, suggesting physiologic stress."

    return {
        "indicator": "Tachycardia / physiologic stress",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
    }


def detect_hemodynamic_signal(features: dict[str, float | int | None]) -> dict[str, Any] | None:
    triage_sbp = _as_float(features.get("triage_sbp"))
    sbp_mean = _as_float(features.get("sbp_mean"))
    sbp_min = _as_float(features.get("sbp_min"))
    sbp_max = _as_float(features.get("sbp_max"))
    bp_slope = _as_float(features.get("bp_slope"))
    hr_slope = _as_float(features.get("hr_slope"))
    hypotension_count = _as_float(features.get("hypotension_count")) or 0.0

    if all(value is None for value in (triage_sbp, sbp_mean, sbp_min, sbp_max, bp_slope)) and hypotension_count == 0:
        return None

    severity = "unknown"
    if any(value is not None and value < 90 for value in (triage_sbp, sbp_min)) or hypotension_count > 0 or (bp_slope is not None and bp_slope < 0 and hr_slope is not None and hr_slope > 0):
        severity = "high"
    elif bp_slope is not None and bp_slope < 0 or any(value is not None and 90 <= value <= 100 for value in (triage_sbp, sbp_mean, sbp_min)):
        severity = "moderate"
    elif any(value is not None and 100 < value <= 105 for value in (triage_sbp, sbp_mean, sbp_min)):
        severity = "low"

    if severity == "unknown":
        return None

    evidence: list[str] = []
    _add_evidence(evidence, "Triage SBP", triage_sbp, " mmHg")
    _add_evidence(evidence, "Mean SBP", sbp_mean, " mmHg")
    _add_evidence(evidence, "Minimum SBP", sbp_min, " mmHg")
    _add_evidence(evidence, "Maximum SBP", sbp_max, " mmHg")
    _add_evidence(evidence, "Blood-pressure slope", bp_slope)
    if hypotension_count > 0:
        evidence.append(f"Hypotension episodes: {hypotension_count:g}")

    if bp_slope is not None and bp_slope < 0 and hr_slope is not None and hr_slope > 0:
        message = "Falling systolic blood pressure with rising heart rate may indicate hemodynamic deterioration."
    elif any(value is not None and value < 90 for value in (triage_sbp, sbp_min)) or hypotension_count > 0:
        message = "Lowest systolic blood pressure suggests a hypotensive episode."
    else:
        message = "Review hemodynamic stability and perfusion status."

    return {
        "indicator": "Hemodynamic instability signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
    }


def detect_fever_signal(features: dict[str, float | int | None]) -> dict[str, Any] | None:
    triage_temp = _as_float(features.get("triage_temperature"))
    temp_mean = _as_float(features.get("temperature_mean"))
    temp_min = _as_float(features.get("temperature_min"))
    temp_max = _as_float(features.get("temperature_max"))
    fever_count = _as_float(features.get("fever_count")) or 0.0
    triage_hr = _as_float(features.get("triage_heartrate"))
    triage_rr = _as_float(features.get("triage_resprate"))
    shock_index = _as_float(features.get("shock_index_max")) or _as_float(features.get("shock_index"))
    sbp_min = _as_float(features.get("sbp_min")) or _as_float(features.get("triage_sbp"))
    spo2_min = _as_float(features.get("spo2_min")) or _as_float(features.get("triage_o2sat"))

    if all(value is None for value in (triage_temp, temp_mean, temp_min, temp_max)) and fever_count == 0:
        return None

    fever_present = any(value is not None and value >= 100.4 for value in (triage_temp, temp_mean, temp_max)) or fever_count > 0
    high_fever = any(value is not None and value >= 102.2 for value in (triage_temp, temp_max))
    hypothermia = any(value is not None and value < 96.8 for value in (triage_temp, temp_min))
    tachy = triage_hr is not None and triage_hr > 100
    tachypnea = triage_rr is not None and triage_rr > 20
    instability = (shock_index is not None and shock_index >= 0.9) or (sbp_min is not None and sbp_min < 90) or (spo2_min is not None and spo2_min < 92)

    severity = "unknown"
    if high_fever or (fever_present and instability):
        severity = "high"
    elif fever_present and (tachy or tachypnea):
        severity = "moderate"
    elif fever_present or hypothermia:
        severity = "low"

    if severity == "unknown":
        return None

    evidence: list[str] = []
    _add_evidence(evidence, "Triage temperature", triage_temp, " F")
    _add_evidence(evidence, "Mean temperature", temp_mean, " F")
    _add_evidence(evidence, "Minimum temperature", temp_min, " F")
    _add_evidence(evidence, "Maximum temperature", temp_max, " F")
    if fever_count > 0:
        evidence.append(f"Fever episodes: {fever_count:g}")

    if fever_present and (tachy or tachypnea):
        message = "Fever combined with tachycardia or tachypnea may suggest systemic physiologic stress."
    elif fever_present:
        message = "Fever signal present; review infection source and clinical context."
    else:
        message = "Temperature pattern should be reviewed in clinical context."

    return {
        "indicator": "Fever / infection-context signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
    }


def detect_shock_index_signal(features: dict[str, float | int | None]) -> dict[str, Any] | None:
    triage_shock = _as_float(features.get("triage_shock_index"))
    shock_index = _as_float(features.get("shock_index"))
    shock_index_max = _as_float(features.get("shock_index_max"))

    values = [value for value in (triage_shock, shock_index, shock_index_max) if value is not None]
    if not values:
        return None

    peak = max(values)
    if peak >= 1.0:
        severity = "high"
    elif peak >= 0.9:
        severity = "moderate"
    elif peak >= 0.7:
        severity = "low"
    else:
        return None

    evidence: list[str] = []
    _add_evidence(evidence, "Triage shock index", triage_shock)
    _add_evidence(evidence, "Shock index", shock_index)
    _add_evidence(evidence, "Maximum shock index", shock_index_max)

    return {
        "indicator": "Elevated shock index / hemodynamic strain",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": "Shock index is elevated; review hemodynamic status in clinical context.",
    }


def detect_deterioration_trend(features: dict[str, float | int | None], risk_score: float) -> dict[str, Any] | None:
    hr_slope = _as_float(features.get("hr_slope"))
    bp_slope = _as_float(features.get("bp_slope"))
    vital_row_count = _as_float(features.get("vital_row_count"))
    spo2_min = _as_float(features.get("spo2_min"))
    spo2_mean = _as_float(features.get("spo2_mean"))

    if all(value is None for value in (hr_slope, bp_slope, vital_row_count, spo2_min, spo2_mean)):
        return None

    severity = "unknown"
    if hr_slope is not None and hr_slope > 0 and bp_slope is not None and bp_slope < 0 and risk_score >= 0.70:
        severity = "high"
    elif hr_slope is not None and hr_slope > 0 or bp_slope is not None and bp_slope < 0:
        severity = "moderate"
    elif spo2_min is not None and spo2_mean is not None and spo2_min < spo2_mean:
        severity = "low"

    if severity == "unknown":
        return None

    evidence: list[str] = []
    _add_evidence(evidence, "Heart-rate slope", hr_slope)
    _add_evidence(evidence, "Blood-pressure slope", bp_slope)
    _add_evidence(evidence, "Vital observations", vital_row_count)
    if spo2_min is not None and spo2_mean is not None and spo2_min < spo2_mean:
        evidence.append(f"Oxygen saturation dips: min {spo2_min:g}% vs mean {spo2_mean:g}%")

    if vital_row_count is not None and vital_row_count < 2:
        message = "Trend confidence is limited because few vital measurements are available."
    else:
        message = "Vitals show a possible deterioration pattern and should be reviewed in clinical context."

    return {
        "indicator": "Deterioration trend signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
    }


def detect_acuity_signal(features: dict[str, float | int | None], risk_score: float) -> dict[str, Any] | None:
    acuity = _as_float(features.get("acuity"))
    acuity_missing = _as_float(features.get("acuity_missing")) or 0.0

    if acuity is None and acuity_missing == 0:
        return None

    if acuity_missing >= 1:
        return {
            "indicator": "Triage acuity concern",
            "severity": "moderate",
            "evidence": ["Triage acuity unavailable"],
            "doctor_message": "Acuity is unavailable, so the prediction relies more heavily on vitals and missingness patterns.",
        }

    if acuity is None:
        return None

    if acuity <= 2:
        severity = "high"
        message = "Triage acuity indicates high initial urgency."
    elif acuity == 3 and risk_score >= 0.70:
        severity = "moderate"
        message = "Model risk is higher than acuity alone suggests; review physiologic contributors."
    elif acuity >= 4 and risk_score >= 0.70:
        severity = "low"
        message = "Model risk is elevated despite lower triage acuity; review the contributing physiologic signals."
    else:
        return None

    return {
        "indicator": "Triage acuity concern",
        "severity": severity,
        "evidence": [f"Triage acuity: {acuity:g}"],
        "doctor_message": message,
    }


def detect_sparse_data_signal(features: dict[str, float | int | None]) -> dict[str, Any] | None:
    vital_row_count = _as_float(features.get("vital_row_count"))
    triage_missing_count = sum(
        int((_as_float(features.get(name)) or 0) >= 1)
        for name in (
            "triage_temperature_missing",
            "triage_heartrate_missing",
            "triage_resprate_missing",
            "triage_o2sat_missing",
            "triage_sbp_missing",
            "triage_dbp_missing",
        )
    )
    missing_rates = {
        name: _as_float(features.get(name))
        for name in (
            "temperature_missing_rate",
            "heartrate_missing_rate",
            "resprate_missing_rate",
            "o2sat_missing_rate",
            "sbp_missing_rate",
            "dbp_missing_rate",
        )
    }
    high_missing = [name for name, value in missing_rates.items() if value is not None and value > 0.5]
    acuity_missing = (_as_float(features.get("acuity_missing")) or 0.0) >= 1

    if vital_row_count is None and triage_missing_count == 0 and not high_missing and not acuity_missing:
        return None

    severity = "unknown"
    if (vital_row_count is not None and vital_row_count < 2 and (triage_missing_count >= 3 or acuity_missing)) or len(high_missing) >= 2:
        severity = "high"
    elif vital_row_count is not None and vital_row_count < 2 or high_missing or triage_missing_count >= 3:
        severity = "moderate"
    elif triage_missing_count > 0 or acuity_missing:
        severity = "low"

    if severity == "unknown":
        return None

    evidence: list[str] = []
    _add_evidence(evidence, "Vital observations", vital_row_count)
    if triage_missing_count:
        evidence.append(f"Missing triage vital fields: {triage_missing_count}")
    for name in high_missing:
        evidence.append(f"{FEATURE_TO_CLINICAL_MEANING.get(name, name)} > 50%")
    if acuity_missing:
        evidence.append("Triage acuity unavailable")

    return {
        "indicator": "Sparse data / low confidence signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": "Prediction confidence may be limited by sparse or missing clinical measurements.",
    }


def detect_composite_patterns(
    features: dict[str, float | int | None],
    base_indicators: list[dict[str, Any]],
    risk_score: float,
) -> list[dict[str, Any]]:
    indicator_names = {indicator["indicator"] for indicator in base_indicators}
    triage_hr = _as_float(features.get("triage_heartrate"))
    hr_mean = _as_float(features.get("hr_mean"))
    triage_rr = _as_float(features.get("triage_resprate"))
    rr_mean = _as_float(features.get("rr_mean"))
    shock_index = _as_float(features.get("shock_index_max")) or _as_float(features.get("shock_index"))
    sbp_min = _as_float(features.get("sbp_min")) or _as_float(features.get("triage_sbp"))
    triage_o2sat = _as_float(features.get("triage_o2sat"))
    spo2_min = _as_float(features.get("spo2_min"))
    hypoxia_count = _as_float(features.get("hypoxia_count")) or 0.0
    fever_count = _as_float(features.get("fever_count")) or 0.0
    temp_max = _as_float(features.get("temperature_max")) or _as_float(features.get("triage_temperature"))
    hr_slope = _as_float(features.get("hr_slope"))
    bp_slope = _as_float(features.get("bp_slope"))

    tachy = any(value is not None and value > 100 for value in (triage_hr, hr_mean))
    tachypnea = any(value is not None and value > 20 for value in (triage_rr, rr_mean))
    fever = fever_count > 0 or temp_max is not None and temp_max >= 100.4
    low_spo2 = any(value is not None and value < 94 for value in (triage_o2sat, spo2_min))
    severe_spo2 = any(value is not None and value < 90 for value in (triage_o2sat, spo2_min))
    shock_borderline = shock_index is not None and shock_index >= 0.9
    shock_high = shock_index is not None and shock_index >= 1.0
    hypotension = sbp_min is not None and sbp_min < 90

    patterns: list[dict[str, Any]] = []

    if fever and tachy and tachypnea:
        severity = "high" if shock_borderline or hypotension or low_spo2 or risk_score >= 0.70 else "moderate"
        patterns.append(
            {
                "indicator": "Sepsis-like physiology pattern",
                "severity": severity,
                "evidence": [
                    "Fever signal present",
                    "Tachycardia signal present",
                    "Tachypnea signal present",
                ],
                "doctor_message": "Pattern suggests infection or sepsis-like physiology. Review clinical context, suspected source, available labs, and clinician assessment.",
            }
        )

    if low_spo2 or hypoxia_count > 0 or tachypnea:
        severity = "high" if severe_spo2 or (low_spo2 and tachypnea) else "moderate"
        patterns.append(
            {
                "indicator": "Respiratory deterioration pattern",
                "severity": severity,
                "evidence": [
                    "Reduced oxygen saturation or hypoxia episodes detected",
                    "Respiratory rate elevation present" if tachypnea else "Respiratory status signal present",
                ],
                "doctor_message": "Respiratory compromise pattern detected. Review oxygen requirement, work of breathing, and relevant clinical context.",
            }
        )

    if hypotension or shock_high or (bp_slope is not None and bp_slope < 0 and hr_slope is not None and hr_slope > 0):
        patterns.append(
            {
                "indicator": "Hemodynamic instability pattern",
                "severity": "high" if shock_high or hypotension else "moderate",
                "evidence": [
                    "Low or falling systolic blood pressure pattern",
                    "Elevated shock index or rising heart rate pattern",
                ],
                "doctor_message": "Hemodynamic instability pattern detected. Review perfusion status and escalation need in clinical context.",
            }
        )

    if (tachy and low_spo2) or (tachy and tachypnea) or shock_borderline:
        patterns.append(
            {
                "indicator": "Cardiopulmonary stress pattern",
                "severity": "high" if shock_high or (tachy and severe_spo2) else "moderate",
                "evidence": [
                    "Heart-rate stress signal present",
                    "Oxygenation or respiratory stress signal present",
                ],
                "doctor_message": "Cardiopulmonary stress pattern detected. Model risk appears driven by heart-rate, oxygenation, and blood-pressure physiology.",
            }
        )

    if fever and (tachy or tachypnea or shock_borderline or hypotension):
        patterns.append(
            {
                "indicator": "Fever with physiologic instability",
                "severity": "high" if hypotension or shock_high or low_spo2 else "moderate",
                "evidence": [
                    "Fever signal present",
                    "Physiologic instability signal present",
                ],
                "doctor_message": "Fever is present together with physiologic instability signals. Review infection, inflammatory state, dehydration, pain, or other clinical causes.",
            }
        )

    deduped: list[dict[str, Any]] = []
    seen = set(indicator_names)
    for pattern in patterns:
        if pattern["indicator"] in seen:
            continue
        seen.add(pattern["indicator"])
        deduped.append(pattern)
    return deduped


def build_dominant_clinical_drivers(top_contributors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    for contributor in top_contributors:
        feature_name = str(contributor.get("feature_name"))
        feature_value = contributor.get("feature_value")
        direction = str(contributor.get("contribution_direction", "unknown"))
        meaning = FEATURE_TO_CLINICAL_MEANING.get(feature_name, feature_name.replace("_", " ").title())

        if direction == "increases_risk":
            message = f"{meaning} contributed to elevated model-estimated risk."
        elif direction == "decreases_risk":
            message = f"{meaning} reduced the model-estimated risk."
        else:
            message = f"{meaning} was included in the model context."

        drivers.append(
            {
                "feature_name": feature_name,
                "clinical_meaning": meaning,
                "feature_value": feature_value,
                "contribution_direction": direction,
                "doctor_message": message,
            }
        )
    return drivers


def _build_review_focus(indicators: list[dict[str, Any]]) -> list[str]:
    focus: list[str] = []
    for indicator in indicators:
        for item in REVIEW_FOCUS_MAP.get(indicator["indicator"], []):
            if item not in focus:
                focus.append(item)
    return focus


def _summary_interpretation(risk_score: float, indicators: list[dict[str, Any]]) -> str:
    if risk_score < 0.40 and not indicators:
        return "No strong model signal of escalation risk."

    phrases = []
    for indicator in indicators:
        name = indicator["indicator"]
        if "Respiratory" in name:
            phrases.append("respiratory")
        elif "Tachycardia" in name:
            phrases.append("heart-rate")
        elif "Hemodynamic" in name or "shock index" in name.lower():
            phrases.append("hemodynamic strain")
        elif "Fever" in name or "Sepsis-like" in name:
            phrases.append("infection-context")
        elif "Sparse data" in name:
            phrases.append("data-quality")
        if len(phrases) == 3:
            break

    if not phrases:
        return _risk_priority(risk_score)[1]

    unique_phrases = []
    for phrase in phrases:
        if phrase not in unique_phrases:
            unique_phrases.append(phrase)
    return f"High predicted risk with {', '.join(unique_phrases)} signals." if risk_score >= 0.70 else f"Risk signal influenced by {', '.join(unique_phrases)} findings."


def build_clinical_interpretation(
    risk_score: float,
    risk_label: str,
    feature_snapshot: list[dict[str, Any]],
    top_contributors: list[dict[str, Any]],
) -> dict[str, Any]:
    features = _feature_map(feature_snapshot)
    risk_score = round(float(risk_score), 3)
    clinical_priority, default_interpretation = _risk_priority(risk_score)

    base_indicators = [
        indicator
        for indicator in (
            detect_respiratory_signal(features),
            detect_tachycardia_signal(features),
            detect_hemodynamic_signal(features),
            detect_fever_signal(features),
            detect_shock_index_signal(features),
            detect_deterioration_trend(features, risk_score),
            detect_acuity_signal(features, risk_score),
            detect_sparse_data_signal(features),
        )
        if indicator is not None
    ]
    all_indicators = base_indicators + detect_composite_patterns(features, base_indicators, risk_score)
    all_indicators.sort(key=lambda item: SEVERITY_RANK[item["severity"]], reverse=True)

    dominant_drivers = build_dominant_clinical_drivers(top_contributors)
    review_focus = _build_review_focus(all_indicators)

    return {
        "risk_summary": {
            "risk_label": risk_label,
            "risk_score": risk_score,
            "clinical_priority": clinical_priority,
            "interpretation": _summary_interpretation(risk_score, all_indicators) or default_interpretation,
        },
        "clinical_indicators": all_indicators,
        "dominant_clinical_drivers": dominant_drivers,
        "recommended_review_focus": review_focus,
        "safety_note": SAFETY_NOTE,
    }
