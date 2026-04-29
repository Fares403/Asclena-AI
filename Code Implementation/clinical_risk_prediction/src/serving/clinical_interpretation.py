"""Deterministic clinical interpretation layer for risk model outputs."""

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

DATA_CONTEXT_FEATURES = {
    "vital_row_count",
    "temperature_missing_rate",
    "heartrate_missing_rate",
    "resprate_missing_rate",
    "o2sat_missing_rate",
    "sbp_missing_rate",
    "dbp_missing_rate",
    "triage_temperature_missing",
    "triage_heartrate_missing",
    "triage_resprate_missing",
    "triage_o2sat_missing",
    "triage_sbp_missing",
    "triage_dbp_missing",
    "acuity_missing",
}

REVIEW_FOCUS_MESSAGES = {
    "respiratory": "Review respiratory status, oxygen requirement, and work of breathing",
    "tachycardia": "Review causes of tachycardia such as pain, fever, dehydration, hypoxia, or infection",
    "hemodynamic": "Review hemodynamic stability and perfusion status",
    "trends": "Review repeated vital-sign trends for deterioration",
    "triage": "Review triage urgency in context of current physiology",
    "infection": "Review infection or fever context if clinically relevant",
    "sepsis": "Review sepsis context and available labs if clinically relevant",
    "data_quality": "Review missing or sparse vital-sign data before relying on model output",
}

RISK_SCORE_NOTE = (
    "Calibrated probability rounded for display; this should not be interpreted as absolute certainty."
)

SAFETY_NOTE = "This output is clinical decision support only and does not replace clinician judgment."


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_map(feature_snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    return {item["feature_name"]: item.get("feature_value") for item in feature_snapshot}


def _severity_rank(value: str) -> int:
    return {"unknown": 0, "low": 1, "moderate": 2, "high": 3}.get(value, 0)


def _max_severity(*values: str) -> str:
    ranked = max(values, key=_severity_rank, default="unknown")
    return ranked


def _display_risk_score(risk_score: float) -> str:
    if risk_score >= 0.995:
        return ">0.99"
    return f"{risk_score:.3f}"


def _risk_priority(risk_label: str) -> str:
    if risk_label == "HIGH":
        return "High concern - clinician review recommended"
    if risk_label == "MODERATE":
        return "Intermediate concern - review contributing clinical signals"
    return "Lower concern - no strong model signal of escalation risk"


def _format_number(value: float | None, decimals: int = 1) -> str | None:
    if value is None:
        return None
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.{decimals}f}"


def _evidence_line(label: str, value: float | None, unit: str = "", decimals: int = 1) -> str | None:
    number = _format_number(value, decimals=decimals)
    if number is None:
        return None
    suffix = unit if not unit else f" {unit}"
    return f"{label}: {number}{suffix}"


def _build_data_quality(features: dict[str, Any]) -> dict[str, Any]:
    vital_count = _as_float(features.get("vital_row_count"))
    triage_missing_flags = [
        "triage_temperature_missing",
        "triage_heartrate_missing",
        "triage_resprate_missing",
        "triage_o2sat_missing",
        "triage_sbp_missing",
        "triage_dbp_missing",
        "acuity_missing",
    ]
    missing_rate_features = [
        ("temperature_missing_rate", "temperature"),
        ("heartrate_missing_rate", "heart rate"),
        ("resprate_missing_rate", "respiratory rate"),
        ("o2sat_missing_rate", "oxygen saturation"),
        ("sbp_missing_rate", "systolic blood pressure"),
        ("dbp_missing_rate", "diastolic blood pressure"),
    ]
    triage_missing_count = sum(1 for name in triage_missing_flags if _as_float(features.get(name)) == 1.0)
    missing_features = [
        label for name, label in missing_rate_features if (_as_float(features.get(name)) or 0.0) > 0.5
    ]
    missingness_concern = bool(missing_features or triage_missing_count >= 3)

    if vital_count is None:
        trend_interpretability = "unknown"
    elif vital_count >= 5:
        trend_interpretability = "high"
    elif vital_count >= 3:
        trend_interpretability = "moderate"
    elif vital_count < 2:
        trend_interpretability = "low"
    else:
        trend_interpretability = "low"

    if missingness_concern:
        note = "Prediction confidence may be limited by missing or sparse vital-sign data."
    elif vital_count is not None and vital_count >= 3:
        note = (
            "Repeated vital-sign observations are available; trend interpretation is possible but should be "
            "reviewed in clinical context."
        )
    elif vital_count is not None and vital_count >= 1:
        note = "Limited repeated vital-sign data are available; trend interpretation should be cautious."
    else:
        note = "Vital-sign observation density is unclear from the available feature set."

    return {
        "vital_observation_count": None if vital_count is None else int(vital_count),
        "trend_interpretability": trend_interpretability,
        "missingness_concern": missingness_concern,
        "missing_features": missing_features,
        "data_quality_note": note,
    }


def detect_respiratory_signal(features: dict[str, Any]) -> dict[str, Any] | None:
    triage_o2 = _as_float(features.get("triage_o2sat"))
    spo2_mean = _as_float(features.get("spo2_mean"))
    spo2_min = _as_float(features.get("spo2_min"))
    triage_rr = _as_float(features.get("triage_resprate"))
    rr_mean = _as_float(features.get("rr_mean"))
    hypoxia_count = _as_float(features.get("hypoxia_count")) or 0.0

    low_spo2 = min(value for value in [triage_o2, spo2_mean, spo2_min] if value is not None) if any(
        value is not None for value in [triage_o2, spo2_mean, spo2_min]
    ) else None
    high_rr = max(value for value in [triage_rr, rr_mean] if value is not None) if any(
        value is not None for value in [triage_rr, rr_mean]
    ) else None

    if low_spo2 is None and high_rr is None and hypoxia_count <= 0:
        return None

    severity = "unknown"
    if low_spo2 is not None:
        if low_spo2 < 90:
            severity = "high"
        elif low_spo2 < 92:
            severity = _max_severity(severity, "moderate")
        elif low_spo2 <= 94:
            severity = _max_severity(severity, "low")
    if high_rr is not None:
        if high_rr >= 30:
            severity = "high"
        elif high_rr >= 24:
            severity = _max_severity(severity, "moderate")
        elif high_rr > 20:
            severity = _max_severity(severity, "low")
    if hypoxia_count >= 2:
        severity = "high" if low_spo2 is not None and low_spo2 < 90 else _max_severity(severity, "moderate")
    elif hypoxia_count > 0:
        severity = _max_severity(severity, "moderate")
    if severity == "unknown":
        return None

    evidence = [
        _evidence_line("Triage SpO2", triage_o2, "%"),
        _evidence_line("Mean SpO2", spo2_mean, "%"),
        _evidence_line("Minimum SpO2", spo2_min, "%"),
        _evidence_line("Triage respiratory rate", triage_rr, "breaths/min"),
        _evidence_line("Mean respiratory rate", rr_mean, "breaths/min"),
        _evidence_line("Hypoxia episodes", hypoxia_count, "episodes", decimals=0) if hypoxia_count > 0 else None,
    ]
    evidence = [item for item in evidence if item is not None]

    if (low_spo2 is not None and low_spo2 < 95) and (high_rr is not None and high_rr > 20):
        message = "Reduced oxygen saturation with elevated respiratory rate suggests a respiratory compromise signal."
    elif low_spo2 is not None and low_spo2 < 95:
        message = "Oxygen saturation is below ideal range and should be reviewed in clinical context."
    elif high_rr is not None and high_rr > 20:
        message = "Respiratory rate is elevated and may reflect respiratory or systemic physiologic stress."
    else:
        message = "Respiratory status should be reviewed in clinical context."

    return {
        "indicator": "Respiratory compromise signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"respiratory"},
    }


def detect_tachycardia_signal(features: dict[str, Any]) -> dict[str, Any] | None:
    triage_hr = _as_float(features.get("triage_heartrate"))
    hr_mean = _as_float(features.get("hr_mean"))
    hr_min = _as_float(features.get("hr_min"))
    hr_slope = _as_float(features.get("hr_slope"))
    tachy_count = _as_float(features.get("tachycardia_count")) or 0.0
    bp_slope = _as_float(features.get("bp_slope"))

    high_hr = max(value for value in [triage_hr, hr_mean, hr_min] if value is not None) if any(
        value is not None for value in [triage_hr, hr_mean, hr_min]
    ) else None
    if high_hr is None and hr_slope is None and tachy_count <= 0:
        return None

    severity = "unknown"
    if high_hr is not None:
        if high_hr >= 120:
            severity = "high"
        elif high_hr >= 110:
            severity = "moderate"
        elif high_hr > 100:
            severity = "low"
    if tachy_count > 0 and (hr_min is not None and hr_min > 100):
        severity = _max_severity(severity, "moderate")
    elif tachy_count > 0:
        severity = _max_severity(severity, "low")
    if (hr_slope or 0.0) > 0 and (bp_slope or 0.0) < 0:
        severity = "high"
    elif (hr_slope or 0.0) > 0:
        severity = _max_severity(severity, "moderate")
    if severity == "unknown":
        return None

    evidence = [
        _evidence_line("Triage heart rate", triage_hr, "bpm"),
        _evidence_line("Mean heart rate", hr_mean, "bpm"),
        _evidence_line("Minimum heart rate", hr_min, "bpm"),
        _evidence_line("Heart-rate trend value", hr_slope, decimals=2) if hr_slope is not None and hr_slope > 0 else None,
        _evidence_line("Tachycardia episodes", tachy_count, "episodes", decimals=0) if tachy_count > 0 else None,
    ]
    evidence = [item for item in evidence if item is not None]

    if high_hr is not None and high_hr >= 120:
        message = "Heart rate is markedly elevated, suggesting physiologic stress and possible deterioration risk."
    elif hr_min is not None and hr_min > 100:
        message = "Heart rate is persistently elevated, suggesting physiologic stress."
    else:
        message = "Heart rate is elevated and may reflect pain, fever, dehydration, hypoxia, or hemodynamic stress."

    return {
        "indicator": "Tachycardia / physiologic stress",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"tachycardia"},
    }


def detect_hemodynamic_signal(features: dict[str, Any]) -> dict[str, Any] | None:
    triage_sbp = _as_float(features.get("triage_sbp"))
    sbp_mean = _as_float(features.get("sbp_mean"))
    sbp_min = _as_float(features.get("sbp_min"))
    bp_slope = _as_float(features.get("bp_slope"))
    hr_slope = _as_float(features.get("hr_slope"))
    hypotension_count = _as_float(features.get("hypotension_count")) or 0.0

    lowest_sbp = min(value for value in [triage_sbp, sbp_mean, sbp_min] if value is not None) if any(
        value is not None for value in [triage_sbp, sbp_mean, sbp_min]
    ) else None
    if lowest_sbp is None and bp_slope is None and hypotension_count <= 0:
        return None

    severity = "unknown"
    if lowest_sbp is not None:
        if lowest_sbp < 90:
            severity = "high"
        elif lowest_sbp <= 100:
            severity = "low"
    if hypotension_count > 0:
        severity = "high"
    if (bp_slope or 0.0) < 0:
        severity = _max_severity(severity, "moderate")
    if (bp_slope or 0.0) < 0 and (hr_slope or 0.0) > 0:
        severity = "high"
    if severity == "unknown":
        return None

    evidence = [
        _evidence_line("Triage SBP", triage_sbp, "mmHg"),
        _evidence_line("Mean SBP", sbp_mean, "mmHg"),
        _evidence_line("Minimum SBP", sbp_min, "mmHg"),
        _evidence_line("Blood-pressure trend value", bp_slope, decimals=2) if bp_slope is not None and bp_slope < 0 else None,
        _evidence_line("Hypotension episodes", hypotension_count, "episodes", decimals=0)
        if hypotension_count > 0
        else None,
    ]
    evidence = [item for item in evidence if item is not None]

    if lowest_sbp is not None and lowest_sbp < 90:
        message = "Lowest systolic blood pressure suggests a hypotensive episode and possible hemodynamic instability."
    elif (bp_slope or 0.0) < 0:
        message = "Falling systolic blood pressure trend may indicate hemodynamic deterioration."
    else:
        message = "Blood-pressure findings should be reviewed for hemodynamic stability and perfusion status."

    return {
        "indicator": "Hemodynamic instability signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"hemodynamic"},
    }


def detect_fever_signal(features: dict[str, Any]) -> dict[str, Any] | None:
    triage_temp = _as_float(features.get("triage_temperature"))
    temp_mean = _as_float(features.get("temperature_mean"))
    temp_max = _as_float(features.get("temperature_max"))
    fever_count = _as_float(features.get("fever_count")) or 0.0
    high_temp = max(value for value in [triage_temp, temp_mean, temp_max] if value is not None) if any(
        value is not None for value in [triage_temp, temp_mean, temp_max]
    ) else None
    shock_index = _as_float(features.get("shock_index"))
    low_spo2 = _as_float(features.get("spo2_min"))
    sbp_min = _as_float(features.get("sbp_min"))
    tachy = max(_as_float(features.get("triage_heartrate")) or 0.0, _as_float(features.get("hr_mean")) or 0.0)
    tachypnea = max(_as_float(features.get("triage_resprate")) or 0.0, _as_float(features.get("rr_mean")) or 0.0)

    if high_temp is None and fever_count <= 0:
        return None

    severity = "unknown"
    if high_temp is not None:
        if high_temp >= 102.2:
            severity = "high"
        elif high_temp >= 100.4:
            severity = "low"
        elif high_temp < 96.8:
            severity = "moderate"
    if fever_count > 0:
        severity = _max_severity(severity, "low")
    if fever_count > 0 and (tachy > 100 or tachypnea > 20):
        severity = _max_severity(severity, "moderate")
    if fever_count > 0 and (
        (shock_index or 0.0) >= 0.9 or (sbp_min is not None and sbp_min < 90) or (low_spo2 is not None and low_spo2 < 92)
    ):
        severity = "high"
    if severity == "unknown":
        return None

    evidence = [
        _evidence_line("Triage temperature", triage_temp, "°F"),
        _evidence_line("Mean temperature", temp_mean, "°F"),
        _evidence_line("Maximum temperature", temp_max, "°F"),
        _evidence_line("Fever episodes", fever_count, "episodes", decimals=0) if fever_count > 0 else None,
    ]
    evidence = [item for item in evidence if item is not None]

    if fever_count > 0 and (tachy > 100 or tachypnea > 20):
        message = "Fever is present together with physiologic stress signals and should be reviewed in clinical context."
    elif high_temp is not None and high_temp >= 100.4:
        message = "Fever signal present; review infection source and clinical context."
    else:
        message = "Temperature pattern may support inflammatory or infectious physiology and should be reviewed."

    return {
        "indicator": "Fever / infection-context signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"infection"},
    }


def detect_shock_index_signal(features: dict[str, Any]) -> dict[str, Any] | None:
    triage_si = _as_float(features.get("triage_shock_index"))
    shock_index = _as_float(features.get("shock_index"))
    shock_index_max = _as_float(features.get("shock_index_max"))
    highest_si = max(value for value in [triage_si, shock_index, shock_index_max] if value is not None) if any(
        value is not None for value in [triage_si, shock_index, shock_index_max]
    ) else None
    if highest_si is None:
        return None

    if highest_si >= 1.0:
        severity = "high"
    elif highest_si >= 0.9:
        severity = "moderate"
    elif highest_si >= 0.7:
        severity = "low"
    else:
        return None

    evidence = [
        _evidence_line("Triage shock index", triage_si, decimals=3),
        _evidence_line("Shock index", shock_index, decimals=3),
        _evidence_line("Maximum shock index", shock_index_max, decimals=3),
    ]
    evidence = [item for item in evidence if item is not None]

    if highest_si >= 1.0:
        message = "Shock index is near or above 1.0, which may suggest hemodynamic strain in clinical context."
    else:
        message = "Shock index is elevated and should be reviewed for hemodynamic strain."

    return {
        "indicator": "Elevated shock index / hemodynamic strain",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"hemodynamic"},
    }


def detect_deterioration_trend(features: dict[str, Any], data_quality: dict[str, Any]) -> dict[str, Any] | None:
    hr_slope = _as_float(features.get("hr_slope"))
    bp_slope = _as_float(features.get("bp_slope"))
    vital_count = data_quality["vital_observation_count"]

    if hr_slope is None and bp_slope is None and vital_count is None:
        return None

    evidence = []
    severity = "unknown"
    if hr_slope is not None and hr_slope > 0:
        evidence.append(f"Heart-rate trend value: {_format_number(hr_slope, 2)}")
        severity = _max_severity(severity, "moderate")
    if bp_slope is not None and bp_slope < 0:
        evidence.append(f"Blood-pressure trend value: {_format_number(bp_slope, 2)}")
        severity = _max_severity(severity, "moderate")
    if vital_count is not None:
        evidence.append(f"Vital observations: {vital_count}")
        if vital_count < 2:
            severity = _max_severity(severity, "low")
    if (hr_slope or 0.0) > 0 and (bp_slope or 0.0) < 0:
        severity = "high"

    if not evidence:
        return None

    if vital_count is not None and vital_count < 2:
        message = "Trend confidence is limited because few vital measurements are available."
    elif (hr_slope or 0.0) > 0 and (bp_slope or 0.0) < 0:
        message = "Vitals show a possible deterioration pattern with rising heart rate and falling blood pressure."
    else:
        message = "Repeated vital signs suggest a pattern that should be reviewed for possible deterioration."

    return {
        "indicator": "Deterioration trend signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"trends"},
    }


def detect_acuity_signal(features: dict[str, Any], risk_label: str) -> dict[str, Any] | None:
    acuity = _as_float(features.get("acuity"))
    acuity_missing = _as_float(features.get("acuity_missing"))
    if acuity is None and acuity_missing != 1.0:
        return None

    severity = "unknown"
    evidence = []
    if acuity_missing == 1.0:
        severity = "moderate"
        evidence.append("Triage acuity unavailable")
        message = "Acuity is unavailable; the prediction relies more heavily on vitals and missingness patterns."
    else:
        evidence.append(f"Triage acuity: {int(acuity)}")
        if acuity in {1.0, 2.0}:
            severity = "high"
            message = "Triage acuity indicates high initial urgency."
        elif acuity == 3.0 and risk_label == "HIGH":
            severity = "moderate"
            message = "Model risk is higher than acuity alone suggests; review physiologic contributors."
        else:
            severity = "low"
            message = "Triage acuity should be reviewed in context of the current physiologic pattern."

    return {
        "indicator": "Triage acuity concern",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"triage"},
    }


def detect_sparse_data_signal(features: dict[str, Any], data_quality: dict[str, Any]) -> dict[str, Any] | None:
    vital_count = data_quality["vital_observation_count"]
    missingness_concern = data_quality["missingness_concern"]
    missing_features = data_quality["missing_features"]
    triage_missing_flags = sum(
        1
        for name in [
            "triage_temperature_missing",
            "triage_heartrate_missing",
            "triage_resprate_missing",
            "triage_o2sat_missing",
            "triage_sbp_missing",
            "triage_dbp_missing",
            "acuity_missing",
        ]
        if _as_float(features.get(name)) == 1.0
    )
    if not missingness_concern and not (vital_count is not None and vital_count < 2):
        return None

    severity = "low"
    if missingness_concern and (vital_count is not None and vital_count < 2):
        severity = "high"
    elif missingness_concern or (vital_count is not None and vital_count < 2):
        severity = "moderate"

    evidence = []
    if vital_count is not None:
        evidence.append(f"Vital observations: {vital_count}")
    if missing_features:
        evidence.extend(f"High missingness: {name}" for name in missing_features)
    if triage_missing_flags >= 3:
        evidence.append(f"Missing triage elements: {triage_missing_flags}")

    return {
        "indicator": "Sparse data / low confidence signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": data_quality["data_quality_note"],
        "_focus": {"data_quality"},
    }


def detect_composite_patterns(
    features: dict[str, Any],
    base_indicators: list[dict[str, Any]],
    risk_label: str,
) -> list[dict[str, Any]]:
    indicator_names = {item["indicator"] for item in base_indicators}
    patterns = []
    triage_rr = _as_float(features.get("triage_resprate")) or 0.0
    rr_mean = _as_float(features.get("rr_mean")) or 0.0
    triage_hr = _as_float(features.get("triage_heartrate")) or 0.0
    hr_mean = _as_float(features.get("hr_mean")) or 0.0
    sbp_min = _as_float(features.get("sbp_min"))
    shock_index = max(
        _as_float(features.get("shock_index")) or 0.0,
        _as_float(features.get("shock_index_max")) or 0.0,
        _as_float(features.get("triage_shock_index")) or 0.0,
    )
    low_spo2 = min(
        [value for value in [_as_float(features.get("triage_o2sat")), _as_float(features.get("spo2_min"))] if value is not None],
        default=None,
    )
    hypoxia_count = _as_float(features.get("hypoxia_count")) or 0.0
    fever_count = _as_float(features.get("fever_count")) or 0.0
    bp_slope = _as_float(features.get("bp_slope")) or 0.0
    hr_slope = _as_float(features.get("hr_slope")) or 0.0

    tachypnea = triage_rr > 20 or rr_mean > 20
    tachycardia = triage_hr > 100 or hr_mean > 100
    respiratory_signal = "Respiratory compromise signal" in indicator_names
    hemodynamic_signal = "Hemodynamic instability signal" in indicator_names
    fever_signal = "Fever / infection-context signal" in indicator_names

    if respiratory_signal and ((low_spo2 is not None and low_spo2 < 92) or hypoxia_count > 0 or tachypnea):
        patterns.append(
            {
                "pattern": "Respiratory deterioration pattern",
                "severity": "high" if (low_spo2 is not None and low_spo2 < 90) or (tachypnea and low_spo2 is not None and low_spo2 < 92) else "moderate",
                "evidence": [
                    "Reduced oxygen saturation or hypoxia episodes detected",
                    "Respiratory rate elevation present" if tachypnea else "Respiratory findings should be reviewed",
                ],
                "doctor_message": (
                    "Respiratory compromise pattern detected. Review oxygen requirement, work of breathing, "
                    "and relevant clinical context."
                ),
                "_focus": {"respiratory"},
            }
        )

    if hemodynamic_signal and ((sbp_min is not None and sbp_min < 90) or shock_index >= 1.0 or (bp_slope < 0 and hr_slope > 0)):
        patterns.append(
            {
                "pattern": "Hemodynamic instability pattern",
                "severity": "high",
                "evidence": [
                    "Low or falling systolic pressure detected",
                    "Elevated shock index or rising heart rate with falling blood pressure present",
                ],
                "doctor_message": (
                    "Hemodynamic instability pattern detected: low or falling systolic pressure, elevated shock index, "
                    "or rising heart rate with falling blood pressure. Review perfusion status and escalation need."
                ),
                "_focus": {"hemodynamic", "trends"},
            }
        )

    if (tachycardia and ((low_spo2 is not None and low_spo2 < 95) or tachypnea)) or shock_index >= 0.9:
        patterns.append(
            {
                "pattern": "Cardiopulmonary stress pattern",
                "severity": "high" if shock_index >= 1.0 or (low_spo2 is not None and low_spo2 < 92) else "moderate",
                "evidence": [
                    "Heart-rate stress signal present",
                    "Oxygenation or respiratory-rate concern present" if ((low_spo2 is not None and low_spo2 < 95) or tachypnea) else "Elevated shock index present",
                ],
                "doctor_message": (
                    "Cardiopulmonary stress pattern detected. Model risk appears driven by heart-rate, oxygenation, "
                    "and blood-pressure physiology."
                ),
                "_focus": {"respiratory", "tachycardia", "hemodynamic"},
            }
        )

    if fever_signal and tachycardia and tachypnea:
        severity = "high" if shock_index >= 0.9 or (sbp_min is not None and sbp_min < 90) or respiratory_signal or risk_label == "HIGH" else "moderate"
        patterns.append(
            {
                "pattern": "Sepsis-like physiology pattern",
                "severity": severity,
                "evidence": [
                    "Fever signal present",
                    "Tachycardia signal present",
                    "Tachypnea signal present",
                ],
                "doctor_message": (
                    "Pattern suggests infection or sepsis-like physiology: fever, tachycardia, and tachypnea. "
                    "Review clinical context, suspected infection source, available labs, and clinician assessment."
                ),
                "_focus": {"infection", "sepsis"},
            }
        )

    if fever_count > 0 and (tachycardia or tachypnea or shock_index >= 0.9 or (sbp_min is not None and sbp_min < 90)):
        patterns.append(
            {
                "pattern": "Fever with physiologic instability",
                "severity": "high" if shock_index >= 0.9 or (sbp_min is not None and sbp_min < 90) else "moderate",
                "evidence": [
                    "Fever episodes detected",
                    "Physiologic instability signal present",
                ],
                "doctor_message": (
                    "Fever is present together with physiologic instability signals. Review infection, inflammatory state, "
                    "dehydration, pain, or other clinical causes."
                ),
                "_focus": {"infection", "hemodynamic", "tachycardia"},
            }
        )

    return patterns


def build_dominant_clinical_drivers(top_contributors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drivers = []
    for item in top_contributors:
        feature_name = item["feature_name"]
        driver_type = "data_context" if feature_name in DATA_CONTEXT_FEATURES else "clinical_signal"
        clinical_meaning = FEATURE_TO_CLINICAL_MEANING.get(feature_name, feature_name.replace("_", " ").title())
        direction = item.get("contribution_direction", "neutral")
        feature_value = item.get("feature_value")

        if feature_name == "vital_row_count":
            message = (
                "Multiple vital-sign observations were available, allowing the model to capture repeated or evolving "
                "physiologic abnormalities."
            )
        elif feature_name in {"spo2_mean", "spo2_min", "triage_o2sat"} and direction == "increases_risk":
            message = f"{clinical_meaning} is below ideal range and contributed to elevated respiratory risk."
        elif feature_name in {"shock_index", "shock_index_max", "triage_shock_index"} and direction == "increases_risk":
            message = f"{clinical_meaning} is elevated and contributed to increased hemodynamic strain risk."
        elif direction == "increases_risk":
            message = f"{clinical_meaning} contributed to elevated model-estimated risk."
        elif direction == "decreases_risk":
            message = f"{clinical_meaning} reduced the model-estimated risk."
        else:
            message = f"{clinical_meaning} was included in the model context."

        drivers.append(
            {
                "feature_name": feature_name,
                "clinical_meaning": clinical_meaning,
                "feature_value": feature_value,
                "driver_type": driver_type,
                "contribution_direction": direction,
                "doctor_message": message,
            }
        )
    return drivers


def _dedupe_review_focus(focus_keys: set[str]) -> list[str]:
    ordered_keys = ["respiratory", "tachycardia", "hemodynamic", "trends", "triage", "infection", "sepsis", "data_quality"]
    if "sepsis" in focus_keys:
        focus_keys.discard("infection")
    messages = [REVIEW_FOCUS_MESSAGES[key] for key in ordered_keys if key in focus_keys]
    return messages[:6]


def _build_interpretation_sentence(risk_label: str, indicators: list[dict[str, Any]], patterns: list[dict[str, Any]]) -> str:
    if risk_label == "LOW" and not indicators and not patterns:
        return "No strong model signal of escalation risk."

    domains = []
    indicator_names = {item["indicator"] for item in indicators}
    if "Respiratory compromise signal" in indicator_names:
        domains.append("respiratory")
    if "Tachycardia / physiologic stress" in indicator_names:
        domains.append("heart-rate")
    if (
        "Hemodynamic instability signal" in indicator_names
        or "Elevated shock index / hemodynamic strain" in indicator_names
    ):
        domains.append("hemodynamic strain")
    if "Fever / infection-context signal" in indicator_names:
        domains.append("infection-context")

    if not domains:
        if risk_label == "HIGH":
            return "High predicted risk; clinician review recommended."
        if risk_label == "MODERATE":
            return "Intermediate risk; review contributing clinical signals."
        return "No strong model signal of escalation risk."

    if len(domains) == 1:
        joined = domains[0]
    elif len(domains) == 2:
        joined = f"{domains[0]} and {domains[1]}"
    else:
        joined = ", ".join(domains[:-1]) + f", and {domains[-1]}"

    if risk_label == "HIGH":
        return f"High predicted risk with {joined} signals."
    if risk_label == "MODERATE":
        return f"Intermediate predicted risk with {joined} signals."
    return f"Lower predicted risk overall, with {joined} findings to review in context."


def build_clinical_interpretation(
    risk_score: float,
    risk_label: str,
    feature_snapshot: list[dict[str, Any]],
    top_contributors: list[dict[str, Any]],
) -> dict[str, Any]:
    features = _feature_map(feature_snapshot)
    data_quality = _build_data_quality(features)

    indicators = []
    for detector in (
        detect_respiratory_signal,
        detect_tachycardia_signal,
        detect_hemodynamic_signal,
        detect_fever_signal,
        detect_shock_index_signal,
    ):
        indicator = detector(features)
        if indicator is not None:
            indicators.append(indicator)

    trend_indicator = detect_deterioration_trend(features, data_quality)
    if trend_indicator is not None:
        indicators.append(trend_indicator)

    acuity_indicator = detect_acuity_signal(features, risk_label)
    if acuity_indicator is not None:
        indicators.append(acuity_indicator)

    sparse_indicator = detect_sparse_data_signal(features, data_quality)
    if sparse_indicator is not None:
        indicators.append(sparse_indicator)

    patterns = detect_composite_patterns(features, indicators, risk_label)
    focus_keys = set()
    for item in indicators + patterns:
        focus_keys.update(item.get("_focus", set()))

    clean_indicators = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in indicators
    ]
    clean_patterns = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in patterns
    ]

    return {
        "risk_summary": {
            "risk_label": risk_label,
            "risk_score": round(risk_score, 6),
            "display_risk_score": _display_risk_score(risk_score),
            "risk_score_note": RISK_SCORE_NOTE,
            "clinical_priority": _risk_priority(risk_label),
            "interpretation": _build_interpretation_sentence(risk_label, clean_indicators, clean_patterns),
        },
        "clinical_indicators": clean_indicators,
        "clinical_patterns": clean_patterns,
        "dominant_clinical_drivers": build_dominant_clinical_drivers(top_contributors),
        "data_quality": data_quality,
        "recommended_review_focus": _dedupe_review_focus(focus_keys),
        "safety_note": SAFETY_NOTE,
    }
