"""Deterministic clinical interpretation layer for risk model outputs."""

from __future__ import annotations

from typing import Any


HR_SLOPE_THRESHOLD = 1.0
BP_SLOPE_THRESHOLD = -1.0

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
    "limited_trends": "Review limited repeated vital-sign trend data in clinical context",
    "triage": "Review triage urgency in context of current physiology",
    "infection": "Review infection or fever context if clinically relevant",
    "sepsis": "Review sepsis context and available labs if clinically relevant",
    "data_quality": "Review missing or sparse vital-sign data before relying on model output",
    "context": "Review model output in clinical context",
    "drivers": "Review model drivers, data completeness, and clinical context",
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


def _display_risk_score(risk_score: float) -> str:
    return ">0.99" if risk_score >= 0.995 else f"{risk_score:.3f}"


def _format_number(value: float | None, decimals: int = 1) -> str | None:
    if value is None:
        return None
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.{decimals}f}"


def _evidence_line(label: str, value: float | None, unit: str = "", decimals: int = 1) -> str | None:
    number = _format_number(value, decimals)
    if number is None:
        return None
    suffix = f" {unit}" if unit else ""
    return f"{label}: {number}{suffix}"


def _present(values: list[float | None], fn=max) -> float | None:
    items = [value for value in values if value is not None]
    return fn(items) if items else None


def _build_data_quality(features: dict[str, Any]) -> dict[str, Any]:
    vital_count = _as_float(features.get("vital_row_count"))
    triage_missing_names = [
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
    triage_missing_count = sum(1 for name in triage_missing_names if _as_float(features.get(name)) == 1.0)
    missing_features = [label for name, label in missing_rate_features if (_as_float(features.get(name)) or 0.0) > 0.5]
    missingness_concern = bool(missing_features or triage_missing_count >= 3)

    if vital_count is None:
        trend_interpretability = "unknown"
    elif vital_count >= 5:
        trend_interpretability = "high"
    elif vital_count >= 3:
        trend_interpretability = "moderate"
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
        note = "Trend interpretation is limited because only a small number of repeated vital-sign observations are available."
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
    rr_max = _as_float(features.get("rr_max"))
    hypoxia_count = _as_float(features.get("hypoxia_count")) or 0.0

    low_spo2 = _present([triage_o2, spo2_mean, spo2_min], min)
    high_rr = _present([triage_rr, rr_mean, rr_max], max)

    respiratory_trigger = (
        (low_spo2 is not None and low_spo2 < 95)
        or hypoxia_count > 0
        or (high_rr is not None and high_rr > 20)
    )
    if not respiratory_trigger:
        return None

    severity = "low"
    if (low_spo2 is not None and low_spo2 < 90) or (high_rr is not None and high_rr >= 30) or hypoxia_count >= 2:
        severity = "high"
    elif (low_spo2 is not None and low_spo2 <= 94) or (high_rr is not None and high_rr > 20) or hypoxia_count > 0:
        severity = "moderate"

    evidence = [
        _evidence_line("Triage SpO2", triage_o2, "%"),
        _evidence_line("Mean SpO2", spo2_mean, "%"),
        _evidence_line("Minimum SpO2", spo2_min, "%"),
        _evidence_line("Triage respiratory rate", triage_rr, "breaths/min"),
        _evidence_line("Mean respiratory rate", rr_mean, "breaths/min"),
        _evidence_line("Peak respiratory rate", rr_max, "breaths/min"),
        _evidence_line("Hypoxia episodes", hypoxia_count, "episodes", 0) if hypoxia_count > 0 else None,
    ]
    evidence = [item for item in evidence if item is not None]

    if (low_spo2 is not None and low_spo2 < 95) and (high_rr is not None and high_rr > 20):
        message = "Reduced oxygen saturation with elevated respiratory rate suggests a respiratory compromise signal."
    elif low_spo2 is not None and low_spo2 < 95:
        message = "Oxygen saturation is below ideal range and should be reviewed in clinical context."
    else:
        message = "Respiratory rate is elevated and should be reviewed in clinical context."

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
    hr_max = _as_float(features.get("hr_max"))
    tachy_count = _as_float(features.get("tachycardia_count")) or 0.0
    hypoxia_count = _as_float(features.get("hypoxia_count")) or 0.0
    hypotension_count = _as_float(features.get("hypotension_count")) or 0.0
    fever_count = _as_float(features.get("fever_count")) or 0.0

    high_hr = _present([triage_hr, hr_mean, hr_max], max)
    persistent_tachy = (hr_min is not None and hr_min > 100) or (hr_mean is not None and hr_mean > 100)
    tachy_trigger = (
        (triage_hr is not None and triage_hr > 100)
        or (hr_mean is not None and hr_mean > 100)
        or (hr_max is not None and hr_max > 100)
        or tachy_count > 0
    )
    if not tachy_trigger:
        return None

    severity = "low"
    if (high_hr is not None and high_hr >= 120) or (
        high_hr is not None and high_hr > 100 and (hypoxia_count > 0 or hypotension_count > 0 or fever_count > 0)
    ):
        severity = "high"
    elif (high_hr is not None and high_hr >= 110) or persistent_tachy:
        severity = "moderate"

    evidence = [
        _evidence_line("Triage heart rate", triage_hr, "bpm"),
        _evidence_line("Mean heart rate", hr_mean, "bpm"),
        _evidence_line("Minimum heart rate", hr_min, "bpm"),
        _evidence_line("Peak heart rate", hr_max, "bpm"),
        _evidence_line("Tachycardia episodes", tachy_count, "episodes", 0) if tachy_count > 0 else None,
    ]
    evidence = [item for item in evidence if item is not None]

    if high_hr is not None and high_hr >= 120:
        message = "Heart rate is markedly elevated, suggesting physiologic stress."
    elif persistent_tachy:
        message = "Heart rate is persistently elevated, suggesting physiologic stress."
    else:
        message = "Heart rate is elevated and should be reviewed in clinical context."

    return {
        "indicator": "Tachycardia / physiologic stress",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"tachycardia"},
    }


def detect_hemodynamic_signal(features: dict[str, Any], risk_label: str) -> dict[str, Any] | None:
    triage_sbp = _as_float(features.get("triage_sbp"))
    sbp_mean = _as_float(features.get("sbp_mean"))
    sbp_min = _as_float(features.get("sbp_min"))
    hypotension_count = _as_float(features.get("hypotension_count")) or 0.0
    shock_index = _present(
        [
            _as_float(features.get("triage_shock_index")),
            _as_float(features.get("shock_index")),
            _as_float(features.get("shock_index_max")),
        ],
        max,
    )
    tachy_trigger = (
        (_as_float(features.get("triage_heartrate")) or 0.0) > 100
        or (_as_float(features.get("hr_mean")) or 0.0) > 100
        or (_as_float(features.get("hr_max")) or 0.0) > 100
        or (_as_float(features.get("tachycardia_count")) or 0.0) > 0
    )
    lowest_sbp = _present([triage_sbp, sbp_mean, sbp_min], min)

    hemodynamic_trigger = (
        (lowest_sbp is not None and lowest_sbp < 100)
        or hypotension_count > 0
        or (shock_index is not None and shock_index >= 0.9)
    )
    if not hemodynamic_trigger:
        return None

    severity = "low"
    if (shock_index is not None and shock_index >= 1.0) or (
        lowest_sbp is not None and lowest_sbp < 90 and tachy_trigger
    ) or (hypotension_count > 0 and risk_label == "HIGH"):
        severity = "high"
    elif (lowest_sbp is not None and lowest_sbp < 90) or (shock_index is not None and shock_index >= 0.9):
        severity = "moderate"

    evidence = [
        _evidence_line("Triage SBP", triage_sbp, "mmHg"),
        _evidence_line("Mean SBP", sbp_mean, "mmHg"),
        _evidence_line("Minimum SBP", sbp_min, "mmHg"),
        _evidence_line("Hypotension episodes", hypotension_count, "episodes", 0) if hypotension_count > 0 else None,
        _evidence_line("Maximum shock index", shock_index, decimals=3) if shock_index is not None else None,
    ]
    evidence = [item for item in evidence if item is not None]

    if lowest_sbp is not None and lowest_sbp < 90:
        message = "Systolic blood pressure is below expected range and may suggest hemodynamic instability."
    elif shock_index is not None and shock_index >= 0.9:
        message = "Hemodynamic strain markers are elevated and should be reviewed in clinical context."
    else:
        message = "Borderline-low systolic blood pressure should be reviewed in clinical context."

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
    temp_min = _as_float(features.get("temperature_min"))
    temp_max = _as_float(features.get("temperature_max"))
    fever_count = _as_float(features.get("fever_count")) or 0.0
    high_temp = _present([triage_temp, temp_mean, temp_max], max)
    low_temp = _present([triage_temp, temp_min], min)

    fever_trigger = (
        (high_temp is not None and high_temp >= 100.4)
        or fever_count > 0
        or (low_temp is not None and low_temp < 96.8)
    )
    if not fever_trigger:
        return None

    tachy_trigger = (
        (_as_float(features.get("triage_heartrate")) or 0.0) > 100
        or (_as_float(features.get("hr_mean")) or 0.0) > 100
    )
    tachypnea_trigger = (
        (_as_float(features.get("triage_resprate")) or 0.0) > 20
        or (_as_float(features.get("rr_mean")) or 0.0) > 20
    )
    shock_index = _present(
        [_as_float(features.get("triage_shock_index")), _as_float(features.get("shock_index")), _as_float(features.get("shock_index_max"))],
        max,
    )
    sbp_min = _as_float(features.get("sbp_min"))
    low_spo2 = _present([_as_float(features.get("triage_o2sat")), _as_float(features.get("spo2_min"))], min)

    severity = "low"
    if (high_temp is not None and high_temp >= 102.2) or (
        fever_count > 0 and ((shock_index or 0.0) >= 0.9 or (sbp_min is not None and sbp_min < 90) or (low_spo2 is not None and low_spo2 < 92))
    ):
        severity = "high"
    elif (fever_count > 0 and (tachy_trigger or tachypnea_trigger)) or (high_temp is not None and high_temp >= 100.4):
        severity = "moderate"

    evidence = [
        _evidence_line("Triage temperature", triage_temp, "°F"),
        _evidence_line("Mean temperature", temp_mean, "°F"),
        _evidence_line("Minimum temperature", temp_min, "°F"),
        _evidence_line("Maximum temperature", temp_max, "°F"),
        _evidence_line("Fever episodes", fever_count, "episodes", 0) if fever_count > 0 else None,
    ]
    evidence = [item for item in evidence if item is not None]

    if high_temp is not None and high_temp >= 100.4:
        message = "Fever signal present; review infection source and clinical context."
    else:
        message = "Temperature pattern is outside expected range and should be reviewed in context."

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
    highest_si = _present([triage_si, shock_index, shock_index_max], max)
    if highest_si is None or highest_si < 0.7:
        return None

    if highest_si >= 1.0:
        severity = "high"
    elif highest_si >= 0.9:
        severity = "moderate"
    else:
        severity = "low"

    evidence = [
        _evidence_line("Triage shock index", triage_si, decimals=3),
        _evidence_line("Shock index", shock_index, decimals=3),
        _evidence_line("Maximum shock index", shock_index_max, decimals=3),
    ]
    evidence = [item for item in evidence if item is not None]

    message = (
        "Shock index is near or above 1.0, which may suggest hemodynamic strain in clinical context."
        if highest_si >= 1.0
        else "Shock index is elevated and should be reviewed for hemodynamic strain."
    )
    return {
        "indicator": "Elevated shock index / hemodynamic strain",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"hemodynamic"},
    }


def detect_trend_context_or_signal(features: dict[str, Any], data_quality: dict[str, Any]) -> dict[str, Any] | None:
    vital_count = data_quality["vital_observation_count"]
    hr_slope = _as_float(features.get("hr_slope"))
    bp_slope = _as_float(features.get("bp_slope"))
    has_slope = hr_slope is not None or bp_slope is not None
    if vital_count is None or not has_slope:
        return None

    evidence = [f"Vital observations: {vital_count}"]
    if hr_slope is not None:
        evidence.append(f"Heart-rate trend value: {_format_number(hr_slope, 2)}")
    if bp_slope is not None:
        evidence.append(f"Blood-pressure trend value: {_format_number(bp_slope, 2)}")

    meaningful_hr = hr_slope is not None and hr_slope >= HR_SLOPE_THRESHOLD
    meaningful_bp = bp_slope is not None and bp_slope <= BP_SLOPE_THRESHOLD

    if vital_count < 3:
        return {
            "indicator": "Limited trend data context",
            "severity": "low",
            "evidence": evidence,
            "doctor_message": f"Trend interpretation is limited because only {vital_count} vital-sign observations are available.",
            "_focus": {"limited_trends"},
        }

    if not meaningful_hr and not meaningful_bp:
        return None

    severity = "high" if meaningful_hr and meaningful_bp else "moderate"
    message = (
        "Vitals show a possible deterioration pattern with rising heart rate and falling blood pressure."
        if meaningful_hr and meaningful_bp
        else "Repeated vital signs show a trend that should be reviewed for possible deterioration."
    )
    return {
        "indicator": "Deterioration trend signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": message,
        "_focus": {"trends"},
    }


def detect_acuity_indicator(features: dict[str, Any], risk_label: str, has_other_abnormal_indicators: bool) -> dict[str, Any] | None:
    acuity = _as_float(features.get("acuity"))
    acuity_missing = _as_float(features.get("acuity_missing"))
    if acuity is None and acuity_missing != 1.0:
        return None

    if acuity_missing == 1.0:
        return {
            "indicator": "Triage acuity concern",
            "severity": "moderate",
            "evidence": ["Triage acuity unavailable"],
            "doctor_message": "Acuity is unavailable; the prediction relies more heavily on vitals and missingness patterns.",
            "_focus": {"triage", "data_quality"},
        }

    acuity_int = int(acuity)
    if acuity_int in {1, 2}:
        return {
            "indicator": "Triage acuity concern",
            "severity": "high",
            "evidence": [f"Triage acuity: {acuity_int}"],
            "doctor_message": "Triage acuity indicates high initial urgency.",
            "_focus": {"triage"},
        }
    if acuity_int == 3 and risk_label in {"MODERATE", "HIGH"}:
        return {
            "indicator": "Triage acuity concern",
            "severity": "moderate",
            "evidence": [f"Triage acuity: {acuity_int}"],
            "doctor_message": "Triage acuity indicates moderate urgency and should be reviewed with the current physiologic findings.",
            "_focus": {"triage"},
        }
    if acuity_int in {4, 5}:
        if acuity_int == 5 and not has_other_abnormal_indicators:
            message = "Triage acuity suggests low initial urgency and reduced the model-estimated risk."
        else:
            message = "Triage acuity suggests lower initial urgency and should be interpreted alongside current clinical findings."
        return {
            "indicator": "Triage acuity context",
            "severity": "low",
            "evidence": [f"Triage acuity: {acuity_int}"],
            "doctor_message": message,
            "_focus": {"triage"} if has_other_abnormal_indicators else set(),
        }
    return None


def detect_sparse_data_signal(features: dict[str, Any], data_quality: dict[str, Any]) -> dict[str, Any] | None:
    vital_count = data_quality["vital_observation_count"]
    missingness_concern = data_quality["missingness_concern"]
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

    severity = "high" if missingness_concern and (vital_count is not None and vital_count < 2) else "moderate"
    evidence = []
    if vital_count is not None:
        evidence.append(f"Vital observations: {vital_count}")
    evidence.extend(f"High missingness: {name}" for name in data_quality["missing_features"])
    if triage_missing_flags >= 3:
        evidence.append(f"Missing triage elements: {triage_missing_flags}")
    return {
        "indicator": "Sparse data / low confidence signal",
        "severity": severity,
        "evidence": evidence,
        "doctor_message": data_quality["data_quality_note"],
        "_focus": {"data_quality"},
    }


def detect_composite_patterns(features: dict[str, Any], indicators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {item["indicator"] for item in indicators}
    patterns = []
    triage_rr = _as_float(features.get("triage_resprate")) or 0.0
    rr_mean = _as_float(features.get("rr_mean")) or 0.0
    triage_hr = _as_float(features.get("triage_heartrate")) or 0.0
    hr_mean = _as_float(features.get("hr_mean")) or 0.0
    shock_index = _present(
        [_as_float(features.get("triage_shock_index")), _as_float(features.get("shock_index")), _as_float(features.get("shock_index_max"))],
        max,
    ) or 0.0
    sbp_min = _as_float(features.get("sbp_min"))
    low_spo2 = _present([_as_float(features.get("triage_o2sat")), _as_float(features.get("spo2_min"))], min)
    hypoxia_count = _as_float(features.get("hypoxia_count")) or 0.0
    fever_count = _as_float(features.get("fever_count")) or 0.0
    vital_count = _as_float(features.get("vital_row_count")) or 0.0
    hr_slope = _as_float(features.get("hr_slope")) or 0.0
    bp_slope = _as_float(features.get("bp_slope")) or 0.0

    tachypnea = triage_rr > 20 or rr_mean > 20
    tachycardia = triage_hr > 100 or hr_mean > 100 or (_as_float(features.get("tachycardia_count")) or 0.0) > 0

    if "Respiratory compromise signal" in names and ((low_spo2 is not None and low_spo2 < 92) or hypoxia_count > 0 or tachypnea):
        patterns.append(
            {
                "pattern": "Respiratory deterioration pattern",
                "severity": "high" if (low_spo2 is not None and low_spo2 < 90) or hypoxia_count >= 2 else "moderate",
                "evidence": [
                    "Reduced oxygen saturation or hypoxia episodes detected",
                    "Respiratory rate elevation present" if tachypnea else "Respiratory findings should be reviewed",
                ],
                "doctor_message": "Respiratory compromise pattern detected. Review oxygen requirement, work of breathing, and relevant clinical context.",
                "_focus": {"respiratory"},
            }
        )

    if "Hemodynamic instability signal" in names and (
        (sbp_min is not None and sbp_min < 90) or shock_index >= 1.0 or (vital_count >= 3 and hr_slope >= HR_SLOPE_THRESHOLD and bp_slope <= BP_SLOPE_THRESHOLD)
    ):
        patterns.append(
            {
                "pattern": "Hemodynamic instability pattern",
                "severity": "high",
                "evidence": [
                    "Low or falling systolic pressure detected",
                    "Elevated shock index or concerning combined trend is present",
                ],
                "doctor_message": "Hemodynamic instability pattern detected: low or falling systolic pressure, elevated shock index, or concerning combined trends. Review perfusion status and escalation need.",
                "_focus": {"hemodynamic", "trends"},
            }
        )

    if ("Respiratory compromise signal" in names and "Tachycardia / physiologic stress" in names) or shock_index >= 0.9:
        patterns.append(
            {
                "pattern": "Cardiopulmonary stress pattern",
                "severity": "high" if shock_index >= 1.0 or (low_spo2 is not None and low_spo2 < 92) else "moderate",
                "evidence": [
                    "Heart-rate stress signal present" if "Tachycardia / physiologic stress" in names else "Hemodynamic strain marker present",
                    "Oxygenation or respiratory-rate concern present" if "Respiratory compromise signal" in names else "Elevated shock index present",
                ],
                "doctor_message": "Cardiopulmonary stress pattern detected. Model risk appears related to heart-rate, oxygenation, and blood-pressure physiology.",
                "_focus": {"respiratory", "tachycardia", "hemodynamic"},
            }
        )

    if "Fever / infection-context signal" in names and "Tachycardia / physiologic stress" in names and tachypnea:
        patterns.append(
            {
                "pattern": "Sepsis-like physiology pattern",
                "severity": "high" if shock_index >= 0.9 or (sbp_min is not None and sbp_min < 90) else "moderate",
                "evidence": ["Fever signal present", "Tachycardia signal present", "Tachypnea signal present"],
                "doctor_message": "Pattern suggests infection or sepsis-like physiology: fever, tachycardia, and tachypnea. Review clinical context, suspected infection source, available labs, and clinician assessment.",
                "_focus": {"sepsis"},
            }
        )

    if fever_count > 0 and (tachycardia or tachypnea or shock_index >= 0.9 or (sbp_min is not None and sbp_min < 90)):
        patterns.append(
            {
                "pattern": "Fever with physiologic instability",
                "severity": "high" if shock_index >= 0.9 or (sbp_min is not None and sbp_min < 90) else "moderate",
                "evidence": ["Fever episodes detected", "Physiologic instability signal present"],
                "doctor_message": "Fever is present together with physiologic instability signals. Review infection, inflammatory state, dehydration, pain, or other clinical causes.",
                "_focus": {"infection"},
            }
        )

    return patterns


def _driver_message(feature_name: str, feature_value: Any, direction: str, features: dict[str, Any]) -> str:
    clinical_meaning = FEATURE_TO_CLINICAL_MEANING.get(feature_name, feature_name.replace("_", " ").title())
    hr_values = [_as_float(features.get(name)) for name in ("triage_heartrate", "hr_mean", "hr_min", "hr_max")]
    hr_normal = all(value is None or 60 <= value <= 100 for value in hr_values)
    sbp_values = [_as_float(features.get(name)) for name in ("triage_sbp", "sbp_mean", "sbp_min", "sbp_max")]
    sbp_normal = all(value is None or value >= 100 for value in sbp_values)
    dbp_values = [_as_float(features.get(name)) for name in ("triage_dbp", "dbp_mean", "dbp_min", "dbp_max")]
    dbp_expected = all(value is None or 60 <= value <= 90 for value in dbp_values)
    spo2_values = [_as_float(features.get(name)) for name in ("triage_o2sat", "spo2_mean", "spo2_min", "spo2_max")]
    spo2_normal = all(value is None or value >= 95 for value in spo2_values)
    shock_values = [_as_float(features.get(name)) for name in ("triage_shock_index", "shock_index", "shock_index_max")]
    shock_normal = all(value is None or value < 0.7 for value in shock_values)

    if feature_name == "vital_row_count":
        count = _as_float(feature_value)
        if count is not None and count >= 3:
            return "Multiple vital-sign observations were available, allowing the model to assess repeated or evolving physiologic patterns."
        return "Only limited repeated vital-sign data were available; trend-based interpretation should be cautious."
    if feature_name == "hr_slope":
        if hr_normal:
            return "Heart-rate trend contributed to the model estimate, but absolute heart-rate values remain within expected range."
        return "Heart-rate trend contributed to the model estimate alongside abnormal absolute heart-rate values."
    if feature_name == "bp_slope":
        if sbp_normal:
            return "Blood-pressure trend contributed to the model estimate, but available blood-pressure values do not meet hypotension thresholds."
        return "Blood-pressure trend contributed to the model estimate alongside low or borderline-low systolic blood pressure."
    if feature_name == "acuity":
        acuity = _as_float(feature_value)
        if acuity == 5:
            return "Triage acuity suggests low initial urgency and reduced the model-estimated risk."
        if acuity in {1.0, 2.0}:
            return "Triage acuity indicates high initial urgency and contributed to the model context."
    if feature_name in {"dbp_min", "dbp_mean", "dbp_max"} and direction == "decreases_risk" and dbp_expected:
        return f"{clinical_meaning} remained within expected range and reduced the model-estimated risk."
    if feature_name in {"spo2_mean", "spo2_min", "triage_o2sat"}:
        if direction == "increases_risk" and not spo2_normal:
            return f"{clinical_meaning} was below ideal range and contributed to elevated respiratory risk."
        if direction == "increases_risk":
            return f"{clinical_meaning} contributed to the model estimate, but available oxygenation values remain within expected range."
    if feature_name in {"shock_index", "shock_index_max", "triage_shock_index"}:
        if direction == "increases_risk" and not shock_normal:
            return f"{clinical_meaning} was elevated and contributed to hemodynamic strain risk."
        if direction == "increases_risk":
            return f"{clinical_meaning} contributed to the model estimate, but available shock index values remain within expected range."
    if direction == "decreases_risk":
        return f"{clinical_meaning} reduced the model-estimated risk."
    if direction == "increases_risk":
        return f"{clinical_meaning} contributed to the model estimate and should be interpreted in clinical context."
    return f"{clinical_meaning} was included in the model context."


def build_dominant_clinical_drivers(top_contributors: list[dict[str, Any]], features: dict[str, Any]) -> list[dict[str, Any]]:
    drivers = []
    for item in top_contributors:
        feature_name = item["feature_name"]
        drivers.append(
            {
                "feature_name": feature_name,
                "clinical_meaning": FEATURE_TO_CLINICAL_MEANING.get(feature_name, feature_name.replace("_", " ").title()),
                "feature_value": item.get("feature_value"),
                "driver_type": "data_context" if feature_name in DATA_CONTEXT_FEATURES else "model_signal",
                "contribution_direction": item.get("contribution_direction", "neutral"),
                "doctor_message": _driver_message(
                    feature_name,
                    item.get("feature_value"),
                    item.get("contribution_direction", "neutral"),
                    features,
                ),
            }
        )
    return drivers


def _validate_conflicts(indicators: list[dict[str, Any]], features: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = []
    hr_values = [_as_float(features.get(name)) for name in ("triage_heartrate", "hr_mean", "hr_min", "hr_max")]
    spo2_values = [_as_float(features.get(name)) for name in ("triage_o2sat", "spo2_mean", "spo2_min", "spo2_max")]
    sbp_values = [_as_float(features.get(name)) for name in ("triage_sbp", "sbp_mean", "sbp_min", "sbp_max")]
    temp_values = [_as_float(features.get(name)) for name in ("triage_temperature", "temperature_mean", "temperature_min", "temperature_max")]
    shock_values = [_as_float(features.get(name)) for name in ("triage_shock_index", "shock_index", "shock_index_max")]
    vital_count = _as_float(features.get("vital_row_count")) or 0.0
    for item in indicators:
        name = item["indicator"]
        if name == "Tachycardia / physiologic stress":
            if all(value is None or value <= 100 for value in hr_values) and (_as_float(features.get("tachycardia_count")) or 0.0) == 0:
                continue
        if name == "Respiratory compromise signal":
            if all(value is None or value >= 95 for value in spo2_values) and (_as_float(features.get("hypoxia_count")) or 0.0) == 0 and all(
                (_as_float(features.get(name2)) or 0.0) <= 20 for name2 in ("triage_resprate", "rr_mean", "rr_max")
            ):
                continue
        if name == "Hemodynamic instability signal":
            if all(value is None or value >= 100 for value in sbp_values) and (_as_float(features.get("hypotension_count")) or 0.0) == 0 and all(
                value is None or value < 0.9 for value in shock_values
            ):
                continue
        if name == "Fever / infection-context signal":
            if all(value is None or (96.8 <= value < 100.4) for value in temp_values) and (_as_float(features.get("fever_count")) or 0.0) == 0:
                continue
        if name == "Elevated shock index / hemodynamic strain":
            if all(value is None or value < 0.7 for value in shock_values):
                continue
        if name == "Deterioration trend signal":
            hr_slope = _as_float(features.get("hr_slope")) or 0.0
            bp_slope = _as_float(features.get("bp_slope")) or 0.0
            if vital_count < 3 or (hr_slope < HR_SLOPE_THRESHOLD and bp_slope > BP_SLOPE_THRESHOLD):
                continue
        if name == "Triage acuity concern" and _as_float(features.get("acuity")) == 5.0:
            continue
        filtered.append(item)
    return filtered


def _dedupe_review_focus(focus_keys: set[str], risk_label: str, has_abnormal_indicators: bool) -> list[str]:
    if not has_abnormal_indicators:
        if "limited_trends" in focus_keys:
            return [REVIEW_FOCUS_MESSAGES["limited_trends"]]
        if risk_label == "HIGH":
            return [REVIEW_FOCUS_MESSAGES["drivers"]]
        return [REVIEW_FOCUS_MESSAGES["context"]]

    ordered_keys = ["respiratory", "tachycardia", "hemodynamic", "trends", "triage", "infection", "sepsis", "data_quality"]
    if "sepsis" in focus_keys:
        focus_keys.discard("infection")
    messages = [REVIEW_FOCUS_MESSAGES[key] for key in ordered_keys if key in focus_keys]
    deduped: list[str] = []
    for message in messages:
        if message not in deduped:
            deduped.append(message)
    return deduped[:6]


def _risk_priority(risk_label: str) -> str:
    if risk_label == "HIGH":
        return "High concern - clinician review recommended"
    if risk_label == "MODERATE":
        return "Intermediate concern - review contributing clinical signals"
    return "Lower concern - no strong model signal of escalation risk"


def _build_interpretation_sentence(
    risk_label: str,
    indicators: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    data_quality: dict[str, Any],
) -> str:
    abnormal_indicator_names = {
        item["indicator"]
        for item in indicators
        if item["indicator"] not in {"Triage acuity context", "Limited trend data context", "Sparse data / low confidence signal"}
    }
    limited_trends = any(item["indicator"] == "Limited trend data context" for item in indicators)

    if not abnormal_indicator_names and not patterns:
        if risk_label == "LOW":
            base = "Lower predicted risk overall, with no major abnormal vital-sign indicators detected."
            if limited_trends:
                return f"{base} Trend interpretation is limited by sparse repeated measurements."
            return base
        if risk_label == "MODERATE":
            return "Intermediate model risk without major threshold-based vital-sign abnormalities; review model drivers and clinical context."
        return "High model-estimated risk, but no major threshold-based vital-sign abnormality is detected in the provided features. Review model drivers, data completeness, and clinical context."

    domains = []
    if "Respiratory compromise signal" in abnormal_indicator_names:
        domains.append("respiratory")
    if "Tachycardia / physiologic stress" in abnormal_indicator_names:
        domains.append("heart-rate")
    if "Hemodynamic instability signal" in abnormal_indicator_names or "Elevated shock index / hemodynamic strain" in abnormal_indicator_names:
        domains.append("hemodynamic strain")
    if "Fever / infection-context signal" in abnormal_indicator_names:
        domains.append("infection-context")

    if len(domains) == 1:
        joined = domains[0]
    elif len(domains) == 2:
        joined = f"{domains[0]} and {domains[1]}"
    else:
        joined = ", ".join(domains[:-1]) + f", and {domains[-1]}" if domains else "clinical"

    if risk_label == "HIGH":
        return f"High predicted risk with {joined} signals."
    if risk_label == "MODERATE":
        return f"Intermediate predicted risk; review {joined} signals."
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
        lambda values: detect_hemodynamic_signal(values, risk_label),
        detect_fever_signal,
        detect_shock_index_signal,
    ):
        indicator = detector(features)
        if indicator is not None:
            indicators.append(indicator)

    trend_indicator = detect_trend_context_or_signal(features, data_quality)
    if trend_indicator is not None:
        indicators.append(trend_indicator)

    sparse_indicator = detect_sparse_data_signal(features, data_quality)
    if sparse_indicator is not None:
        indicators.append(sparse_indicator)

    has_other_abnormal_indicators = any(
        item["indicator"]
        not in {"Limited trend data context", "Sparse data / low confidence signal"}
        for item in indicators
    )
    acuity_indicator = detect_acuity_indicator(features, risk_label, has_other_abnormal_indicators)
    if acuity_indicator is not None:
        indicators.append(acuity_indicator)

    indicators = _validate_conflicts(indicators, features)
    patterns = detect_composite_patterns(features, indicators)

    focus_keys: set[str] = set()
    for item in indicators + patterns:
        focus_keys.update(item.get("_focus", set()))

    clean_indicators = [{k: v for k, v in item.items() if not k.startswith("_")} for item in indicators]
    clean_patterns = [{k: v for k, v in item.items() if not k.startswith("_")} for item in patterns]
    has_abnormal_indicators = any(
        item["indicator"] not in {"Triage acuity context", "Limited trend data context", "Sparse data / low confidence signal"}
        for item in clean_indicators
    ) or bool(clean_patterns)

    return {
        "risk_summary": {
            "risk_label": risk_label,
            "risk_score": round(risk_score, 6),
            "display_risk_score": _display_risk_score(risk_score),
            "risk_score_note": RISK_SCORE_NOTE,
            "clinical_priority": _risk_priority(risk_label),
            "interpretation": _build_interpretation_sentence(risk_label, clean_indicators, clean_patterns, data_quality),
        },
        "clinical_indicators": clean_indicators,
        "clinical_patterns": clean_patterns,
        "dominant_clinical_drivers": build_dominant_clinical_drivers(top_contributors, features),
        "data_quality": data_quality,
        "recommended_review_focus": _dedupe_review_focus(focus_keys, risk_label, has_abnormal_indicators),
        "safety_note": SAFETY_NOTE,
    }
