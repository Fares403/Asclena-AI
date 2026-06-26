
# Feature Engineering Documentation

## Overview

After completing the data cleaning phase, the next step was transforming the cleaned clinical data into machine-learning-ready features.

The objective was to create a single feature row for every Emergency Department (ED) visit, combining information from:

* Patient demographics
* Triage assessment
* Vital signs
* Diagnoses
* Medication history
* Previous hospital visits
* Previous risk history

This process produced the final dataset called:

```text
patient_feature_store
```

where each row represents one patient visit and each column represents a measurable characteristic (feature) that can be used by machine learning models. 

---

# 1. Demographic Features

Basic patient information was transformed into numerical features.

### Original Data

| Gender  |
| ------- |
| Male    |
| Female  |
| Unknown |

### Generated Features

| gender_male | gender_female | gender_unknown |
| ----------- | ------------- | -------------- |
| 1           | 0             | 0              |

| gender_male | gender_female | gender_unknown |
| ----------- | ------------- | -------------- |
| 0           | 1             | 0              |

This encoding allows machine learning algorithms to process categorical information. 

---

# 2. Length of Stay Feature

The duration of the emergency visit was calculated.

### Formula

```text
Length of Stay = Out Time – In Time
```

### Example

| Arrival | Discharge |
| ------- | --------- |
| 08:00   | 14:00     |

Generated Feature:

```text
length_of_stay_hours = 6
```

Although this feature was calculated, it was excluded from the initial model because it can leak outcome information. 

---

# 3. Triage Features

The first measurements taken when the patient arrived were retained as predictive features.

Generated Features:

* triage_temperature
* triage_heartrate
* triage_resprate
* triage_o2sat
* triage_sbp
* triage_dbp
* acuity

### Example

| Temperature | Heart Rate | O2Sat |
| ----------- | ---------- | ----- |
| 101.2       | 130        | 89    |

These values help estimate patient severity immediately upon arrival. 

---

# 4. Missingness Features

Instead of ignoring missing values, the pipeline created indicators showing whether information was missing.

### Example

| Temperature |
| ----------- |
| NULL        |

Generated Feature:

| temperature_missing |
| ------------------- |
| 1                   |

This allows the model to learn patterns related to missing clinical information. 

---

# 5. Shock Index

A clinically meaningful feature called Shock Index was generated.

### Formula

Shock\ Index = \frac{Heart\ Rate}{Systolic\ Blood\ Pressure}

### Example

Heart Rate:

```text
120
```

Systolic Blood Pressure:

```text
100
```

Result:

```text
Shock Index = 1.2
```

Higher values often indicate circulatory instability and increased clinical risk. 

---

# 6. Vital Sign Aggregation Features

Patients often have multiple vital sign measurements during a visit.

Instead of keeping every measurement separately, summary statistics were generated.

### Heart Rate Example

Measurements:

```text
90
95
110
105
```

Generated Features:

| Feature | Value |
| ------- | ----- |
| hr_mean | 100   |
| hr_min  | 90    |
| hr_max  | 110   |

The same process was applied to:

* Temperature
* Heart Rate
* Respiratory Rate
* Oxygen Saturation
* Systolic Blood Pressure
* Diastolic Blood Pressure



---

# 7. Clinical Event Counters

Several medically important events were counted.

### Tachycardia Count

Number of times:

```text
Heart Rate ≥ 120
```

### Hypotension Count

Number of times:

```text
SBP < 90
```

### Hypoxia Count

Number of times:

```text
O2Sat < 92%
```

### Fever Count

Number of times:

```text
Temperature ≥ 100.4°F
```

### Example

Heart Rates:

```text
110
125
135
90
```

Generated:

```text
tachycardia_count = 2
```

These features capture repeated physiological deterioration. 

---

# 8. Trend Features (Patient Improvement or Deterioration)

The pipeline measures whether vital signs improved or worsened over time.

### Heart Rate Trend

Generated Features:

* hr_first
* hr_last
* hr_slope

### Example

| Time  | HR  |
| ----- | --- |
| 08:00 | 80  |
| 12:00 | 120 |

Generated:

```text
hr_slope = +10 beats/hour
```

Positive slope indicates worsening tachycardia.

The same logic was applied to blood pressure using:

```text
bp_slope
```



---

# 9. Missing Rate Features

The percentage of missing measurements was calculated.

### Example

10 heart-rate records expected.

Available:

```text
7
```

Missing:

```text
3
```

Generated Feature:

```text
heartrate_missing_rate = 0.30
```

This acts as a quality and completeness indicator. 

---

# 10. Diagnosis-Based Features

Diagnosis information was transformed into disease-category indicators.

### Cardiovascular Disease

Generated:

```text
has_cardiovascular_dx
```

### Respiratory Disease

Generated:

```text
has_respiratory_dx
```

### Endocrine Disease

Generated:

```text
has_endocrine_dx
```

### Renal Disease

Generated:

```text
has_renal_dx
```

### Example

Diagnosis:

```text
I21 Acute Myocardial Infarction
```

Generated:

```text
has_cardiovascular_dx = 1
```



---

# 11. Comorbidity Score

The number of diagnoses associated with a patient visit was calculated.

### Example

Diagnoses:

```text
Hypertension
Diabetes
Chronic Kidney Disease
```

Generated:

```text
comorbidity_score = 3
```

This approximates disease burden. 

---

# 12. Medication Features

Medication activity was summarized.

Generated Features:

* pyxis_med_count
* pyxis_distinct_med_count
* med_recon_count
* med_recon_distinct_med_count

### Example

Patient received:

```text
Aspirin
Insulin
Metformin
```

Generated:

```text
pyxis_med_count = 3
```



---

# 13. Medication Intensity Score

A combined medication burden score was generated.

### Formula

```text
Medication Intensity =
Pyxis Medications +
Medication Reconciliation Entries
```

### Example

```text
Pyxis = 4
Med Recon = 3
```

Generated:

```text
medication_intensity_score = 7
```



---

# 14. Patient History Features

One of the strongest feature groups in the project.

The model was given access to the patient's historical visits.

Generated Features:

* prior_ed_visit_count
* prior_ed_visit_count_30d
* prior_ed_visit_count_90d
* prior_admission_count
* prior_admission_count_1y
* prior_icu_or_death_count
* time_since_last_ed_visit_days

### Example

Patient history:

```text
3 ED visits in last 30 days
```

Generated:

```text
prior_ed_visit_count_30d = 3
```

Frequent ED visits often correlate with higher clinical risk. 

---

# 15. Historical Diagnosis Features

Previous diagnoses from earlier visits were summarized.

Generated Features:

* prior_cardiovascular_dx_count
* prior_respiratory_dx_count
* prior_endocrine_dx_count
* prior_renal_dx_count
* prior_distinct_diagnosis_count

### Example

Previous visits contained:

```text
Heart Failure
Hypertension
Atrial Fibrillation
```

Generated:

```text
prior_cardiovascular_dx_count = 3
```



---

# 16. Previous Risk History Features

If the patient had previous risk predictions, they were transformed into features.

Generated:

* prior_high_risk_prediction_count
* last_risk_score
* avg_prior_risk_score
* max_prior_risk_score

### Example

Previous Scores:

```text
0.82
0.75
0.91
```

Generated:

```text
avg_prior_risk_score = 0.83
max_prior_risk_score = 0.91
```

These features provide longitudinal risk awareness. 

---

# Final Target Variable

The model predicts:

```text
risk_target
```

A patient is considered High Risk if the ED visit resulted in:

* Hospital Admission
* Transfer
* ICU Escalation
* Death-related Outcome

Example:

| Outcome         | risk_target |
| --------------- | ----------- |
| Discharged Home | 0           |
| Admitted        | 1           |
| ICU Transfer    | 1           |



---

# Features Excluded from the Initial XGBoost Model

Several features were intentionally excluded to prevent data leakage.

Examples:

* disposition
* has_hadm_id
* length_of_stay_hours
* diagnosis_count
* medication_intensity_score

Reason:

These variables may reveal information that becomes available only after the clinical outcome has already occurred. Including them would artificially inflate model performance. 

---

## Summary

The Feature Engineering pipeline transformed raw clinical observations into a comprehensive patient representation containing:

* Demographic features
* Triage features
* Vital-sign statistics
* Clinical deterioration indicators
* Diagnosis features
* Medication features
* Historical utilization features
* Historical diagnosis features
* Historical risk features

The final result was a machine-learning-ready **Patient Feature Store** containing one row per ED visit and more than 100 engineered features designed to support early risk prediction in emergency care.
