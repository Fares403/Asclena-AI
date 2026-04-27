# Asclena Clinical AI — Full Data Dictionary

> **System Overview:** Asclena is a clinical AI system designed to simulate a real Emergency Department (ED) data pipeline. It models the full patient journey from arrival through AI-based risk prediction.

---

## Pipeline at a Glance

```
ED_STAYS → TRIAGE → VITAL_SIGN → DIAGNOSIS → PYXIS / MED_RECON → PATIENT_FEATURE_STORE → ML MODEL
```

| Step | Table | Role |
|------|-------|------|
| 1 | `ED_STAYS` | Patient enters the system — master visit record |
| 2 | `TRIAGE` | First medical assessment on arrival |
| 3 | `VITAL_SIGN` | Continuous monitoring throughout the stay |
| 4 | `DIAGNOSIS` | Official disease codes assigned by physicians |
| 5 | `PYXIS` | Medications administered during the visit |
| 6 | `MED_RECON` | Pre-admission medication history |
| 7 | `PATIENT_FEATURE_STORE` | AI-ready feature layer for ML models |

---

## Table 1 — `ED_STAYS` · Patient Visit Master

**Purpose:** Central table of the entire system. Every other table links back here via `stay_id`. One row = one emergency department visit.

| Column | Type | Description | Example / Notes |
|--------|------|-------------|-----------------|
| `stay_id` | `INT` (PK) | Unique identifier for each ED visit. Primary join key across all tables. | `30004567` |
| `subject_id` | `INT` (FK) | Unique patient identifier. One patient can have multiple visits (multiple `stay_id` values). | Patient A → visits 1, 2, 3 |
| `hadm_id` | `INT` (FK, nullable) | Hospital Admission ID. Populated **only** when the patient is admitted from the ED to an inpatient ward. `NULL` for discharged patients. | Links to inpatient tables |
| `intime` | `TIMESTAMP` | Arrival timestamp — when the patient entered the ED. Critical anchor for all time-based calculations. | `2024-06-10 14:30:00` |
| `outtime` | `TIMESTAMP` | Departure timestamp — when the patient left the ED. Used to derive length of stay and workflow metrics. | `2024-06-10 21:15:00` |
| `gender` | `VARCHAR` | Biological or recorded gender. Used in demographic analysis and feature engineering. | `M`, `F` |
| `race` | `VARCHAR` | Patient racial/ethnic background. Used for population-level statistics only — **not** used as a clinical predictor. | `White`, `Black`, `Hispanic`, `Asian` |
| `arrival_transport` | `VARCHAR` | Mode of arrival to the ED. Serves as a proxy for urgency level. | `Ambulance` (emergency), `Walk-in` (self-referred), `Police transport` |
| `disposition` | `VARCHAR` | Final outcome of the ED visit — what happened to the patient at discharge. | `Discharged` (sent home), `Admitted` (moved to ward), `Transferred` (sent to another facility), `Deceased` |
| `created_at` | `TIMESTAMP` | System timestamp when this record was inserted into the database. Not a clinical event time. | Auto-generated |

**Derived Metric:**
- **Length of Stay** = `outtime − intime` (in hours)

---

## Table 2 — `TRIAGE` · Initial Emergency Assessment

**Purpose:** Captures the first medical evaluation performed within minutes of patient arrival. The most important dataset for emergency severity estimation, initial risk scoring, and fast clinical decision-making. One row per ED visit.

| Column | Type | Description | Normal Range | Clinical Significance |
|--------|------|-------------|--------------|----------------------|
| `stay_id` | `INT` (FK) | Links to `ED_STAYS`. | — | Join key |
| `temperature` | `FLOAT` | Body temperature in °F at time of triage. | 97 – 99 °F | Values > 100.4 °F indicate fever; may signal infection or sepsis |
| `heartrate` | `INT` | Heart beats per minute (BPM) at triage. | 60 – 100 BPM | Tachycardia (> 100) indicates stress, pain, or hemodynamic compromise; bradycardia (< 60) may indicate cardiac dysfunction |
| `resprate` | `INT` | Respiratory rate — breaths per minute at triage. | 12 – 20 breaths/min | Elevated rates (> 20) indicate respiratory distress, sepsis, or acidosis |
| `o2sat` | `FLOAT` | Peripheral oxygen saturation (SpO₂) as a percentage. | 95 – 100% | Values < 90% indicate hypoxemia requiring immediate intervention; < 95% warrants monitoring |
| `sbp` | `INT` | Systolic blood pressure — the peak pressure when the heart contracts (mmHg). | 90 – 120 mmHg | > 140 = hypertension; < 90 = hypotension / shock risk |
| `dbp` | `INT` | Diastolic blood pressure — the pressure when the heart relaxes (mmHg). | 60 – 80 mmHg | Persistently high values indicate chronic hypertension |
| `pain` | `VARCHAR` | Patient-reported pain level or description. Can be numeric (0–10 scale) or free text. | N/A | Guides analgesic decisions and urgency classification |
| `acuity` | `INT` | Emergency Severity Index (ESI) score assigned at triage. Drives care prioritization. | — | `1` = Critical (life-threatening); `2` = High urgency; `3` = Moderate; `4` = Low; `5` = Non-urgent |
| `chiefcomplaint` | `TEXT` | Primary reason the patient came to the hospital. Free text entered by the triage nurse or patient. | — | Examples: `"chest pain"`, `"shortness of breath"`, `"abdominal pain"`, `"fall"` |
| `created_at` | `TIMESTAMP` | Timestamp when this triage record was created. | — | System-generated |

**Acuity Reference:**

| Score | Label | Typical Action |
|-------|-------|----------------|
| 1 | Critical | Immediate resuscitation |
| 2 | High urgency | Seen within 15 minutes |
| 3 | Moderate | Seen within 30 minutes |
| 4 | Low | Seen within 60 minutes |
| 5 | Non-urgent | Seen within 120 minutes |

---

## Table 3 — `VITAL_SIGN` · Continuous Patient Monitoring

**Purpose:** Time-series table of repeated vital sign measurements taken throughout a patient's ED stay. Unlike `TRIAGE` (a single snapshot), this table captures the full physiological trend, enabling detection of patient deterioration or recovery.

| Column | Type | Description | Normal Range | Clinical Significance |
|--------|------|-------------|--------------|----------------------|
| `stay_id` | `INT` (FK) | Links to `ED_STAYS`. Multiple rows per visit. | — | Join key |
| `charttime` | `TIMESTAMP` | Exact timestamp when this measurement was recorded. Used to build time-series trends. | — | Essential for trajectory analysis |
| `temperature` | `FLOAT` | Body temperature (°F) at this time point. | 97 – 99 °F | Rising temperature may indicate worsening infection |
| `heartrate` | `INT` | Heart rate (BPM) at this time point. | 60 – 100 BPM | Trending up = deterioration; trending down = response to treatment |
| `resprate` | `INT` | Respiratory rate (breaths/min) at this time point. | 12 – 20 | Useful for tracking sepsis or respiratory failure progression |
| `o2sat` | `FLOAT` | Oxygen saturation (%) at this time point. | 95 – 100% | Declining SpO₂ over time is a critical deterioration signal |
| `sbp` | `INT` | Systolic blood pressure (mmHg) at this time point. | 90 – 120 | Declining SBP trend = hemodynamic instability |
| `dbp` | `INT` | Diastolic blood pressure (mmHg) at this time point. | 60 – 80 | — |
| `rhythm` | `VARCHAR` | Qualitative description of cardiac rhythm at this time point. | `Normal sinus rhythm` | Abnormal values: `Irregular rhythm`, `Atrial fibrillation`, `Arrhythmia` |
| `pain` | `VARCHAR` | Patient-reported pain score or description at this time. | 0 (none) | Used to evaluate effectiveness of pain management over time |
| `created_at` | `TIMESTAMP` | System insertion timestamp for this record. | — | System-generated; differs from `charttime` |

> **Note:** `charttime` ≠ `created_at`. `charttime` is when the nurse recorded the measurement; `created_at` is when the database entry was written.

---

## Table 4 — `DIAGNOSIS` · Medical Conditions

**Purpose:** Stores all official diagnoses assigned by physicians during or after the ED visit, coded using the International Classification of Diseases (ICD) standard. Multiple diagnoses can exist per visit.

| Column | Type | Description | Example | Notes |
|--------|------|-------------|---------|-------|
| `stay_id` | `INT` (FK) | Links to `ED_STAYS`. | — | Multiple rows per visit |
| `seq_num` | `INT` | Order of diagnostic importance for this visit. | — | `1` = primary (main) diagnosis; `2+` = secondary / comorbid conditions |
| `icd_code` | `VARCHAR` | Standardized disease code per the ICD classification system. | `I10`, `E11`, `J18.9` | Used for billing, research, and clinical analytics |
| `icd_version` | `INT` | Version of the ICD coding system used. | `9`, `10` | ICD-10 is the current standard; ICD-9 appears in older records |
| `icd_title` | `VARCHAR` | Human-readable name of the diagnosis corresponding to the ICD code. | `"Essential hypertension"`, `"Type 2 diabetes mellitus"` | Useful for display and NLP tasks |
| `created_at` | `TIMESTAMP` | Record insertion timestamp. | — | System-generated |

**Common ICD-10 Examples:**

| ICD Code | Title |
|----------|-------|
| `I10` | Essential hypertension |
| `E11` | Type 2 diabetes mellitus |
| `J18.9` | Pneumonia, unspecified |
| `N39.0` | Urinary tract infection |
| `I21.9` | Acute myocardial infarction |

---

## Table 5 — `PYXIS` · In-Hospital Medication Administration

**Purpose:** Tracks every drug dispensed and administered to the patient **inside the hospital** during their ED stay. Data is sourced from the Pyxis automated medication dispensing system.

| Column | Type | Description | Example | Notes |
|--------|------|-------------|---------|-------|
| `stay_id` | `INT` (FK) | Links to `ED_STAYS`. | — | Multiple rows per visit |
| `charttime` | `TIMESTAMP` | Exact time the medication was dispensed / administered. | `2024-06-10 15:45:00` | Used to sequence medication events |
| `med_rn` | `INT` | Internal sequential record number for this medication event. | — | Uniquely identifies a single administration event |
| `name` | `VARCHAR` | Common or brand name of the drug administered. | `Paracetamol`, `Morphine`, `Insulin`, `Aspirin` | Free text; may require normalization |
| `gsn` | `INT` | Generic Sequence Number — drug classification identifier from the First Databank system. | — | Groups drugs by pharmacological class |
| `gsn_rn` | `INT` | Record number within the GSN group for this administration. | — | Handles multiple administrations of the same drug class |
| `created_at` | `TIMESTAMP` | System timestamp when this record was inserted. | — | System-generated |

---

## Table 6 — `MED_RECON` · Pre-Admission Medication History

**Purpose:** Records the medications a patient was taking **before arriving at the hospital**. Captured during the medication reconciliation process — a standard step to ensure hospital treatments don't conflict with home medications.

| Column | Type | Description | Example | Notes |
|--------|------|-------------|---------|-------|
| `stay_id` | `INT` (FK) | Links to `ED_STAYS`. | — | Multiple rows per visit |
| `charttime` | `TIMESTAMP` | Timestamp when the medication history was recorded by clinical staff. | — | Often captured early in the visit |
| `name` | `VARCHAR` | Name of the home medication the patient was taking. | `Metformin`, `Lisinopril`, `Atorvastatin` | Free text; may need normalization |
| `gsn` | `INT` | Generic Sequence Number for drug classification. | — | Matches the GSN system used in `PYXIS` |
| `ndc` | `VARCHAR` | National Drug Code — official 10-digit US pharmaceutical identifier assigned by the FDA. | `0069-0105-68` | Uniquely identifies a drug product, manufacturer, and package size |
| `etcdescription` | `VARCHAR` | Description of the medication category or therapeutic class. | `"Antidiabetic"`, `"ACE Inhibitor"`, `"Statin"` | Useful for grouping drugs by therapeutic intent |
| `created_at` | `TIMESTAMP` | System insertion timestamp. | — | System-generated |

---

## Table 7 — `PATIENT_FEATURE_STORE` · AI Feature Layer

**Purpose:** The final machine-learning-ready dataset. Compresses all raw clinical tables into **one row per visit** — a flat, numeric representation suitable for ML model training and inference. Each column is an engineered clinical signal.

### Demographics

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `stay_id` | `INT` (FK) | Links to `ED_STAYS`. | One row per visit |
| `age` | `INT` | Patient age at time of visit in years. | Derived from patient DOB and `intime` |
| `gender` | `INT` | Encoded gender. | Numeric encoding, e.g. `0` = Female, `1` = Male |

### Heart Rate Features

| Column | Type | Description | Clinical Signal |
|--------|------|-------------|-----------------|
| `hr_mean` | `FLOAT` | Mean heart rate across all `VITAL_SIGN` measurements during the stay. | Represents baseline cardiovascular load |
| `hr_max` | `FLOAT` | Maximum heart rate recorded. | Captures peak physiological stress |
| `hr_min` | `FLOAT` | Minimum heart rate recorded. | Represents lowest stable state; very low values may indicate bradycardia |

### Blood Pressure Features

| Column | Type | Description | Clinical Signal |
|--------|------|-------------|-----------------|
| `sbp_mean` | `FLOAT` | Mean systolic blood pressure across the stay. | Represents average cardiovascular status |
| `sbp_min` | `FLOAT` | Lowest systolic blood pressure recorded. | Key shock indicator — values < 90 mmHg are critical |

### Oxygen Saturation Features

| Column | Type | Description | Clinical Signal |
|--------|------|-------------|-----------------|
| `spo2_mean` | `FLOAT` | Mean SpO₂ across the stay. | Reflects overall respiratory adequacy |
| `spo2_min` | `FLOAT` | Lowest SpO₂ recorded. | Critical dip indicator — values < 90% flag hypoxemia episodes |

### Respiratory Features

| Column | Type | Description | Clinical Signal |
|--------|------|-------------|-----------------|
| `rr_mean` | `FLOAT` | Mean respiratory rate across the stay. | Elevated mean may indicate sepsis, pain, or metabolic disturbance |

### Engineered AI Signals

| Column | Type | Description | Formula / Logic |
|--------|------|-------------|-----------------|
| `shock_index` | `FLOAT` | Ratio of heart rate to systolic blood pressure. A key hemodynamic instability indicator. | `shock_index = hr_mean / sbp_mean`; values > 1.0 suggest shock |
| `hr_slope` | `FLOAT` | Linear trend of heart rate over time during the visit. | Positive = worsening (rising HR); Negative = recovery (falling HR) |
| `bp_slope` | `FLOAT` | Linear trend of systolic blood pressure over time during the visit. | Negative = worsening (falling BP); Positive = stabilization |

### Clinical Severity Scores

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `acuity` | `INT` | ESI acuity score from `TRIAGE`. Carried forward as a feature. | 1 (critical) to 5 (non-urgent) |
| `comorbidity_score` | `INT` | Count of pre-existing diseases derived from `DIAGNOSIS` (secondary diagnoses). | Higher score = more complex patient |
| `medication_intensity_score` | `FLOAT` | Composite score reflecting the strength and volume of medications used during the visit (`PYXIS`). | Higher score = more aggressive treatment |

### Outcome Variables

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `length_of_stay_hours` | `FLOAT` | Total time spent in the ED in hours. Derived from `outtime − intime`. | Regression target for LOS prediction models |
| `risk_target` | `INT` | Binary AI prediction label. The primary outcome variable for classification models. | `0` = Stable patient (low risk); `1` = High-risk patient (requires escalation) |

---

## Cross-Table Relationships

```
ED_STAYS (stay_id)
    ├── TRIAGE           (stay_id) — 1:1
    ├── VITAL_SIGN       (stay_id) — 1:many
    ├── DIAGNOSIS        (stay_id) — 1:many
    ├── PYXIS            (stay_id) — 1:many
    ├── MED_RECON        (stay_id) — 1:many
    └── PATIENT_FEATURE_STORE (stay_id) — 1:1
```

---

## Glossary

| Term | Definition |
|------|------------|
| **ESI** | Emergency Severity Index — standardized 5-level triage scoring system |
| **ICD** | International Classification of Diseases — global standard for diagnostic codes |
| **NDC** | National Drug Code — FDA identifier for pharmaceutical products |
| **GSN** | Generic Sequence Number — First Databank drug classification ID |
| **SpO₂** | Peripheral oxygen saturation measured by pulse oximetry |
| **SBP / DBP** | Systolic / Diastolic Blood Pressure |
| **Shock Index** | HR ÷ SBP; values > 1.0 suggest hemodynamic instability |
| **Pyxis** | Automated medication dispensing cabinet system used in hospitals |
| **Medication Reconciliation** | Process of comparing a patient's home medications against hospital orders |
| **Feature Store** | Pre-computed, ML-ready table that aggregates raw clinical data into model inputs |