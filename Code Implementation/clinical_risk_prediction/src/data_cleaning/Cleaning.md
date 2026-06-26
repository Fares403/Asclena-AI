# Data Cleaning Documentation

## Introduction

Healthcare data is often collected from multiple systems and healthcare professionals over long periods of time. As a result, raw clinical datasets usually contain missing values, duplicate records, inconsistent text formats, invalid measurements, and records that cannot be linked to a valid patient encounter.

The objective of this cleaning process was not simply to remove bad data, but to improve data quality while preserving clinically meaningful information. Every cleaning decision was designed to maintain medical accuracy and prepare the dataset for feature engineering and machine learning.

The cleaning process was applied to the following tables:

* ED Stays
* Triage
* Vital Signs
* Diagnosis
* Pyxis Medication Records
* Medication Reconciliation

Each cleaned table was stored separately from the original data to preserve auditability and reproducibility.

---

# 1. Raw Data Backup

Before applying any cleaning operation, complete backups of all raw tables were created.

### Why?

This ensures that:

* Original data remains unchanged.
* Cleaning operations can be reversed if needed.
* Results can be audited and reproduced.

### Example

Before Cleaning:

| Table      |
| ---------- |
| ed_stays   |
| triage     |
| vital_sign |

After Backup:

| Raw Table  | Backup Table          |
| ---------- | --------------------- |
| ed_stays   | ed_stays_raw_backup   |
| triage     | triage_raw_backup     |
| vital_sign | vital_sign_raw_backup |

---

# 2. Removing Records Without Essential Identifiers

Every clinical observation must belong to a patient and a hospital visit.

Records missing:

* Stay ID
* Subject ID

were removed.

### Why?

Without these identifiers, there is no reliable way to determine which patient generated the record.

### Example

Before:

| stay_id | subject_id | temperature |
| ------- | ---------- | ----------- |
| NULL    | 1001       | 98.6        |
| 2001    | NULL       | 99.2        |
| 2002    | 1002       | 98.4        |

After:

| stay_id | subject_id | temperature |
| ------- | ---------- | ----------- |
| 2002    | 1002       | 98.4        |

---

# 3. Duplicate Record Removal

Duplicate records frequently occur because data may be imported multiple times or entered by different systems.

The cleaning pipeline identified duplicate records and retained only the most complete version.

### Example

Before:

| stay_id | gender | race  |
| ------- | ------ | ----- |
| 5001    | Male   | White |
| 5001    | Male   | NULL  |

After:

| stay_id | gender | race  |
| ------- | ------ | ----- |
| 5001    | Male   | White |

### Why?

The first record contains more complete information and therefore provides greater analytical value.

---

# 4. Text Standardization

Many text fields contained formatting inconsistencies.

### Problems Identified

* Extra spaces
* Empty strings
* Different capitalization styles

### Example

Before:

| Medication Name |
| --------------- |
| " Aspirin "     |
| "Aspirin"       |
| "Aspirin   "    |

After:

| Medication Name |
| --------------- |
| Aspirin         |
| Aspirin         |
| Aspirin         |

### Benefit

The same medication is no longer treated as multiple different values.

---

# 5. Gender Standardization

Gender information appeared in several formats.

### Before

| Raw Value |
| --------- |
| M         |
| Male      |
| MALE      |
| F         |
| FEMALE    |

### After

| Cleaned Value |
| ------------- |
| Male          |
| Female        |

Any unknown or invalid values were assigned:

```text
Unknown
```

### Benefit

This prevents category fragmentation during statistical analysis.

---

# 6. Clinical Range Validation

Vital signs were validated using clinically acceptable physiological ranges.

Measurements outside these ranges were considered impossible or highly unreliable.

### Examples

#### Temperature

Accepted Range:

90°F – 110°F

Before:

| Temperature |
| ----------- |
| 98.6        |
| 102         |
| 250         |

After:

| Temperature |
| ----------- |
| 98.6        |
| 102         |
| NULL        |

---

#### Heart Rate

Accepted Range:

20 – 250 bpm

Before:

| Heart Rate |
| ---------- |
| 75         |
| 90         |
| 500        |

After:

| Heart Rate |
| ---------- |
| 75         |
| 90         |
| NULL       |

### Why not delete the entire row?

The patient record may still contain valid blood pressure, oxygen saturation, or other useful information.

Only the invalid measurement was removed.

---

# 7. Missing Value Handling

Missing values were treated carefully because missing information can itself be clinically meaningful.

Instead of automatically filling every missing value, the pipeline preserved many missing observations.

### Example

Before:

| Temperature |
| ----------- |
| NULL        |

After:

| Temperature |
| ----------- |
| NULL        |

### Reason

A missing measurement is different from a normal measurement.

Replacing it with an estimated value could introduce bias.

---

# 8. Missing Value Indicators

Special indicator columns were added to record whether a measurement was originally missing.

### Example

Before

| Temperature |
| ----------- |
| NULL        |

After

| Temperature | Temperature_Missing |
| ----------- | ------------------- |
| NULL        | TRUE                |

Another example:

| Temperature | Temperature_Missing |
| ----------- | ------------------- |
| 98.6        | FALSE               |

### Benefit

Machine learning models can learn patterns associated with missing clinical information.

---

# 9. Triage Data Cleaning

The triage table contains the patient's first assessment upon arrival.

Several cleaning steps were performed:

* Invalid vital signs converted to NULL.
* Duplicate triage records removed.
* Missing pain values standardized.
* Missing chief complaints standardized.
* Missing indicators created.

### Example

Before

| Chief Complaint |
| --------------- |
| NULL            |

After

| Chief Complaint |
| --------------- |
| Unknown         |

This preserves the record while clearly indicating missing information.

---

# 10. Vital Signs Cleaning

Vital sign measurements are collected repeatedly during a patient's stay.

Records where every vital measurement was missing were removed.

### Example

Before

| Temp | HR   | RR   | O2Sat | SBP  | DBP  |
| ---- | ---- | ---- | ----- | ---- | ---- |
| NULL | NULL | NULL | NULL  | NULL | NULL |

After

Record removed.

### Why?

Such records provide no clinical information and only increase noise.

---

# 11. Diagnosis Cleaning

Diagnosis records were standardized and validated.

### ICD Code Standardization

Before

| ICD Code |
| -------- |
| i10      |
| I10      |

After

| ICD Code |
| -------- |
| I10      |
| I10      |

---

### Empty Diagnosis Removal

Before

| ICD Code | Diagnosis Title |
| -------- | --------------- |
| NULL     | NULL            |

After

Record removed.

---

### Missing Diagnosis Title

Before

| ICD Code | Diagnosis Title |
| -------- | --------------- |
| I10      | NULL            |

After

| ICD Code | Diagnosis Title   |
| -------- | ----------------- |
| I10      | Unknown Diagnosis |

---

# 12. Medication Data Cleaning

Medication-related tables often contain incomplete entries.

Records without meaningful medication information were removed.

### Example

Before

| Name | GSN  | NDC  |
| ---- | ---- | ---- |
| NULL | NULL | NULL |

After

Record removed.

### Reason

The record contains no useful medication information.

---

# 13. Referential Integrity Validation

Every cleaned table was linked back to a valid emergency department visit.

### Example

Suppose a diagnosis record references:

```text
stay_id = 99999
```

but no matching visit exists.

The record is removed.

### Why?

Every clinical observation must belong to a valid patient encounter.

---

# 14. Exclusion of Rhythm from Machine Learning

The Rhythm column contained extensive missing data.

Instead of replacing missing values with:

```text
Normal
```

the project team chose to exclude this variable from model training.

### Example

Before

| Rhythm |
| ------ |
| NULL   |
| NULL   |
| Normal |

Artificial Imputation:

| Rhythm |
| ------ |
| Normal |
| Normal |
| Normal |

This would falsely suggest that patients had normal cardiac rhythm measurements.

### Final Decision

* Keep Rhythm in the database.
* Preserve original values.
* Exclude it from feature engineering.
* Exclude it from machine learning.

### Benefit

Avoids introducing medically inaccurate information.

---

# Final Outcome

After cleaning, the dataset became:

* More consistent.
* More reliable.
* Free from major duplicates.
* Free from impossible physiological values.
* Easier to analyze.
* Better suited for machine learning.
* Clinically trustworthy.

The final cleaning strategy prioritizes data quality while preserving as much authentic clinical information as possible, ensuring that future analytical and predictive models are built on reliable foundations.
