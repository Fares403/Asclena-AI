
-- Asclena Clinical AI Database Schema
-- PostgreSQL Production-Oriented Schema
-- Primary unit: stay_id

CREATE SCHEMA IF NOT EXISTS asclena;

-- =========================================================
-- 1. ED STAYS (Master encounter table)
-- =========================================================

CREATE TABLE IF NOT EXISTS asclena.ed_stays (
    stay_id BIGINT PRIMARY KEY,
    subject_id BIGINT NOT NULL,
    hadm_id BIGINT,
    intime TIMESTAMP NOT NULL,
    outtime TIMESTAMP,
    gender VARCHAR(20),
    race TEXT,
    arrival_transport TEXT,
    disposition TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ed_stays_subject_id
ON asclena.ed_stays(subject_id);

CREATE INDEX IF NOT EXISTS idx_ed_stays_intime
ON asclena.ed_stays(intime);


-- =========================================================
-- 2. TRIAGE (Initial ER assessment)
-- One stay should usually have one triage snapshot
-- =========================================================

CREATE TABLE IF NOT EXISTS asclena.triage (
    triage_id BIGSERIAL PRIMARY KEY,
    stay_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,

    temperature NUMERIC(5,2),
    heartrate NUMERIC(6,2),
    resprate NUMERIC(6,2),
    o2sat NUMERIC(6,2),
    sbp NUMERIC(6,2),
    dbp NUMERIC(6,2),
    pain TEXT,
    acuity INTEGER,
    chiefcomplaint TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_triage_stay
        FOREIGN KEY (stay_id)
        REFERENCES asclena.ed_stays(stay_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_triage_stay_id
ON asclena.triage(stay_id);







-- =========================================================
-- 3. VITAL SIGN (Time-series monitoring)
-- Multiple rows per stay
-- =========================================================

CREATE TABLE IF NOT EXISTS asclena.vital_sign (
    vital_id BIGSERIAL PRIMARY KEY,
    stay_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,
    charttime TIMESTAMP NOT NULL,

    temperature NUMERIC(5,2),
    heartrate NUMERIC(6,2),
    resprate NUMERIC(6,2),
    o2sat NUMERIC(6,2),
    sbp NUMERIC(6,2),
    dbp NUMERIC(6,2),
    rhythm TEXT,
    pain TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_vital_stay
        FOREIGN KEY (stay_id)
        REFERENCES asclena.ed_stays(stay_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vital_stay_id
ON asclena.vital_sign(stay_id);

CREATE INDEX IF NOT EXISTS idx_vital_charttime
ON asclena.vital_sign(charttime);


-- =========================================================
-- 4. DIAGNOSIS (ICD conditions)
-- =========================================================

CREATE TABLE IF NOT EXISTS asclena.diagnosis (
    diagnosis_id BIGSERIAL PRIMARY KEY,
    stay_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,

    seq_num INTEGER,
    icd_code VARCHAR(50),
    icd_version INTEGER,
    icd_title TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_diagnosis_stay
        FOREIGN KEY (stay_id)
        REFERENCES asclena.ed_stays(stay_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_diagnosis_stay_id
ON asclena.diagnosis(stay_id);

CREATE INDEX IF NOT EXISTS idx_diagnosis_icd_code
ON asclena.diagnosis(icd_code);


-- =========================================================
-- 5. PYXIS (Medication administered during stay)
-- =========================================================

CREATE TABLE IF NOT EXISTS asclena.pyxis (
    pyxis_id BIGSERIAL PRIMARY KEY,
    stay_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,
    charttime TIMESTAMP,

    med_rn TEXT,
    name TEXT,
    gsn_rn TEXT,
    gsn TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_pyxis_stay
        FOREIGN KEY (stay_id)
        REFERENCES asclena.ed_stays(stay_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pyxis_stay_id
ON asclena.pyxis(stay_id);

CREATE INDEX IF NOT EXISTS idx_pyxis_charttime
ON asclena.pyxis(charttime);


-- =========================================================
-- 6. MED RECON (Medication history before admission)
-- =========================================================

CREATE TABLE IF NOT EXISTS asclena.med_recon (
    med_recon_id BIGSERIAL PRIMARY KEY,
    stay_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,
    charttime TIMESTAMP,

    name TEXT,
    gsn TEXT,
    ndc TEXT,
    etc_rn TEXT,
    etccode TEXT,
    etcdescription TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_med_recon_stay
        FOREIGN KEY (stay_id)
        REFERENCES asclena.ed_stays(stay_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_med_recon_stay_id
ON asclena.med_recon(stay_id);

CREATE INDEX IF NOT EXISTS idx_med_recon_ndc
ON asclena.med_recon(ndc);


-- =========================================================
-- 7. FEATURE STORE (Final ML-ready table)
-- One row per stay_id
-- =========================================================

CREATE TABLE IF NOT EXISTS asclena.patient_feature_store (
    stay_id BIGINT PRIMARY KEY,

    subject_id BIGINT NOT NULL,

    -- Demographics
    age INTEGER,
    gender VARCHAR(20),

    -- Vitals engineered features
    hr_mean NUMERIC(8,3),
    hr_max NUMERIC(8,3),
    hr_min NUMERIC(8,3),

    sbp_mean NUMERIC(8,3),
    sbp_min NUMERIC(8,3),

    spo2_mean NUMERIC(8,3),
    spo2_min NUMERIC(8,3),

    rr_mean NUMERIC(8,3),

    shock_index NUMERIC(8,4),
    hr_slope NUMERIC(8,4),
    bp_slope NUMERIC(8,4),

    -- Triage
    acuity INTEGER,

    -- Clinical severity
    comorbidity_score INTEGER,
    medication_intensity_score INTEGER,

    -- Encounter
    length_of_stay_hours NUMERIC(10,2),

    -- ML target placeholder
    risk_target INTEGER,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_feature_store_stay
        FOREIGN KEY (stay_id)
        REFERENCES asclena.ed_stays(stay_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feature_store_target
ON asclena.patient_feature_store(risk_target);

--===============================