CREATE TABLE IF NOT EXISTS asclena.risk_predictions (
    risk_prediction_id BIGSERIAL PRIMARY KEY,
    stay_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    risk_score NUMERIC(6,5) NOT NULL,
    predicted_target INTEGER NOT NULL,
    risk_label VARCHAR(20) NOT NULL,
    severity_index INTEGER,
    severity_label VARCHAR(80),
    severity_description TEXT,
    severity_scale_name VARCHAR(80),
    threshold_used NUMERIC(6,5),
    top_features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE asclena.risk_predictions
ADD COLUMN IF NOT EXISTS severity_index INTEGER;

ALTER TABLE asclena.risk_predictions
ADD COLUMN IF NOT EXISTS severity_label VARCHAR(80);

ALTER TABLE asclena.risk_predictions
ADD COLUMN IF NOT EXISTS severity_description TEXT;

ALTER TABLE asclena.risk_predictions
ADD COLUMN IF NOT EXISTS severity_scale_name VARCHAR(80);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_stay_id
ON asclena.risk_predictions(stay_id);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_model
ON asclena.risk_predictions(model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_score
ON asclena.risk_predictions(risk_score);
