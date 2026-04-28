CREATE TABLE IF NOT EXISTS asclena.risk_predictions (
    risk_prediction_id BIGSERIAL PRIMARY KEY,
    stay_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    risk_score NUMERIC(6,5) NOT NULL,
    predicted_target INTEGER NOT NULL,
    risk_label VARCHAR(20) NOT NULL,
    threshold_used NUMERIC(6,5),
    top_features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_stay_id
ON asclena.risk_predictions(stay_id);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_model
ON asclena.risk_predictions(model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_score
ON asclena.risk_predictions(risk_score);
