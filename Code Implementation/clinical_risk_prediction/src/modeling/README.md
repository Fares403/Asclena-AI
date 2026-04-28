# Asclena XGBoost Risk Model

This pipeline trains a binary XGBoost classifier using
`asclena.patient_feature_store`.

The model learns:

- `risk_target = 1`: high-risk ED visit
- `risk_target = 0`: low-risk or stable ED visit

The system output is a probability:

```python
risk_scores = model.predict_proba(X)[:, 1]
```

Risk labels:

- `risk_score < 0.40`: `LOW`
- `0.40 <= risk_score < 0.70`: `MODERATE`
- `risk_score >= 0.70`: `HIGH`

Run from `Code Implementation/clinical_risk_prediction`:

```bash
python src/modeling/train_xgboost_risk_model.py
```

Outputs:

- model artifact in `models/`
- reports in `reports/modeling/<model_version>/`
- predictions in `asclena.risk_predictions`

Excluded from training:

- identifiers: `stay_id`, `subject_id`
- target/leakage columns: `disposition`, `has_hadm_id`,
  `length_of_stay_hours`, `risk_target`
- raw categorical text: `race`, `arrival_transport`
- diagnosis and medication counts in this MVP, because they may not be
  available at early triage time or may leak downstream care intensity
