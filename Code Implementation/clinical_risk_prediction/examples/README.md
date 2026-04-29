# Clinical Risk API Examples

## Purpose

These example files give a stable request and an expected response shape for quick manual verification of the deployed API.

Files:

- `sample_predict_request_high_risk.json`
- `sample_predict_expected_high_risk.json`
- `sample_predict_request_borderline_high_risk.json`
- `sample_predict_expected_borderline_high_risk.json`
- `sample_predict_request_low_risk.json`
- `sample_predict_expected_low_risk.json`
- `sample_predict_request_invalid_missing_feature.json`
- `sample_predict_expected_invalid_missing_feature.json`

## How to use

Start the API, then send:

```bash
curl -X POST "http://127.0.0.1:8002/v1/predict" \
  -H "Content-Type: application/json" \
  --data @examples/sample_predict_request_high_risk.json
```

## Expected key result

For the current trained artifact `asclena_xgboost_risk_20260429T184534Z.joblib`, the important expected values are:

- `risk_score = 1.0`
- `predicted_target = 1`
- `risk_label = HIGH`
- `calibration_method = isotonic`

Expected leading contributors:

- `rr_mean`
- `acuity`
- `hr_min`
- `vital_row_count`
- `spo2_min`

## Additional cases

### Borderline but still high-risk case

Expected:

- `risk_score = 0.885167`
- `predicted_target = 1`
- `risk_label = HIGH`

This case is useful because it shows the calibrated model still pushes some clinically concerning mid-severity cases into the `HIGH` band.

### Low-risk case

Expected:

- `risk_score = 0.049485`
- `predicted_target = 0`
- `risk_label = LOW`

### Invalid contract case

Expected:

- HTTP `400`
- `error_code = invalid_feature_contract`

This case is useful for backend integration testing when the caller forgets required features.

## Important note

The full response includes `top_contributors` by default.
Use `include_feature_snapshot=true` when you also need the full ordered feature snapshot for debugging.

When verifying manually, focus on:

- `prediction.risk_score`
- `prediction.predicted_target`
- `prediction.risk_label`
- top contributor ordering

If the model artifact version changes, this expected output should be regenerated.
The calibrated model keeps the same request and response schema, but probability values can shift because `risk_score` is now calibrated on the validation set.
