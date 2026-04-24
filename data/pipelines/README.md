# Feature Pipeline Storage

This directory stores serialized fitted feature pipelines used at inference
time by `PredictionService`.

Expected layout:

```text
data/pipelines/
  <feature_schema_version>/
    config.json
    numerical_scaler.joblib
    categorical_encoder.joblib
```

Notes:
- Pipelines are produced by the training pipeline after fitting.
- Inference loads from this path and falls back to basic form features if
  no fitted pipeline is available.
- Serialized pipeline artifacts are gitignored.
