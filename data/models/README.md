# Model Artifact Storage

This directory stores versioned prediction model artifacts created by
`TrainingPipeline` and calibration workflows.

Expected layout:

```text
data/models/
  <model_type>/
    <version>/
      model.pkl
      metadata.json
```

Notes:
- Artifacts are managed by `ModelRegistry` in `algobet/predictions/models/registry.py`.
- Avoid manual file edits; use CLI/API model management operations.
- Model artifact files are gitignored.
