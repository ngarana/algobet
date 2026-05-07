# Odds-Free Model Training And Prediction Generation

## Summary

Make new model training and prediction generation fully odds-free. Remove the `odds_blend` / `odds_blend_weight` training path, stop using implied-odds feature generators, remove odds-required training filters, and require retraining before prediction generation can use a model.

## Interface Changes

- `POST /api/v1/ml/train` no longer defines `odds_blend`, `odds_blend_weight`, or `require_odds`.
- Training `feature_groups` excludes `"odds"`; allowed groups are `team_form`, `head_to_head`, `temporal`, and `standings`.
- New trained models use `feature_schema_version = "v2.0_odds_free"`.
- Prediction generation rejects old/missing-schema models with a clear “retrain required” error.

## Implementation Changes

- Backend training:
  - Remove odds fields from `TrainModelRequest` and `TrainingConfig`.
  - Remove `odds_blend=request.odds_blend` and `odds_blend_weight=request.odds_blend_weight`.
  - Remove `OddsAnchoredBlender` and its exports.
  - Stop passing `require_odds` into training data selection.

- Feature pipeline:
  - Remove odds generators from the default model feature set.
  - Remove `"odds"` from feature-group construction and validate unsupported groups.
  - Save the fitted feature pipeline beside each trained model artifact and store its path in model hyperparameters.
  - Update prediction generation to load the model’s saved feature pipeline, not the global `data/pipelines/v1.0` pipeline.

- Frontend:
  - Remove the “Require betting odds available” training control.
  - Remove the “Market Odds” feature group option.
  - Remove `requireOdds` from training config/types/defaults and request builders.
  - Remove `require_odds` from `TrainModelRequestSchema`.

## Test Plan

- Update/add backend tests for:
  - Training config/request no longer contains odds blend or require-odds fields.
  - Default feature names contain no `odds`, `implied_prob`, `bookmaker`, `favorite`, or market-derived fields.
  - `feature_groups=["odds"]` is rejected or fails validation.
  - Prediction generation rejects pre-`v2.0_odds_free` models.
  - Prediction generation succeeds for a retrained odds-free model with its saved feature pipeline.

- Update frontend checks:
  - `pnpm --filter algobet-frontend typecheck`
  - Existing model-training UI tests if present.

- Backend verification:
  - `uv run pytest tests/unit/predictions/test_training_pipeline.py tests/predictions/test_feature_pipeline.py tests/unit/services/test_prediction_service.py tests/integration/test_ml_operations.py tests/integration/test_predictions_router.py -q`
  - Note: baseline currently has stale failures in `test_training_pipeline.py` around removed calibration-policy helpers; refresh those expectations while touching this file.

## Assumptions

- Odds remain available for scraping, match display, value-bet calculations, and betting/backtest metrics.
- Odds are removed only from model training features and prediction-generation features.
- Existing trained models are intentionally unsupported until retrained under `v2.0_odds_free`.
