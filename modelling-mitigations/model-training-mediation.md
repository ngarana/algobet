# EPL Model Fine-Tuning Implementation Plan

## Problem Statement

The last EPL-only XGBoost run from `payload.md` did not fail because it was
odds-free. It failed because the feature/model stack did not preserve enough
signal for away wins, draws, and enriched team strength.

Known symptoms:

- Latest xgboost model, predicted Burnley vs Brighton
  `19589` as `HOME` at about `46%` even though the match finished `0-2`.
- Stored holdout log loss trailed the market benchmark:
  `test_log_loss=1.0809` vs `test_market_log_loss=0.9637`.
- The selected feature list contained no `enriched_stats` columns even though
  the payload enabled `enriched_stats`.
- The EPL-only backtest diagnosis showed class collapse symptoms: Home-only
  predictions with `F1 Away=0.0` and `F1 Draw=0.0`.

This plan keeps the production model pure: no odds features, no implied-odds
blend, and no market anchoring in training, tuning, calibration, or prediction.
Odds are allowed only as a post-hoc diagnostic benchmark.

## Success Criteria

Do not activate a new model unless all of these pass on an EPL-only temporal or
season-aware holdout:

- `predicted_classes == 3` on validation and test.
- `max_prediction_share <= 0.70` on validation and test.
- Draw and away recall are both non-zero.
- Test log loss beats the current XGBoost baseline from `payload.md`.
- Test log loss narrows the gap to `test_market_log_loss`; beating the market is
  not required for the first pass, but losing by a wider margin is a reject.
- Feature selection report shows retained features from:
  - draw/low-scoring family,
  - away-strength family,
  - team-form differential family,
  - enriched coverage or enriched-stat family when enrichment coverage exists.
- The model artifact stores the exact feature pipeline, selected features,
  base-model metrics, ensemble weights, calibration method, and EPL evaluation
  metrics.

## Phase 1: Make Evaluation Trustworthy

Files:

- `algobet/api/routers/ml_operations.py`
- `algobet/predictions/features/pipeline.py`
- `scripts/evaluate_epl_model.py` (new)
- `tests/integration/test_ml_operations.py`
- `tests/predictions/test_feature_pipeline.py`

Implementation:

1. Add a repo script `scripts/evaluate_epl_model.py` that accepts
   `--model-version`, `--tournament-id 359`, `--start-date`, `--end-date`, and
   `--min-matches`.
2. The script must load the saved model-specific `feature_pipeline_path`, not a
   default feature pipeline.
3. It must print and save:
   - log loss,
   - market log loss when odds exist,
   - confusion matrix,
   - per-class precision/recall/F1,
   - predicted-class distribution,
   - model-vs-market favorite agreement,
   - selected feature list,
   - grouped feature counts.
4. Add an integration assertion that `/api/v1/ml/backtest` uses the saved
   feature pipeline when `feature_pipeline_path` exists.
5. Add a stale-row guard in the evaluation script: only score rows with
   `status='FINISHED'`, non-null scores, non-null `season_id`, and the requested
   `tournament_id`.

Acceptance:

- Running the evaluator against `xgboost_20260508_035804` reproduces the failure
  profile before any retrain.
- The evaluator and API backtest agree on sample count, feature count, and
  predicted-class distribution.

## Phase 2: Build a Feature Audit Before Training

Files:

- `scripts/audit_epl_features.py` (new)
- `algobet/predictions/features/pipeline.py`
- `algobet/predictions/training/pipeline.py`
- `tests/predictions/test_feature_pipeline.py`

Implementation:

1. Add `scripts/audit_epl_features.py` for tournament `359`.
2. Generate raw features once with the requested feature groups.
3. Export `reports/epl_feature_audit.json` and `reports/epl_feature_audit.csv`.
4. Include, per feature:
   - raw null rate,
   - zero rate,
   - variance,
   - correlation cluster,
   - feature group,
   - family label such as `draw`, `away`, `low_scoring`, `enriched`,
     `coverage`, `standings`, `temporal`,
   - train-only univariate mutual information or permutation signal,
   - whether the feature survived selection.
5. Add grouped summaries:
   - group coverage,
   - selected count by group,
   - selected count by outcome family,
   - top positive/negative correlations with target encodings.

Acceptance:

- The audit explains why `enriched_stats` was dropped.
- The audit identifies redundant 3/5/10 form-window features before retraining.
- The audit can be rerun without training a model.

## Phase 3: Repair the Feature Set

Files:

- `algobet/predictions/features/generators.py`
- `algobet/predictions/training/pipeline.py`
- `tests/predictions/test_feature_pipeline.py`

Schema:

- Bump feature schema to `v3.0_epl_feature_tuning`.
- Keep the request feature group names stable for now:
  `team_form`, `head_to_head`, `temporal`, `standings`, `enriched_stats`.

Feature additions:

### Team Form

Add these for each rolling window `3`, `5`, and `10`:

- `home_draw_rate_{w}`, `away_draw_rate_{w}`
- `home_loss_rate_{w}`, `away_loss_rate_{w}`
- `home_clean_sheet_rate_{w}`, `away_clean_sheet_rate_{w}`
- `home_failed_to_score_rate_{w}`, `away_failed_to_score_rate_{w}`
- `home_btts_rate_{w}`, `away_btts_rate_{w}`
- `home_low_scoring_rate_{w}`, `away_low_scoring_rate_{w}`
- `home_goal_variance_{w}`, `away_goal_variance_{w}`
- `home_points_volatility_{w}`, `away_points_volatility_{w}`

Add venue-specific draw/away signals:

- `home_home_draw_rate_{w}`
- `away_away_draw_rate_{w}`
- `away_away_win_rate_{w}`
- `away_away_clean_sheet_rate_{w}`

### Head To Head

Add:

- `h2h_draw_rate`
- `h2h_away_win_rate`
- `h2h_low_scoring_rate`
- `h2h_btts_rate`
- `h2h_goal_diff_avg_from_home_perspective`
- `h2h_recency_weighted_home_points`

H2H must stay capped and low weight because it is sparse. It should help draw
and low-scoring priors, not dominate the model.

### Standings

Current standings are computed as-of match time. Preserve that leakage-safe
behavior and add:

- `home_draw_rate_season`, `away_draw_rate_season`
- `home_loss_rate_season`, `away_loss_rate_season`
- `draw_rate_diff_season`
- `loss_rate_diff_season`
- `points_per_game_diff`
- `home_top_six`, `away_top_six`
- `home_bottom_six`, `away_bottom_six`

### Enriched Stats

Change the default `enriched_stats` generator to include differential features
or add an explicit internal option for the EPL tuning run:

- `include_diffs=True`

Add enriched derived features:

- `xg_diff_avg_{w}`
- `npxg_diff_avg_{w}`
- `xg_allowed_diff_avg_{w}`
- `shot_quality_for_avg_{w}` = xG for / shots for
- `shot_quality_against_avg_{w}` = xG against / shots against
- `shots_on_target_rate_for_avg_{w}`
- `shots_on_target_rate_against_avg_{w}`
- `xg_conversion_for_avg_{w}` = goals for - xG for
- `xg_conversion_against_avg_{w}` = goals against - xG against
- `ppda_diff_avg_{w}`
- `deep_completions_diff_avg_{w}`

Expand player rollups already present in the database:

- `player_saves`
- `player_goals_conceded`
- `player_fouls_committed`
- `player_fouls_suffered`
- `player_yellow_cards`
- `player_red_cards`
- `player_offsides`
- `starter_minutes`
- `starter_count`

Missingness and coverage:

- Keep existing coverage features.
- Add `home_has_enriched_match_stats_{w}`, `away_has_enriched_match_stats_{w}`.
- Add `home_has_player_stats_{w}`, `away_has_player_stats_{w}`.
- Do not silently turn missing enrichment into a false zero signal without a
  coverage flag beside it.

Acceptance:

- Feature generation produces no odds-derived fields.
- New draw, away, and enriched-diff features exist in
  `FeaturePipeline.feature_names`.
- Unit tests prove standings snapshots are strictly before match date.
- Unit tests prove enriched player fields include saves and offsides when the DB
  rows contain them.

## Phase 4: Replace Naive Feature Selection With Group-Aware Selection

Files:

- `algobet/predictions/training/pipeline.py`
- `algobet/predictions/training/feature_selection.py` (new)
- `tests/unit/predictions/test_training_pipeline.py`

Implementation:

1. Add correlation pruning before model-importance pruning:
   - default `max_feature_correlation=0.94`,
   - keep the stronger validation-signal feature inside each correlated group.
2. Add feature family tagging by name pattern:
   - `draw`,
   - `away`,
   - `low_scoring`,
   - `enriched`,
   - `coverage`,
   - `standings`,
   - `form`,
   - `temporal`.
3. Add minimum retention guards for EPL tuning:
   - at least 3 draw-family features when available,
   - at least 3 away-family features when available,
   - at least 5 enriched or coverage features when enriched coverage is non-zero,
   - at least 2 low-scoring features when available.
4. The guards must not keep all-zero or near-constant features.
5. Store a `feature_selection_report` in model hyperparameters:
   - selected features,
   - dropped features,
   - drop reason,
   - group counts before and after,
   - correlation clusters,
   - retained protected features.

Acceptance:

- The selection pass can no longer drop the entire enriched family without
  recording a clear reason.
- A collapsed Home-only model is rejected before activation.
- The report is visible from model metadata.

## Phase 5: Tune XGBoost And LightGBM Separately

Files:

- `algobet/predictions/training/tuner.py`
- `algobet/predictions/training/pipeline.py`
- `algobet/predictions/training/classifiers.py`
- `tests/unit/predictions/test_training_tuner.py`
- `tests/unit/predictions/test_training_classifiers.py`

Current issue:

- The existing ensemble path passes a shared hyperparameter dictionary through
  `resolve_training_hyperparameters()` for all base models. XGBoost and LightGBM
  need separate search spaces and separate best params.

Implementation:

1. Add `per_model_hyperparameters` to `TrainingConfig`.
2. Add `per_model_tuning_results` to `TrainingResult`.
3. When `use_ensemble=true`, tune each base model independently:
   - XGBoost gets XGBoost search space.
   - LightGBM gets LightGBM search space.
4. Change tuner objective from plain log loss to guarded log loss:
   - base objective: validation `log_loss`,
   - add penalty if predicted classes < 3,
   - add penalty if `max_prediction_share > 0.70`,
   - add small penalty for high ECE.
5. Recommended XGBoost search space for EPL:
   - `max_depth`: 2 to 5
   - `learning_rate`: 0.01 to 0.08
   - `n_estimators`: 400 to 2000
   - `min_child_weight`: 3 to 30
   - `gamma`: 0.0 to 3.0
   - `reg_alpha`: 0.0 to 8.0
   - `reg_lambda`: 2.0 to 30.0
   - `subsample`: 0.55 to 0.90
   - `colsample_bytree`: 0.40 to 0.85
6. Recommended LightGBM search space for EPL:
   - `num_leaves`: 7 to 63
   - `max_depth`: 2 to 6
   - `learning_rate`: 0.01 to 0.08
   - `n_estimators`: 400 to 2500
   - `min_child_samples`: 20 to 150
   - `min_split_gain`: 0.0 to 2.0
   - `reg_alpha`: 0.0 to 10.0
   - `reg_lambda`: 2.0 to 40.0
   - `subsample`: 0.55 to 0.90
   - `colsample_bytree`: 0.40 to 0.85
7. Keep `outcome_balance=false` for the main probability model. Use collapse
   recovery only as a rescue path, and record it if triggered.

Acceptance:

- A tuned XGBoost-only model and a tuned LightGBM-only model can be trained from
  the same feature pipeline.
- Their params and metrics are stored separately.
- The tuner refuses a candidate that wins log loss by collapsing to one class.

## Phase 6: Implement Weighted XGBoost + LightGBM Ensemble

Files:

- `algobet/predictions/training/ensemble.py` (new)
- `algobet/predictions/training/classifiers.py`
- `algobet/predictions/training/pipeline.py`
- `algobet/predictions/models/registry.py`
- `algobet/api/routers/ml_operations.py`
- `frontend/lib/types/ml-operations.ts`
- `frontend/components/models/EnsembleSection.tsx`
- `tests/unit/predictions/test_training_classifiers.py`
- `tests/unit/predictions/test_training_pipeline.py`
- `tests/integration/test_ml_operations.py`

Implementation:

1. Add `EnsembleWeightOptimizer`.
2. Input:
   - validation probabilities from each fitted base model,
   - validation labels,
   - objective metric `log_loss`.
3. For two models, use deterministic grid search:
   - `xgb_weight` from `0.05` to `0.95` by `0.01` or `0.02`,
   - `lgbm_weight = 1.0 - xgb_weight`.
4. Reject weights that produce class collapse.
5. Store:
   - `ensemble_strategy=weighted_soft_vote`,
   - `ensemble_weights`,
   - `base_model_metrics`,
   - `ensemble_validation_metrics`.
6. Save ensemble artifacts under `data/models/ensemble/<version>` instead of
   pretending the model is only `xgboost`.
7. In the API response and UI, display the model as `ensemble`.
8. Split validation internally for ensemble runs:
   - first half: early stopping and ensemble weight search,
   - second half: final probability calibration,
   - test split: untouched final evaluation.

Acceptance:

- Equal-weight ensemble remains available as a fallback.
- Weighted ensemble beats both base models on validation log loss before it is
  allowed to proceed.
- The final saved model loads and predicts through the existing model registry.
- The UI can request `use_ensemble=true` with `["xgboost", "lightgbm"]`.

## Phase 7: Training Payload After Implementation

Run with `activate=false` first. Activate only after the acceptance gates pass.

```bash
curl -sS -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "description": "EPL v3 tuned weighted XGBoost LightGBM ensemble",
    "activate": false,
    "tournament_ids": [359],
    "feature_groups": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats"],
    "feature_selection": true,
    "feature_selection_threshold": 0.002,
    "min_samples_per_feature": 60,
    "min_matches": 1000,
    "outcome_balance": false,
    "tune_hyperparameters": true,
    "tuning_trials": 150,
    "calibrate_probabilities": true,
    "calibration_method": "sigmoid",
    "use_ensemble": true,
    "ensemble_types": ["xgboost", "lightgbm"],
    "split_strategy": "season_aware",
    "train_seasons": 5,
    "val_seasons": 1,
    "test_seasons": 1,
    "tags": {
      "model_scope": "epl",
      "odds_policy": "pure_model",
      "feature_schema": "v3.0_epl_feature_tuning",
      "ensemble": "weighted_xgboost_lightgbm"
    }
  }'
```

Fallback if season-aware split has too few complete seasons:

```json
{
  "split_strategy": "temporal",
  "train_ratio": 0.70,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "gap_days": 7
}
```

## Phase 8: Ablation Matrix

Run every row with the same EPL date range and split.

| Run | Model | Features | Selection | Purpose |
| --- | --- | --- | --- | --- |
| A | Existing XGBoost payload | v2 features | current | Baseline failure reproduction |
| B | XGBoost | v3 features | off | Check whether new features help without pruning |
| C | XGBoost | v3 features | group-aware | Check feature selection effect |
| D | LightGBM | v3 features | group-aware | Independent base learner |
| E | XGBoost + LightGBM | v3 features | group-aware | Equal soft-vote control |
| F | XGBoost + LightGBM | v3 features | group-aware | Weighted soft-vote target |
| G | Weighted ensemble | no enriched_stats | group-aware | Prove enriched contribution |
| H | Weighted ensemble | enriched_stats only plus form | group-aware | Isolate enriched signal quality |

The target model is F only if it beats C, D, and E on test log loss without
class collapse.

## Verification Commands

Backend unit tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/predictions/test_feature_pipeline.py \
  tests/unit/predictions/test_training_pipeline.py \
  tests/unit/predictions/test_training_tuner.py \
  tests/unit/predictions/test_training_classifiers.py \
  -q
```

API integration:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/integration/test_ml_operations.py \
  tests/integration/test_predictions_router.py \
  -q
```

Frontend contract checks if UI fields change:

```bash
cd frontend
pnpm typecheck
pnpm test -- lib/types components/models
```

Live Docker verification:

```bash
docker exec algobet-api python scripts/audit_epl_features.py --tournament-id 359
docker exec algobet-api python scripts/evaluate_epl_model.py \
  --model-version xgboost_20260508_035804 \
  --tournament-id 359
```

Then train with the Phase 7 payload and evaluate the returned model version with
the same evaluator.

## Implementation Order

1. Evaluation script and saved-pipeline backtest assertion.
2. Feature audit script.
3. Feature generator additions and schema bump.
4. Group-aware feature selection report.
5. Per-model tuning for XGBoost and LightGBM.
6. Weighted ensemble optimizer.
7. API and UI support for ensemble metadata.
8. Ablation run and activation decision.

## Activation Rule

The first ensemble run should be trained with `activate=false`. Activate only
after the evaluator shows:

- all three classes predicted,
- non-zero draw and away recall,
- lower test log loss than the last XGBoost payload run,
- no odds-derived features in the training feature list,
- saved `feature_selection_report`,
- saved `ensemble_weights`,
- a reproducible EPL-only backtest result.
