# Ablation & Permutation Feature Importance Guide

This guide explains how to use the `/api/v1/ml/ablation` endpoint to measure feature family contributions in your prediction models.

## Overview

The endpoint provides two complementary methods for understanding which feature families matter most:

| Method | Speed | What it does |
|--------|-------|--------------|
| **`permutation`** | Fast (seconds) | Shuffles feature columns per family on a trained model and measures performance drop |
| **`ablation`** | Slow (minutes) | Retrains the model excluding each feature group (leave-one-out) and compares metrics |

## Endpoint

```
POST /api/v1/ml/ablation
```

## Quick Start

### Permutation Importance (Recommended First Step)

```json
{
  "method": "permutation",
  "model_version": "xgboost_20260510_123233",
  "n_repeats": 10,
  "group_by": "family",
  "min_matches": 100
}
```

This loads the trained model, splits data 70/30, and for each feature family shuffles those columns 10 times, averaging the log loss increase. No retraining needed.

### Leave-One-Out Ablation

```json
{
  "method": "ablation",
  "feature_families": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats"],
  "min_matches": 500,
  "ablation_model_config": {
    "model_type": "xgboost",
    "calibrate_probabilities": true,
    "random_seed": 42
  }
}
```

This trains 6 models: one baseline with all groups, then one for each group excluded. Each training run follows the same pipeline as `/api/v1/ml/train`.

## Request Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | `"permutation"` | `"permutation"` or `"ablation"` |
| `model_version` | string \| null | null | Model version to analyse. `null` uses the active model. |
| `n_repeats` | int | 10 | Number of shuffle repeats (permutation only, 1–100) |
| `random_state` | int | 42 | Random seed for reproducibility |
| `group_by` | string | `"family"` | `"family"` groups by sub-family patterns (draw, away, form, etc.), `"generator"` groups by feature generator (team_form, head_to_head, etc.) |
| `feature_families` | list \| null | null | Which families/groups to evaluate. `null` = all available. |
| `min_matches` | int | 100 | Minimum matches required in evaluation data |

### Data Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | datetime \| null | 1 year ago | Start of evaluation window |
| `end_date` | datetime \| null | now | End of evaluation window |
| `tournament_ids` | list \| null | null | Filter to specific tournaments |

### Train/Test Split (shared by both methods)

| Parameter | Type | Default | Description |
|-----------
| `train_ratio` | float | 0.7 | Training split ratio |
| `val_ratio` | float | 0.15 | Validation split ratio |
| `test_ratio` | float | 0.15 | Test split ratio |
| `gap_days` | int | 0 | Gap between train and test (days) |

### Ablation-Specific Model Config

The `ablation_model_config` object controls how each retraining run is configured:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | string | `"xgboost"` | `"xgboost"`, `"lightgbm"`, or `"random_forest"` |
| `tune_hyperparameters` | bool | false | Whether to tune hyperparameters |
| `early_stopping_rounds` | int | 50 | Early stopping patience |
| `calibrate_probabilities` | bool | true | Whether to calibrate output probabilities |
| `calibration_method` | string | `"sigmoid"` | `"sigmoid"` or `"isotonic"` |
| `random_seed` | int | 42 | Random seed for reproducibility |

## Grouping Strategies

### `group_by: "family"` (Sub-family Patterns)

Groups features by semantic sub-family using pattern matching. The families are:

- **draw** — draw-related features (draw_rate, home_draw, h2h_draw, etc.)
- **away** — away performance features (away_win, away_clean_sheet, etc.)
- **low_scoring** — defensive/low-scoring features (clean_sheet, btts, etc.)
- **enriched** — advanced stats (xg, shots, ppda, player stats, etc.)
- **coverage** — data coverage indicators (has_enriched, has_player, etc.)
- **standings** — league table features (league_position, points_per_game, etc.)
- **form** — recent form features (points_last, win_rate, goal_diff, etc.)
- **temporal** — time-based features (day_of_week, rest_days, season_progress, etc.)
- **h2h** — head-to-head features (h2h_win, h2h_draw, etc.)
- **other** — features not matching any pattern

### `group_by: "generator"` (Feature Generator Groups)

Groups features by which generator produced them. These match the `feature_groups` parameter used in training:

- **team_form** — rolling form, venue-specific form, trends
- **head_to_head** — head-to-head records
- **temporal** — day of week, rest days, fixture density
- **standings** — league position, points per game, promotion/relegation zones
- **enriched_stats** — xG, shots, PPDA, player statistics

This grouping is useful when you want to understand which entire feature generator to keep or remove, since it maps directly to the `feature_groups` training parameter.

## Response Formats

### Permutation Response

```json
{
  "method": "permutation",
  "model_version": "xgboost_20260510_123233",
  "num_samples": 450,
  "n_repeats": 10,
  "baseline_log_loss": 1.05,
  "baseline_accuracy": 0.52,
  "families": [
    {
      "family": "form",
      "features_in_family": ["home_points_last_5", "away_points_last_5", ...],
      "features_found": ["home_points_last_5", "away_points_last_5", ...],
      "baseline_log_loss": 1.05,
      "permuted_log_loss": 1.12,
      "log_loss_increase": 0.07,
      "baseline_accuracy": 0.52,
      "permuted_accuracy": 0.47,
      "accuracy_decrease": 0.05,
      "importance_score": 0.45,
      "importance_rank": 1
    },
    {
      "family": "standings",
      "features_in_family": ["league_position", "points_per_game", ...],
      "features_found": ["league_position", "points_per_game", ...],
      "baseline_log_loss": 1.05,
      "permuted_log_loss": 1.08,
      "log_loss_increase": 0.03,
      "baseline_accuracy": 0.52,
      "permuted_accuracy": 0.50,
      "accuracy_decrease": 0.02,
      "importance_score": 0.19,
      "importance_rank": 2
    }
  ],
  "raw_feature_importance": {
    "home_points_last_5": 0.15,
    "league_position": 0.08,
    ...
  }
}
```

**Key metrics:**
- `log_loss_increase` — How much log loss rises when this family is permuted. Higher = more important.
- `accuracy_decrease` — How much accuracy drops. Higher = more important.
- `importance_score` — Normalized relative importance (sums to 1.0 across families).
- `importance_rank` — 1 = most important family.
- `raw_feature_importance` — The model's native feature importance (gain-based for XGBoost/LightGBM).

### Ablation Response

```json
{
  "method": "ablation",
  "baseline_model_version": "xgboost_20260511_001233",
  "baseline_num_features": 120,
  "baseline_train_metrics": {"accuracy": 0.70, "log_loss": 0.85},
  "baseline_val_metrics": {"accuracy": 0.55, "log_loss": 1.05},
  "baseline_test_metrics": {"accuracy": 0.54, "log_loss": 1.08},
  "families": [
    {
      "family": "team_form",
      "features_excluded": ["home_points_last_5", "away_points_last_5", ...],
      "num_features_used": 110,
      "model_version": "xgboost_20260511_001234",
      "train_metrics": {"accuracy": 0.67, "log_loss": 0.90},
      "val_metrics": {"accuracy": 0.53, "log_loss": 1.10},
      "test_metrics": {"accuracy": 0.52, "log_loss": 1.12},
      "log_loss_delta": 0.04,
      "accuracy_delta": -0.02
    },
    {
      "family": "temporal",
      "features_excluded": ["day_of_week", "month", ...],
      "num_features_used": 118,
      "model_version": "xgboost_20260511_001235",
      "train_metrics": {"accuracy": 0.69, "log_loss": 0.87},
      "val_metrics": {"accuracy": 0.54, "log_loss": 1.06},
      "test_metrics": {"accuracy": 0.53, "log_loss": 1.09},
      "log_loss_delta": 0.01,
      "accuracy_delta": -0.01
    }
  ]
}
```

**Key metrics:**
- `log_loss_delta` — How much test log loss increased vs baseline. Positive = worse without this group.
- `accuracy_delta` — How much accuracy changed. Negative = worse without this group.
- `features_excluded` — Which features were left out for this run.
- Each family entry includes a full `model_version` so you can load and inspect the ablated model.

## Interpreting Results

### Which Method Should I Use?

| Question | Method |
|----------|--------|
| Quick check: which feature families matter most? | **Permutation** |
| Does the model rely on specific features or the overall group signal? | **Permutation** (look at `log_loss_increase` vs family size) |
| How much would model performance drop if we remove a data source entirely? | **Ablation** |
| Can we simplify data collection by dropping a generator? | **Ablation** with `group_by: "generator"` |
| Comparing two generators' standalone predictive power? | **Ablation** (leave-one-out with just that generator) |

### Reading Permutation Results

1. **Rank families by `importance_rank`** — Lower rank = more important.
2. **Compare `log_loss_increase`** — A value near 0 means the family contributes little. Positive values mean the model relies on those features.
3. **Watch for negative `log_loss_increase`** — This can happen if permuting the family actually improves predictions, indicating the features may be adding noise.
4. **Cross-reference with `raw_feature_importance`** — The model's native importance tells you which individual features matter, while permutation tells you which *families* matter functionally.

### Reading Ablation Results

1. **Positive `log_loss_delta`** means removing this group hurt performance (the group is valuable).
2. **Negative `log_loss_delta`** means removing this group *improved* performance (the group may be adding noise — consider dropping it).
3. **Near-zero delta** means the group isn't contributing much either way.
4. **Compare feature counts** — A group with fewer features but a large delta is highly efficient per feature.

## Example Workflows

### 1. Quick Feature Audit

```bash
# Start with permutation to identify top families
curl -X POST http://localhost:8010/api/v1/ml/ablation \
  -H "Content-Type: application/json" \
  -d '{
    "method": "permutation",
    "model_version": "xgboost_20260510_123233",
    "n_repeats": 10,
    "group_by": "generator",
    "min_matches": 500
  }'
```

Look at `importance_rank` to quickly see which generators matter.

### 2. EPL Feature Optimization

```bash
# First: permutation audit at sub-family level
curl -X POST http://localhost:8010/api/v1/ml/ablation \
  -H "Content-Type: application/json" \
  -d '{
    "method": "permutation",
    "group_by": "family",
    "tournament_ids": [359],
    "n_repeats": 20
  }'

# Then: confirm with ablation at generator level
curl -X POST http://localhost:8010/api/v1/ml/ablation \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ablation",
    "feature_families": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats"],
    "tournament_ids": [359],
    "min_matches": 1000,
    "ablation_model_config": {
      "model_type": "xgboost",
      "calibrate_probabilities": true,
      "calibration_method": "sigmoid"
    },
    "split_strategy": "season_aware",
    "train_seasons": 5,
    "val_seasons": 1,
    "test_seasons": 1
  }'
```

### 3. Validate a Simplified Feature Set

If permutation shows `temporal` and `coverage` families are unimportant, train a simplified model using only the important generators:

```bash
curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "feature_groups": ["team_form", "head_to_head", "standings", "enriched_stats"],
    "tournament_ids": [359],
    "min_matches": 500
  }'
```

Then compare test metrics with your baseline.

## Caveats

- **Permutation** is model-dependent: it measures how much the *trained* model relies on each family. A model that never uses a feature (e.g., because it was pruned during training) will show 0 importance for that family, even if the underlying signal is strong.
- **Ablation** retrains from scratch each time. Results depend on the training configuration (hyperparameters, split strategy, calibration). Use the same `ablation_model_config` as your baseline for fair comparisons.
- **Correlated features**: Permutation shuffles one family at a time, so if two families encode similar information, permuting one may not cause a large drop because the other compensates. Ablation avoids this by removing the entire group during training.
- **Single-feature permutation** is not currently supported. Use the model's `feature_importance` from the train response for per-feature importance.
