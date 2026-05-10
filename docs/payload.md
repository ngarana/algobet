# EPL Odds-Free Training Payload

This document explains every part of the following training request.

The goal of this payload is:

- Train an EPL-only model instead of a cross-league model.
- Keep the model pure odds-free.
- Optimize for probability quality rather than forcing the output toward market odds.
- Use Optuna hyperparameter search for a slower, accuracy-first training run.

## Full Command

```bash
curl -sS -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "description": "EPL odds-free tuned probability model",
    "activate": true,
    "tournament_ids": [359],
    "feature_groups": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats"],
    "feature_selection": true,
    "feature_selection_threshold": 0.005,
    "min_samples_per_feature": 40,
    "min_matches": 150,
    "outcome_balance": false,
    "tune_hyperparameters": true,
    "tuning_trials": 100,
    "calibrate_probabilities": true,
    "calibration_method": "sigmoid",
    "tags": {
      "model_scope": "epl",
      "odds_policy": "pure_model",
      "search_mode": "optuna"
    }
  }'
```

## Command-Level Explanation

| Component | Meaning | Why it is used here |
| --- | --- | --- |
| `curl` | Sends an HTTP request from the terminal. | This is the quickest way to trigger training from the API directly. |
| `-sS` | `-s` runs quietly, `-S` still shows errors if the request fails. | Keeps output clean without hiding failures. |
| `-X POST` | Uses the `POST` HTTP method. | Model training creates a new trained model, so this is a write operation. |
| `http://localhost:8010/api/v1/ml/train` | The local FastAPI training endpoint. | This is the API route that accepts model-training requests. |
| `-H "Content-Type: application/json"` | Declares that the request body is JSON. | The backend expects a JSON payload for training configuration. |
| `-d '...'` | Sends the JSON body. | This is where all model-training options are defined. |

## Payload Fields

| Field | Value | Meaning | Why it is used here |
| --- | --- | --- | --- |
| `model_type` | `"xgboost"` | Chooses the training algorithm. | XGBoost is the strongest default option in the current pipeline for structured football features and probability outputs. |
| `description` | `"EPL odds-free tuned probability model"` | Human-readable label for the training run. | Makes the trained model easy to identify later in the UI, metadata, and registry. |
| `activate` | `true` | Marks the new model as the active model after training completes. | Useful if this run is intended to replace the currently active production model immediately. |
| `tournament_ids` | `[359]` | Restricts training data to one tournament. | This makes the model EPL-specific. Narrow league scope usually improves calibration and outcome realism over a broad cross-league blend. |
| `feature_groups` | `["team_form", "head_to_head", "temporal", "standings", "enriched_stats"]` | Selects which feature generators are included. | This uses the full current odds-free feature stack while keeping market odds out of training. |
| `feature_selection` | `true` | Enables the two-pass feature pruning flow. | The pipeline first trains a probe model, measures feature importance, then retrains using only the selected subset. |
| `feature_selection_threshold` | `0.005` | Minimum normalized importance needed for a feature to survive pruning. | This is a permissive threshold that removes nearly useless features without being too aggressive. |
| `min_samples_per_feature` | `40` | Caps how many features can survive relative to dataset size. | This helps prevent a small league-specific dataset from carrying too many predictors. |
| `min_matches` | `150` | Requires at least 150 historical matches before training proceeds. | Prevents the run from training on too little data after the EPL filter is applied. |
| `outcome_balance` | `false` | Disables inverse-frequency class weighting. | This is important for probability quality. The current odds-free tuning path avoids class weights unless you explicitly want them. |
| `tune_hyperparameters` | `true` | Enables Optuna hyperparameter search before final training. | This makes the run slower, but it gives the best chance of finding a more accurate EPL model than a fixed-parameter baseline. |
| `tuning_trials` | `100` | Runs 100 Optuna trials. | This is a serious search budget: large enough to matter, but still practical compared with very long exploratory runs. |
| `calibrate_probabilities` | `true` | Applies a probability calibration step after model fitting. | Raw gradient-boosted probabilities are often not well calibrated. This improves how trustworthy the output probabilities are. |
| `calibration_method` | `"sigmoid"` | Uses sigmoid calibration instead of isotonic. | Sigmoid is the current production-oriented default because it is usually more stable on modest football datasets. |
| `tags` | `{...}` | Stores metadata with the model version. | These tags make it easier to filter, compare, and remember why the model was trained. |
| `tags.model_scope` | `"epl"` | Labels the model as EPL-specific. | Helps distinguish it from broader or different-league models. |
| `tags.odds_policy` | `"pure_model"` | Records that the model does not use odds in training or served probabilities. | Useful when comparing this model to any market-aware or blended experiments. |
| `tags.search_mode` | `"optuna"` | Records that the run used hyperparameter search. | Makes it easy to separate tuned runs from fixed-hyperparameter baselines. |

## Feature Groups Explained

| Feature Group | What it adds |
| --- | --- |
| `team_form` | Recent points, win rates, goals scored, goals conceded, and short-term trend signals. |
| `head_to_head` | Historical matchup patterns between the two teams. |
| `temporal` | Match timing information such as rest days, fixture density, and season progress. |
| `standings` | League-position context such as points, rank, and table-based team strength. |
| `enriched_stats` | Rolling xG, shots, corners, PPDA, and player-derived match statistics already stored in the database. |

## Why This Payload Is Accuracy-Oriented

This request is aimed at model quality rather than speed:

- It trains on one league instead of mixing leagues with different scoring and tactical patterns.
- It enables feature pruning so weak signals do not dilute the model.
- It disables class balancing because calibrated probabilities are more important than forcing class symmetry.
- It enables Optuna search with 100 trials instead of relying only on fixed defaults.
- It calibrates the final probabilities with sigmoid calibration.

## What This Payload Intentionally Does Not Do

These omissions are part of the design:

- It does not include any odds-derived feature group.
- It does not include any odds blend or implied-odds anchoring.
- It does not force team filters, date filters, or venue-only filters.
- It does not provide a custom `hyperparameters` block, because Optuna needs freedom to search the space.

## Defaults Still Applied Because They Are Omitted

The backend will still apply its normal defaults for fields not included in this payload.

| Omitted Field | Effective Default |
| --- | --- |
| `train_ratio` | `0.7` |
| `val_ratio` | `0.15` |
| `test_ratio` | `0.15` |
| `random_seed` | `42` |
| `early_stopping_rounds` | `50` |
| `split_strategy` | `"temporal"` |
| `gap_days` | `0` |
| `start_date` | no lower bound |
| `end_date` | no upper bound |
| `team_ids` | no team filter |
| `venue_filter` | no venue restriction |
| `min_total_goals` | no minimum-goals filter |
| `max_total_goals` | no maximum-goals filter |
| `use_ensemble` | `false` |
| `feature_schema_version` | `v2.0_odds_free` |

## Operational Notes

- `tournament_ids: [359]` assumes tournament `359` is the EPL in the current database.
- Because `activate` is `true`, the newly trained model will become the default model if training succeeds.
- Because `tune_hyperparameters` is `true`, this run can take much longer than a fixed-hyperparameter request.
- If you want a faster but more repeatable baseline, use the fixed-hyperparameter payload in [model-tuning.md](/home/arch/Coding/algobet/model-tuning.md:1).
