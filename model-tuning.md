# Pure Odds-Free Probability Tuning Plan

## Summary

The model is not “wrong because it ignored odds” so much as it is currently a weak, under-tuned probability model. For match `19589`, normalized market probabilities were about `HOME 17.9% / DRAW 22.4% / AWAY 59.7%`, while the model gave `HOME 45.9% / DRAW 24.6% / AWAY 29.5%`. On a 60-match EPL sample, the market still beat the model on log loss (`1.0236` vs `1.0819`).

We will keep the model pure: no odds features, no odds blend, no implied-odds anchoring. Odds may be used only as diagnostics after prediction.

## Key Changes

- Make `feature_selection=true` actually work.
  - Current request accepts the flag but the pipeline never prunes features.
  - Add a two-pass feature selection step: train a probe model, select features by normalized gain importance, apply `feature_selection_threshold`, enforce `min_samples_per_feature`, then retrain with only selected features.
  - Save the selected feature names in model hyperparameters and in the fitted feature pipeline.

- Fix probability-oriented training defaults.
  - Change `outcome_balance` default behavior so class weights are only applied when explicitly set to `true`; inverse-frequency weights distort calibrated probabilities.
  - Prefer `sigmoid` calibration for production probability stability; keep `isotonic` available.
  - Use stronger conservative XGBoost defaults for odds-free football probabilities: shallower trees, higher regularization, lower learning rate, and early stopping.

- Fix feature-pipeline reproducibility.
  - `FeaturePipeline.load()` currently recreates default generators instead of faithfully reconstructing the saved generator setup.
  - Persist and reload the exact non-odds feature groups and selected feature subset so prediction uses the same feature shape as training.

- Fix standings leakage.
  - `get_team_standings(... before_date=...)` currently returns the latest cached standings snapshot, ignoring `before_date`.
  - Add an `as_of` timestamp to standings snapshots and return the latest snapshot strictly before the match date.

- Add market diagnostics without training on odds.
  - During evaluation only, compute normalized implied-odds benchmark metrics where odds exist: market log loss, market favorite accuracy, model-vs-market probability MAE, and favorite agreement.
  - Store these as model metrics for review; do not feed them into training, tuning, calibration, or prediction.

## Recommended Retrain Request

After the code fixes, retrain an EPL-specific pure model. This is the best stable production payload for the current code path:

```bash
curl -sS -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "description": "EPL odds-free calibrated probability model",
    "tournament_ids": [359],
    "feature_groups": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats"],
    "feature_selection": true,
    "feature_selection_threshold": 0.005,
    "min_samples_per_feature": 40,
    "min_matches": 150,
    "outcome_balance": false,
    "tune_hyperparameters": false,
    "calibrate_probabilities": true,
    "calibration_method": "sigmoid",
    "hyperparameters": {
      "max_depth": 3,
      "learning_rate": 0.03,
      "n_estimators": 1200,
      "min_child_weight": 10,
      "gamma": 1.0,
      "reg_alpha": 2.0,
      "reg_lambda": 10.0,
      "subsample": 0.7,
      "colsample_bytree": 0.5
    },
    "tags": {"model_scope": "epl", "odds_policy": "pure_model"}
  }'
```

If Optuna is installed and you want a slower search-heavy run, switch to:

```json
"tune_hyperparameters": true,
"tuning_trials": 100
```

and remove the explicit `hyperparameters` block so the tuner can search freely. That path is the best shot at squeezing out more holdout performance, but the fixed request above is the most repeatable production baseline.

## Test Plan

- Unit tests:
  - `feature_selection=true` reduces feature count and respects `min_samples_per_feature`.
  - selected feature names are saved and restored with the model’s feature pipeline.
  - `outcome_balance` defaults to no class weights unless explicitly enabled.
  - standings lookup returns the latest snapshot before the match date, not the final-season snapshot.
  - no trained feature names contain `odds`, `implied_prob`, `bookmaker`, `favorite`, or `market`.

- Integration tests:
  - training with the recommended EPL request succeeds and stores selected features.
  - prediction generation loads the selected-feature pipeline without feature-count mismatch.
  - model metrics include market diagnostics when odds exist, while prediction probabilities remain odds-free.
  - direct service prediction for match `19589` works with the retrained model.

## Assumptions

- We are keeping the user-selected `Pure Model Only` policy: odds never influence training, calibration, or served probabilities.
- “Closer to implied odds” means fewer large probability mistakes and better calibrated outcome probabilities, not forced market anchoring.
- EPL model quality matters more here than cross-league generality, so `tournament_ids: [359]` is the recommended training scope.
