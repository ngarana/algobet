# Address Draw Correction Limitations

Fixes the three known limitations documented in [DRAW_CORRECTION_IMPLEMENTATION.md](file:///home/arch/Coding/algobet/DRAW_CORRECTION_IMPLEMENTATION.md#L335-L339):

1. **Feature Set Constraint**: Only `team_form` features viable (others collinear)
2. **Dixon-Coles Uses Dummy Features**: Currently learns league-wide priors only (not match-specific)
3. **Draw Recall Still Suboptimal**: Needs richer features or ensemble approach for production

---

## User Review Required

> [!IMPORTANT]
> This plan introduces a **stacking meta-learner** as an alternative to the current `EnsemblePredictor` weighted averaging. This is a meaningful architectural addition — please confirm this direction.

> [!WARNING]
> Limitation 1 (collinearity) is partially inherent to the data domain. The plan mitigates it via curated feature groups and draw-signal features, but will not eliminate all inter-group correlation. The existing correlation pruning (`max_feature_correlation=0.94`) handles residual collinearity.

---

## Proposed Changes

### Limitation 1: Feature Set Constraint (Collinearity)

The current problem is that combining multiple feature groups (e.g., `team_form` + `standings` + `elo_rating`) introduces high collinearity that hurts model performance. Only `team_form` is viable alone.

**Approach**: Create **draw-signal composite features** — new features specifically engineered to capture draw conditions — that are complementary to (not collinear with) existing features. These features compress draw-relevant signals from multiple groups into a small set of orthogonal features.

---

#### [NEW] [draw_signal_generator.py](file:///home/arch/Coding/algobet/algobet/predictions/features/draw_signal_generator.py)

A new `DrawSignalFeatureGenerator` implementing `FeatureGenerator` that produces ~15 draw-targeted features:

| Feature | Signal | Rationale |
|---------|--------|-----------|
| `strength_parity` | `abs(home_form - away_form)` | Small values predict draws |
| `combined_draw_rate_{3,5,10}` | `(home_draw_rate + away_draw_rate) / 2` | Teams that draw frequently should draw against each other |
| `defensive_balance` | `abs(home_goals_against - away_goals_against)` | Similar defensive strengths → draws |
| `low_scoring_probability` | `home_low_scoring_rate * away_low_scoring_rate` | Interaction term: both teams tending to low-scoring games |
| `clean_sheet_interaction` | `home_clean_sheet_rate * away_failed_to_score_rate + ...` | Cross-match defensive dominance signals |
| `goal_convergence_{3,5}` | `1 / (1 + abs(home_gf - away_gf))` | Similar goal rates → draw |
| `volatility_sum_{3,5}` | `home_goal_variance + away_goal_variance` | Low volatility → more predictable draws |
| `h2h_draw_boost` | Combines H2H draw rate with sample size weighting | H2H draws are sticky |
| `xg_parity_{3,5}` | `1 / (1 + abs(home_xg_for - away_xg_against))` | xG balance predicts draws (uses enriched stats if available) |

**Why this is non-collinear**: These are *interaction* and *transformation* features (products, absolute differences, reciprocals). Raw `draw_rate_5` is a team-level feature; `combined_draw_rate_5` is a match-level interaction. Parity features are nonlinear transforms that capture *closeness* rather than raw magnitude.

The generator will be built from values already computed by other generators (team_form, h2h, enriched_stats), so it doesn't make redundant DB queries — it computes from the `matches` DataFrame and repository like all other generators.

---

#### [MODIFY] [composite.py](file:///home/arch/Coding/algobet/algobet/predictions/features/composite.py)

- Register `draw_signals` in `create_generators_by_names()` factory
- Add `draw_signals` to `create_default_generators()` list

---

#### [MODIFY] [config.py](file:///home/arch/Coding/algobet/algobet/predictions/training/config.py)

- Add `"draw_signals"` to `ALLOWED_FEATURE_GROUPS`

---

#### [MODIFY] [feature_selection.py](file:///home/arch/Coding/algobet/algobet/predictions/training/feature_selection.py)

- Add `"draw_signal"` family to `FEATURE_FAMILIES` mapping so draw-signal features get grouped/protected correctly during feature selection

---

### Limitation 2: Match-Specific Dixon-Coles

The current `DixonColesPredictor` is trained with dummy features (`np.arange(n).reshape(-1,1)`) in the training script, so it learns only league-wide average goal rates. Every match gets nearly the same ~25% draw probability regardless of team strength.

**Approach**: Make Dixon-Coles match-specific by using real features, and integrate it into the standard training pipeline so it trains alongside XGBoost.

---

#### [MODIFY] [classifiers.py](file:///home/arch/Coding/algobet/algobet/predictions/training/classifiers.py) — `DixonColesPredictor`

- Add a new `fit(X, y, X_val, y_val)` implementation that extracts goals from the matches dataframe via a stored mapping, removing the need for the dummy-feature workaround. This makes `fit()` functional so Dixon-Coles can participate in the standard pipeline.
- Alternatively: store goals in `fit_with_scores()` and make `fit()` delegate to `fit_with_scores()` when goals are available from the training context.

The better approach is to make the `runner.py` explicitly pass goals data when `dc_model_path` is set or when `fit_draw_aware_calibrator` is enabled:

---

#### [MODIFY] [runner.py](file:///home/arch/Coding/algobet/algobet/predictions/training/runner.py) — Step 6c

Replace the current approach (loading a pre-trained DC model from disk) with in-pipeline training:

1. Extract `home_goals` and `away_goals` arrays from `self._train_df` (which has `home_score` / `away_score` columns)
2. Create a `DixonColesPredictor`, call `fit_with_scores(X_train, y_train, home_goals, away_goals, X_val, y_val)`
3. This DC model now uses the same real features as XGBoost, producing match-specific draw probabilities
4. Fit `DrawAwareCalibrator` using XGBoost's val probas and the match-specific DC val probas
5. Save the DC model alongside the XGBoost model

This eliminates the `dc_model_path` config requirement — Dixon-Coles trains inline. Keep `dc_model_path` as an optional override for backward compatibility.

---

#### [MODIFY] [scripts/train_dixon_coles.py](file:///home/arch/Coding/algobet/scripts/train_dixon_coles.py)

Update to use real features from the FeaturePipeline instead of dummy features:
- Load matches, build features via `FeaturePipeline`, train DC with `fit_with_scores()`
- This makes the standalone script produce a quality match-specific DC model

---

### Limitation 3: Draw Recall — Stacking Ensemble

The current `DrawAwareCalibrator` uses a simple linear blend `(1-α)*XGB + α*DC` which is too constrained. We need a meta-learner that can learn nonlinear combinations.

**Approach**: Implement a **stacking meta-learner** that feeds base model outputs through a lightweight calibrated classifier.

---

#### [NEW] [stacking.py](file:///home/arch/Coding/algobet/algobet/predictions/training/stacking.py)

A `StackingEnsemble` class:

```
Base models:     XGBoost → 3 probas
                 LightGBM → 3 probas (optional)
                 DixonColes → 3 probas

Meta-learner:    LogisticRegression(C=1.0, multi_class='multinomial')
                 Input: concatenated base probabilities (6–9 features)
                 Output: calibrated 3-class probabilities
```

Key design:
- Base models train on `X_train`, predict on `X_val`
- Meta-learner trains on `X_val` base predictions, tested on `X_test`
- Uses isotonic calibration on meta-learner output
- Collapse guards: reject if <3 classes predicted
- `fit(X_train, y_train, X_val, y_val)` and `predict_proba(X)` interface

This is more expressive than the current `EnsemblePredictor` (which only does weighted averaging) and the `DrawAwareCalibrator` (which only does linear blend).

---

#### [MODIFY] [config.py](file:///home/arch/Coding/algobet/algobet/predictions/training/config.py)

Add config fields:
- `use_stacking_ensemble: bool = False` — enable stacking meta-learner
- `stacking_base_models: list[str] = ["xgboost", "dixon_coles"]` — which base models to stack

---

#### [MODIFY] [runner.py](file:///home/arch/Coding/algobet/algobet/predictions/training/runner.py)

When `use_stacking_ensemble` is enabled:
1. Train base models (XGBoost, optionally LightGBM, Dixon-Coles) on `X_train`
2. Collect base predictions on `X_val`
3. Train meta-learner on val base predictions
4. Evaluate on `X_test` using full stack
5. Save all base models + meta-learner

---

### Draw-Focused Tuning Objective

#### [MODIFY] [tuner.py](file:///home/arch/Coding/algobet/algobet/predictions/training/tuner.py)

Add draw recall into the CV guard objective:
- After computing `base_ll`, also compute draw recall on the fold
- Add a penalty term `max(0, 0.10 - draw_recall) * 2.0` — penalize configs that predict zero draws
- This steers hyperparameter search away from draw-collapsing configurations

---

## Open Questions

1. **Stacking vs. weighted average**: Should the stacking ensemble *replace* or *coexist alongside* the current `EnsemblePredictor`? I recommend coexistence — `use_ensemble` remains for weighted averaging, `use_stacking_ensemble` enables the meta-learner approach.

2. **Inline Dixon-Coles training time**: Fitting `HistGradientBoostingRegressor` on ~200 features × ~2000 rows takes ~5–10 seconds. Is this acceptable added time per training run?

3. **Feature schema version bump**: Adding `draw_signals` features changes the feature count. Should we bump `MODEL_FEATURE_SCHEMA_VERSION` to `v3.2_draw_signals`? Old models won't be compatible with the new schema.

---

## Verification Plan

### Automated Tests

```bash
# Run existing test suite to ensure no regressions
cd /home/arch/Coding/algobet
source .venv/bin/activate

# Type checking
cd frontend && pnpm typecheck && cd ..

# Backend linting
ruff check algobet/

# Backend type checking
mypy algobet/predictions/training/stacking.py algobet/predictions/features/draw_signal_generator.py

# Run the draw evaluation script against a model trained with new features
python scripts/evaluate_draw_calibration.py \
  --model-version <new_model> \
  --tournament-id 359
```

### Integration Testing

1. Train a model with `draw_signals` feature group enabled:
   ```bash
   curl -X POST http://localhost:8010/api/v1/ml/train \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "xgboost",
       "tournament_ids": [359],
       "feature_groups": ["team_form", "draw_signals"],
       "fit_draw_aware_calibrator": true,
       "outcome_balance": true,
       "tune_hyperparameters": true,
       "tuning_trials": 20,
       "calibrate_probabilities": true
     }'
   ```

2. Compare draw recall and F1 between old and new models via backtest
3. Verify stacking ensemble trains without collapse
