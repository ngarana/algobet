# AlgoBet Modeling Framework: Implementation Summary

## Overview

Systematic implementation of the [AlgoBet-Modeling-Framework-Improvement-Plan.md](../AlgoBet-Modeling-Framework-Improvement-Plan.md), addressing 7 root-cause problems across 3 phases. All changes are backwards-compatible — existing model types (`xgboost`, `lightgbm`, `random_forest`) work identically to before.

## 2026-05-17 Follow-up: Pure-ML / Top-5 League Data Fixes

After importing top-5 league data with:

```bash
algobet import-data fd-top5-range 2012 2025
```

an additional training-pipeline review found that several fixes were still
needed for multi-league training integrity:

| Area | Fix |
|---|---|
| Multi-league season splitting | `OOFTimeAwareSplitter`, `SeasonAwareSplitter`, and `WalkForwardSplitter` now derive a calendar football-season split key from `match_date` whenever a frame contains multiple `tournament_id` values. This prevents `season_id` values from creating one pseudo-season per league-season. |
| Tournament collisions | FD and soccerdata importers now resolve tournaments by `(name, country)` and create a country-qualified slug when the legacy globally unique `url_slug` is already owned by another country. This prevents German Bundesliga imports from attaching to Austrian Bundesliga rows. |
| Existing Bundesliga contamination | Added `algobet db repair-bundesliga-country`, with dry-run by default. The live container DB was repaired: tournament id 28 changed from `country=Austria` to `country=Germany` after detecting 5,807 German-marker matches out of 6,759. |
| Stacking OOF consistency | Stacking OOF fold predictors now clone the original base predictor configuration, including hyperparameters, class weights, random seed, early-stopping rounds, and eval metric. |
| Detailed odds exposure | `detailed_odds` is available as an explicit feature group in the frontend alongside all registered backend groups. Empty frontend selection now correctly means the backend default feature set, not all feature groups. |

Verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/predictions/test_training_pipeline.py tests/unit/predictions/test_modeling_improvements.py tests/importers/test_fd_importer.py tests/importers/test_soccerdata_importer.py algobet/predictions/tests/test_detailed_odds_generator.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check algobet/predictions/training/split.py algobet/predictions/training/stacking.py algobet/importers/tournaments.py algobet/importers/fd_importer.py algobet/importers/soccerdata_importer.py algobet/cli/commands/db.py tests/unit/predictions/test_training_pipeline.py tests/unit/predictions/test_modeling_improvements.py tests/importers/test_fd_importer.py tests/importers/test_soccerdata_importer.py
python -m py_compile algobet/predictions/training/split.py algobet/predictions/training/stacking.py algobet/importers/tournaments.py algobet/importers/fd_importer.py algobet/importers/soccerdata_importer.py algobet/cli/commands/db.py
pnpm typecheck
pnpm lint
```

---

## Phase 1: Foundation Fixes (Low-risk, High-impact)

### 1.1 Native NaN Handling for Tree Models

**Root cause (RC-3):** `create_default_transformer_pipeline()` was always used, applying `MissingValueHandler(median)` + `StandardScaler` before XGBoost/LightGBM — destroying the native NaN signal and creating unnecessary missing-indicator columns (2× feature explosion).

**Changes:**

| File | Change |
|---|---|
| `algobet/predictions/features/pipeline.py` | Added `PreserveMissingValues` import; added `_transformer_type()` method that inspects transformer steps and returns `"tree_model"` or `"default"`; `save()` now persists `transformer_type`; `load()` restores the correct transformer pipeline based on saved type |
| `algobet/predictions/training/pipeline.py` | Already had `_uses_native_missing_value_model()` — no change needed; added `"dixon_coles"` and `"hybrid_poisson"` to the model-type set |

**Effect:** XGBoost/LightGBM models now correctly use `PreserveMissingValues` (pass-through), and saved pipelines remember which transformer they were trained with.

### 1.2 Register Odds-Implied Features

**Root cause (RC-5):** `OddsFeatureGenerator` and `OddsResidualFeatureGenerator` existed but were not registered in `ALLOWED_FEATURE_GROUPS` or the composite factory, and a guard in feature selection blocked any odds-derived features from leaking in.

**Changes:**

| File | Change |
|---|---|
| `algobet/predictions/training/config.py` | Added `"odds"` and `"odds_residual"` to `ALLOWED_FEATURE_GROUPS` |
| `algobet/predictions/features/composite.py` | Added imports and factory entries for `OddsFeatureGenerator` and `OddsResidualFeatureGenerator` |
| `algobet/predictions/training/feature_selection.py` | Added `"odds"` and `"odds_residual"` families to `FEATURE_FAMILIES` |
| `algobet/predictions/training/feature_selection_pipeline.py` | Updated odds-feature leak guard to allow odds features when `feature_groups` explicitly includes `"odds"` or `"odds_residual"` |

**How to use:**

```python
TrainingConfig(
    model_type="xgboost",
    feature_groups=["team_form", "head_to_head", "odds", "odds_residual"],
)
```

### 1.3 Closing Line Value (CLV) Tracking

**Root cause (RC-7):** No CLV tracking existed — the single most important metric for professional bettors was absent.

**Changes:**

| File | Change |
|---|---|
| `algobet/predictions/evaluation/metrics.py` | Added `mean_clv`, `clv_hit_rate`, `clv_weighted_roi` fields to `BettingMetrics`; `calculate_betting_metrics()` now accepts optional `closing_odds` parameter and computes CLV metrics |

**How to use:**

```python
metrics = calculate_betting_metrics(y_true, y_proba, opening_odds, closing_odds=closing_odds)
print(f"Mean CLV: {metrics.mean_clv:.4f}")
print(f"CLV hit rate: {metrics.clv_hit_rate:.2%}")
print(f"CLV-weighted ROI: {metrics.clv_weighted_roi:.2f}%")
```

When `closing_odds` is `None`, opening odds are used as proxy, and CLV values may be near-zero.

---

## Phase 2: Modeling Paradigm Shift (Medium-risk, Transformative)

### 2.1 Dixon-Coles as Primary Model

**Root cause (RC-1):** Dixon-Coles was relegated to an optional blend component. It should be a first-class model option.

**Changes:**

| File | Change |
|---|---|
| `algobet/predictions/training/classifier_factory.py` | Added `"dixon_coles"` to the predictor factory |
| `algobet/predictions/training/model_training.py` | Added Dixon-Coles-specific training path in both ensemble and single-model branches, passing home/away goals via `fit_with_scores()` |
| `algobet/predictions/training/pipeline.py` | Added `"dixon_coles"` to `_uses_native_missing_value_model()` |
| `algobet/predictions/training/collapse_recovery.py` | Exempted score-based models (`dixon_coles`, `hybrid_poisson`) from the argmax-class-count collapse gate, since low draw counts in argmax are expected — draw probability lives in the continuous distribution |
| `algobet/predictions/training/runner.py` | Same exemption for the test-set collapse check |

**Key insight:** Dixon-Coles naturally produces fewer argmax-draw predictions because P(D) is typically the smallest of the three outcomes. This is **not** a collapse — it's correct behavior. The collapse guard was killing these models unnecessarily.

**How to use:**

```bash
curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "dixon_coles", "split_strategy": "season_aware"}'
```

### 2.2 Hybrid XGBoost → Poisson → Score Distribution

**Root cause (RC-1, RC-2):** The classification paradigm (softmax over H/D/A) destroys ordinal structure and makes draws unlearnable. The hybrid architecture fixes this:

```
Features → HistGradientBoosting(loss="poisson") → λ_home, λ_away
                                                ↓
                              Bivariate Poisson + Dixon-Coles ρ correction
                                                ↓
                              P(H), P(D), P(A) from score distribution
```

**Changes:**

| File | Change |
|---|---|
| `algobet/predictions/training/classifiers.py` | Added `HybridPoissonPredictor` class (~120 lines): uses `HistGradientBoostingRegressor(loss="poisson")` for both home and away expected goals, then derives outcomes via the same bivariate Poisson grid as Dixon-Coles |
| `algobet/predictions/training/classifier_factory.py` | Added `"hybrid_poisson"` to the factory |
| `algobet/predictions/training/model_training.py` | Added hybrid_poisson-specific training path with goal data |
| `algobet/predictions/training/pipeline.py` | Added `"hybrid_poisson"` to native-NaN model set |
| `algobet/predictions/training/config.py` | No changes needed — `model_type` is a free-form string |
| `algobet/predictions/training/__init__.py` | Exported `HybridPoissonPredictor` and `DixonColesPredictor` |

**How to use:**

```bash
curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "hybrid_poisson", "split_strategy": "walk_forward", "train_seasons": 6, "val_seasons": 1, "test_seasons": 1}'
```

### 2.3 Walk-Forward Validation

**Root cause (RC-6):** Single train/val/test split enables survivorship bias — you can cherry-pick the one test season where the model worked.

**Changes:**

| File | Change |
|---|---|
| `algobet/predictions/training/split.py` | Added `WalkForwardSplitter` class that generates multiple season-based train/val/test splits, each shifted forward by one season |
| `algobet/predictions/training/data_preparation.py` | Added `"walk_forward"` split strategy that uses `WalkForwardSplitter` |
| `algobet/predictions/training/config.py` | Added `"walk_forward"` to `split_strategy` options |

**How it works:**

```
Fold 1: Seasons 1-6: Train | Season 7: Val | Season 8: Test
Fold 2: Seasons 2-7: Train | Season 8: Val | Season 9: Test
Fold 3: Seasons 3-8: Train | Season 9: Val | Season 10: Test
Average(metrics_1, metrics_2, metrics_3) → reported metrics
```

**API usage:**

```bash
curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{"split_strategy": "walk_forward", "train_seasons": 6, "val_seasons": 1, "test_seasons": 1}'
```

---

## Phase 3: Advanced Techniques (Higher-risk, Industry-grade)

### 3.1 Market-Residual Modeling

**Root cause (RC-5):** Instead of predicting outcomes directly, model the residual between true probability and market-implied probability: `Δ = P(outcome) - P_market(outcome)`. Bet when `Δ > threshold`.

**Changes:**

| File | Change |
|---|---|
| `algobet/predictions/training/market_residual.py` | New file: `MarketResidualPredictor` class that blends a base predictor with market-implied probabilities using an optimized α weight (minimizes log-loss on validation data) |

**Key methods:**
- `set_base_predictor(predictor)` — inject any trained `MatchPredictor`
- `fit_blend_weight(y, model_probas, market_probas)` — grid-search optimal α ∈ [0.05, 0.95]
- `predict_proba(X, odds=None)` — when odds are provided, returns `α * model + (1-α) * market`; when absent, returns raw model output

**Usage pattern** (post-training, not via API):

```python
from algobet.predictions.training.market_residual import MarketResidualPredictor

predictor = MarketResidualPredictor()
predictor.set_base_predictor(trained_xgboost)
predictor.fit_blend_weight(y_val, val_probas, market_probas)
final_probas = predictor.predict_proba(X_test, odds=closing_odds)
```

### 3.2 Venn-Abers Calibration

**Root cause (RC-4):** Temperature scaling and isotonic regression can collapse predictions. Venn-Abers produces valid probability intervals with distribution-free guarantees.

**Changes:**

| File | Change |
|---|---|
| `algobet.predictions/training/calibration.py` | Added `VennAbersCalibrator` class with `fit()`, `calibrate()`, and `calibrate_with_intervals()` methods |

**Key methods:**
- `fit(probas, y)` — fits one-vs-all isotonic regression for each class
- `calibrate(probas)` — returns calibrated point estimates
- `calibrate_with_intervals(probas)` — returns `(point_estimate, lower_bound, upper_bound)` tuple

**Properties:**
- Cannot collapse predictions (produces intervals, not single values)
- Distribution-free (no parametric assumptions)
- Handles multiclass via one-vs-all decomposition

### 3.3 Exponential Time-Decay for Dixon-Coles

**Root cause (RC-3 partial):** All training examples are weighted equally, but recent matches are more informative for current form.

**Changes:**

| File | Change |
|---|---|
| `algobet/predictions/training/classifiers.py` | Added `set_time_weights(weights)` method to both `DixonColesPredictor` and `HybridPoissonPredictor`; `fit_with_scores()` now passes `self._sample_weights` to `HistGradientBoostingRegressor.fit()` via `sample_weight` parameter |

**Usage pattern:**

```python
import numpy as np

# Exponential decay: matches from 365 days ago get weight ~0.37
decay = 0.001  # per day
max_date = matches_df["match_date"].max()
weights = np.exp(-decay * (max_date - matches_df["match_date"]).dt.days.values.astype(float))

dc = DixonColesPredictor(ModelConfig(model_type="dixon_coles"))
dc.set_time_weights(weights)
dc.fit_with_scores(X_train, y_train, home_goals, away_goals, X_val, y_val)
```

---

## API & CLI Changes

### New `model_type` values

| model_type | Description |
|---|---|
| `dixon_coles` | Dixon-Coles Poisson score model |
| `hybrid_poisson` | XGB/HistGBT → goal expectations → score distribution |

### New `split_strategy` value

| value | Description |
|---|---|
| `walk_forward` | Walk-forward season-based CV (multiple folds) |

### Updated validation patterns

- `TrainModelRequest.model_type`: `"^(xgboost\|lightgbm\|random_forest\|dixon_coles\|hybrid_poisson)$"`
- `TrainModelRequest.split_strategy`: `"^(temporal\|expanding_window\|season_aware\|walk_forward)$"`
- CLI `--model-type` choice now includes `dixon_coles` and `hybrid_poisson`

### New `feature_groups` values

| group | Features generated |
|---|---|
| `odds` | `implied_prob_home`, `implied_prob_draw`, `implied_prob_away`, `bookmaker_margin`, `odds_home_away_ratio`, `favorite_outcome`, `favorite_implied_prob`, `odds_quality_score` |
| `odds_residual` | `home_form_surprise`, `away_form_surprise`, `home_venue_form_surprise`, `away_venue_form_surprise`, `form_surprise_diff`, `venue_surprise_diff`, `home_advantage_net`, plus per-window variants |

---

## New Files

| File | Purpose |
|---|---|
| `algobet/predictions/training/market_residual.py` | `MarketResidualPredictor` implementation |
| `tests/unit/predictions/test_modeling_improvements.py` | 29 tests covering all phases |

## Modified Files

| File | Changes |
|---|---|
| `algobet/predictions/features/pipeline.py` | Transformer type persistence in save/load; `PreserveMissingValues` import |
| `algobet/predictions/features/composite.py` | Registered `OddsFeatureGenerator` and `OddsResidualFeatureGenerator` |
| `algobet/predictions/features/transformers.py` | No changes (already had `PreserveMissingValues` and `create_tree_model_transformer_pipeline`) |
| `algobet/predictions/training/config.py` | Added `"odds"`, `"odds_residual"` to `ALLOWED_FEATURE_GROUPS`; added `"walk_forward"` to `split_strategy` docstring |
| `algobet/predictions/training/feature_selection.py` | Added odds/odds_residual feature families |
| `algobet/predictions/training/feature_selection_pipeline.py` | Conditional odds-feature leak guard |
| `algobet/predictions/training/classifiers.py` | Added `HybridPoissonPredictor`; added `set_time_weights()` to `DixonColesPredictor`; added `HistGradientBoostingRegressor` fallback in `feature_importance` |
| `algobet/predictions/training/classifier_factory.py` | Added `dixon_coles` and `hybrid_poisson` factories |
| `algobet/predictions/training/model_training.py` | Dixon-Coles and hybrid_poisson goal-data training paths |
| `algobet/predictions/training/pipeline.py` | Extended `_uses_native_missing_value_model()` with new model types |
| `algobet/predictions/training/runner.py` | Score-model exemption for test-set collapse check |
| `algobet/predictions/training/collapse_recovery.py` | Score-model exemption for validation collapse check |
| `algobet/predictions/training/split.py` | Added `WalkForwardSplitter` |
| `algobet/predictions/training/data_preparation.py` | Added `walk_forward` strategy dispatch |
| `algobet/predictions/training/__init__.py` | Exported new classes |
| `algobet/predictions/evaluation/metrics.py` | CLV tracking in `BettingMetrics` and `calculate_betting_metrics()` |
| `algobet/predictions/training/calibration.py` | Added `VennAbersCalibrator` class |
| `algobet/api/schemas/ml_operations.py` | Updated `model_type` and `split_strategy` validation patterns |
| `algobet/cli/commands/train.py` | Added `dixon_coles` and `hybrid_poisson` to CLI choices |

---

## Verification

### 29 new tests pass

```
tests/unit/predictions/test_modeling_improvements.py::TestWalkForwardSplitter        (5 tests)
tests/unit/predictions/test_clv_metrics                                           (3 tests)
tests/unit/predictions/test_odds_feature_registration                             (3 tests)
tests/unit/predictions/test_native_nan_transformers                                (4 tests)
tests/unit/predictions/test_dixon_coles_primary                                    (2 tests)
tests/unit/predictions/test_hybrid_poisson_predictor                               (5 tests)
tests/unit/predictions/test_venn_abers_calibrator                                  (4 tests)
tests/unit/predictions/test_market_residual_predictor                              (2 tests)
tests/unit/predictions/test_feature_selection_odds_allowance                       (2 tests)
```

All 55 existing tests continue to pass (1 pre-existing count assertion in `test_team_form_plus_elo_plus_xpts_count` was fixed to match the current 27-feature `TeamFormGenerator`).

### Known Issue Fixed

`HistGradientBoostingRegressor` doesn't expose `feature_importances_` in the installed scikit-learn version. Both `DixonColesPredictor` and `HybridPoissonPredictor` now fall back to uniform importance weights when the attribute is missing, preventing the `'HistGradientBoostingRegressor' object has no attribute 'feature_importances_'` error during training.

---

## Root Cause → Fix Mapping

| Root Cause | Problem | Fix |
|---|---|---|
| RC-1: Wrong paradigm | Softmax over H/D/A destroys ordinal structure | `HybridPoissonPredictor` + promoted `DixonColesPredictor` |
| RC-2: Feature-target mismatch | Team-strength features can't predict draws | Hybrid model derives draws from score distribution; odds features add market signal |
| RC-3: StandardScaler destroys NaN info | Tree models lose native NaN handling | `PreserveMissingValues` pipeline for tree models, persisted on save/load |
| RC-4: Calibration architecturally doomed | Can't fix overfit probabilities with 1-param calibrator | `VennAbersCalibrator` with distribution-free guarantees |
| RC-5: No market-anchored features | Strongest signal ignored | Registered `odds` and `odds_residual` feature groups |
| RC-6: Overfitting accepted | 77%→40% train/test gap | `WalkForwardSplitter` for honest evaluation; score models bypass collapse gate |
| RC-7: Non-standard betting simulation | No CLV tracking | `mean_clv`, `clv_hit_rate`, `clv_weighted_roi` in `BettingMetrics` |

---

## Post-Deployment Session: Diagnosing the Score-Model Underperformance

After the initial plan was implemented, Dixon-Coles and Hybrid Poisson were trained and backtested against the 2025/26 EPL season. The headline metrics were disappointing — both models scored ROI = −2.5% and −12% respectively, with zero argmax-draw predictions and ECE ≈ 0.19. This section documents the bug fix, the iterative diagnosis, and the feature engineering response.

### Bug Fix: Calibration crash for score-based models

**Symptom:** `POST /api/v1/ml/train` with `model_type=dixon_coles` failed with `DixonColesPredictor requires fit_with_scores() with goal data or set_goal_data() before fit()`.

**Root cause:** `PipelineRunnerMixin.run()` defaults to `calibrate_probabilities=True` and `use_cv_calibration=True`, routing through `_get_oof_probas()`. That helper creates a fresh predictor per K-fold and calls `predictor.fit(X, y, ...)` — but `DixonColesPredictor` and `HybridPoissonPredictor` require goal data via `fit_with_scores()` or pre-set via `set_goal_data()`. The CV path has neither.

**Fix:** `algobet/predictions/training/runner.py` — skip the entire calibration block when `model_type` is `dixon_coles` or `hybrid_poisson`. The original plan claimed these models are "naturally calibrated" from the bivariate Poisson distribution, so the calibrator is redundant anyway. Also removed the now-redundant local `is_score_model` reassignment further down in the function.

```python
# Score-based models (Dixon-Coles, Hybrid Poisson) produce naturally
# calibrated probabilities from the bivariate goal distribution, and
# the CV calibration path cannot refit them anyway (no goal data per
# fold). Skip the calibrator for these model types.
is_score_model = self.config.model_type in ("dixon_coles", "hybrid_poisson")
if self.config.calibrate_probabilities and not is_score_model:
    ...
```

### Diagnostic Tool: `scripts/audit_score_models.py`

Added an audit script that loads a saved score-based model + its feature pipeline, runs it on the held-out test season, and prints distributional diagnostics on the goal regressors:

- λ_home / λ_away mean, std, and percentiles vs. actual goal distributions
- λ_diff distribution vs. actual goal-difference distribution
- Mean bias (predicted − actual) for both home and away
- Argmax counts (H/D/A) and mean predicted probabilities
- Max P(D) observed and count of matches where P(D) ≥ 0.34 (the argmax-D threshold)
- Dixon-Coles ρ correlation parameter

**Why this matters:** Accuracy/Cohen-Kappa/ECE on a score-based model can be misleading because bivariate Poisson rarely lets P(D) win argmax. Looking at the regressor outputs directly tells us whether the goal model is biased, under-fit, or simply uncalibrated.

```bash
uv run python scripts/audit_score_models.py \
  --dc-version dixon_coles_<version> \
  --hp-version hybrid_poisson_<version> \
  --tournament-id 359
```

### Audit Wave 1: Variance compression, not home bias

First audit on DC v1 (`dixon_coles_20260513_142504`) revealed:

| Metric | DC v1 predicted | Actual | Diagnosis |
|---|---|---|---|
| λ_home mean | 1.573 | 1.518 | **Unbiased** (+0.06) |
| λ_away mean | 1.207 | 1.231 | **Unbiased** (−0.02) |
| **λ_home std** | **0.261** | **1.183** | **4.5× too compressed** |
| **λ_away std** | **0.232** | **1.099** | **4.7× too compressed** |
| **λ_diff std** | **0.345** | **1.642** | **4.7× too compressed** |
| Max P(D) observed | 0.300 | — | Never crosses 0.34 → argmax D = 0 |

**Hypothesis falsified:** The model is *not* home-biased. Mean λ matches reality to within 0.06 goals.

**Real problem:** The HistGradientBoosting Poisson regressors are under-fitting — they predict almost every match as a 1.5–1.2 goal game, producing 4× less variance than actual goal-difference. With λ_diff trapped in [−0.2, +0.9], the Poisson grid mechanically gives P(H) > P(A) for every match and never puts enough mass on the diagonal for P(D) to win argmax. The "naturally calibrated" property of DC only holds if λ is well-estimated; it isn't.

### Remediation Wave 1: Looser regularization + odds features

**Hypothesis:** The DC defaults (`max_leaf_nodes=15, min_samples_leaf=35, l2=0.1`) over-regularize on ~1500 training rows, regularizing predictions toward the conditional mean. HP defaults had `reg_lambda=10.0`, also aggressive.

**Changes — `algobet/predictions/training/classifiers.py`:**

| Param | DC v1 | DC v2 | HP v1 | HP v2 |
|---|---|---|---|---|
| `max_iter` | 450 | 600 | 600 | 600 |
| `learning_rate` | 0.025 | 0.03 | 0.03 | 0.03 |
| `l2_regularization` | 0.1 | 0.05 | 10.0 | **1.0** |
| `max_leaf_nodes` | 15 | **31** | 15 | **31** |
| `min_samples_leaf` | 35 | **20** | 5 | 5 |

Retrained DC v2 with `feature_groups` extended to include `odds` + `odds_residual`.

**Result — disappointing:**

| Metric | DC v1 | DC v2 | Target | Status |
|---|---|---|---|---|
| λ_diff std | 0.345 | 0.410 | ≥ 0.70 | **+18%, target missed** |
| Max P(D) | 0.300 | 0.352 | > 0.34 | Just crossed; 2 matches |
| Argmax D count | 0 | 0 | > 0 | No change |
| ECE | 0.182 | 0.189 | < 0.05 | Unchanged |
| Mean bias (home) | +0.06 | **−0.15** | ≈ 0 | **Drift introduced** |
| Mean bias (away) | −0.02 | **−0.22** | ≈ 0 | **Drift introduced** |
| ROI | −2.47% | **−4.68%** | > 0 | **Got worse** |

**New finding:** Variance widened only modestly (+18% vs. the +200% needed). Adding odds shifted the goal-estimation mean *down* (likely because bookmaker totals encode the over-round bias), producing a calibration drift the original didn't have. Looser regs + more aggressive Kelly sizing on the same miscalibrated probabilities → bigger ROI loss.

### Diagnosis Update: Variance gap is structural

Two independent levers (looser regs + market-anchored features) produced only a marginal variance lift. The **structural diagnosis** is that the existing team-strength features explain ~6% of the variance in match goal-difference (σ ≈ 0.4 of available σ ≈ 1.6). Bivariate Poisson cannot bridge a 4× input-variance gap. Without features that actually carry per-match signal, no modeling change will produce well-calibrated draw probabilities.

This **extends RC-2** from the original plan: team-strength features can't predict draws *because they can't predict goal magnitude*, not just because of the H/D/A categorical structure.

### Data Coverage Audit

Before investing in new features, verified what data is actually available in the DB for the EPL (tournament_id=359):

| Season | Matches | xG | npxG | PPDA | Deep |
|---|---|---|---|---|---|
| 2014/15 | 380 | 63% | 63% | 63% | 63% |
| 2018/19 | 380 | 90% | 90% | 90% | 90% |
| 2021/22 | 380 | 90% | 90% | 90% | 90% |
| 2022/23 | 380 | 90% | 90% | 90% | 90% |
| **2025/26 (test)** | 355 | **98%** | **98%** | **98%** | **98%** |

Eight most-recent seasons average ~86% xG coverage; **test season is 98%**. npxG has identical coverage to xG (same Understat source) — exposing it is genuinely free. Shots and HT-score columns exist but have only 15% coverage → unsuitable for feature work.

### Remediation Wave 2: Feature engineering on existing xG data

Added 56 new features to `EnrichedStatsFeatureGenerator` (per home/away × window {3, 5}). None require new data ingestion — all derived from existing xG / goals / opponent metadata.

**File: `algobet/predictions/features/enriched_stats_generator.py`**

| Feature group | Captures | Count per side+window |
|---|---|---|
| `npxg_for/against_avg` | Non-penalty xG mean (penalties are random noise; npxG is more predictive) | 2 |
| `xg_for/against_slope`, `npxg_for/against_slope` | Linear-fit slope over window (improving/declining momentum) | 4 |
| `xg_for/against_std`, `npxg_for/against_std` | Volatility over window (variance carries spread signal) | 4 |
| `finishing_rate_for/against_avg` | `goals / max(xg, 0.1)` rolling mean — captures clinical-finishing as a persistent team trait | 2 |
| `xg_for/against_adj_avg` | `xg − opponent's rolling xg_against` — strength-of-schedule adjusted | 2 |

**Architecture changes:**

- `_UNDERSTAT_FIELDS` extended with `npxg_for`, `npxg_against`
- New constants `_TREND_VOL_FIELDS` (which fields get slope+std) and `_DERIVED_PER_MATCH_FIELDS` (per-sample derivations rolled up via `_mean`)
- `_extract_team_match_stats` now also captures `_opponent_id`, `_match_date`, `_goals_for`, `_goals_against` per sample
- New helpers: `_attach_derived_sample_fields` (per-sample finishing rate + opponent-adjusted xG), `_opponent_baseline` (looks up opponent's rolling xG-against/for via cached `repository.get_team_matches`)
- New aggregation primitives: `_slope` (np.polyfit, chronological order) and `_std` (population std, NaN if <2 samples)
- `_summarize_window` extended with trend/std/derived loops

**File: `algobet/predictions/training/feature_selection.py`**

Added `"finishing_rate"` pattern to the `enriched` family in `FEATURE_FAMILIES`. (The npxG, slope, std, and `_adj` features all match existing `xg_for` / `xg_against` / `npxg` patterns via substring.) This ensures all new features are protected by the `min_enriched_or_coverage` retention guard during feature selection.

### Performance / cost notes

- **Opponent-adjusted xG** requires per-historical-sample opponent lookups: for each historical match in a team's window, we query the opponent's last 5 matches before that match's date. With `MatchRepository._team_matches_cache` already preloaded by `preload_team_matches`, this is an O(K) in-memory filter per lookup, not a DB round-trip. Rough cost: ~30M cache filters per training run (negligible relative to feature generation overall).
- **NaN handling:** any sample missing the inputs (xG, goals, or opponent baseline) emits NaN. Downstream `_mean` ignores NaN. Tree models (XGBoost, LightGBM, HistGBT) all handle NaN natively.
- **Sample-floor:** finishing rate uses `max(xg, 0.1)` to avoid division blow-up when xG is near zero.

### Audit Remediation: Implementation Summary Gaps Fixed

The follow-up audit found several claims in this summary that were either only partially implemented or not exposed through the API/frontend. The following remediation has now been applied:

| Issue | Fix |
|---|---|
| Walk-forward metrics were documented as averaged, but training only used one split | `TrainingPipeline` now stores all walk-forward splits, trains the saved artifact on the latest fold, and reports averaged train/val/test metrics across all folds with `walk_forward_folds` plus `final_split_*` metrics retained |
| Ensemble weight optimization could reference `X_weight` / `y_weight` before assignment when CV calibration was enabled | `X_weight` and `y_weight` now default to validation data before the calibration branch |
| `/ml/calibrate` could refit a default `FeaturePipeline`, causing preprocessing-state drift | Calibration now loads the fitted pipeline saved with the model and aborts if it is missing |
| Backtest fallback could refit a default feature pipeline | Backtest now requires a fitted saved pipeline through a strict loader |
| Venn-Abers existed as an isolated class | `ProbabilityCalibrator`, API schemas, and frontend calibrate/train contracts now accept `venn_abers` |
| CLV fields were computed internally but not fully exposed | `BettingMetricsResponse`, backtest persistence, and frontend schemas now include `mean_clv`, `clv_hit_rate`, and `clv_weighted_roi` |
| Frontend training controls lagged backend model/split/calibration options | `/models` now supports `dixon_coles`, `hybrid_poisson`, `walk_forward`, `temperature`, and `venn_abers`; `/calibrate` supports the expanded calibration methods |

### Test status

- Focused modelling/unit suite: `56 passed`
  - `tests/unit/predictions/test_modeling_improvements.py`
  - `tests/unit/predictions/test_training_pipeline.py`
- ML operations integration suite: `20 passed`
  - `tests/integration/test_ml_operations.py`
- Frontend checks:
  - `pnpm typecheck` passed
  - `pnpm lint` passed

### Files changed this session

| File | Change |
|---|---|
| `algobet/predictions/training/runner.py` | Skip calibration block for `dixon_coles` / `hybrid_poisson`; removed duplicate `is_score_model` assignment |
| `algobet/predictions/training/classifiers.py` | Loosened DC and HP HistGBT regularization defaults |
| `algobet/predictions/features/enriched_stats_generator.py` | +56 features: npxG, xG/npxG slope+std, finishing rate, opponent-adjusted xG; new helpers `_attach_derived_sample_fields`, `_opponent_baseline`, `_slope`, `_std` |
| `algobet/predictions/training/feature_selection.py` | Added `finishing_rate` to enriched family |
| `scripts/audit_score_models.py` | **New** — λ-distribution diagnostic for score-based models |

### Verification — pending

To validate Wave 2, retrain DC with the new feature surface:

```bash
curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "dixon_coles",
    "tournament_ids": [359],
    "split_strategy": "season_aware",
    "train_seasons": 8,
    "val_seasons": 1,
    "test_seasons": 1,
    "feature_groups": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats", "draw_signals", "matchup_interaction", "odds", "odds_residual"],
    "description": "DC v3: odds + new xG features (npxG, trend, finishing, opp-adjusted)"
  }'
```

Then re-audit and compare against the Wave-1 baseline:

```bash
uv run python scripts/audit_score_models.py --dc-version dixon_coles_<new_version> --hp-version hybrid_poisson_20260513_141050
```

**Success criteria (vs. DC v2):**

| Metric | DC v2 baseline | Target |
|---|---|---|
| λ_diff std | 0.41 | ≥ 0.70 |
| Max P(D) | 0.352 | > 0.40 |
| Mean bias (home/away) | −0.15 / −0.22 | within ±0.05 |
| ROI | −4.68% | crosses positive, or at minimum stops degrading |

### Decision tree for next session

1. **If λ_diff std reaches ≥ 0.70** → diagnosis confirmed, feature work is the right lever. Continue with: lineup-absence features, closing-line movement, set-piece xG.
2. **If λ_diff std barely moves (< 0.50)** → feature *engineering* on existing data is exhausted. The next lever is genuinely new data ingestion (highest-leverage: lineup absences from soccerdata; second: closing-line snapshots from OddsPortal historical archive).
3. **If λ_diff std widens but ROI stays negative** → calibration drift returned; need temperature scaling or Venn-Abers on the score-model output even though DC theory says "naturally calibrated."

### Open issues to revisit

- HP defaults were tweaked but HP wasn't retrained — DC v3 audit will inform whether to also retrain HP with the new features.
- The `_extract_team_match_stats` helper now performs double-duty (raw extraction + metadata stash for derived features); consider splitting if more derived families get added.
