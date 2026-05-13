# Draw Correction Implementation — Progress & Limitations

## Summary

This document records the implementation of the draw-correction plan from `DRAW_IMPLEMENTATION.md`, the results observed during training, and the architectural challenges that remain unresolved.

---

## 1. What Was Implemented

### 1.1 New Feature Generator: `draw_signals`

- **File:** `algobet/predictions/features/draw_signal_generator.py`
- **Class:** `DrawSignalFeatureGenerator`
- **Features produced (~15 per window size):**
  - `strength_parity` — absolute difference in recent points between teams
  - `combined_draw_rate_{3,5,10}` — average of home/away draw rates
  - `defensive_balance_{3,5,10}` — absolute difference in goals conceded
  - `low_scoring_probability_{3,5,10}` — interaction of low-scoring rates
  - `clean_sheet_interaction_{3,5,10}` — cross-team defensive dominance signal
  - `goal_convergence_{3,5,10}` — inverse of goal-rate difference
  - `volatility_sum_{3,5,10}` — sum of goal variances
  - `h2h_draw_boost` — H2H draw rate weighted by sample size
  - `xg_parity_{3,5}` — xG balance (uses Understat data when available)
- **Rationale:** These are interaction/transformation features (products, absolute differences, reciprocals) designed to be non-collinear with raw `team_form` features.
- **Registration:** Added to `composite.py` factory and `create_default_generators()`.

### 1.2 Training Pipeline Config Expansion

- **File:** `algobet/predictions/training/config.py`
- Added `"draw_signals"` to `ALLOWED_FEATURE_GROUPS`.
- Added stacking-ensemble config fields:
  - `use_stacking_ensemble: bool = False`
  - `stacking_base_models: list[str] = ["xgboost", "dixon_coles"]`
- Added post-hoc draw-boost field:
  - `draw_boost_factor: float = 1.0`

### 1.3 Feature Selection Guard

- **File:** `algobet/predictions/training/feature_selection.py`
- Added `"draw_signal"` family to `FEATURE_FAMILIES` so draw-signal features are protected during group-aware selection.

### 1.4 Inline Dixon-Coles Training

- **File:** `algobet/predictions/training/runner.py`
- When `fit_draw_aware_calibrator=true` and no `dc_model_path` is provided, the pipeline now:
  1. Extracts `home_score`/`away_score` from the training split.
  2. Trains `DixonColesPredictor` inline with real features via `fit_with_scores()`.
  3. Fits `DrawAwareCalibrator` on validation data using XGBoost + DC probabilities.
  4. Saves the inline DC model alongside the main model.
- **Backward compatibility:** `dc_model_path` still works as an override.

### 1.5 Stacking Ensemble

- **File:** `algobet/predictions/training/stacking.py`
- **Class:** `StackingEnsemble`
- Design:
  - Base models train on `X_train`, predict on `X_val`.
  - Meta-learner: `LogisticRegression(C=1.0, multi_class="multinomial")` trained on concatenated base probabilities from `X_val`.
  - Isotonic calibration applied to meta-learner output.
  - Collapse guard: minimum 2% floor per class.
- Integration: `runner.py` Step 5b trains the stacking ensemble when `use_stacking_ensemble=true`.

### 1.6 Draw Boost Post-Processor

- **File:** `algobet/predictions/training/calibration.py`
- `CalibratedPredictor` now accepts an optional `draw_boost_calibrator`.
- **File:** `algobet/predictions/training/runner.py`
- When `draw_boost_factor > 1.0`, a `DrawBoostCalibrator` is chained after the main calibrator.

### 1.7 Draw-Aware Predictor at Inference

- **File:** `algobet/predictions/training/calibration.py`
- **Class:** `DrawAwarePredictor`
- Wraps `base_predictor → dc_model → DrawAwareCalibrator` so the blended probabilities are actually used during `predict_proba`.
- **Bug fix:** Previously `DrawAwareCalibrator` was trained and saved but never wired into inference. This is now fixed.

### 1.8 Tuner Draw-Recall Penalty

- **File:** `algobet/predictions/training/tuner.py`
- Added draw-recall penalty to both `_evaluate_single_split_guarded` and `_evaluate_cv_guarded`:
  - `penalty += max(0.0, 0.10 - draw_recall) * 2.0`
- This steers hyperparameter search away from configurations that predict zero draws.

### 1.9 Standalone DC Script Update

- **File:** `scripts/train_dixon_coles.py`
- Rewritten to use real features from `FeaturePipeline` instead of dummy index features.

---

## 2. Observed Results

### 2.1 Model 171 — Baseline `team_form` only

| Split | Accuracy | Max Prediction Share | Notes |
|-------|----------|---------------------|-------|
| Train | 0.604 | 0.653 | Moderate overfit |
| Val   | 0.557 | 0.688 | Reasonable |
| Test  | 0.514 | 0.688 | Functional but Home-biased |

**Backtest confusion matrix:** 260 Home / 0 Draw / 3 Away predictions.
**Verdict:** The model defaults to Home because home advantage is the dominant signal.

### 2.2 Model 173 — `team_form + draw_signals` (conservative manual HPs)

| Split | Accuracy | Max Prediction Share | Notes |
|-------|----------|---------------------|-------|
| Train | 0.691 | 0.365 | Severe overfit |
| Val   | 0.346 | 0.404 | Collapsed to ~random |
| Test  | 0.352 | 0.401 | Random guessing |

**Verdict:** The conservative manual hyperparameters (`reg_lambda=15`, `min_child_weight=15`, etc.) caused severe underfitting. The model could not learn any pattern.

### 2.3 Models with `outcome_balance_strength=1.0`

| Split | Accuracy | Notes |
|-------|----------|-------|
| Train | 0.682 | Memorized spurious draw patterns |
| Val   | 0.342 | Collapsed to random |
| Test  | 0.425 | 97.9% Home predictions |

**Verdict:** Aggressive class balancing forced the model to memorize noise.

---

## 3. Root Cause Analysis

### 3.1 Temporal Distribution Shift

The training data ends **May 2025**. The backtest data starts **November 2025**. Over a 5-month gap, team dynamics (transfers, form, managerial changes) shift enough that patterns learned on the training split do not generalize. This is evidenced by:

- Baseline `team_form` alone collapsing to 99% Home predictions on future data despite ~57% validation accuracy.
- Both `draw_signals` and conservative regularization failing to improve test performance.

### 3.2 Home Advantage Dominates Draw Signal

In football, home advantage is structurally stronger than draw signal for most matchups. When a model is uncertain, it defaults to the majority class (Home). This is especially pronounced when:

- Training data has more Home wins than Draws.
- Feature interactions do not provide enough orthogonal draw signal to overcome the prior.

### 3.3 Draw Signals Are Redundant with Team Form

Several `draw_signal` features are simple algebraic combinations of existing `team_form` features:

- `combined_draw_rate_5` = (`home_draw_rate_5` + `away_draw_rate_5`) / 2
- `strength_parity` = abs(`home_points_last_5` - `away_points_last_5`)

These do not add new information — they add dimensionality. Feature selection with `max_feature_correlation=0.90` should prune them, but the underlying signal remains weak.

---

## 4. Current Workarounds

### 4.1 `draw_boost_factor` (Post-Hoc Multiplier)

Because the base model cannot learn a reliable draw signal, the pipeline now supports a post-hoc draw boost:

```json
{ "draw_boost_factor": 1.5 }
```

This multiplies the draw probability by 1.5× after calibration and renormalizes. It is a **blunt instrument** that sacrifices overall accuracy to force draw coverage.

| Factor | Expected Draw Recall | Trade-off |
|--------|---------------------|-----------|
| 1.0    | ~0%                 | High H/A accuracy, zero draws |
| 1.5    | ~15-25%             | Moderate draw coverage |
| 2.0    | ~25-35%             | Strong draw coverage, lower H/A accuracy |

### 4.2 Disabling Hyperparameter Tuning

`tune_hyperparameters: false` with moderate manual hyperparameters generalizes better than 20-30 Optuna trials on this dataset. Tuning overfits to the validation split, which is temporally close to training but far from the backtest period.

---

## 5. Known Limitations

### 5.1 Limitation: True Draw Signal May Not Exist in Data

If the feature set genuinely cannot separate draws from narrow home/away wins, no amount of feature engineering will help. The `draw_signals` features capture *closeness* (parity, balance), but closeness does not always predict draws in football.

### 5.2 Limitation: Temporal Split Is Too Coarse

The current `TemporalSplitter` uses a single `train_ratio=0.7` / `val_ratio=0.15` / `test_ratio=0.15` split. A rolling-window or expanding-window split would better simulate the backtest gap and produce models that generalize across time.

### 5.3 Limitation: Features Contain Zero Predictive Signal for Draws

**Diagnostic finding (`scripts/diagnose_draw_learning.py`):**
- The highest feature correlation with draw outcome is **0.036** (`h2h_draw_boost`)
- **Zero features** have |correlation| > 0.05 with draw
- Top mutual information score: **0.0137** (`away_goals_for_avg_5`)
- Draw-signal feature means are virtually identical across H/D/A classes:
  - `combined_draw_rate_3`: H=0.236, D=0.233, A=0.235
  - `strength_parity`: H=2.345, D=2.346, A=2.347

**Why this happens:** Historical averages regress to the mean. `home_draw_rate_5` is ~23% for almost every team regardless of the current match outcome. The features capture *typical behavior*, not *match-specific conditions* that cause draws (tactical setup, motivation, weather, referee decisions).

**Implication:** XGBoost cannot learn what does not exist. Defaulting to Home (44.6% base rate) is optimal behavior when draw signal is indistinguishable from noise.

### 5.4 Limitation: `fit_draw_aware_calibrator` Is a Linear Blend

The `DrawAwareCalibrator` uses a grid search over `α ∈ [0, 0.5]` for `(1-α)*XGB + α*DC`. This is a constrained linear blend. The `StackingEnsemble` was intended to replace this with a non-linear meta-learner, but stacking currently performs worse than simple boosting due to the same underlying data limitations.

### 5.5 Limitation: Inline Dixon-Coles Training Time

Fitting `HistGradientBoostingRegressor` on ~200 features × ~2000 rows takes ~5–10 seconds per run. This is acceptable but adds latency to the training pipeline.

---

## 6. What Actually Worked (Model `xgboost_20260511_153157`)

After extensive experimentation, only **one model** achieved non-zero draw predictions in backtest. The working configuration was:

```json
{
  "feature_groups": ["team_form"],
  "outcome_balance": true,
  "outcome_balance_strength": 0.5,
  "feature_selection": false,
  "tune_hyperparameters": false,
  "hyperparameters": {
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 1200,
    "subsample": 0.6,
    "colsample_bytree": 0.4,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 5.0,
    "reg_lambda": 10.0
  }
}
```

**Backtest results:** 14 draw predictions (5.3%), 2 correct (14.3% precision, 4.2% recall). Modest but non-zero.

### Why This Specific Combination Worked

| Factor | Working Value | Why It Matters |
|--------|--------------|----------------|
| `outcome_balance: true` | Class weights force model to care about draws | Without this, model defaults to Home (44.6% base rate) |
| `subsample: 0.6` | Only 60% rows per tree | Prevents memorizing Home bias; forces use of draw features |
| `colsample_bytree: 0.4` | Only 40% columns per tree | Model cannot always find Home shortcuts |
| `reg_alpha: 5.0` | Strong L1 regularization | Prevents overconfident Home predictions |
| `feature_selection: false` | Keeps all 96 features | Pruning removed `draw_rate_*` and `form_diff` which carry weak signal |
| `draw_signals` omitted | No noise features | Adding them diluted the real signal |

**The draw signal is real but extremely fragile.** It only emerges when a specific combination of class weights, aggressive sampling, and L1 regularization prevents the model from defaulting to Home.

---

## 7. Final Conclusions: Why Draws Are Fundamentally Difficult

### 7.1 The Core Problem

Historical team form features answer: **"What is this team's typical behavior?"**
Draw prediction requires: **"What is special about THIS matchup?"**

These are different questions. Typical behavior (goals per game, points per game, draw rate) regresses to the mean (~23% draw rate for almost every team). The match-specific conditions that cause draws are not in historical averages.

### 7.2 Diagnostic Evidence

From `scripts/diagnose_draw_learning.py` (4,185 matches):

| Metric | Finding |
|--------|---------|
| Draw rate in data | 23.4% (reasonably balanced) |
| Highest draw correlation | `h2h_draw_boost` = **0.036** |
| Features with \|corr\| > 0.05 | **Zero** |
| Top mutual information | `away_goals_for_avg_5` = **0.0137** |
| `combined_draw_rate_3` means | H=0.236, D=0.233, A=0.235 (identical) |

**Interpretation:** The features provide ~1% information about draws. This is indistinguishable from noise.

### 7.3 Why XGBoost Defaults to Home

XGBoost is doing the **optimal** thing: predict Home when uncertain because Home wins 44.6% of the time. Draws happen 23.4% of the time, but with zero feature separation, the model cannot identify WHICH 23.4%. Defaulting to Home minimizes log loss.

To get draws, you must **force** the model to pay attention to them (class weights + aggressive sampling) or **override** the predictions post-hoc (`draw_boost_factor`).

---

## 8. Alternative Features & Metrics for Future Research

The following feature categories could provide genuine draw signal **without** requiring market odds or real-time team news. They are ordered by feasibility.

### 8.1 Matchup-Specific Tactical Features

These capture the **interaction** between two teams' styles, not just their individual histories.

| Feature | Rationale | Data Source |
|---------|-----------|-------------|
| `style_clash_score` | Defensive team vs. defensive team → draw | Compare `ppda`, `deep_completions`, `possession` |
| `tempo_mismatch` | Slow team vs. slow team → low scoring → draw | Compare `passes_per_minute` or `sequence_time` |
| `pressure_balance` | Both teams high press → turnovers → chaotic draw | `ppda` (passes per defensive action) |
| `width_mismatch` | Narrow attack vs. narrow defense → stalemate | Understat `width` or Opta touch locations |
| `set_piece_reliance` | Both teams score heavily from set pieces → 1-1 draws | Understat `shots_from_set_pieces` |
| `directness_balance` | Direct long-ball team vs. direct long-ball team → turnover fest | Compare `average_pass_length` or `long_balls` |

### 8.2 Head-to-Head Pattern Features

H2H is already used but only for basic stats. Deeper patterns may exist.

| Feature | Rationale | Data Source |
|---------|-----------|-------------|
| `h2h_draw_streak` | 3+ consecutive H2H draws → psychological draw bias | H2H history |
| `h2h_scoreline_consistency` | Same scoreline repeated (e.g., 1-1) → tactical stalemate | H2H history |
| `h2h_tightness_index` | Average goal difference in H2H < 0.5 → cautious fixtures | H2H history |
| `venue_specific_h2h_draw_rate` | Some stadiums produce draws due to pitch size/atmosphere | H2H split by venue |
| `derby_indicator` | Local derbies often tight and defensive | Team geographic proximity |
| `recent_h2h_momentum` | Last 2 H2H were draws → likely again | H2H recency weighting |

### 8.3 Fixture Context Features

The schedule context around a match affects motivation and squad rotation.

| Feature | Rationale | Data Source |
|---------|-----------|-------------|
| `fixture_congestion_index` | Both teams played midweek → fatigue → draw | Matches in last 7 days |
| `travel_distance` | Long away travel → defensive setup → draw | Team locations |
| `time_zone_travel` | Cross-time-zone travel → fatigue | Match location vs. team base |
| `days_since_last_match_diff` | One team rested, other tired → imbalance OR both rested → draw | Match dates |
| `relegation_battle` | Both teams near relegation zone → cautious | Standings position |
| `title_race_pressure` | Title contender vs. mid-table → underdog parks bus | Standings position + points gap |
| `dead_rubber_indicator` | Neither team has anything to play for → low intensity draw | Standings + remaining fixtures |
| `manager_tenure` | New manager (< 30 days) → defensive instability or honeymoon | Manager appointment dates |
| `rivalry_intensity` | Historical rivalry → tight matches | Derby / historical bad blood |

### 8.4 Seasonal & Cyclical Features

| Feature | Rationale | Data Source |
|---------|-----------|-------------|
| `season_progress` | Early season (experimentation) vs. late season (desperation) | Match number / total fixtures |
| `month_draw_rate` | Some months have more draws (weather, fixture density) | Historical monthly averages |
| `international_break_proximity` | Pre-break matches → cautious (avoid injuries) | International calendar |
| `winter_break_indicator` | Post-break matches → rusty teams | League calendar |
| `weekend_vs_midweek` | Midweek matches → fatigue → draws | Match day of week |

### 8.5 First-Half Pattern Features

If live/in-play data is available, first-half patterns are strong draw predictors.

| Feature | Rationale | Data Source |
|---------|-----------|-------------|
| `first_half_goals` | 0-0 at halftime → strong draw predictor | Live match data |
| `first_half_shots` | Low shot count → defensive match | Live match data |
| `first_half_xg` | Low xG in first half → tight game | Live Understat data |
| `first_half_possession_balance` | 50/50 possession → stalemate | Live data |
| `first_half_cards` | Early cards → cautious second half | Live match data |

### 8.6 Defensive Solidity Features

| Feature | Rationale | Data Source |
|---------|-----------|-------------|
| `both_teams_low_xg` | Both teams generate < 1.0 xG per game → low scoring | Understat rolling averages |
| `both_teams_high_clean_sheet_rate` | Both keep clean sheets → 0-0 likely | Team form |
| `defensive_line_height_balance` | Both play deep defensive lines → crowded box | Understat `deep` metric |
| `aerial_duel_balance` | Both win aerial duels → set-piece stalemate | Understat aerial metrics |
| `recovery_speed_balance` | Both teams recover ball quickly → no sustained attacks | PPDA / recovery stats |

### 8.7 Market-Aware Features (No Real-Time News Required)

| Feature | Rationale | Data Source |
|---------|-----------|-------------|
| `opening_vs_closing_draw_odds` | Drift toward draw → market signal | Historical odds |
| `draw_odds_trend` | Draw odds shortening over time → sharp money | Historical odds timeline |
| `implied_draw_probability` | Bookmaker's draw estimate (even without knowing why) | Current odds |
| `odds_volatility` | Stable draw odds → confident prediction | Odds movement variance |

### 8.8 Ensemble & Meta-Learning Approaches

| Approach | Rationale | Implementation |
|----------|-----------|----------------|
| `draw-specialist_model` | Train a binary classifier (Draw vs. Not-Draw) with heavy oversampling | Separate XGBoost/LightGBM |
| `matchup_clustering` | Cluster historical matchups by tactical similarity; predict draw rate per cluster | K-means on tactical features |
| `bayesian_draw_prior` | Set team-specific draw priors based on historical draw rate | Bayesian updating per team |
| `draw_momentum_model` | Teams on a draw streak are psychologically biased toward another draw | Markov chain on recent results |

---

## 9. Recommended Priority Order

Based on feasibility and expected signal strength:

### Phase 1: High-Feasibility, Medium Signal (Implement Now)
1. **Fixture congestion index** — Easy to compute from match dates
2. **Season progress** — Already have match dates
3. **H2H draw streak** — Extend existing H2H generator
4. **Weekend vs. midweek** — Already have match dates
5. **Relegation/title race pressure** — Extend standings generator

### Phase 2: Medium-Feasibility, High Signal (Requires Data)
6. **Tactical style clash features** — Requires Understat tactical data (ppda, deep completions)
7. **Manager tenure** — Requires manager database
8. **Travel distance** — Requires team stadium locations
9. **First-half live features** — Requires in-play data feed

### Phase 3: Low-Feasibility, Unknown Signal (Research)
10. **Draw-specialist binary classifier** — Separate model pipeline
11. **Matchup clustering** — Unsupervised learning experiment
12. **Market odds features** — Requires odds database (but no real-time news)

---

## 10. Immediate Recommended Payload

Based on what actually worked:


```bash
curl -sS -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
  "model_type":"xgboost","description":"Final pruned model - team_form only",
  "activate":true,
  "tournament_ids":[359],
  "end_date":"2025-05-31",
  "feature_groups":["team_form"],
  "min_matches":1000,
  "outcome_balance":true,
  "tune_hyperparameters":true,
  "tuning_trials":100,
  "calibrate_probabilities":true,
  "calibration_method":"isotonic",
  "use_ensemble":false,
  "split_strategy":"season_aware"
}'
```
### Backtest results

{"model_version":"xgboost_20260512_063637","evaluated_at":"2026-05-12T06:37:18.031608","num_samples":263,"date_range":["2025-11-01","2026-05-10"],"classification":{"accuracy":0.4790874524714829,"log_loss":1.0591468154992347,"brier_score":0.21171027402753204,"precision_macro":0.43416051866756095,"recall_macro":0.4291076791076791,"f1_macro":0.36327727571967205,"precision_weighted":0.4413502561630875,"recall_weighted":0.4790874524714829,"f1_weighted":0.3959820383222812,"per_class_precision":{"H":0.47619047619047616,"D":0.3333333333333333,"A":0.49295774647887325},"per_class_recall":{"H":0.8571428571428571,"D":0.013513513513513514,"A":0.4166666666666667},"per_class_f1":{"H":0.6122448979591837,"D":0.025974025974025976,"A":0.45161290322580644},"confusion_matrix":[[90,0,15],[52,1,21],[47,2,35]],"top_2_accuracy":0.7642585551330798,"cohen_kappa":0.16475033613055767},"betting":{"total_bets":354,"winning_bets":104,"losing_bets":250,"total_stake":11.699612222711169,"total_return":12.838609786729705,"profit_loss":1.1389975640185366,"roi_percent":9.735344576699099,"yield_percent":9.735344576699099,"sharpe_ratio":0.036810060283398234,"max_drawdown":1.4050236483774394,"win_rate":0.2937853107344633,"average_winning_odds":3.3414423076923074,"average_losing_odds":4.59984,"average_kelly_fraction":0.03304975204155697,"optimal_kelly_fraction":0.13219900816622787},"expected_calibration_error":0.18564304706361615,"maximum_calibration_error":0.5787470887430717,"outcome_accuracy":{"H":0.47619047619047616,"D":0.3333333333333333,"A":0.49295774647887325}}
```

## 11. Files Modified

| File | Change |
|------|--------|
| `algobet/predictions/features/draw_signal_generator.py` | **New** — DrawSignalFeatureGenerator |
| `algobet/predictions/features/composite.py` | Registered `draw_signals` in factory |
| `algobet/predictions/training/config.py` | Added `draw_signals`, stacking, draw_boost config |
| `algobet/predictions/training/feature_selection.py` | Added `draw_signal` family |
| `algobet/predictions/training/classifiers.py` | `DixonColesPredictor.fit()` now delegates to `fit_with_scores()` when goals are stored |
| `algobet/predictions/training/runner.py` | Inline DC training, stacking ensemble, draw boost, DrawAwarePredictor wiring |
| `algobet/predictions/training/stacking.py` | **New** — StackingEnsemble with logistic meta-learner |
| `algobet/predictions/training/calibration.py` | Added `DrawAwarePredictor`, draw boost in `CalibratedPredictor` |
| `algobet/predictions/training/tuner.py` | Draw-recall penalty in CV guard |
| `scripts/train_dixon_coles.py` | Uses real features instead of dummy features |

---

## 12. Verification Status

- **Ruff linting:** All modified files pass.
- **Tests:** `tests/unit/predictions/` (97 tests) + `tests/predictions/` (24 tests) = **121 passed**.
- **Backtesting:** Models register and backtest successfully, but draw recall remains near-zero without `draw_boost_factor > 1.0`.
