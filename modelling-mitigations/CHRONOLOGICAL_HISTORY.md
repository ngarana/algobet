# AlgoBet Modelling & Feature Engineering — Chronological History

> This document consolidates all modelling attempts, feature engineering efforts, and fine-tuning sessions documented in `modelling-mitigations/`, ordered chronologically from earliest to latest.

---

## 1. Initial Model Collapse Debug (2026-05-08)

**Context:** The EPL-only XGBoost model (`xgboost_20260508_035804`) produced class collapse — predicting HOME for every match with F1 Draw = 0.0 and F1 Away = 0.0. Test log loss (1.0809) trailed the market benchmark (0.9637). The selected feature list contained no `enriched_stats` columns despite them being enabled.

**Response:** The EPL Model Fine-Tuning Implementation Plan (`model-training-mediation.md`) was created with 8 phases:

| Phase | Focus |
|---|---|
| 1 | Make evaluation trustworthy — saved-pipeline backtest, stale-row guards |
| 2 | Build feature audit script — null rates, variance, correlation clusters, family tagging |
| 3 | Repair feature set — add draw/away/low-scoring rates, venue-specific signals, enriched diffs, player rollups |
| 4 | Replace naive feature selection with group-aware selection — correlation pruning, family retention guards |
| 5 | Tune XGBoost and LightGBM separately — per-model search spaces, guarded log-loss objective |
| 6 | Implement weighted XGBoost + LightGBM ensemble — `EnsembleWeightOptimizer`, validation split |
| 7 | Training payload with `activate=false` first |
| 8 | Ablation matrix (8 runs A–H) |

**Key additions planned:**
- Team form: draw rates, loss rates, clean sheet rates, BTTS rates, low-scoring rates, goal variance, points volatility, venue-specific draw/away rates
- H2H: draw rate, away win rate, low-scoring rate, recency-weighted points
- Standings: draw/loss rates, top-six/bottom-six indicators
- Enriched: xG/npxG diffs, shot quality, xG conversion, PPDA diffs, deep completion diffs, player saves/fouls/cards/offsides

---

## 2. Backtest Collapse & Calibration Experiments (2026-05-12)

**Session:** `session_work_summary.md`

### Problem
The backtest endpoint returned all-HOME predictions. Kelly bet sizing was based on broken probabilities.

### Root Causes Found & Fixed

| # | Root Cause | Fix |
|---|---|---|
| 1 | Missing `preload_season_standings()` in backtest — all standings features were NaN | Added full preload block before `feature_pipeline.transform()` |
| 2 | Adaptive regularization overriding tuner — `max_depth=3`, `min_child_weight=10` too restrictive for DRAW class (5:1 sample/feature ratio) | Relaxed to `max_depth=4`, `min_child_weight=5` |
| 3 | Calibration collapse not disabled — sigmoid calibrator collapsed to HOME, code only warned but kept it | Set `self._calibrator = None` when collapse detected |
| 4 | Tuner search space too restrictive | Adjusted `max_depth: (4,7)`, `min_child_weight: (1,6)`, strengthened draw recall penalty |

### Feature Selection Breakthrough
With 159 features and ~3040 samples, all feature importances were flat (~0.66% each). Solution: `min_samples_per_feature=75` capped features at 40, raising sample/feature ratio to 217:1.

### Model `xgboost_20260512_180433` — First Working Model

| Metric | Value |
|---|---|
| Accuracy | 40.3% |
| Cohen's Kappa | 0.065 (was 0.000) |
| DRAW recall | 17.7% (was 0%) |
| AWAY recall | 38.0% (was 0%) |
| Betting ROI | +4.48% |

### Player Quality Feature Generator
Created `player_quality_generator.py` capturing **squad continuity** (XI stability, starting pool size) — orthogonal to existing form/xG/standings features.

### Model `xgboost_20260512_184948` — Player Quality Added
Accuracy improved to 41.6% but betting ROI collapsed to −0.77% because calibration worsened (ECE 0.224 → 0.340) and Kelly triggered 538 bets on 375 matches (1.44/match).

### Calibration Experiments (Afternoon Session)

| Model | Calibration | Accuracy | Kappa | D recall | ROI |
|---|---|---|---|---|---|
| 180433 | None (baseline) | 40.3% | +0.065 | 17.7% | +6.87% |
| 185942 | Single-T temperature | 38.1% | +0.020 | 15.6% | −5.49% |
| 192746 | Sigmoid + tight search | 38.7% | −0.012 | 6.3% | −13.4% |
| 194212 | Per-class T + 8× draw penalty | 34.9% | −0.001 | 30.2% | −11.7% |
| 195921 | Per-class T + 3× draw penalty | 33.6% | −0.022 | 28.1% | −12.4% |
| 200943 | None | 41.3% | +0.045 | 7.3% | −2.75% |
| **201906** | **None + mild balance (0.3)** | **40.3%** | **+0.037** | **12.5%** | **+4.79%** |

### Key Fixes in This Session

| Fix | Result |
|---|---|
| **Fix 5:** Temperature scaling (single-T and per-class) | **Failed** — every calibration method destroyed discriminative signal |
| **Fix 6:** Single-best-edge betting (max 1 bet/match) | **Success** — cleaned up Kelly over-betting |
| **Fix 7:** Per-class temperature scaling | **Failed** — Kappa went negative |
| **Fix 8:** Tuner search space & penalty iterations | **Mixed** — all changes reverted to 180433 recipe |
| **Fix 9:** Dixon-Coles blend calibration | **Failed** — collapsed on test set (0 DRAWs), validation-tuned blend didn't transfer |
| **Fix 10:** No calibration + mild class weighting (strength 0.3) | **Winner** — model `xgboost_20260512_201906` |

### Key Lessons Learned
1. **Do not calibrate XGBoost probabilities for football betting** — every method (sigmoid, isotonic, single-T, per-class T) destroyed Kappa-to-ECE trade-off
2. **Distribution shift between EPL seasons is real** — validation-tuned calibrators don't transfer
3. **HOME-dominant overfit is profitable** — 77%→40% train/test gap accepted as "feature, not bug"
4. **Single-best-edge betting is the right simulation** — caps at 1 bet/match

---

## 3. Modeling Framework Improvement Plan (Post May 12)

**Document:** `AlgoBet-Modeling-Framework-Improvement-Plan.md`

After reading every file in the prediction pipeline, 7 root causes were identified:

| RC | Problem | Severity |
|---|---|---|
| RC-1 | Wrong modeling paradigm — classifying H/D/A instead of modeling scores | Critical |
| RC-2 | Feature-target mismatch — team strength features can't predict draws | Critical |
| RC-3 | StandardScaler + median imputation destroys NaN information for tree models | High |
| RC-4 | Calibration architecturally doomed — calibrating overfit softmax probabilities | High |
| RC-5 | No market-anchored features — ignoring the strongest signal (closing line) | High |
| RC-6 | Overfitting accepted as "feature" — 77%→40% gap rationalized | Medium |
| RC-7 | Non-standard betting simulation — no CLV tracking | Medium |

### Proposed 3-Phase Plan

**Phase 1 — Foundation Fixes:**
- Switch to native NaN handling for tree models (`PreserveMissingValues`)
- Register odds-implied features (`odds`, `odds_residual` feature groups)
- Add CLV tracking to evaluation metrics

**Phase 2 — Paradigm Shift:**
- Promote Dixon-Coles to primary model (bivariate Poisson with ρ correction)
- Hybrid architecture: XGBoost → λ_home, λ_away → score distribution → P(H/D/A)
- Walk-forward validation replacing single train/val/test split

**Phase 3 — Advanced Techniques:**
- Market-residual modeling (predict Δ = P(outcome) − P_market(outcome))
- Venn-Abers calibration (distribution-free probability intervals)
- Bayesian score modeling with time decay

---

## 4. Implementation of Improvement Plan

**Document:** `IMPLEMENTATION_SUMMARY.md`

All 3 phases were implemented with backwards-compatible changes:

### Phase 1 Completed
- **Native NaN handling:** XGBoost/LightGBM now use `PreserveMissingValues` pipeline; transformer type persisted on save/load
- **Odds features registered:** `OddsFeatureGenerator` and `OddsResidualFeatureGenerator` added to composite factory, config, and feature selection
- **CLV tracking:** `mean_clv`, `clv_hit_rate`, `clv_weighted_roi` added to `BettingMetrics`

### Phase 2 Completed
- **Dixon-Coles primary:** `model_type="dixon_coles"` works end-to-end; exempted from collapse gates (low argmax-D is expected behavior)
- **Hybrid Poisson:** `HybridPoissonPredictor` using `HistGradientBoostingRegressor(loss="poisson")` for both goal expectations, then bivariate Poisson grid
- **Walk-forward validation:** `WalkForwardSplitter` generates multiple season-shifted folds

### Phase 3 Completed
- **Market-residual:** `MarketResidualPredictor` blends base predictor with market-implied probabilities via optimized α
- **Venn-Abers:** `VennAbersCalibrator` with one-vs-all isotonic regression, producing intervals not point estimates
- **Time decay:** `set_time_weights()` added to Dixon-Coles and Hybrid Poisson; exponential decay passed to `sample_weight`

### New API Values
- `model_type`: `dixon_coles`, `hybrid_poisson`
- `split_strategy`: `walk_forward`
- `feature_groups`: `odds`, `odds_residual`
- `calibration_method`: `venn_abers`

### Verification
29 new tests covering all phases; 55 existing tests continue to pass.

---

## 5. Post-Deployment: Score-Model Underperformance Diagnosis

After implementing the plan, Dixon-Coles and Hybrid Poisson were trained and backtested on 2025/26 EPL. Both underperformed: ROI = −2.5% and −12% respectively, zero argmax-draw predictions, ECE ≈ 0.19.

### Bug Fix
Calibration crash for score-based models — `DixonColesPredictor` requires `fit_with_scores()` with goal data, but CV calibration path creates fresh predictors per fold without goal data. **Fix:** Skip calibration block entirely for `dixon_coles` and `hybrid_poisson`.

### Audit Wave 1: Variance Compression

| Metric | DC v1 predicted | Actual | Diagnosis |
|---|---|---|---|
| λ_home mean | 1.573 | 1.518 | Unbiased (+0.06) |
| λ_away mean | 1.207 | 1.231 | Unbiased (−0.02) |
| **λ_home std** | **0.261** | **1.183** | **4.5× too compressed** |
| **λ_away std** | **0.232** | **1.099** | **4.7× too compressed** |
| **λ_diff std** | **0.345** | **1.642** | **4.7× too compressed** |

**Finding:** The model is not home-biased — mean λ matches reality. The HistGradientBoosting Poisson regressors are under-fitting, predicting almost every match as 1.5–1.2 goals with 4× less variance than actual goal-difference.

### Remediation Wave 1: Looser Regularization + Odds Features

| Param | DC v1 | DC v2 |
|---|---|---|
| `max_iter` | 450 | 600 |
| `learning_rate` | 0.025 | 0.03 |
| `l2_regularization` | 0.1 | 0.05 |
| `max_leaf_nodes` | 15 | 31 |
| `min_samples_leaf` | 35 | 20 |

**Result — disappointing:**

| Metric | DC v1 | DC v2 | Target |
|---|---|---|---|
| λ_diff std | 0.345 | 0.410 | ≥ 0.70 |
| Max P(D) | 0.300 | 0.352 | > 0.34 |
| ROI | −2.47% | −4.68% | > 0 |

Variance widened only 18% (needed 200%). Adding odds shifted goal-estimation mean down (bookmaker totals encode over-round bias), introducing calibration drift.

### Structural Diagnosis
Team-strength features explain ~6% of variance in match goal-difference (σ ≈ 0.4 of available σ ≈ 1.6). Bivariate Poisson cannot bridge a 4× input-variance gap. **RC-2 extended:** team-strength features can't predict draws because they can't predict goal magnitude, not just because of H/D/A categorical structure.

### Remediation Wave 2: Feature Engineering on Existing xG Data

Added 56 new features to `EnrichedStatsFeatureGenerator` — all derived from existing xG/goals/opponent metadata:

| Feature Group | Captures | Count |
|---|---|---|
| `npxg_for/against_avg` | Non-penalty xG (penalties are noise) | 2 |
| `xg/npxg slope` | Linear-fit slope (momentum) | 4 |
| `xg/npxg std` | Volatility over window | 4 |
| `finishing_rate_for/against` | `goals / max(xg, 0.1)` — clinical finishing | 2 |
| `xg_adj_avg` | `xg − opponent's rolling xg_against` — strength-of-schedule adjusted | 2 |

Per home/away × window {3, 5} = 56 total new features.

### Audit Remediation: Implementation Summary Gaps Fixed

| Issue | Fix |
|---|---|
| Walk-forward metrics only used one split | Now stores all folds, trains on latest, reports averaged metrics |
| Ensemble weight optimization referenced unassigned vars | `X_weight`/`y_weight` default to validation data |
| `/ml/calibrate` could refit default pipeline | Now loads fitted pipeline saved with model |
| Backtest fallback could refit default pipeline | Now requires fitted saved pipeline |
| Venn-Abers was isolated class | Wired into `ProbabilityCalibrator`, API schemas, frontend |
| CLV fields not fully exposed | Added to `BettingMetricsResponse`, backtest persistence, frontend |
| Frontend lagged backend options | Added `dixon_coles`, `hybrid_poisson`, `walk_forward`, `temperature`, `venn_abers` |

---

## 6. Pure ML Improvement Plan Update

**Document:** `PURE-ML-IMPROVEMENT-PLAN-UPDATE.md`

A critical review of the improvement plan with emphasis on the **pure-ML constraint**: no odds-derived training features, no implied-odds blending. Market odds only for post-hoc diagnostics.

### Key Changes to Original Plan
- Replace fixed expected gains with falsifiable gates against baseline
- Remove unsupported claims about specific syndicates
- No shuffled K-fold OOF — must be chronological/season-aware
- Don't call L2-regularized validation-weight optimization "Bayesian model averaging"
- Defer custom betting loss (conflicts with pure-ML boundary)
- Treat CatBoost as ablation, not guaranteed improvement

### Updated 8-Phase Plan

| Phase | Focus | Status |
|---|---|---|
| 0 | Replace claims with gates — name baseline, EPL scope, holdout policy | Done |
| 1 | Repair OOF and validation boundaries — time-aware splitter, stop validation-set reuse | Done |
| 2 | Expose ensemble/stacking controls correctly through API | Done |
| 3 | Harden weighted XGBoost + LightGBM ensemble — generalize optimizer, rejection gates | Done |
| 4 | Implement time-aware OOF stacking with logistic meta-learner | Done |
| 5 | Add CatBoost as controlled ablation | Partial |
| 6 | Calibration and decision layer — sample gates, log-loss/ECE gates | Done |
| 7 | Recency and temporal ensembling | Pending |
| 8 | Custom betting loss research only | Deferred |

---

## 7. Pure ML Improvements — Implementation Status

**Document:** `IMPLEMENTATION-STATUS.md` (2026-05-17)

### What Was Achieved

| Phase | Status |
|---|---|
| 1. Time-aware OOF & validation boundaries | ✅ `OOFTimeAwareSplitter`, LightGBM callback fix, sklearn 1.7 compatibility |
| 2. API schema & persistence | ✅ Stacking config fields, correct `model_type` persistence |
| 3. Blend rejection gates | ✅ Class-collapse rejection in `EnsembleWeightOptimizer` and `StackingEnsemble` |
| 4. OOF stacking with Logistic/MLP meta-learner | ✅ LogisticRegression default, MLP opt-in |
| 5. CatBoost integration | ⚠️ Partial — implemented but collapses to 1 class (all HOME) |
| 6. Calibration hardening | ✅ Sample gate (isotonic ≥ 1000 samples), log-loss gate, collapse guard |

### What's Not Working

**1. Stacking Ensemble Collapses to 2 Classes (HOME/AWAY)**
- Meta-learner (logistic regression) ignores draw class entirely
- `max_prediction_share` up to 0.98 on test set
- Root cause: draws are ~25% of matches; meta-learner optimizes for overall accuracy and learns to split only HOME/AWAY

| Model | Train Acc | Val Acc | Test Acc | Test Log Loss | Classes | ECE |
|---|---|---|---|---|---|---|
| XGBoost baseline | 0.517 | 0.403 | 0.425 | 1.081 | 3 | 0.076 |
| Stacking (XGB+LGB+DC) | 0.577 | 0.418 | 0.423 | 1.084 | 2 | 0.102 |
| Stacking (XGB+LGB) | 0.478 | 0.413 | 0.428 | 1.084 | 2 | 0.084 |

**2. CatBoost Collapses to 1 Class (All HOME)**
- Even with heavy regularization (`depth=3`, `l2_leaf_reg=20.0`, `iterations=300`)
- Ordered boosting + class imbalance (HOME ~45%) causes convergence on majority class
- **Conclusion:** CatBoost doesn't work with current feature surface

**3. No Model Beats the Market**
- None achieve test log loss lower than market (1.007)
- Consistent with finding that team-strength features explain ~6% of variance

### Current Best Model
**XGBoost baseline** remains best:
- Test accuracy: 42.5%
- Test log loss: 1.081 (vs. market 1.007)
- Predicts all 3 classes
- ECE: 0.076

---

## 8. Top-5 League Data Import & Detailed Odds Comparison (2026-05-17)

**Document:** `TOP5_DETAILED_ODDS_TRAINING_COMPARISON.md`

Top-5 league data imported via `algobet import-data fd-top5-range 2012 2025`. Two XGBoost models trained to isolate `detailed_odds` feature group impact.

### Configuration
- Tournaments: EPL (359), La Liga (545), Serie A (98), Bundesliga (28), Ligue 1 (123)
- Walk-forward: 8 train seasons, 1 val, 1 test (4 folds)
- Feature selection: threshold 0.005, min samples per feature 40

### Results

| Metric | Baseline | + Detailed Odds | Delta |
|---|---|---|---|
| Accuracy | 0.3276 | 0.3127 | **−0.0149** |
| F1 macro | 0.3235 | 0.2978 | **−0.0257** |
| Brier score | 0.2224 | 0.2227 | +0.0003 |
| ECE | 0.2064 | 0.2137 | +0.0073 |
| Max prediction share | 0.4110 | 0.5251 | +0.1141 |

Only 6 of 101 selected features were from `detailed_odds` (avg implied prob draw, odds disagreement home/away, OU vs 1x2 diff, over/under odds).

**Conclusion:** Adding `detailed_odds` did not improve the top-5 walk-forward run. Reduced accuracy by 1.49pp, F1 by 2.57pp, worsened calibration, and made predictions more concentrated.

### Multi-League Fixes Applied
| Area | Fix |
|---|---|
| Multi-league season splitting | Splitters now derive calendar football-season split key from `match_date` when multiple `tournament_id` values present |
| Tournament collisions | FD and soccerdata importers resolve by `(name, country)` with country-qualified slugs |
| Bundesliga contamination | Tournament id 28 repaired from `country=Austria` to `country=Germany` (5,807 German-marker matches out of 6,759) |
| Stacking OOF consistency | Fold predictors now clone original base config including hyperparameters, class weights, random seed |
| Detailed odds exposure | Available as explicit feature group in frontend; empty selection correctly means backend default |

---

## 9. Market Mediation Model Design (Post May 17)

**Document:** `Market_Mediation_Model.md`

A shift in strategy: instead of forcing a prediction on every match, the model learns when AlgoBet has measurable edge and returns `ABSTAIN` otherwise.

### Key Design
- **Dual-lane system:** Pure lane (pre-market prior) + market mediation lane
- **Two heads:** Probability residual head (corrects market probabilities) + CLV gate head (predicts whether current price beats closing price)
- **Selective production:** Emit `BET_CANDIDATE` only when lower confidence bound of expected CLV is positive; otherwise `ABSTAIN`
- **Closing odds:** Labels/evaluation data only, never available as inference features

### Data Requirements
- Real opening/closing odds support (not just `odds_home/draw/away` as closing proxy)
- Alembic migration for `Match` columns: opening/closing 1X2 odds, Asian handicap, over/under
- FDImporter updated to map opening from non-`C` columns and closing from `B365CH/B365CD/B365CA`, `AvgCH/AvgCD/AvgCA`, etc.

### Activation Gates
- Closing-odds coverage ≥ 80% in every test fold
- Selected bets ≥ 200 pooled across folds
- Mean selected-bet CLV > 0 in at least 3 of 4 walk-forward folds
- Pooled bootstrap 95% lower bound for selected-bet CLV > 0
- Final probability log loss not worse than opening market log loss

---

## 10. Market Mediation Debug Session (2026-05-18)

**Document:** `market_mediation_debug_session.md`

**Starting model:** `market_mediation_20260517_191919` (model_id 234)
**Final model:** `market_mediation_20260518_042022` (model_id 243)

### Starting State
```
selected_bets: 0
abstention_share: 1.0
num_features: 17
mediation_features_in_importance: 3
```

### Bugs Found & Fixed

| Bug | Root Cause | Fix |
|---|---|---|
| **1. Expected CLV always zero** | `_expected_clv` used unconditional CLV mean; `max(..., 0)` clipped to zero since avg CLV ≈ −0.3% | Store conditional means: E[CLV | CLV > 0] ≈ +6%, E[CLV | CLV ≤ 0] ≈ −5.8% |
| **2. CLV head trained on anti-signal** | Included `residual = pure_probas - market_probas`; when pure model thinks market is wrong, market is usually right (LL gap ~0.07) | `_build_clv_features` excludes pure-model residuals entirely; CLV head uses only market-structure features (24-dim input) |
| **3a. X column index mismatch** | Forced mediation features removed from `prunable_names` but X matrix not sliced — `prunable_names[i]` no longer matched `X[:, i]` | Build explicit column index mapping and slice X before correlation pruning |
| **3b. Uniform importance rejects all features** | `MarketMediationPredictor.feature_importance` returns uniform `1/N ≈ 0.004`; default threshold `0.005` rejects everything | Use `threshold=0.0` for `market_mediation`; let correlation pruning and family minimums drive selection |

### Results After All Fixes

| Metric | Before | After |
|---|---|---|
| `num_features` | 17 | 205 |
| Mediation features | 3 | 21 |
| Selected bets (walk-forward pooled) | 0 | 9,775 |
| CLV hit rate | 0.0 | 0.404 |
| Pooled mean CLV | 0.0 | −0.0028 |
| Positive CLV folds | 0/7 | 1/7 |
| Activation | Not activated | Not activated |

### Remaining Limitation — CLV Signal

**The CLV base rate is 46%, not 50%.** Opening prices are systematically tighter than closing prices because bookmakers post with larger margin, sharp money compresses it by close. This means:
- E[CLV] at base rate = 0.46 × 6% + 0.54 × (−5.8%) = **−0.37%** (structurally negative)
- Finding positive expected CLV requires hit rate > 50% (4+ pp above base)

The CLV classifier gets **40.4% hit rate vs 46% base** — finding a systematic anti-pattern, not edge.

### What Would Give Positive CLV Signal

| Data Source | Why |
|---|---|
| Intraday line movement (T-6h, T-2h odds) | Predicts direction of close movement |
| Asian handicap volume/liquidity | Sharp money indicator |
| Pinnacle opening vs public book spread | Sharp vs soft book disagreement |
| Bet365 / Asian market price difference | Divergence signals sharp action |
| Historical team-specific CLV patterns | Some teams consistently attract/lose sharp money |

**Without these, the model cannot outperform random selection of opening-price bets.**

---

## 11. Ablation & Permutation Feature Importance Tooling

**Document:** `ablation_guide.md`

Added `/api/v1/ml/ablation` endpoint with two methods:

| Method | Speed | What it does |
|---|---|---|
| **Permutation** | Fast (seconds) | Shuffles feature columns per family, measures performance drop |
| **Ablation** | Slow (minutes) | Retrains model excluding each feature group (leave-one-out) |

Supports grouping by `family` (sub-family patterns: draw, away, form, enriched, etc.) or `generator` (feature generator groups: team_form, head_to_head, etc.).

---

## 12. Model Training Limitations — Historical Record

**Document:** `MODEL_TRAINING_LIMITATIONS.md`

Documents that the vast majority of historical limitations have been resolved:

### Currently Supported
- Multi-tournament filtering, team filtering, venue filtering, goal range filters
- Multiple algorithms: XGBoost, LightGBM, Random Forest
- Ensemble training (XGBoost + LightGBM)
- Hyperparameter tuning via Optuna
- Three split strategies: Temporal, Expanding Window, Season-Aware
- Feature group selection, automatic feature pruning, outcome balancing
- Model tags and descriptions

---

## Summary Timeline

| Date | Event | Outcome |
|---|---|---|
| ~May 8 | EPL model class collapse diagnosed | Fine-tuning plan created (8 phases) |
| May 12 | Backtest collapse debug session | First working model (40.3% acc, +4.48% ROI); all calibration methods failed |
| May 12 (afternoon) | 7 calibration/betting experiments | Winner: no calibration + mild class weighting (0.3) |
| Post May 12 | Root cause analysis (7 RCs identified) | 3-phase improvement plan created |
| Post May 12 | Implementation of all 3 phases | Dixon-Coles, Hybrid Poisson, Walk-forward, Venn-Abers, Market-residual, CLV tracking |
| Post deployment | Score-model audit | Variance compression diagnosed (4× too compressed); Wave 1 remediation failed |
| Post deployment | Wave 2: 56 new xG-derived features | npxG, slope, std, finishing rate, opponent-adjusted xG |
| Post deployment | Implementation gaps fixed | Walk-forward averaging, pipeline loading, Venn-Abers wiring, CLV exposure |
| ~May 17 | Pure ML plan update | 8-phase plan with falsifiable gates, pure-ML constraint enforced |
| May 17 | Implementation status report | Stacking collapses to 2 classes; CatBoost collapses to 1; XGBoost baseline remains best |
| May 17 | Top-5 league data import | Multi-league fixes; `detailed_odds` adds no value (accuracy −1.49pp) |
| Post May 17 | Market mediation model designed | Dual-lane system with abstention policy |
| May 18 | Market mediation debug session | 3 bugs fixed; model selects 9,775 bets but CLV still negative (−0.28%); activation gates correctly prevent deployment |
| Ongoing | Ablation tooling added | Permutation and leave-one-out ablation endpoints |

---

## Current State (as of May 18, 2026)

**Best model:** XGBoost baseline — 42.5% test accuracy, 1.081 log loss (market: 1.007), predicts all 3 classes, ECE 0.076.

**Key structural limitation:** Team-strength features explain ~6% of variance in goal-difference. No model trained on these features alone can beat the market.

**Next levers (in order):**
1. New data sources: lineup absences, closing-line movement, intraday odds snapshots, Asian handicap liquidity
2. If λ_diff std reaches ≥ 0.70 with current feature engineering → continue with lineup-absence features, set-piece xG
3. If λ_diff std barely moves (< 0.50) → feature engineering on existing data is exhausted; must ingest new data
4. Market mediation model needs intraday odds or AH divergence signal to achieve positive CLV
