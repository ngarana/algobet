# AlgoBet Modeling Framework: Diagnostic Analysis & Improvement Plan

## Problem Statement

The session work summary documents a model that achieves **40% accuracy** (barely above the ~45% home-win base rate), **Cohen's Kappa of 0.065** (near-random), **ECE of 0.224** (severely miscalibrated), and a **77% → 40% train/test gap** (massive overfitting). Every calibration method tested (sigmoid, isotonic, single-T temperature, per-class temperature, Dixon-Coles blend) either destroyed discriminative signal or collapsed predictions. The "winning" model achieved +4.79% ROI only by disabling all calibration and relying on raw softmax overconfidence — a fragile strategy that cannot survive distribution shift.

---

## Root Cause Analysis

After reading every file in the prediction pipeline (`features/`, `training/`, `evaluation/`), I've identified **7 structural problems**, grouped from most to least fundamental.

### RC-1: Wrong Modeling Paradigm — Classifying Outcomes Instead of Modeling Scores

> [!CAUTION]
> This is the single largest architectural mistake and the root cause of every downstream problem.

The entire framework treats football prediction as a **3-class classification problem** (H/D/A), which is fundamentally wrong for two reasons:

1. **Draws are not a "class" — they're a boundary condition.** A draw occurs when both teams score the same number of goals. It's not an intrinsic property of a match — it's a consequence of the score being tied. Treating it as a class forces the model to learn "draw-ness" from features that were designed to predict team strength, which is why draw recall is always catastrophic.

2. **Softmax destroys the ordinal structure.** The outcomes H/D/A have an ordinal relationship: H → D → A corresponds to a continuum of goal difference (from strongly positive to zero to strongly negative). Softmax treats them as independent, unrelated categories. The model can't learn that a match that's "almost a draw" should have draw probability smoothly interpolated between H and A.

**Evidence from the codebase:**
- [classifiers.py:377](file:///home/arch/Coding/algobet/algobet/predictions/training/classifiers.py#L377): `"objective": "multi:softprob"` — pure categorical classification
- Every feature generator ([draw_signal_generator.py](file:///home/arch/Coding/algobet/algobet/predictions/features/draw_signal_generator.py), [team_form_generator.py](file:///home/arch/Coding/algobet/algobet/predictions/features/team_form_generator.py)) computes *team-level* statistics, but the model target is a *match-level* categorical outcome
- The `DixonColesPredictor` ([classifiers.py:919-1122](file:///home/arch/Coding/algobet/algobet/predictions/training/classifiers.py#L919-L1122)) already models goals correctly via Poisson regression, but it's relegated to an optional blend component rather than being the primary model

**What syndicates do:** Professional betting operations model **expected goals (λ_home, λ_away)** via Poisson regression, then derive H/D/A probabilities from the bivariate goal distribution. The Dixon-Coles ρ parameter corrects for score correlation at low goals. This naturally produces well-calibrated draw probabilities because draws emerge from the score distribution rather than being a learned class.

---

### RC-2: Feature-Target Mismatch — Team Strength Features Can't Predict Draws

The feature generators produce signals about team strength differentials:
- `form_diff`, `points_last_5`, `win_rate_10` → measure which team is stronger
- `standings_position_diff`, `elo_diff` → measure strength gap
- `xg_for/against` → measure attacking/defensive quality

These features are **informative for separating H from A** but **uninformative for predicting D**. A draw requires both teams to be *similarly strong* **and** the match to produce a tied score — the second condition is stochastic and not capturable by pre-match features alone.

**Evidence:**
- [feature_selection.py:11-117](file:///home/arch/Coding/algobet/algobet/predictions/training/feature_selection.py#L11-L117): The `FEATURE_FAMILIES` classification shows the vast majority of features are team-strength metrics
- [draw_signal_generator.py](file:///home/arch/Coding/algobet/algobet/predictions/features/draw_signal_generator.py): The draw-specific features (`goal_convergence`, `strength_parity`, `xg_parity`) are just transformations of the same team-strength signals — they can't add genuinely new information
- Session summary: "With 159 features... all XGBoost feature importances were flat (~0.66% normalized gain each) — no discriminative signal" — this is the smoking gun. The features are redundant and uninformative for the target.

---

### RC-3: StandardScaler + Median Imputation Destroys Information for Tree Models

> [!WARNING]
> The default transformer pipeline is actively harmful for XGBoost/LightGBM.

```python
# transformers.py:553-567
def create_default_transformer_pipeline() -> TransformerPipeline:
    return TransformerPipeline(steps=[
        ("imputer", MissingValueHandler(numeric_strategy="median", add_indicator=True)),
        ("scaler", FeatureScaler(with_mean=True, with_std=True)),
    ])
```

XGBoost and LightGBM **natively handle NaN values** by learning optimal split directions for missing data. Replacing NaNs with medians before training:
1. **Removes the "data is missing" signal** that the tree can learn from (enriched stats coverage varies by season — missingness itself is informative)
2. **StandardScaler is a no-op for tree models** — splits are rank-based, so mean-centering and variance-scaling have zero effect on tree decision boundaries
3. The `add_indicator=True` flag adds binary missing-indicator columns, but these are a poor substitute for native NaN handling

The codebase already has `create_tree_model_transformer_pipeline()` ([transformers.py:570-572](file:///home/arch/Coding/algobet/algobet/predictions/features/transformers.py#L570-L572)) using `PreserveMissingValues`, but **it's never used** — the training pipeline always calls `create_default_transformer_pipeline()`.

---

### RC-4: Calibration is Architecturally Doomed — Wrong Sequence, Wrong Data

The calibration approach has fundamental design flaws:

1. **Calibrating softmax probabilities from an overfit model.** The model has 77% train accuracy but 40% test accuracy. The softmax outputs are already distorted by overfitting. Calibrating distorted probabilities with a 1-parameter (temperature) or 6-parameter (per-class sigmoid) model cannot fix the underlying prediction quality.

2. **Validation-set calibration doesn't transfer across seasons.** The session summary confirms: "Distribution shift between adjacent EPL seasons is large enough to break a validation-tuned blend weight." This is expected — season-to-season variation in team compositions, tactics, and league dynamics means any calibrator fitted on season N will be stale on season N+1.

3. **The calibration pipeline conflicts with betting evaluation.** From [runner.py:287-295](file:///home/arch/Coding/algobet/algobet/predictions/training/runner.py#L287-L295), the evaluation uses `apply_calibration=False` for all splits — meaning the calibrator is fitted but never actually evaluated properly. The betting simulation in [metrics.py](file:///home/arch/Coding/algobet/algobet/predictions/evaluation/metrics.py) uses raw `model.predict_proba()` output without distinguishing calibrated vs uncalibrated.

---

### RC-5: No Market-Anchored Features — Ignoring the Strongest Signal

> [!IMPORTANT]
> The most powerful predictor of match outcomes is the **closing line** — the final betting odds before kickoff, which aggregate millions of dollars of sharp-money information.

The codebase stores odds (`odds_home`, `odds_draw`, `odds_away`) in the match table and uses them for betting simulation, but **never feeds them into the model as features**. The `OddsTransformer` in [transformers.py:357-435](file:///home/arch/Coding/algobet/algobet/predictions/features/transformers.py#L357-L435) exists but is never instantiated.

The `odds_generator.py` and `odds_residual_generator.py` files exist but are **not registered in the default generators** ([composite.py:96-112](file:///home/arch/Coding/algobet/algobet/predictions/features/composite.py#L96-L112)) and **not in `ALLOWED_FEATURE_GROUPS`** ([config.py:12-23](file:///home/arch/Coding/algobet/algobet/predictions/training/config.py#L12-L23)).

Professional syndicates either:
- Use odds-implied probabilities as the primary anchor, then model the **residual** (deviation from market consensus)
- Or use their own model to identify discrepancies against the market line (CLV approach)

Either way, ignoring the market is leaving the single most informative signal on the table.

---

### RC-6: Overfitting Accepted as "Feature" — 77% Train → 40% Test

The session summary states: *"HOME-dominant overfit is profitable. The 77%→40% train/test gap is a feature, not a bug."*

This is a dangerous rationalization. A 37-point accuracy gap means the model has memorized training data patterns that don't generalize. The positive ROI on the test set is almost certainly survivorship bias — the "winning" configuration was selected *because* it happened to show positive ROI on this specific test season. On a different test season, the same model would likely produce negative ROI.

**Evidence of overfitting acceptance in code:**
- [tuner.py:379-434](file:///home/arch/Coding/algobet/algobet/predictions/training/tuner.py#L379-L434): `_evaluate_cv_guarded()` had an overfit penalty that was **explicitly reverted** because it reduced training accuracy
- The adaptive regularization ([classifiers.py:230-312](file:///home/arch/Coding/algobet/algobet/predictions/training/classifiers.py#L230-L312)) was weakened from `max_depth=3` to `max_depth=4` and `min_child_weight=10` to `5` because the stronger regularization prevented draw learning — the correct fix was changing the modeling paradigm, not weakening regularization

---

### RC-7: Evaluation Uses Non-Standard Betting Simulation

The betting simulation in [metrics.py:239-368](file:///home/arch/Coding/algobet/algobet/predictions/evaluation/metrics.py#L239-L368) has several issues:

1. **Kelly criterion with uncalibrated probabilities.** Kelly sizing assumes perfectly calibrated probabilities. With ECE=0.224, Kelly will systematically over-bet on overconfident predictions and under-bet on underconfident ones.

2. **No Closing Line Value (CLV) tracking.** The gold standard metric for professional bettors — whether your bet price beats the closing line — is completely absent. ROI over 375 matches is a noisy, high-variance metric; CLV converges much faster.

3. **Single-best-edge per match is a band-aid.** The fix to bet only the highest-edge outcome per match (session fix #6) masks the calibration problem rather than solving it.

---

## Proposed Improvements

### Phase 1: Foundation Fixes (Low-risk, high-impact)

#### 1.1 Switch to Native NaN Handling for Tree Models

**Files:** [pipeline.py](file:///home/arch/Coding/algobet/algobet/predictions/features/pipeline.py), [data_preparation.py](file:///home/arch/Coding/algobet/algobet/predictions/training/data_preparation.py)

Use `create_tree_model_transformer_pipeline()` instead of `create_default_transformer_pipeline()` when the model type is `xgboost` or `lightgbm`. This is a one-line change that will:
- Preserve native NaN handling (XGBoost learns optimal split direction for missing values)
- Remove meaningless StandardScaler step
- Eliminate the 2× feature explosion from missing-indicator columns

#### 1.2 Register Odds-Implied Features

**Files:** [composite.py](file:///home/arch/Coding/algobet/algobet/predictions/features/composite.py), [config.py](file:///home/arch/Coding/algobet/algobet/predictions/training/config.py)

Add `"odds"` and `"odds_residual"` to `ALLOWED_FEATURE_GROUPS` and register the existing generators. These features capture market consensus and provide an anchor for the model's predictions.

> [!WARNING]
> Using odds as features means the model can only predict matches where odds are available. For pre-market predictions, a separate odds-free model is needed. The user should decide which use-case to prioritize.

#### 1.3 Add CLV Tracking to Evaluation

**File:** [metrics.py](file:///home/arch/Coding/algobet/algobet/predictions/evaluation/metrics.py)

Add `closing_line_value` to `BettingMetrics`:
- For each bet placed, compute `clv = (model_implied_odds / closing_odds) - 1`
- Track mean CLV, CLV hit rate, and CLV-weighted ROI
- This requires storing closing odds alongside opening odds in the match table (or using current odds as proxy)

---

### Phase 2: Modeling Paradigm Shift (Medium-risk, transformative)

#### 2.1 Promote Dixon-Coles to Primary Model

**Files:** [classifiers.py](file:///home/arch/Coding/algobet/algobet/predictions/training/classifiers.py), [runner.py](file:///home/arch/Coding/algobet/algobet/predictions/training/runner.py)

The existing `DixonColesPredictor` already implements bivariate Poisson with ρ correction. Promote it from an optional blend component to a first-class model option:

1. Make `model_type="dixon_coles"` work end-to-end through the training API
2. Use `HistGradientBoostingRegressor(loss="poisson")` for both home and away goal expectations — this is already implemented but underutilized
3. Derive H/D/A probabilities from the score distribution — this naturally produces well-calibrated draw probabilities
4. Grid-search ρ on validation data — already implemented

#### 2.2 Hybrid Architecture: XGBoost Features → Poisson Goals → Score Distribution

The most powerful approach, used by professional syndicates:

```
Features → XGBoost → λ_home, λ_away (goal expectations)
                    ↓
         Bivariate Poisson + ρ correction
                    ↓
         P(H), P(D), P(A) from score distribution
```

This is architecturally different from the current approach:
- **Current:** Features → XGBoost → softmax → P(H|X), P(D|X), P(A|X)
- **Proposed:** Features → XGBoost → E[goals_home|X], E[goals_away|X] → score distribution → P(H), P(D), P(A)

The advantage is that P(D) naturally emerges from the diagonal of the bivariate Poisson grid, meaning draws don't need to be "learned" as a class — they're a mathematical consequence of two teams having similar goal expectations.

#### 2.3 Walk-Forward Validation

**Files:** [split.py](file:///home/arch/Coding/algobet/algobet/predictions/training/split.py), [tuner.py](file:///home/arch/Coding/algobet/algobet/predictions/training/tuner.py)

Replace the single train/val/test split with proper walk-forward validation:

```
Season 1-6: Train  | Season 7: Val  | Season 8: Test  → metrics_1
Season 2-7: Train  | Season 8: Val  | Season 9: Test  → metrics_2
Season 3-8: Train  | Season 9: Val  | Season 10: Test → metrics_3
Average(metrics_1, metrics_2, metrics_3) → reported metrics
```

This provides:
- Honest out-of-sample performance estimates across multiple season boundaries
- Detection of distribution shift (if metrics degrade in later windows, the model doesn't transfer)
- Resistance to survivorship bias (can't cherry-pick the one test season where the model worked)

---

### Phase 3: Advanced Techniques (Higher-risk, industry-grade)

#### 3.1 Market-Residual Modeling

Instead of predicting outcomes directly, model the **residual** between true outcome probability and market-implied probability:

```
target = actual_outcome - odds_implied_probability
model predicts: Δ = P(outcome) - P_market(outcome)
bet when: Δ > threshold
```

This reframes the problem from "predict football matches" (hard, markets are efficient) to "find where the market is wrong" (still hard, but better-defined).

#### 3.2 Venn-Abers Calibration

Replace temperature scaling with Venn-Abers predictors — the only calibration method with theoretical guarantees of validity. Unlike isotonic/sigmoid/temperature calibration:
- Produces valid probability intervals, not point estimates
- Is distribution-free (no parametric assumptions)
- Handles multiclass natively via one-vs-all decomposition
- Cannot "collapse" predictions because it produces intervals, not single values

#### 3.3 Bayesian Score Modeling with Time Decay

Extend the Dixon-Coles model with:
- **Exponential time weighting** (recent matches count more) — standard in production DC implementations
- **Hierarchical priors** on team attack/defense parameters — prevents overfitting for teams with few matches
- **Posterior predictive sampling** via MCMC — produces genuine uncertainty estimates rather than point probabilities

---

## Industry Approaches Comparison

| Approach | Used By | Key Advantage | AlgoBet Status |
|---|---|---|---|
| **Bivariate Poisson + DC ρ** | Pinnacle, Smart Odds, Starlizard | Naturally calibrated draws | ✅ Implemented but sidelined |
| **Market-implied features** | All professional syndicates | Anchors model to efficient market | ❌ Ignored (generators exist but unregistered) |
| **CLV-based evaluation** | All professional syndicates | Honest edge measurement | ❌ Missing entirely |
| **Walk-forward CV** | Standard ML practice | Honest performance estimates | ❌ Single split only |
| **Poisson goal regression** | Academic standard (Maher 1982, DC 1997) | Continuous target, natural draws | ⚠️ Exists as blend option only |
| **Kelly with calibrated probs** | Professional bankroll management | Optimal growth rate | ❌ Used with uncalibrated probs |
| **Native NaN handling** | XGBoost documentation | Better splits on sparse data | ❌ Destroyed by imputer+scaler |
| **Exponential time decay** | Betfair, Pinnacle internal models | Adapts to form changes | ❌ Not implemented |
| **SHAP-based feature audit** | ML best practice | Identifies spurious correlations | ❌ Not implemented |

---

## Open Questions

> [!IMPORTANT]
> These decisions will shape the implementation direction. Please advise before I proceed.

1. **Primary use-case: pre-market or post-market?**
   - If you want to predict matches *before* odds are published → odds-free model, Dixon-Coles primary
   - If you want to find value *after* odds are published → odds-anchored model, market-residual approach
   - Both can coexist, but they have different architectures

2. **Closing odds data availability:**
   - Do you have closing-line odds stored, or only opening odds?
   - CLV tracking requires both; if only opening odds are available, we can approximate using line movement data

3. **Scope of leagues:**
   - Are you targeting EPL only, or expanding to other leagues?
   - Multi-league models need league-specific parameters (home advantage varies significantly across leagues)

4. **Acceptable timeline:**
   - Phase 1 (foundation fixes): ~2-3 hours, low risk
   - Phase 2 (paradigm shift): ~6-8 hours, medium risk
   - Phase 3 (advanced): ~12-16 hours, higher risk
   - Which phases do you want to proceed with?

5. **The 77%→40% gap:**
   - Do you agree this is overfitting that should be addressed, or do you want to maintain the current "profitable overfit" approach?
   - My recommendation: fix the overfitting. The +4.79% ROI on one test season is not statistically significant over 82 bets (p ≈ 0.35 under a null hypothesis of random betting).

---

## Verification Plan

### Automated Tests
- Run `pnpm quality-gates` for frontend (no frontend changes)
- Run existing Python tests: `pytest tests/`
- Add new tests for:
  - Walk-forward CV split correctness
  - Dixon-Coles primary model end-to-end
  - CLV calculation accuracy
  - Native NaN transformer behavior

### Model Validation
- Train 3 models: (a) Current Best — XGBoost (with native NaN handling)
```bash
  curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "split_strategy": "season_aware",
    "train_seasons": 8,
    "val_seasons": 1,
    "test_seasons": 1,
    "feature_groups": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats", "draw_signals", "matchup_interaction"],
    "description": "XGBoost baseline - current best approach"
  }'
```

 (b) DC primary
```bash
    curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "dixon_coles",
    "split_strategy": "season_aware",
    "train_seasons": 8,
    "val_seasons": 1,
    "test_seasons": 1,
    "feature_groups": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats", "draw_signals", "matchup_interaction", "odds"],
    "calibrate_probabilities": false,
    "description": "Dixon-Coles primary with odds features"
  }'

```
 (c) hybrid XGB→Poisson
```bash
    curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "hybrid_poisson",
    "split_strategy": "walk_forward",
    "train_seasons": 6,
    "val_seasons": 1,
    "test_seasons": 1,
    "feature_groups": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats", "draw_signals", "matchup_interaction", "odds"],
    "calibrate_probabilities": false,
    "description": "Hybrid Poisson: XGB→λ_home,λ_away→score distribution"
  }'
```
<!-- Training the 3 Models
(a) Current Best — XGBoost (with native NaN handling)
algobet train run \
  --model-type xgboost \
  --description "XGBoost baseline with native NaN handling"

(b) Dixon-Coles Primary
algobet train run \
  --model-type dixon_coles \
  --description "Dixon-Coles primary model"

(c) Hybrid XGB→Poisson
algobet train run \
  --model-type hybrid_poisson \
  --description "Hybrid Poisson: XGB goals → score distribution" -->


- Evaluate all three with walk-forward CV across 3+ season boundaries
- Compare: accuracy, kappa, ECE, CLV, and ROI per model
- Statistical significance test (DeLong test for AUC, paired t-test for CLV)

### Manual Verification
- Backtest each model on EPL 2024/25 season (held-out)
- Verify draw calibration curve (bin predicted draw probability vs actual draw frequency)
- Spot-check SHAP values for 10 matches to verify feature contributions make football sense

### Key differences between approaches:
 	(a) XGBoost	(b) Dixon-Coles	(c) Hybrid Poisson
Target	H/D/A class	Home/Away goals → H/D/A probs	Home/Away goals → H/D/A probs
Draw handling	Learned as class	Emerges from score grid	Emerges from score grid
NaN handling	Native (PreserveMissingValues)	Native (HistGBTRegressor)	Native
Odds features	Optional	Recommended	Recommended
Calibration	Needed (temperature)	Naturally calibrated	Naturally calibrated
