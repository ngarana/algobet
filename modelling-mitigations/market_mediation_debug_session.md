# Market Mediation Model — Debug Session Report

**Branch:** `feat/pure-ml-improvements`
**Date:** 2026-05-18
**Starting model version:** `market_mediation_20260517_191919` (model_id 234)
**Final working model:** `market_mediation_20260518_042022` (model_id 243)

---

## Starting State

Model 234 produced the following results at the start of the session:

```
selected_bets: 0
abstention_share: 1.0
walk_forward_closing_coverage_min: 0.8214
num_features: 17
mediation_features_in_importance: 3
```

The model was correctly not activated (no selected bets = no CLV to evaluate). The goal was to understand why and fix the root causes.

---

## Bugs Found and Fixed

### Bug 1 — Expected CLV always zero or negative

**File:** `algobet/predictions/training/market_mediation.py`

**Root cause:** The `_expected_clv` formula used the unconditional CLV mean:

```python
# OLD (broken)
positive_mean = max(unconditional_mean, 0)  # ≈ 0 when avg_clv ≈ -0.003
expected_clv = positive_probs * positive_mean  # always near 0
```

With bookmaker margin the unconditional CLV average is slightly negative (~-0.3%). `max(..., 0)` clipped it to zero, making `expected_clv` always ≤ 0, so no bet ever passed `min_expected_clv > 0`.

**Fix:** Store and use conditional means separately:

```python
# E[CLV | CLV > 0] per outcome class ≈ +6%
self._clv_positive_mean_by_class

# E[CLV | CLV ≤ 0] per outcome class ≈ -5.8%
self._clv_negative_mean_by_class

def _expected_clv(self, positive_probs):
    return (
        positive_probs * self._clv_positive_mean_by_class
        + (1.0 - positive_probs) * self._clv_negative_mean_by_class
    )
```

**Actual values from DB data:**
- HOME: +6.1% / -5.8%
- DRAW: +4.8% / -4.3%
- AWAY: +7.8% / -7.7%

---

### Bug 2 — CLV head trained on anti-signal features

**File:** `algobet/predictions/training/market_mediation.py`

**Root cause:** The CLV classifier was originally trained on meta-features that included `residual = pure_probas - market_probas`. When the pure model thinks HOME is undervalued (residual > 0), the market typically corrects by moving HOME odds DOWN further — meaning the market is RIGHT and the pure model is WRONG (pure model log-loss is ~0.07 worse than the market). Residuals are therefore anti-correlated with positive CLV.

**Fix:** Added `_build_clv_features` that excludes pure-model residuals entirely:

```python
def _build_clv_features(self, X, market_probas):
    # CLV head uses only market-structure features.
    # Pure-model residuals are anti-signal: when the pure model thinks
    # the market is wrong, the market is usually right (LL gap ~0.07).
    market_features = X[:, self._market_feature_indices]
    return np.hstack([market_probas, market_features])
```

The CLV head now trains on: `[market_probas(3)] + [mediation_features(21)]` = 24-dimensional input.

---

### Bug 3 — Feature selection discarding all mediation features (main bug)

This was actually **two stacked bugs** producing the same symptom: 17 features total, only 3 mediation features surviving.

#### Bug 3a — X column index mismatch in correlation pruning

**File:** `algobet/predictions/training/feature_selection.py`

**Root cause:** When `min_market_mediation_features > 0`, forced mediation features were removed from `prunable_names` before calling `prune_correlation`. But the full X matrix (N + M missing-indicator columns) was passed unchanged. The `prune_correlation` call then used `prunable_names[i]` alongside `X[:, i]` — those no longer referred to the same features.

```python
# BROKEN: X not sliced to match prunable_names
pruned_names, corr_drops = prune_correlation(prunable_names, X, max_correlation)
```

**Fix:** Build an explicit column index mapping and slice X:

```python
all_name_to_idx = {n: i for i, n in enumerate(feature_names)}
prunable_col_indices = [all_name_to_idx[n] for n in prunable_names]
X_prunable = X[:, prunable_col_indices]
pruned_names, corr_drops = prune_correlation(prunable_names, X_prunable, max_correlation)
```

#### Bug 3b — Uniform importance causes all features to fail the threshold

**File:** `algobet/predictions/training/feature_selection_pipeline.py`

**Root cause:** `MarketMediationPredictor.feature_importance` always returns uniform importance `{name: 1/N for name in feature_names}`. With N ≈ 253 features (across all generators), each feature gets `1/253 ≈ 0.004`. The default `feature_selection_threshold = 0.005` therefore **rejects every single feature** (`0.004 < 0.005`).

The code falls back to:
```python
selected = [max(pruned_names, key=lambda n: normalized[n])]  # only 1 feature
```
Then family guards restore: 3 draw + 3 away + 2 low_scoring + 5 enriched = 13 features. The hard-coded required mediation features add 3. Total: **17 features**.

**Fix:** Use `threshold=0.0` for `market_mediation`, letting correlation pruning and family minimums drive selection:

```python
effective_threshold = (
    0.0
    if self.config.model_type == "market_mediation"
    else self.config.feature_selection_threshold
)
```

**Why 0.0 is safe here:** `MarketMediationPredictor` is a dual-head model (HistGradientBoostingClassifier + Ridge residual + LogisticRegression CLV). It has no tree-based feature ranking. All selection for market_mediation is now driven by correlation pruning (removes redundant features) and family minimum guards (ensures representation).

---

## Results After All Fixes

| Metric | Before | After |
|--------|--------|-------|
| `num_features` | 17 | 205 |
| Mediation features in model | 3 | 21 |
| `selected_bets` (walk-forward pooled) | 0 | 9,775 |
| `clv_hit_rate` | 0.0 | 0.404 |
| `pooled_mean_clv` | 0.0 | -0.0028 |
| `positive_clv_folds` | 0/7 | 1/7 |
| Activation | Not activated | Not activated |

The model now selects bets and computes real CLV, but the activation gates correctly prevent it from going live because the CLV is still negative.

**Mediation features now flowing to the CLV head:**
- `opening_implied_prob_{home,draw,away}` (3)
- `bookmaker_disagreement_{home,draw,away}` (3)
- `max_implied_prob_{home,draw}` (2)
- `opening_overround`, `opening_entropy`, `opening_favorite_prob`, `opening_favorite_outcome`, `opening_home_away_ratio` (5)
- `asian_handicap_{home,away}_odds`, `asian_handicap_line` (3)
- `over_25_odds`, `under_25_odds`, `over_under_implied_total`, `over_under_line` (4)
- `market_data_quality` (1)

---

## Remaining Limitation — CLV Signal

### What the numbers say

Walk-forward (7 folds, season_aware 8+1+1):
- 9,775 selected bets, closing coverage 74.4%
- CLV hit rate: **40.4%** (base rate for random selection: ~46%)
- Pooled mean CLV: **-0.28%**
- Positive CLV folds: **1 out of 7**
- Z-score vs base rate: **~11** (statistically significant, not noise)

### Root cause of negative CLV

**The CLV base rate is 46%, not 50%.** Opening prices are systematically tighter than closing prices because:

1. Bookmakers post opening lines with larger margin (overround)
2. Sharp money bets in, moving the line toward fair value
3. By close, the margin is compressed — odds generally improve for favorites

This means `E[CLV]` at the base rate = `0.46 × 6% + 0.54 × (-5.8%) = 2.76% - 3.13% = -0.37%` — slightly negative. The structural disadvantage of taking opening prices means **finding positive expected CLV requires identifying a subset with hit rate > 50%**, which is 4+ percentage points above base.

### Why market structure features don't work

The CLV classifier currently uses: overround, entropy, bookmaker disagreement, AH/OU odds. None of these predict the *direction* of line movement with sufficient accuracy:

- **High bookmaker disagreement** → uncertain price → could move either way
- **High overround** → more margin → margin compression would help ALL outcomes equally, not a specific direction
- **AH/OU context** → gives information about expected goal totals, not about which way the 1X2 line will move

The classifier gets 40.4% hit rate vs 46% base — it's finding a systematic anti-pattern (betting into markets where the line moves against you), not finding edge.

### Approaches tested that did not work

| Approach | Result |
|----------|--------|
| Invert CLV output (`1 - prob`) | Worse: 38.3% hit rate, 0/7 positive folds |
| Market-structure features without pure residuals | 40.4% hit rate, 1/7 folds |
| Lower `min_expected_clv` threshold | Selects more bets, all with negative CLV |

### What would give positive CLV signal

The CLV gate would require one or more of the following data sources:

| Data Source | Why It Helps |
|-------------|--------------|
| **Intraday line movement** (odds at T-6h, T-2h before kickoff) | Predicts which way the close will move |
| **Asian handicap volume / liquidity** | Sharp money indicator — AH markets attract professionals |
| **Pinnacle opening vs public book opening spread** | Proxy for sharp vs soft book disagreement |
| **Bet365 / Asian market price difference** | Divergence often signals sharp action |
| **Historical team-specific CLV patterns** | Some teams consistently attract or lose sharp money |

Without these, the model cannot outperform a random selection of opening-price bets.

---

## Activation Gate Status

The activation gates correctly prevent deployment:

```python
# training_runner.py — _passes_activation_gate
if positive_folds < 3.0:       # 1/7 → FAILS
    return False
if mean_clv <= 0.0:            # -0.0028 → FAILS
    return False
if lower_95 <= 0.0:            # -0.0046 → FAILS
    return False
```

This is the intended behavior: the model is built to abstain when it has no measurable edge. The gates are working correctly.

---

## Files Modified

| File | Change |
|------|--------|
| `algobet/predictions/training/market_mediation.py` | Conditional CLV means; `_build_clv_features` (market-only); `_expected_clv_lower_bound` (20% haircut) |
| `algobet/predictions/training/feature_selection.py` | X-slicing fix for prunable features before correlation pruning |
| `algobet/predictions/training/feature_selection_pipeline.py` | `threshold=0.0` for market_mediation; `min_market_mediation_features` wired into selection call |
| `algobet/predictions/training/config.py` | `min_market_mediation_features: int = 0` field added |
| `algobet/services/ml_ops/training_runner.py` | Passes `min_market_mediation_features=15` for market_mediation |

---

## Next Steps

1. **Integrate intraday odds data** — If Football-Data.co.uk or a similar source provides mid-week odds snapshots (e.g., Tuesday and Thursday prices for weekend matches), these could serve as a proxy for line movement direction.

2. **Asian Handicap divergence signal** — Compare AH implied probability vs 1X2 implied probability. A systematic mismatch between AH and 1X2 markets is a known sharp-money indicator.

3. **Bookmaker-specific opening spreads** — If Pinnacle and a soft book (Bet365) have different opening prices, bet the direction Pinnacle prices imply.

4. **Alternatively: pure value-bet approach** — Drop the CLV gate and replace with a pure expected-value detector: select matches where the pure model's probability significantly exceeds the best available market price. Accept that this will be less precise but avoids the CLV data limitation.
