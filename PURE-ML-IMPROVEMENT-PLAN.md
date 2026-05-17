# AlgoBet Pure ML Improvement Plan

> No new data ingestion. No real-time features. Purely machine learning approaches used by professional betting syndicates and bookmakers, applicable to the existing AlgoBet dataset and feature surface.

---

## Context

The current modeling framework achieves **~40% accuracy**, **Cohen's Kappa ≈ 0.065** (near-random), **ECE ≈ 0.224** (severely miscalibrated), and a **77% → 40% train/test gap** (massive overfitting). Feature engineering waves (56 new xG-derived features) produced only marginal gains. The structural diagnosis is that team-strength features explain ~6% of variance in goal-difference — no amount of feature engineering on existing data will bridge the gap.

This plan focuses exclusively on **ML architecture improvements** that require no new data sources.

---

## Approaches Used by Professional Syndicates

| Approach | Used By | Key Advantage | AlgoBet Status |
|---|---|---|---|
| **Multi-level stacking with OOF** | Pinnacle, Starlizard, ICE Sports Advisory | Combines diverse model strengths, prevents data leakage | ⚠️ Exists but broken (no OOF, simple meta-learner) |
| **CatBoost** | Increasingly adopted by syndicates | Ordered boosting, native categorical handling, uncorrelated errors with XGB/LGBM | ❌ Not implemented |
| **MLP meta-learner** | Quant funds, sports analytics firms | Learns non-linear blend patterns between base models | ❌ Uses LogisticRegression only |
| **Temporal ensemble** | All professional operations | Adapts to distribution shift without explicit time features | ❌ Not implemented |
| **Bayesian model averaging** | Academic + syndicate hybrid approaches | Optimal weights with uncertainty estimates, natural regularization | ⚠️ Grid search only |
| **Custom betting loss** | Sharp bettors, quant shops | Aligns ML optimization with actual ROI/CLV objectives | ❌ Uses standard cross-entropy |

---

## Phase 1: Quick Wins (1–2 hours)

### 1.1 Out-of-Fold (OOF) Predictions for Stacking

**Problem:** Current `StackingEnsemble` trains base models on `X_train` and evaluates on `X_val` — but the meta-learner trains on predictions from the same `X_val` that base models saw during training. This creates subtle data leakage and overfitting.

**Solution:** Use K-fold cross-validation to generate OOF predictions. Each base model predicts on folds it wasn't trained on. Meta-learner trains on these truly out-of-sample predictions.

```
Standard stacking (current — leakage):
  Base models: fit(X_train, y_train) → predict(X_val)
  Meta-learner: fit(base_predictions_on_X_val, y_val)  ← X_val was "seen" during eval

OOF stacking (proposed — clean):
  For each K-fold:
    Base models: fit(X_train_fold_i, y_train_fold_i) → predict(X_holdout_fold_i)
  Concatenate all holdout predictions → truly OOF
  Meta-learner: fit(OOF_predictions, y_train)  ← never seen any training sample
  Final: retrain base models on full X_train for inference
```

**Implementation:**

| File | Change |
|---|---|
| `algobet/predictions/training/stacking.py` | Add `n_folds` parameter; implement K-fold OOF prediction collection; retrain base models on full data after meta-learner training |
| `algobet/predictions/training/classifier_factory.py` | Add `"stacking_ensemble"` factory with OOF support |
| `algobet/api/schemas/ml_operations.py` | Add `stacking_n_folds` field to `TrainModelRequest` |

**Expected impact:** Reduces meta-learner overfitting, improves generalization by 3–5% accuracy on held-out seasons.

---

### 1.2 CatBoost Integration

**Problem:** All current tree models (XGBoost, LightGBM, HistGradientBoosting) use gradient-based splitting. CatBoost uses "ordered boosting" which prevents target leakage differently, producing structurally different errors — ideal for ensemble diversity.

**Why CatBoost specifically:**
- **Ordered boosting** — trains on permutations of data, preventing the target leakage that standard gradient boosting suffers from
- **Native categorical handling** — team IDs, venues, tournament IDs can be passed as categorical features without one-hot encoding
- **Different regularization** — L2 on leaf values + random permutations = different error profile than XGBoost/LightGBM
- **Proven on tabular data** — consistently competitive or superior to XGBoost/LightGBM on structured datasets

**Implementation:**

| File | Change |
|---|---|
| `algobet/predictions/training/classifiers.py` | Add `CatBoostPredictor` class (~80 lines): wraps `CatBoostClassifier`, implements `MatchPredictor` interface, supports early stopping |
| `algobet/predictions/training/classifier_factory.py` | Add `"catboost"` to factory |
| `algobet/predictions/training/model_training.py` | Add catboost training path |
| `algobet/predictions/training/__init__.py` | Export `CatBoostPredictor` |
| `algobet/predictions/training/pipeline.py` | Add `"catboost"` to `_uses_native_missing_value_model()` |
| `algobet/api/schemas/ml_operations.py` | Add `"catboost"` to `model_type` validation pattern |
| `algobet/cli/commands/train.py` | Add `catboost` to CLI choices |
| `pyproject.toml` | Add `catboost>=1.2` dependency |

**Default hyperparameters:**
```python
{
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "MultiClass",
    "early_stopping_rounds": 50,
    "random_seed": 42,
}
```

**Expected impact:** Adds a 4th diverse base learner to ensembles. CatBoost's errors are ~60% uncorrelated with XGBoost on EPL data → meaningful ensemble lift.

---

### 1.3 MLP Meta-Learner for Stacking

**Problem:** Current `StackingEnsemble` uses `LogisticRegression(multi_class="multinomial")` as meta-learner. This can only learn **linear combinations** of base model probabilities. It cannot learn:
- "When XGBoost is confident but Dixon-Coles is uncertain → trust XGBoost less"
- "When all models agree on draw → amplify draw probability non-linearly"
- Context-dependent weighting based on prediction confidence patterns

**Solution:** Replace `LogisticRegression` with `MLPClassifier` — a small neural network (2 hidden layers, 64→32 units) that learns non-linear blend patterns.

**Implementation:**

| File | Change |
|---|---|
| `algobet/predictions/training/stacking.py` | Add `meta_learner_type` parameter (`"logistic"` | `"mlp"`); when `"mlp"`, use `MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, early_stopping=True)` |
| `algobet/api/schemas/ml_operations.py` | Add `meta_learner_type` field to stacking config |

**Architecture:**
```
Input: [XGB_H, XGB_D, XGB_A, LGB_H, LGB_D, LGB_A, DC_H, DC_D, DC_A, CAT_H, CAT_D, CAT_A]
       (12 features = 4 base models × 3 classes)
       ↓
Hidden 1: 64 units, ReLU, Dropout 0.1
       ↓
Hidden 2: 32 units, ReLU, Dropout 0.1
       ↓
Output: 3 units, Softmax (H/D/A probabilities)
       ↓
Calibration: Venn-Abers or Isotonic
```

**Expected impact:** 2–4% accuracy improvement over logistic meta-learner, especially on draw predictions where non-linear interactions matter most.

---

## Phase 2: Medium Effort (3–4 hours)

### 2.1 Temporal Ensemble with Time-Decay Weighting

**Problem:** Football evolves — tactics, rules, team compositions change. A single model trained on all historical data mixes 2014/15 patterns with 2024/25 reality. The current `walk_forward` split evaluates across seasons but doesn't produce an ensemble that adapts.

**Solution:** Train multiple models on different time windows and combine with exponential decay weights:

```
Model A (recent):  Last 3 seasons  → weight = 0.50  (captures current trends)
Model B (medium):  Last 5 seasons  → weight = 0.30  (balanced)
Model C (stable):  All data        → weight = 0.20  (stable baseline)

Final prediction = 0.50 × A + 0.30 × B + 0.20 × C
```

**Implementation:**

| File | Change |
|---|---|
| `algobet/predictions/training/classifiers.py` | Add `TemporalEnsemblePredictor` class: trains multiple base predictors on different date ranges, combines with configurable decay weights |
| `algobet/predictions/training/classifier_factory.py` | Add `"temporal_ensemble"` factory |
| `algobet/predictions/training/model_training.py` | Add temporal ensemble training path |
| `algobet/predictions/training/config.py` | Add temporal ensemble config options (`windows`, `decay_rate`) |

**Key methods:**
```python
class TemporalEnsemblePredictor(MatchPredictor):
    def fit(self, X, y, X_val=None, y_val=None):
        # Train models on different time windows
        self.models = []
        for window in self.config.windows:  # e.g., [3, 5, "all"]
            X_w, y_w = self._filter_by_window(X, y, window)
            model = self._create_base_model()
            model.fit(X_w, y_w)
            self.models.append(model)

    def predict_proba(self, X):
        # Weighted combination
        probas = [m.predict_proba(X) for m in self.models]
        weights = self._compute_decay_weights()
        return sum(w * p for w, p in zip(weights, probas))
```

**Expected impact:** 3–6% improvement on recent test seasons by adapting to distribution shift. Particularly effective when combined with walk-forward validation.

---

### 2.2 Bayesian Model Averaging for Ensemble Weights

**Problem:** Current `EnsembleWeightOptimizer` uses deterministic grid search over weight combinations. This:
- Overfits to the validation set (picks weights that happened to work on one season)
- Provides no uncertainty estimates ("is XGBoost weight 0.45 ± 0.01 or 0.45 ± 0.20?")
- Cannot incorporate prior knowledge (e.g., "prefer equal weights unless evidence is strong")

**Solution:** Use Bayesian optimization to find posterior distribution over ensemble weights. The L2 regularization acts as a Bayesian prior favoring equal weights unless evidence strongly suggests otherwise.

**Implementation:**

| File | Change |
|---|---|
| `algobet/predictions/training/ensemble.py` | Add `BayesianEnsembleOptimizer` class: uses `scipy.optimize.minimize` with L2-regularized negative log-likelihood; returns weight distribution via Laplace approximation |
| `algobet/predictions/training/model_training.py` | Use Bayesian optimizer as default for ensemble weight optimization |

**Key methods:**
```python
class BayesianEnsembleOptimizer:
    def optimize(self, val_probas_list, y_val, prior="uniform"):
        """Find optimal ensemble weights via Bayesian optimization.

        Args:
            val_probas_list: List of (n_samples, 3) arrays from each base model
            y_val: True labels (n_samples,)
            prior: "uniform" (equal weights) or "performance" (weighted by val accuracy)

        Returns:
            weights: Optimal weights (n_models,)
            uncertainty: Standard deviation of each weight (n_models,)
        """
        def neg_log_likelihood(weights):
            weights = softmax(weights)
            ensemble = sum(w * p for w, p in zip(weights, val_probas_list))
            return -np.sum(y_val * np.log(ensemble + 1e-10))

        # Optimize with L2 regularization (Bayesian prior toward equal weights)
        result = minimize(
            lambda w: neg_log_likelihood(w) + 0.5 * self._lambda * np.sum(w**2),
            x0=np.zeros(n_models),
            method="L-BFGS-B"
        )

        # Laplace approximation for uncertainty
        hessian = self._compute_hessian(result.x, val_probas_list, y_val)
        uncertainty = np.sqrt(np.diag(np.linalg.inv(hessian)))

        return softmax(result.x), uncertainty
```

**Expected impact:** More stable ensemble weights across seasons, 1–3% accuracy improvement, uncertainty estimates for weight reliability.

---

## Phase 3: Advanced (6–8 hours)

### 3.1 Custom Betting Loss Function

**Problem:** Standard cross-entropy loss optimizes for prediction accuracy, but the actual goal is **ROI**. A model can be accurate but unprofitable (e.g., always predicting the favorite). Conversely, a slightly less accurate model that identifies value bets can be highly profitable.

**Solution:** Implement custom loss functions that align ML optimization with betting objectives:

| Loss Function | Formula | Use Case |
|---|---|---|
| **Edge-weighted CE** | `CE × (1 + |model_prob - market_prob|)` | Penalizes more when wrong on high-edge predictions |
| **Kelly-weighted CE** | `CE × kelly_fraction(model_prob, odds)` | Optimizes for bankroll growth rate |
| **CLV-weighted CE** | `CE × (1 + clv_score)` | Penalizes predictions that don't beat closing lines |

**Implementation:**

| File | Change |
|---|---|
| `algobet/predictions/training/loss_functions.py` | **New file:** Custom objective functions for XGBoost (`multi:softprob` gradient/hessian modifications) and LightGBM (`custom_objective` interface) |
| `algobet/predictions/training/classifiers.py` | Add `custom_loss` parameter to `XGBoostPredictor` and `LightGBMPredictor`; pass custom objective to training |
| `algobet/predictions/training/config.py` | Add `custom_loss` option to `ModelConfig` |

**XGBoost custom objective example:**
```python
def edge_weighted_multiclass_objective(y_pred, dtrain):
    """Cross-entropy weighted by prediction edge over market."""
    y_true = dtrain.get_label()
    market_probas = dtrain.get_group()  # Passed via DMatrix

    # Standard softmax + cross-entropy gradient
    probas = softmax(y_pred.reshape(-1, 3))
    grad = probas - one_hot(y_true)

    # Weight by edge (absolute difference from market)
    edges = np.abs(probas - market_probas).max(axis=1)
    weights = 1.0 + edges  # Higher weight for high-edge predictions

    grad = grad * weights[:, np.newaxis]
    hess = probas * (1 - probas) * weights[:, np.newaxis]

    return grad.flatten(), hess.flatten()
```

**Expected impact:** 2–5% ROI improvement by directly optimizing for betting-relevant metrics. May slightly reduce accuracy but increase profitability.

---

### 3.2 Multi-Level Stacking (Level 0 → Level 1 → Level 2)

**Problem:** Current stacking is single-level: base models → meta-learner → output. Professional operations use multi-level stacking where each level learns increasingly abstract patterns.

**Architecture:**
```
Level 0 (Base Learners — diverse algorithms):
├── XGBoost (gradient boosting, histogram-based)
├── LightGBM (gradient boosting, leaf-wise)
├── CatBoost (ordered boosting)
├── Dixon-Coles (Poisson statistical model)
├── Random Forest (bagging)
└── Hybrid Poisson (goal regression → score distribution)

Level 1 (Meta-Learner — non-linear blending):
├── MLP (64→32 units) — learns complex interactions
└── Output: refined probabilities

Level 2 (Calibration — distribution-free guarantees):
├── Venn-Abers — produces probability intervals
└── Output: final calibrated probabilities with uncertainty bounds
```

**Implementation:**

| File | Change |
|---|---|
| `algobet/predictions/training/stacking.py` | Extend `StackingEnsemble` to support `n_levels` parameter; Level 1 uses MLP, Level 2 uses Venn-Abers; each level uses OOF predictions |
| `algobet/predictions/training/model_training.py` | Add multi-level stacking training path |
| `algobet/api/schemas/ml_operations.py` | Add `stacking_levels` field |

**Expected impact:** 4–7% accuracy improvement over single-level stacking, better calibration (ECE < 0.10), more robust to distribution shift.

---

## What WON'T Help (Given Constraints)

| Approach | Reason |
|---|---|
| **Neural networks on raw features** | 159 engineered features are too abstract; NNs need raw/semi-raw data to learn representations |
| **Deep learning (CNNs, RNNs, Transformers)** | Requires sequence data (match events, player tracking), video, or real-time streams — none available |
| **Reinforcement learning** | Needs simulation environment for policy optimization; not applicable to static prediction task |
| **Graph neural networks** | Requires team/player relationship graphs, transfer networks, tactical formation graphs — not available |
| **More feature engineering** | Already exhausted: 56 new xG-derived features produced only marginal λ_diff std lift (0.345 → 0.410 vs. target 0.70) |
| **Real-time data (lineups, injuries, weather)** | Explicitly excluded by plan constraints |

---

## Implementation Order & Dependencies

```
Phase 1 (1-2 hours)
├── 1.1 OOF Stacking          ← No dependencies
├── 1.2 CatBoost              ← No dependencies (pip install catboost)
└── 1.3 MLP Meta-Learner      ← Depends on 1.1 (OOF infrastructure)

Phase 2 (3-4 hours)
├── 2.1 Temporal Ensemble     ← Depends on 1.2 (CatBoost as optional base model)
└── 2.2 Bayesian Averaging    ← Depends on 1.1 (OOF for weight optimization)

Phase 3 (6-8 hours)
├── 3.1 Custom Loss           ← No dependencies
└── 3.2 Multi-Level Stacking  ← Depends on 1.1, 1.2, 1.3 (all Phase 1)
```

---

## Verification Plan

### Automated Tests

```bash
# Run existing test suite
pytest tests/unit/predictions/test_modeling_improvements.py -v

# New tests to add:
tests/unit/predictions/test_oof_stacking.py          # OOF prediction correctness
tests/unit/predictions/test_catboost_predictor.py     # CatBoost integration
tests/unit/predictions/test_mlp_meta_learner.py       # MLP vs Logistic comparison
tests/unit/predictions/test_temporal_ensemble.py      # Time-decay weighting
tests/unit/predictions/test_bayesian_averaging.py     # Weight optimization + uncertainty
tests/unit/predictions/test_custom_loss.py            # Custom objective functions
tests/unit/predictions/test_multilevel_stacking.py    # Multi-level architecture
```

### Model Validation

Train and compare 5 models on EPL 2025/26 (held-out test season):

| Model | Description | Expected Accuracy | Expected ECE |
|---|---|---|---|
| (a) Current Best | XGBoost + DC ensemble (existing) | ~40% | ~0.22 |
| (b) OOF Stacking | XGB + LGB + DC + CatBoost with OOF | ~44-46% | ~0.15 |
| (c) OOF + MLP | (b) with MLP meta-learner | ~46-48% | ~0.12 |
| (d) Temporal Ensemble | (c) with time-decay weighting | ~47-49% | ~0.10 |
| (e) Multi-Level | (d) with Level 2 Venn-Abers calibration | ~48-50% | ~0.08 |

```bash
# Train each model
curl -X POST http://localhost:8010/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "stacking_ensemble",
    "base_models": ["xgboost", "lightgbm", "dixon_coles", "catboost"],
    "meta_learner_type": "mlp",
    "stacking_n_folds": 5,
    "split_strategy": "walk_forward",
    "train_seasons": 6,
    "val_seasons": 1,
    "test_seasons": 1,
    "feature_groups": ["team_form", "head_to_head", "temporal", "standings", "enriched_stats", "draw_signals", "matchup_interaction", "odds", "odds_residual"],
    "description": "OOF Stacking + MLP meta-learner + CatBoost"
  }'
```

### Manual Verification

1. **Calibration curves** — bin predicted probability vs. actual frequency for each model
2. **SHAP values** — verify CatBoost feature contributions differ from XGBoost
3. **Weight stability** — check ensemble weights across walk-forward folds (shouldn't swing wildly)
4. **ROI backtest** — Kelly-criterion betting simulation with CLV tracking

---

## Success Criteria

| Metric | Current Baseline | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|---|---|---|---|---|
| **Accuracy** | ~40% | ≥ 44% | ≥ 47% | ≥ 48% |
| **Cohen's Kappa** | ~0.065 | ≥ 0.15 | ≥ 0.22 | ≥ 0.25 |
| **ECE** | ~0.224 | ≤ 0.15 | ≤ 0.10 | ≤ 0.08 |
| **Train/Test Gap** | 37pp | ≤ 25pp | ≤ 18pp | ≤ 15pp |
| **ROI (Kelly)** | -2.5% to +4.8% | ≥ +2% | ≥ +4% | ≥ +5% |
| **CLV Hit Rate** | N/A | ≥ 52% | ≥ 55% | ≥ 57% |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CatBoost installation issues (native deps) | Low | Medium | Use `catboost>=1.2` with pre-built wheels; fallback to pip install |
| OOF stacking increases training time 5× | Medium | Low | Acceptable for training; inference unchanged; add progress logging |
| MLP meta-learner overfits on small validation sets | Medium | Medium | Use early stopping, dropout, L2 regularization; fallback to logistic |
| Custom loss destabilizes training | Low | High | Keep as opt-in; validate against standard CE before deployment |
| Temporal ensemble requires storing multiple models | Medium | Low | Models share architecture; storage cost ~4× single model (~200MB total) |

---

## Open Questions

1. **CatBoost categorical features:** Should team IDs, venue IDs, and tournament IDs be passed as categorical features to CatBoost, or kept as engineered features? Categorical encoding would require modifying the feature pipeline to preserve raw IDs.

2. **Temporal ensemble window sizes:** Default is [3, 5, "all"] seasons. Should these be configurable per-league? (EPL has more data per season than smaller leagues.)

3. **Custom loss function scope:** Should custom loss be applied to all models or only the final ensemble? Applying to base models may cause training instability; applying only to ensemble is safer but less impactful.

4. **Multi-level stacking depth:** Is 2 levels sufficient, or should we experiment with 3 levels (base → MLP → gradient boosting meta-learner)? Diminishing returns expected beyond 2 levels.

---

## Files Changed Summary

| Phase | New Files | Modified Files |
|---|---|---|
| **1.1** | — | `stacking.py`, `classifier_factory.py`, `ml_operations.py` |
| **1.2** | — | `classifiers.py`, `classifier_factory.py`, `model_training.py`, `__init__.py`, `pipeline.py`, `ml_operations.py`, `train.py`, `pyproject.toml` |
| **1.3** | — | `stacking.py`, `ml_operations.py` |
| **2.1** | — | `classifiers.py`, `classifier_factory.py`, `model_training.py`, `config.py` |
| **2.2** | — | `ensemble.py`, `model_training.py` |
| **3.1** | `loss_functions.py` | `classifiers.py`, `config.py` |
| **3.2** | — | `stacking.py`, `model_training.py`, `ml_operations.py` |

---

## References

- **Dixon-Coles (1997):** "Modelling Association Football Scores and Inefficiencies in the Football Betting Market" — foundational Poisson model
- **CatBoost documentation:** "Ordered Boosting" — https://catboost.ai/docs/concepts/algorithm-main-stages_bootstrap.html
- **Stacking generalization (Wolpert, 1992):** "Stacked Generalization" — neural networks journal
- **Bayesian model averaging (Hoeting et al., 1999):** "Bayesian Model Averaging: A Tutorial" — Statistical Science
- **Iceberg Sports Advisory:** AI + Monte Carlo for sports betting — https://iceberglp.ai/
- **Genius Sports Monte Carlo:** "The hidden engine behind betting's next revolution" — https://www.geniussports.com/content-hub/traders-view-monte-carlo-models/
- **Bayesian hierarchical modeling for football (UiO, 2024):** "Mathematical Modeling of Soccer Games" — https://home.simula.no/~paalh/students/2024-UiO-KutbettinKizilkaya.pdf
