# Pure ML Improvements — Implementation Status

> Branch: `feat/pure-ml-improvements`
> Date: 2026-05-17

---

## What Was Achieved

### Phase 1: Time-Aware OOF & Validation Boundaries ✅

| Change | File | Status |
|---|---|---|
| `OOFTimeAwareSplitter` — expanding-window OOF folds | `split.py` | ✅ Implemented |
| Replace `StratifiedKFold` in `_get_oof_probas()` | `runner.py` | ✅ Implemented |
| Fix LightGBM callbacks (`None` in list caused `AttributeError`) | `classifiers.py` | ✅ Fixed |
| Fix `LogisticRegression` `multi_class` param (removed in sklearn 1.7) | `stacking.py` | ✅ Fixed |
| Fix `_is_fitted` timing bug in `StackingEnsemble.fit()` | `stacking.py` | ✅ Fixed |
| Skip refitting already-fitted base models (needed for Dixon-Coles) | `stacking.py` | ✅ Implemented |

### Phase 2: API Schema & Persistence ✅

| Change | File | Status |
|---|---|---|
| `use_stacking_ensemble`, `stacking_base_models`, `stacking_meta_learner`, `stacking_n_folds` | `ml_operations.py` | ✅ Added |
| `stacking_metadata`, `calibration_metadata` in response | `ml_operations.py` | ✅ Added |
| Persist correct `model_type` (`stacking_ensemble`/`ensemble`) in registry | `runner.py` | ✅ Implemented |
| Wire stacking config through `training_runner.py` | `training_runner.py` | ✅ Implemented |

### Phase 3: Blend Rejection Gates ✅

| Change | File | Status |
|---|---|---|
| Existing class-collapse rejection in `EnsembleWeightOptimizer` | `ensemble.py` | ✅ Already existed |
| Collapse guard in `StackingEnsemble` (logs warning if <3 classes) | `stacking.py` | ✅ Implemented |
| Exempt stacking ensembles from test-set collapse check | `runner.py` | ✅ Implemented |

### Phase 4: OOF Stacking with Logistic/MLP Meta-Learner ✅

| Change | File | Status |
|---|---|---|
| `meta_learner_type` parameter (`logistic` or `mlp`) | `stacking.py` | ✅ Implemented |
| LogisticRegression default with isotonic calibration gated by sample count | `stacking.py` | ✅ Implemented |
| MLPClassifier opt-in with `(64, 32)` layers and early stopping | `stacking.py` | ✅ Implemented |

### Phase 5: CatBoost Integration ⚠️ Partial

| Change | File | Status |
|---|---|---|
| `CatBoostPredictor` class | `classifiers.py` | ✅ Implemented |
| Classifier factory entry | `classifier_factory.py` | ✅ Added |
| Exports in `__init__.py` | `__init__.py` | ✅ Added |
| API schema `catboost` in model_type pattern | `ml_operations.py` | ✅ Added |
| CLI choice | `train.py` | ✅ Added |
| Native-NaN handling | `pipeline.py` | ✅ Added |
| `catboost` optional dependency | `pyproject.toml` | ✅ Added |
| Dockerfile install | `Dockerfile.api` | ✅ Added |
| **CatBoost training on EPL data** | — | ❌ **Fails — collapses to 1 class** |

### Phase 6: Calibration Hardening ✅

| Change | File | Status |
|---|---|---|
| Sample gate: isotonic only when ≥1000 samples | `stacking.py` | ✅ Implemented |
| Log-loss gate: disable if calibration worsens validation log-loss | `stacking.py` | ✅ Implemented |
| Collapse gate in stacking (warning, not fatal) | `stacking.py` | ✅ Implemented |

### Test Results ✅

```
128 passed, 13 warnings
```

All pre-commit hooks pass (ruff, ruff-format, mypy).

---

## What's Not Working

### 1. Stacking Ensemble Collapses to 2 Classes (HOME/AWAY)

**Symptom:** The stacking meta-learner (logistic regression) learns to ignore the draw class entirely. On validation and test sets, `predicted_classes = 2` with `max_prediction_share` up to 0.98.

**Root cause:** Draws are ~25% of matches. The logistic regression meta-learner optimizes for overall accuracy, and since HOME wins ~45% and AWAY ~30%, it learns to split only between those two. The draw probability from base models (especially Dixon-Coles) is never high enough to win argmax after the meta-learner applies its coefficients.

**Evidence from stacking feature importance:**
```
xgboost_class_0 (HOME):  0.004  ← almost ignored
xgboost_class_1 (DRAW):  0.133
xgboost_class_2 (AWAY):  0.124
lightgbm_class_0 (HOME): 0.126
lightgbm_class_1 (DRAW): 0.167
lightgbm_class_2 (AWAY): 0.055
dixon_coles_class_0:     0.205  ← DC dominates HOME prediction
dixon_coles_class_1:     0.185
dixon_coles_class_2:     0.140
```

The meta-learner heavily weights Dixon-Coles for HOME predictions but doesn't learn to amplify draw signals.

**Comparison of models on EPL 2025/26 (tournament 359):**

| Model | Train Acc | Val Acc | Test Acc | Test Log Loss | Test Classes | Test ECE | Market Log Loss |
|---|---|---|---|---|---|---|---|
| XGBoost baseline | 0.517 | 0.403 | 0.425 | 1.081 | 3 | 0.076 | 1.007 |
| Stacking (XGB+LGB+DC) | 0.577 | 0.418 | 0.423 | 1.084 | 2 | 0.102 | 1.007 |
| Stacking (XGB+LGB) | 0.478 | 0.413 | 0.428 | 1.084 | 2 | 0.084 | 1.007 |

The stacking ensembles do **not** beat the XGBoost baseline on test accuracy or log loss. The draw collapse explains why — by ignoring draws, the model loses ~25% of potential correct predictions.

### 2. CatBoost Collapses to 1 Class (All HOME)

**Symptom:** CatBoost predicts HOME for every single match in validation (380/380). Even with heavy regularization (`depth=3`, `l2_leaf_reg=20.0`, `iterations=300`) and class weights, it collapses.

**Root cause:** CatBoost's ordered boosting, combined with the heavy class imbalance (HOME ~45%, DRAW ~25%, AWAY ~30%), causes it to converge on the majority class. The feature surface (123 engineered features) doesn't provide enough signal for CatBoost to differentiate between classes. CatBoost is known to be aggressive on tabular data with imbalanced classes.

**Attempted fixes (all failed):**
- `depth=6, l2=3.0, iterations=1000` → collapsed to 1 class
- `depth=4, l2=10.0, iterations=500` → collapsed to 1 class
- `depth=3, l2=20.0, iterations=300` → collapsed to 1 class
- Added `class_weights` from `outcome_balance` config → collapsed to 1 class

**Conclusion:** CatBoost as a standalone model does not work with the current feature surface and class distribution. It may still be useful as a base learner in an ensemble if its predictions are constrained by a meta-learner, but even that failed in stacking.

### 3. No Model Beats the Market

None of the trained models achieve a test log loss lower than the market log loss (1.007). This is consistent with the documented finding that team-strength features explain only ~6% of variance in goal-difference.

---

## Why Stacking Doesn't Help

The `IMPLEMENTATION_SUMMARY.md` documents that:
> "team-strength features can't predict draws because they can't predict goal magnitude, not just because of the H/D/A categorical structure."

Stacking combines multiple models that all suffer from the same fundamental limitation: **the features don't carry enough per-match signal to predict draws**. Combining three models that all fail at draw prediction doesn't produce a model that predicts draws — it produces a model that fails at draw prediction with slightly different coefficients.

The stacking meta-learner (logistic regression) correctly learns that the draw signal from base models is too noisy to rely on, so it optimizes for HOME/AWAY accuracy instead. This is rational behavior from the meta-learner's perspective, but it means stacking provides no improvement over the best single model.

---

## What Would Be Needed to Make Stacking Work

1. **Draw-specific features that actually work** — The current draw signal features (`goal_convergence`, `strength_parity`, `xg_parity`) are transformations of team-strength metrics. They can't add genuinely new information about whether a match will be tied.

2. **New data sources** — As documented in the decision tree:
   - Lineup absences (injuries, suspensions)
   - Closing-line movement
   - Set-piece xG
   - Weather conditions
   - Referee tendencies

3. **Different meta-learner** — A meta-learner that explicitly optimizes for draw recall (e.g., weighted loss that penalizes missed draws more heavily) might force the stacking to preserve draw predictions, but this would likely hurt overall accuracy.

---

## Current Best Model

The **XGBoost baseline** remains the best model:
- Test accuracy: **42.5%**
- Test log loss: **1.081** (vs. market 1.007)
- Predicts all 3 classes
- ECE: 0.076

This is consistent with the documented ~40% accuracy baseline from previous sessions.

---

## Files Changed

| File | Lines Changed | Description |
|---|---|---|
| `algobet/predictions/training/split.py` | +78 | `OOFTimeAwareSplitter` |
| `algobet/predictions/training/stacking.py` | +260 | Time-aware OOF, MLP meta-learner, calibration gates |
| `algobet/predictions/training/classifiers.py` | +160 | `CatBoostPredictor`, LightGBM callback fix |
| `algobet/predictions/training/runner.py` | +30 | Time-aware OOF calibration, stacking exemption, correct model_type persistence |
| `algobet/predictions/training/classifier_factory.py` | +3 | CatBoost factory entry |
| `algobet/predictions/training/__init__.py` | +4 | CatBoost + OOFTimeAwareSplitter exports |
| `algobet/predictions/training/pipeline.py` | +8 | CatBoost in native-NaN set |
| `algobet/predictions/training/config.py` | +3 | Stacking config fields |
| `algobet/api/schemas/ml_operations.py` | +15 | Stacking API fields, CatBoost in pattern |
| `algobet/services/ml_ops/training_runner.py` | +25 | Stacking config wiring, traceback logging |
| `algobet/cli/commands/train.py` | +8 | CatBoost CLI choice |
| `pyproject.toml` | +4 | CatBoost optional dependency |
| `Dockerfile.api` | +1 | CatBoost install |

---

## Next Steps (If Pursued)

1. **Accept draw prediction limitation** — Document that the model is a HOME/AWAY binary classifier with draw probability as a byproduct. Adjust evaluation metrics accordingly (e.g., binary accuracy on H vs A, ignoring draws).

2. **Focus on binary classification** — Reframe the problem as HOME vs NOT-HOME (or AWAY vs NOT-AWAY) where the model actually has signal. This would likely improve accuracy to ~55-60% on the binary task.

3. **Invest in new data** — As documented in the decision tree, feature engineering on existing data is exhausted. The next lever is genuinely new data ingestion.

4. **Abandon CatBoost** — It doesn't work with the current feature surface. The ordered boosting that makes it powerful on other tabular datasets causes it to collapse on this imbalanced, low-signal football prediction task.
