# Pure ML Improvement Plan Update

This document reviews `PURE-ML-IMPROVEMENT-PLAN.md` and turns it into a
safer implementation plan for the current AlgoBet codebase. It preserves the
constraint that the production model remains pure ML: no odds-derived training
features and no implied-odds blending in training, tuning, calibration, or
prediction. Market odds may be used only for post-hoc diagnostics and betting
simulation.

## Executive Critique

The original plan has the right direction, but it overstates certainty and
under-specifies leakage controls.

Keep:

- OOF stacking is a valid fix for the current stacking path.
- XGBoost plus LightGBM should remain the first production ensemble target.
- CatBoost is worth testing as a diversity learner.
- Calibration must be treated as a first-class acceptance gate.
- Walk-forward or season-aware evaluation should be the default evidence path.

Change:

- Replace fixed expected gains like `+3-5% accuracy` and `ECE < 0.08` with
  falsifiable gates against the current saved baseline. The available evidence
  does not justify guaranteed lift numbers.
- Remove unsupported claims about specific syndicates. They are not needed for
  implementation and are hard to verify.
- Do not implement standard shuffled K-fold OOF for football match data.
  Stacking and calibration OOF must be chronological, expanding-window, or
  season-aware.
- Do not call L2-regularized validation-weight optimization "Bayesian model
  averaging." It is a regularized stacking or soft-vote optimizer unless it
  computes posterior model probabilities under an explicit Bayesian model.
- Defer custom betting loss. It conflicts with the pure-ML boundary if it uses
  market probabilities directly, and XGBoost custom objectives must satisfy
  smoothness and row-additivity requirements.
- Treat MLP and multi-level stacking as experiments after trustworthy OOF and
  calibration exist. They are not quick wins on a small football dataset.
- Treat CatBoost as an ablation, not a guaranteed improvement. Its categorical
  advantage requires preserving raw categorical columns or explicit categorical
  feature metadata.

## Repository Findings

- `algobet/predictions/training/stacking.py` trains base models using `X_val`
  for early stopping, then trains the meta-learner and isotonic calibrator on
  the same validation predictions. This is validation-set overuse, even if the
  base models are not directly fitted on validation labels as training rows.
- `TrainingConfig` already has `use_stacking_ensemble` and
  `stacking_base_models`, but `algobet/api/schemas/ml_operations.py` does not
  expose them. The API `model_type` pattern also excludes `stacking_ensemble`
  and `catboost`.
- `TrainingPipeline._get_oof_probas()` exists, but it uses shuffled
  `StratifiedKFold`. That is not time-safe for match prediction.
- `EnsembleWeightOptimizer` already exists and is used for XGBoost/LightGBM,
  so the next step is not "add an optimizer." The next step is to make the
  validation boundary clean, generalize metadata, and enforce acceptance gates.
- Feature-family retention guards already exist in `TrainingConfig`, so the
  update should build on them instead of re-planning them as completely new.
- Current calibration defaults to `temperature`, while `StackingEnsemble`
  hardcodes isotonic calibration. The hardcoded isotonic path needs a sample
  gate and a validation log-loss/ECE gate.
- Existing docs in `docs/model-training-mediation.md` already cover saved
  feature pipelines, feature audits, per-model tuning, weighted XGB/LGB
  ensembles, and activation rules. This update should merge with that direction
  instead of creating a competing roadmap.

## Online Evidence Used

- scikit-learn's `StackingClassifier` trains the final estimator from
  cross-validated base-estimator predictions and warns that training a stacking
  model on predictions from models fitted on the same data has high overfitting
  risk:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
- scikit-learn's `TimeSeriesSplit` docs state that time-ordered data needs
  time-aware splitting because other CV methods can train on future data and
  evaluate on past data:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn calibration guidance says isotonic calibration generally needs
  enough data, roughly more than 1000 samples, to avoid overfitting:
  https://scikit-learn.org/stable/modules/calibration.html
- XGBoost custom-objective docs require objectives to be smooth, twice
  differentiable, and additive by row:
  https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/advanced_custom_obj.html
- The CatBoost paper supports the ordered-boosting and categorical-feature
  rationale, but it does not guarantee lift on this dataset:
  https://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf
- Cawley and Talbot show that model selection itself can overfit finite
  validation estimates, so nested or untouched final evaluation is required:
  https://www.jmlr.org/papers/v11/cawley10a.html
- Hoeting et al. define Bayesian model averaging as accounting for model
  uncertainty through Bayesian model probabilities, not just regularized
  validation-weight search:
  https://sites.stat.washington.edu/www/research/online/hoeting1999.pdf
- Gneiting and Raftery motivate proper scoring rules for honest probabilistic
  forecasts, which supports keeping log loss/Brier/calibration ahead of raw ROI
  as training objectives:
  https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

## Updated Implementation Plan

### Phase 0: Replace Claims With Gates

Goal: make the plan falsifiable before adding more model complexity.

Implementation:

- Rewrite expected-impact claims as hypotheses.
- Define the baseline model version and EPL scope explicitly.
- Require `activate=false` for every experimental run.
- Keep `tournament_ids: [359]` for EPL-only evaluation.
- Keep the final holdout untouched by tuning, stacking, calibration, and
  ensemble-weight selection.

Acceptance:

- The plan names the baseline model version, tournament scope, date range, and
  final holdout policy.
- The success criteria are relative to baseline log loss, ECE, class diversity,
  and draw/away recall rather than fixed aspirational accuracy numbers.

### Phase 1: Repair OOF And Validation Boundaries

Goal: stop validation-set reuse before adding CatBoost, MLP, or multi-level
stacking.

Implementation:

- Add a shared time-aware OOF splitter utility that can emit expanding-window
  or season-aware folds.
- Replace `StratifiedKFold(shuffle=True)` in `_get_oof_probas()` with this
  splitter.
- Update `StackingEnsemble` to receive already-generated OOF base probabilities
  or a splitter object, instead of training the meta-learner on the same
  validation set used for early stopping.
- Split validation roles for ensemble runs into separate early-stopping,
  weight-selection, and calibration folds.
- Store fold boundaries in model metadata.

Acceptance:

- Unit tests prove every OOF prediction is produced by a model trained only on
  earlier matches or earlier seasons.
- The same row is not used for early stopping, meta-learning, and calibration
  in one training pass.
- Calibration is disabled if it worsens validation log loss or collapses
  predicted classes.

### Phase 2: Expose Existing Ensemble And Stacking Controls Correctly

Goal: make current capabilities reachable through API/UI contracts before
adding new algorithms.

Implementation:

- Add `use_stacking_ensemble`, `stacking_base_models`, `stacking_cv_strategy`,
  and `stacking_n_folds` to `TrainModelRequest`.
- Decide whether `stacking_ensemble` is a `model_type` or a boolean mode. Do
  not support both ambiguous paths.
- Persist `model_type` as `ensemble` or `stacking_ensemble` when the saved
  artifact is an ensemble, not as a misleading base model.
- Return base-model metrics, ensemble weights, stacking metadata, fold policy,
  and calibration metadata in the API response.
- Add frontend type/schema support only after the backend contract is stable.

Acceptance:

- A request can train a stack without hidden server-side flags.
- Saved ensemble artifacts load through the existing registry and predict
  without special-case scripts.
- API response and model registry identify the real artifact type.

### Phase 3: Harden Weighted XGBoost Plus LightGBM

Goal: finish the current highest-value production path before adding CatBoost
or MLP.

Implementation:

- Generalize `EnsembleWeightOptimizer` from two named models to N base models,
  while keeping XGBoost and LightGBM as the production default.
- Rename any proposed "Bayesian" optimizer to `RegularizedEnsembleOptimizer`
  unless a real BMA posterior is implemented.
- Add an equal-weight control and a best-single-base control.
- Require the optimized blend to beat equal-weight and both base models on the
  weight-selection fold.
- Reject blends with fewer than three predicted classes or
  `max_prediction_share > 0.70`.
- Calibrate only after weights are selected, using a separate calibration fold.

Acceptance:

- Weighted XGB/LGB beats XGB-only, LGB-only, and equal-weight XGB/LGB on the
  same clean validation protocol.
- Stored metadata includes base metrics, equal-weight metrics, optimized
  metrics, weights, and rejection reasons for failed blends.

### Phase 4: Implement Time-Aware OOF Stacking

Goal: add stacking only after the fold protocol is correct.

Implementation:

- Use logistic regression as the default meta-learner.
- Add MLP meta-learner as opt-in only, with minimum OOF sample count and early
  stopping.
- Add `passthrough=false` by default. If raw-feature passthrough is tested,
  treat it as a separate ablation because it increases overfitting risk.
- Avoid hardcoded isotonic calibration in `StackingEnsemble`; use the pipeline
  calibration policy and sample gates.
- Store OOF fold metrics and meta-learner coefficients/importances.

Acceptance:

- Stacking beats weighted soft-vote on clean validation and final holdout before
  it can be considered for activation.
- MLP must beat logistic under the same fold protocol before it becomes a
  default option.

### Phase 5: Add CatBoost As A Controlled Ablation

Goal: test CatBoost for model diversity without assuming it is superior.

Implementation:

- Add `catboost` as an optional dependency and predictor type.
- Add `CatBoostPredictor` to `classifiers.py`, `classifier_factory.py`,
  exports, API schema, CLI choices, and model loading.
- Add `catboost` to native-missing-value handling only after verifying the
  wrapper handles NaN consistently.
- Add a feature-pipeline path for raw categorical fields if using CatBoost's
  categorical advantage. If only engineered numeric features are passed, record
  the experiment as numeric-only CatBoost.
- Include CatBoost in stacking or soft-vote only after it beats or diversifies
  against XGB/LGB on OOF predictions.

Acceptance:

- CatBoost trains and reloads from the registry.
- Numeric-only CatBoost and categorical-aware CatBoost are tracked as separate
  ablations if both are tested.
- CatBoost is included in an ensemble only if it improves validation log loss
  or reduces correlated errors without hurting calibration.

### Phase 6: Calibration And Decision Layer

Goal: protect probability quality before any ROI optimization.

Implementation:

- Keep temperature or sigmoid calibration as the default for small calibration
  folds.
- Permit isotonic or Venn-Abers only when calibration samples are large enough
  and validation log loss/ECE improve.
- Store calibration sample count, method, before/after log loss, before/after
  ECE, and predicted-class distribution.
- Keep ROI, Kelly, and CLV as downstream evaluation metrics, not model-training
  objectives, until probability calibration is stable.

Acceptance:

- A calibrated model is never saved if calibration worsens log loss or causes
  class collapse.
- Final model reports both forecast metrics and betting simulation metrics, but
  activation is gated first by proper probabilistic metrics.

### Phase 7: Recency And Temporal Ensembling

Goal: handle distribution shift with the lowest-complexity time-aware method
that works.

Implementation:

- First test recency sample weighting inside XGBoost and LightGBM.
- Then test windowed models: recent seasons, medium window, all-history model.
- Tune window weights on a clean weight-selection fold.
- Use walk-forward evaluation as the primary evidence.

Acceptance:

- Recency weighting or temporal ensembling beats the non-recency ensemble on
  recent holdout seasons without worsening calibration.
- Window weights are stable across walk-forward folds.

### Phase 8: Custom Betting Loss Research Only

Goal: prevent ROI-chasing from corrupting calibrated probabilities.

Implementation:

- Do not include this in the production path until Phases 1-7 pass.
- If tested, start with a custom evaluation metric or sample-weighting
  experiment, not a custom XGBoost objective.
- Do not use market probabilities in the training objective for a pure-ML model.
- If a custom objective is later added, prove it is smooth, twice
  differentiable, row-additive, and numerically stable for multiclass outputs.

Acceptance:

- Custom-loss experiments must beat the proper-scoring baseline on final
  holdout log loss/ECE or remain research-only.
- ROI improvement alone is not sufficient for activation.

## Revised Verification Matrix

Run these before any live training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/unit/predictions/test_training_pipeline.py \
  tests/unit/predictions/test_training_tuner.py \
  tests/unit/predictions/test_training_classifiers.py \
  tests/predictions/test_feature_pipeline.py \
  -q
```

Run these after API contract changes:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/integration/test_ml_operations.py \
  tests/integration/test_predictions_router.py \
  -q
```

Run these after frontend contract changes:

```bash
cd frontend
pnpm typecheck
pnpm test -- lib/types components/models
```

Run live Docker verification only after unit/integration tests pass:

```bash
docker exec algobet-api python scripts/evaluate_epl_model.py \
  --model-version <baseline_or_candidate> \
  --tournament-id 359
```

## Updated Activation Rule

Activate only if all are true on the untouched EPL final holdout:

- Candidate test log loss beats the named baseline.
- Candidate ECE is not worse than baseline and is below the agreed threshold.
- All three classes are predicted.
- `max_prediction_share <= 0.70`.
- Draw recall and away recall are both non-zero.
- Calibration did not worsen validation log loss.
- The model has no odds-derived training features.
- Saved metadata includes feature pipeline path, selected features,
  feature-family counts, fold policy, base-model metrics, ensemble weights, and
  calibration report.

## Immediate Next Implementation Order

1. Replace shuffled OOF calibration with time-aware OOF.
2. Split validation responsibilities for early stopping, ensemble weights, and
   calibration.
3. Expose and persist stacking/ensemble contract fields correctly.
4. Harden and generalize the existing XGB/LGB weighted ensemble.
5. Add time-aware OOF stacking with logistic meta-learner.
6. Add CatBoost as an optional ablation.
7. Test MLP, temporal ensembling, and custom betting loss only after the above
   evidence path is trustworthy.
