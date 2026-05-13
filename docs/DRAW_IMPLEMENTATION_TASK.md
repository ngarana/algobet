# Draw Correction Limitations — Task List

## Limitation 1: Feature Set Constraint (Collinearity)
- [ ] Create `DrawSignalFeatureGenerator` in `algobet/predictions/features/draw_signal_generator.py`
- [ ] Register `draw_signals` in `composite.py` factory
- [ ] Add `"draw_signals"` to `ALLOWED_FEATURE_GROUPS` in `config.py`
- [ ] Add `"draw_signal"` family to `FEATURE_FAMILIES` in `feature_selection.py`

## Limitation 2: Match-Specific Dixon-Coles
- [ ] Make `DixonColesPredictor.fit()` functional (delegate to `fit_with_scores`)
- [ ] Modify `runner.py` Step 6c — inline DC training with real features
- [ ] Update `scripts/train_dixon_coles.py` to use real features

## Limitation 3: Stacking Ensemble for Draw Recall
- [ ] Create `StackingEnsemble` in `algobet/predictions/training/stacking.py`
- [ ] Add stacking config fields to `config.py`
- [ ] Wire stacking into `runner.py`
- [ ] Add draw-recall penalty to tuner objective in `tuner.py`

## Verification
- [ ] Run `ruff check` on modified/new files
- [ ] Run `mypy` on modified/new files
- [ ] Verify existing tests still pass
