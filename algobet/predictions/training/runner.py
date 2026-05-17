"""Top-level training workflow runner."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from algobet.predictions.training.acceleration import resolve_training_hyperparameters
from algobet.predictions.training.calibration import (
    CalibratedPredictor,
    DrawAwarePredictor,
    ProbabilityCalibrator,
)
from algobet.predictions.training.classifiers import EnsemblePredictor
from algobet.predictions.training.config import TrainingResult
from algobet.predictions.training.ensemble import EnsembleWeightOptimizer
from algobet.predictions.training.split import get_class_weights
from algobet.predictions.training.stacking import StackingEnsemble
from algobet.predictions.training.tuner import HAS_OPTUNA, TuningResult


class PipelineRunnerMixin:
    @staticmethod
    def _average_metric_dicts(
        metric_dicts: list[dict[str, float]],
    ) -> dict[str, float]:
        """Average numeric metric values across folds."""
        averaged: dict[str, float] = {}
        keys = set().union(*(metrics.keys() for metrics in metric_dicts))
        for key in sorted(keys):
            values = [
                float(metrics[key])
                for metrics in metric_dicts
                if key in metrics and isinstance(metrics[key], int | float)
            ]
            if values:
                averaged[key] = float(np.mean(values))
        return averaged

    def _snapshot_runtime_state(self) -> dict[str, Any]:
        """Capture mutable training state before temporary fold evaluation."""
        return {
            "feature_pipeline": self.feature_pipeline,
            "_train_df": self._train_df,
            "_val_df": self._val_df,
            "_test_df": self._test_df,
            "_train_raw_features": self._train_raw_features,
            "_val_raw_features": self._val_raw_features,
            "_test_raw_features": self._test_raw_features,
            "_selected_feature_names": self._selected_feature_names,
            "_feature_selection_report": self._feature_selection_report,
            "_collapse_recovery": self._collapse_recovery,
        }

    def _restore_runtime_state(self, state: dict[str, Any]) -> None:
        """Restore mutable training state after temporary fold evaluation."""
        for key, value in state.items():
            setattr(self, key, value)

    def _evaluate_walk_forward_folds(
        self,
        hyperparameters: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], int] | None:
        """Train/evaluate every walk-forward fold and return averaged metrics."""
        matches_df = getattr(self, "_prepared_matches_df", None)
        splits = list(getattr(self, "_prepared_splits", []) or [])
        if self.config.split_strategy != "walk_forward" or matches_df is None:
            return None
        if len(splits) <= 1:
            return None

        state = self._snapshot_runtime_state()
        train_metrics_by_fold: list[dict[str, float]] = []
        val_metrics_by_fold: list[dict[str, float]] = []
        test_metrics_by_fold: list[dict[str, float]] = []

        try:
            for split in splits:
                self.feature_pipeline = self._new_feature_pipeline()
                (
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    _y_test,
                ) = self._prepare_data_for_split(
                    matches_df,
                    split,
                    cache_features=False,
                )

                if self.config.outcome_balance:
                    fold_class_weights = get_class_weights(
                        y_train,
                        strength=self.config.outcome_balance_strength,
                    )
                else:
                    fold_class_weights = None

                fold_params = dict(hyperparameters)
                if self.config.feature_selection:
                    X_train, X_val, X_test = self._apply_feature_selection(
                        X_train=X_train,
                        X_val=X_val,
                        X_test=X_test,
                        y_train=y_train,
                        y_val=y_val,
                        class_weights=fold_class_weights,
                        hyperparameters=fold_params,
                    )

                (
                    fold_predictor,
                    X_train,
                    X_val,
                    X_test,
                    _fold_class_weights,
                    _fold_params,
                ) = self._train_with_collapse_recovery(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    hyperparameters=fold_params,
                    class_weights=fold_class_weights,
                )

                train_metrics_by_fold.append(
                    self._evaluate(
                        fold_predictor,
                        X_train,
                        y_train,
                        self._train_df,
                        apply_calibration=False,
                    )
                )
                val_metrics_by_fold.append(
                    self._evaluate(
                        fold_predictor,
                        X_val,
                        y_val,
                        self._val_df,
                        apply_calibration=False,
                    )
                )
                test_metrics_by_fold.append(
                    self._evaluate(
                        fold_predictor,
                        X_test,
                        _y_test,
                        self._test_df,
                        apply_calibration=False,
                    )
                )
        finally:
            self._restore_runtime_state(state)

        return (
            self._average_metric_dicts(train_metrics_by_fold),
            self._average_metric_dicts(val_metrics_by_fold),
            self._average_metric_dicts(test_metrics_by_fold),
            len(test_metrics_by_fold),
        )

    def run(self) -> TrainingResult:
        """Execute the complete training pipeline.

        Returns:
            TrainingResult with model info and metrics
        """
        import time

        start_time = time.time()

        # Step 1: Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data()

        # Step 2: Handle class imbalance (opt-in only)
        if self.config.outcome_balance:
            class_weights = get_class_weights(
                y_train,
                strength=self.config.outcome_balance_strength,
            )
        else:
            class_weights = None

        # Step 3: Optional feature selection using a probe model
        if self.config.feature_selection:
            X_train, X_val, X_test = self._apply_feature_selection(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                class_weights=class_weights,
                hyperparameters=self.config.hyperparameters.copy(),
            )

        # Step 4: Hyperparameter tuning (optional)
        tuning_result = None
        best_params = self.config.hyperparameters.copy()
        per_model_tuning: dict[str, TuningResult] = {}

        if self.config.tune_hyperparameters and HAS_OPTUNA:
            if self.config.use_ensemble and self.config.per_model_hyperparameters:
                per_model_tuning = self._tune_per_model(
                    X_train, y_train, X_val, y_val, class_weights
                )
                # Build nested hyperparameter dict for per-model tuning
                best_params = {
                    model_type: result.best_params
                    for model_type, result in per_model_tuning.items()
                }
            else:
                tuning_result = self._tune_hyperparameters(
                    X_train, y_train, X_val, y_val, class_weights
                )
                best_params = tuning_result.best_params

        # Step 5: Train model, refusing to persist one-class collapses.
        (
            predictor,
            X_train,
            X_val,
            X_test,
            class_weights,
            best_params,
        ) = self._train_with_collapse_recovery(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            hyperparameters=best_params,
            class_weights=class_weights,
        )

        # Step 5b: Train stacking ensemble (optional)
        if self.config.use_stacking_ensemble:
            predictor = self._train_stacking_ensemble(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                hyperparameters=best_params,
                class_weights=class_weights,
            )

        # Step 6: Calibrate probabilities (optional)
        # Score-based models (Dixon-Coles, Hybrid Poisson) produce naturally
        # calibrated probabilities from the bivariate goal distribution, and
        # the CV calibration path cannot refit them anyway (no goal data per
        # fold). Skip the calibrator for these model types.
        is_score_model = self.config.model_type in ("dixon_coles", "hybrid_poisson")
        X_weight = X_val
        y_weight = y_val
        if self.config.calibrate_probabilities and not is_score_model:
            self._calibrator = ProbabilityCalibrator(
                method=self.config.calibration_method,
            )

            if self.config.use_cv_calibration:
                # Robust: Use K-fold Out-Of-Fold predictions from training data
                oof_probas, oof_y = self._get_oof_probas(
                    X_train, y_train, class_weights, best_params
                )
                self._calibrator.fit(oof_probas, oof_y)
            elif self.config.use_ensemble:
                # Split val in half: first half for calibration, second for weights
                val_n = len(y_val) // 2
                X_calib, X_weight = X_val[:val_n], X_val[val_n:]
                y_calib, y_weight = y_val[:val_n], y_val[val_n:]
                val_probas = predictor.predict_proba(X_calib)
                self._calibrator.fit(val_probas, y_calib)
            else:
                # Simple: Use validation set
                val_probas = predictor.predict_proba(X_val)
                self._calibrator.fit(val_probas, y_val)

            calibrated_predictor = CalibratedPredictor(
                predictor,
                self._calibrator,
            )
            calibrated_report = self._prediction_class_report(
                calibrated_predictor,
                X_val,
            )
            raw_val_probas = predictor.predict_proba(X_val)
            calibrated_val_probas = calibrated_predictor.predict_proba(X_val)
            raw_val_log_loss = self._validation_log_loss(y_val, raw_val_probas)
            calibrated_val_log_loss = self._validation_log_loss(
                y_val,
                calibrated_val_probas,
            )
            calibration_report = {
                "raw_validation_log_loss": raw_val_log_loss,
                "calibrated_validation_log_loss": calibrated_val_log_loss,
                "calibrated_validation_predictions": calibrated_report,
            }
            if not np.isfinite(calibrated_val_log_loss) or (
                calibrated_val_log_loss > raw_val_log_loss + 1e-6
            ):
                self._calibrator = None
                if self._collapse_recovery is None:
                    self._collapse_recovery = {}
                self._collapse_recovery["calibration_disabled"] = True
                self._collapse_recovery["calibration_disable_reason"] = (
                    "validation_log_loss_worse"
                )
                self._collapse_recovery["calibration_report"] = calibration_report
            elif self._is_prediction_collapsed(calibrated_report):
                # Calibration collapsed predictions to fewer classes — disable it
                # even if log_loss marginally improved. A marginal log_loss gain
                # (e.g. 0.0006) is not worth destroying class diversity.
                self._calibrator = None
                if self._collapse_recovery is None:
                    self._collapse_recovery = {}
                self._collapse_recovery["calibration_disabled"] = True
                self._collapse_recovery["calibration_disable_reason"] = (
                    "calibration_collapsed_predictions"
                )
                self._collapse_recovery["calibration_report"] = calibration_report

        # Step 6bb: Apply post-hoc draw boost if requested
        self._draw_boost_calibrator = None
        if self.config.draw_boost_factor > 1.0:
            from algobet.predictions.training.calibration import DrawBoostCalibrator

            self._draw_boost_calibrator = DrawBoostCalibrator(
                boost_factor=self.config.draw_boost_factor,
            )
            # Wrap predictor so future predictions include the boost
            predictor = CalibratedPredictor(
                predictor,
                self._calibrator,
                self._draw_boost_calibrator,
            )

        # Step 6b: Ensemble weight optimization (for ensembles, post-calibration)
        ensemble_weights = None
        ensemble_val_metrics = None
        if self.config.use_ensemble and isinstance(predictor, EnsemblePredictor):
            try:
                xgb_proba = self._get_ensemble_probas(predictor, "xgboost", X_weight)
                lgbm_proba = self._get_ensemble_probas(predictor, "lightgbm", X_weight)
                if xgb_proba is not None and lgbm_proba is not None:
                    optimizer = EnsembleWeightOptimizer(
                        xgboost_proba=xgb_proba,
                        lightgbm_proba=lgbm_proba,
                        y_val=y_weight,
                    )
                    opt_result = optimizer.optimize()
                    ensemble_weights = {
                        "xgboost": opt_result.xgboost_weight,
                        "lightgbm": opt_result.lightgbm_weight,
                    }
                    ensemble_val_metrics = {
                        "validation_log_loss": opt_result.validation_log_loss,
                        "num_classes": opt_result.num_classes,
                        "max_prediction_share": opt_result.max_prediction_share,
                    }
                    # Update predictor with optimized weights
                    predictor.weights = [
                        opt_result.xgboost_weight,
                        opt_result.lightgbm_weight,
                    ]
            except (ValueError, AttributeError):
                # Fall back to equal weights if optimization fails
                ensemble_weights = {"xgboost": 0.5, "lightgbm": 0.5}
                if isinstance(predictor, EnsemblePredictor):
                    predictor.weights = [0.5, 0.5]

        # Step 6c: Fit DrawAwareCalibrator if enabled
        self._draw_aware_calibrator = None
        self._dc_model = None
        draw_aware_alpha = None
        if self.config.fit_draw_aware_calibrator:
            try:
                from algobet.predictions.training.calibration import DrawAwareCalibrator
                from algobet.predictions.training.classifiers import (
                    DixonColesPredictor,
                    ModelConfig,
                )

                if self.config.dc_model_path:
                    # Backward compatibility: load pre-trained DC model
                    self._dc_model = DixonColesPredictor.load(
                        Path(self.config.dc_model_path)
                    )
                else:
                    # Inline Dixon-Coles training using real features
                    home_goals = self._train_df["home_score"].values.astype(np.float64)
                    away_goals = self._train_df["away_score"].values.astype(np.float64)
                    dc_config = ModelConfig(
                        model_type="dixon_coles",
                        random_seed=self.config.random_seed,
                    )
                    self._dc_model = DixonColesPredictor(dc_config)
                    self._dc_model.fit_with_scores(
                        X_train,
                        y_train,
                        home_goals,
                        away_goals,
                        X_val,
                        y_val,
                    )

                # Get base and DC probabilities on validation set
                base_val_probas = predictor.predict_proba(X_val)
                dc_val_probas = self._dc_model.predict_proba(X_val)

                # Fit blend calibrator
                self._draw_aware_calibrator = DrawAwareCalibrator()
                self._draw_aware_calibrator.fit(base_val_probas, dc_val_probas, y_val)
                draw_aware_alpha = self._draw_aware_calibrator.alpha

                print(f"DrawAwareCalibrator fitted: α={draw_aware_alpha:.3f}")
            except Exception as e:
                print(f"Failed to fit DrawAwareCalibrator: {e}")
                self._draw_aware_calibrator = None
                self._dc_model = None

        # Step 6d: Wrap predictor with DrawAwareCalibrator for inference
        if self._draw_aware_calibrator is not None and self._dc_model is not None:
            predictor = DrawAwarePredictor(
                base_predictor=predictor,
                dc_model=self._dc_model,
                draw_aware_calibrator=self._draw_aware_calibrator,
            )

        # Step 7: Evaluate
        # Train/val use raw probabilities (calibrator is fit on val, so
        # calibrating val again would leak data and make calibration metrics
        # look artificially perfect).
        train_metrics = self._evaluate(
            predictor, X_train, y_train, self._train_df, apply_calibration=False
        )
        val_metrics = self._evaluate(
            predictor, X_val, y_val, self._val_df, apply_calibration=False
        )
        test_metrics = self._evaluate(
            predictor, X_test, y_test, self._test_df, apply_calibration=False
        )

        # Step 7b: Check test set for collapse (reject if collapsed)
        # Score-based models (Dixon-Coles, Hybrid Poisson) and stacking ensembles
        # naturally produce fewer argmax-draw predictions; skip this check for them
        # since it does not indicate real collapse.
        test_report = self._prediction_class_report(predictor, X_test)
        is_score_or_stacking = self.config.model_type in (
            "dixon_coles",
            "hybrid_poisson",
        ) or isinstance(predictor, StackingEnsemble)
        if self._is_prediction_collapsed(test_report) and not is_score_or_stacking:
            raise ValueError(
                f"Model passed validation but collapsed on test set: "
                f"{test_report['num_classes']} classes predicted with counts "
                f"{test_report['counts']}. This indicates overfitting to validation. "
                f"Try: (1) disable hyperparameter tuning, "
                f"(2) increase outcome_balance_strength, "
                f"or (3) add more draw-signal features."
            )

        walk_forward_metrics = self._evaluate_walk_forward_folds(best_params)
        if walk_forward_metrics is not None:
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics
            final_test_metrics = test_metrics
            train_metrics, val_metrics, test_metrics, fold_count = walk_forward_metrics
            train_metrics["walk_forward_folds"] = float(fold_count)
            val_metrics["walk_forward_folds"] = float(fold_count)
            test_metrics["walk_forward_folds"] = float(fold_count)
            for key, value in final_train_metrics.items():
                if isinstance(value, int | float):
                    train_metrics[f"final_split_{key}"] = float(value)
            for key, value in final_val_metrics.items():
                if isinstance(value, int | float):
                    val_metrics[f"final_split_{key}"] = float(value)
            for key, value in final_test_metrics.items():
                if isinstance(value, int | float):
                    test_metrics[f"final_split_{key}"] = float(value)

        # Step 8: Register model
        all_metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        if not self.config.use_ensemble:
            resolved_registry_hyperparameters = {
                **resolve_training_hyperparameters(
                    model_type=self.config.model_type,
                    hyperparameters=best_params,
                ),
                **predictor.effective_hyperparameters,
            }
        else:
            resolved_registry_hyperparameters = dict(best_params)
        model_hyperparameters: dict[str, Any] = {
            **resolved_registry_hyperparameters,
            "feature_names": self.feature_pipeline.feature_names,
            "random_seed": self.config.random_seed,
            "early_stopping_rounds": self.config.early_stopping_rounds,
            "use_ensemble": self.config.use_ensemble,
            "outcome_balance": class_weights is not None,
            "outcome_balance_strength": self.config.outcome_balance_strength,
            "min_prediction_classes": self.config.min_prediction_classes,
        }
        if self._selected_feature_names is not None:
            model_hyperparameters["selected_feature_names"] = (
                self._selected_feature_names
            )
            model_hyperparameters["feature_selection"] = {
                "enabled": True,
                "threshold": self.config.feature_selection_threshold,
                "min_samples_per_feature": self.config.min_samples_per_feature,
                "num_selected": len(self._selected_feature_names),
            }
        if self._feature_selection_report is not None:
            model_hyperparameters["feature_selection_report"] = asdict(
                self._feature_selection_report
            )
        if self.config.use_ensemble:
            model_hyperparameters["ensemble_types"] = self.config.ensemble_types
        if self.config.use_stacking_ensemble:
            model_hyperparameters["use_stacking_ensemble"] = True
            model_hyperparameters["stacking_base_models"] = (
                self.config.stacking_base_models
            )
            model_hyperparameters["stacking_meta_learner"] = (
                self.config.stacking_meta_learner
            )
            model_hyperparameters["stacking_n_folds"] = self.config.stacking_n_folds
            if isinstance(predictor, StackingEnsemble):
                model_hyperparameters["stacking_oof_folds"] = predictor.oof_folds
        if ensemble_weights:
            model_hyperparameters["ensemble_weights"] = ensemble_weights
        if ensemble_val_metrics:
            model_hyperparameters["ensemble_validation_metrics"] = ensemble_val_metrics
        if self.config.calibrate_probabilities:
            model_hyperparameters["calibration_method"] = self.config.calibration_method
            model_hyperparameters["calibration_enabled"] = self._calibrator is not None
        if self.config.draw_boost_factor != 1.0:
            model_hyperparameters["draw_boost_factor"] = self.config.draw_boost_factor
        if self._collapse_recovery is not None:
            model_hyperparameters["collapse_recovery"] = self._collapse_recovery

        # Determine model artifact to persist
        if isinstance(predictor, DrawAwarePredictor):
            model_to_save = predictor
        elif self._calibrator is not None:
            model_to_save = CalibratedPredictor(predictor, self._calibrator)
        else:
            model_to_save = predictor

        # Persist the real artifact type, not the base model type
        if isinstance(predictor, StackingEnsemble):
            registry_model_type = "stacking_ensemble"
        elif isinstance(predictor, EnsemblePredictor):
            registry_model_type = "ensemble"
        else:
            registry_model_type = self.config.model_type

        model_version = self.model_registry.save_model(
            model=model_to_save,
            name=self.config.model_name,
            metrics=all_metrics,
            model_type=registry_model_type,
            feature_schema_version=self.config.feature_schema_version,
            hyperparameters=model_hyperparameters,
            description=self.config.description,
            tags=self.config.tags,
        )
        model_dir = self.models_path / registry_model_type / model_version
        self.feature_pipeline_path = model_dir / "feature_pipeline"
        self.feature_pipeline.save(self.feature_pipeline_path)
        model_hyperparameters["feature_pipeline_path"] = str(self.feature_pipeline_path)

        # Save inline-trained Dixon-Coles model if available
        if self._dc_model is not None and self.config.dc_model_path is None:
            dc_path = model_dir / "dixon_coles_model.joblib"
            self._dc_model.save(dc_path)
            model_hyperparameters["dixon_coles_model_path"] = str(dc_path)

        # Save DrawAwareCalibrator if fitted
        if self._draw_aware_calibrator is not None:
            draw_aware_path = model_dir / "draw_aware_calibrator.joblib"
            self._draw_aware_calibrator.save(draw_aware_path)
            model_hyperparameters["draw_aware_calibrator_path"] = str(draw_aware_path)
            model_hyperparameters["draw_aware_alpha"] = draw_aware_alpha

        self.model_registry.update_model_hyperparameters(
            model_version=model_version,
            hyperparameters=model_hyperparameters,
        )

        # Compile result
        duration = time.time() - start_time

        return TrainingResult(
            model_version=model_version,
            model_type=registry_model_type,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_schema_version=self.config.feature_schema_version,
            num_features=len(self.feature_pipeline.feature_names),
            feature_importance=predictor.feature_importance,
            trained_at=datetime.now(),
            training_duration_seconds=duration,
            config=self.config,
            per_model_tuning_results=(
                per_model_tuning
                if per_model_tuning
                else ({"primary": tuning_result} if tuning_result else None)
            ),
            feature_selection_report=self._feature_selection_report,
            ensemble_weights=ensemble_weights,
            ensemble_validation_metrics=ensemble_val_metrics,
            model_path=model_dir,
        )

    def _train_stacking_ensemble(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        hyperparameters: dict[str, Any],
        class_weights: dict[int, float] | None,
    ) -> StackingEnsemble:
        """Train a stacking ensemble with base models and meta-learner."""
        from algobet.predictions.training.acceleration import (
            resolve_training_hyperparameters,
        )
        from algobet.predictions.training.classifiers import (
            DixonColesPredictor,
            HybridPoissonPredictor,
            ModelConfig,
            create_predictor,
        )

        base_predictors: list[Any] = []
        for model_type in self.config.stacking_base_models:
            if model_type == "dixon_coles":
                dc_config = ModelConfig(
                    model_type="dixon_coles",
                    random_seed=self.config.random_seed,
                )
                dc = DixonColesPredictor(dc_config)
                home_goals = self._train_df["home_score"].values.astype(np.float64)
                away_goals = self._train_df["away_score"].values.astype(np.float64)
                dc.fit_with_scores(
                    X_train, y_train, home_goals, away_goals, X_val, y_val
                )
                base_predictors.append(dc)
            elif model_type == "hybrid_poisson":
                hp_config = ModelConfig(
                    model_type="hybrid_poisson",
                    random_seed=self.config.random_seed,
                )
                hp = HybridPoissonPredictor(hp_config)
                home_goals = self._train_df["home_score"].values.astype(np.float64)
                away_goals = self._train_df["away_score"].values.astype(np.float64)
                hp.fit_with_scores(
                    X_train, y_train, home_goals, away_goals, X_val, y_val
                )
                base_predictors.append(hp)
            else:
                model_params = hyperparameters.get(model_type, {})
                if not model_params:
                    model_params = (
                        hyperparameters if self.config.model_type == model_type else {}
                    )
                if not model_params:
                    model_params = self.config.hyperparameters.copy()
                resolved = resolve_training_hyperparameters(
                    model_type=model_type,
                    hyperparameters=model_params,
                )
                config = ModelConfig(
                    model_type=model_type,
                    hyperparameters=resolved,
                    class_weights=class_weights,
                    random_seed=self.config.random_seed,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                )
                predictor = create_predictor(model_type, config)
                predictor.set_feature_names(self.feature_pipeline.feature_names)
                predictor.fit(X_train, y_train, X_val, y_val)
                base_predictors.append(predictor)

        # Use time-aware OOF if enough seasons are available
        matches_df = getattr(self, "_prepared_matches_df", None)
        oof_folds = self.config.stacking_n_folds if matches_df is not None else None

        ensemble = StackingEnsemble(
            base_predictors=base_predictors,
            meta_learner_type=self.config.stacking_meta_learner,
            oof_folds=oof_folds,
            matches_df=matches_df,
        )
        ensemble.fit(X_train, y_train, X_val, y_val)
        return ensemble

    def _get_oof_probas(
        self,
        X: Any,
        y: Any,
        class_weights: dict[int, float] | None,
        hyperparameters: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate time-aware Out-Of-Fold predictions for calibration.

        Uses expanding-window OOF folds so that each fold's predictions
        are produced by a model trained only on earlier seasons. This
        prevents the future-to-past leakage that shuffled StratifiedKFold
        introduces for football match data.
        """
        from algobet.predictions.training.acceleration import (
            resolve_training_hyperparameters,
        )
        from algobet.predictions.training.classifiers import (
            ModelConfig,
            create_predictor,
        )
        from algobet.predictions.training.split import OOFTimeAwareSplitter

        n_folds = self.config.calibration_cv_folds

        # Reconstruct the matches DataFrame needed for the splitter.
        matches_df = getattr(self, "_prepared_matches_df", None)
        if matches_df is None:
            raise ValueError(
                "Time-aware OOF requires the prepared matches DataFrame. "
                "Ensure data preparation has run before calibration."
            )

        splitter = OOFTimeAwareSplitter(
            n_folds=n_folds,
            season_column="season_id",
            date_column="match_date",
        )

        # If X is a DataFrame, convert to values for indexing
        X_values = X.values if hasattr(X, "values") else X
        y_values = np.asarray(y)

        oof_probas = np.zeros((len(y_values), 3))
        oof_y = np.zeros(len(y_values), dtype=np.int64)
        oof_y[:] = -1  # sentinel for unassigned rows

        folds_used = 0
        for train_idx, oof_idx in splitter.split(matches_df):
            # Map split indices back to X/y positions.
            # X and y may be a subset of matches_df (after feature selection),
            # so we find the intersection.
            if hasattr(X, "index"):
                x_index_set = set(X.index)
                train_pos = [
                    i
                    for i, idx in enumerate(X.index)
                    if idx in set(train_idx) & x_index_set
                ]
                oof_pos = [
                    i
                    for i, idx in enumerate(X.index)
                    if idx in set(oof_idx) & x_index_set
                ]
            else:
                # Fallback: use all data (shouldn't happen with DataFrame X)
                train_pos = list(range(len(X_values)))
                oof_pos = list(range(len(X_values)))

            if not train_pos or not oof_pos:
                continue

            X_fold_train = X_values[train_pos]
            y_fold_train = y_values[train_pos]
            X_fold_val = X_values[oof_pos]
            y_fold_val = y_values[oof_pos]

            resolved = resolve_training_hyperparameters(
                model_type=self.config.model_type,
                hyperparameters=hyperparameters,
            )
            config = ModelConfig(
                model_type=self.config.model_type,
                hyperparameters=resolved,
                class_weights=class_weights,
                random_seed=self.config.random_seed,
                early_stopping_rounds=self.config.early_stopping_rounds,
            )
            fold_predictor = create_predictor(self.config.model_type, config)
            fold_predictor.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

            oof_probas[oof_pos] = fold_predictor.predict_proba(X_fold_val)
            oof_y[oof_pos] = y_fold_val
            folds_used += 1

        if folds_used == 0:
            raise ValueError(
                "No valid OOF folds generated. Ensure enough seasons are "
                f"available for {n_folds} folds."
            )

        # Filter out unassigned rows
        valid_mask = oof_y >= 0
        return oof_probas[valid_mask], oof_y[valid_mask]

    @staticmethod
    def _validation_log_loss(
        y_true: Any,
        probas: Any,
    ) -> float:
        """Return guarded multiclass log loss for calibration decisions."""
        from sklearn.metrics import log_loss

        normalized = np.asarray(probas, dtype=np.float64)
        return float(log_loss(y_true, normalized, labels=[0, 1, 2]))
