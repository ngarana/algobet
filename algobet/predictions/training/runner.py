"""Top-level training workflow runner."""

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
        if self.config.calibrate_probabilities:
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
        if self.config.use_ensemble and self._calibrator is not None:
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
        test_report = self._prediction_class_report(predictor, X_test)
        if self._is_prediction_collapsed(test_report):
            raise ValueError(
                f"Model passed validation but collapsed on test set: "
                f"{test_report['num_classes']} classes predicted with counts "
                f"{test_report['counts']}. This indicates overfitting to validation. "
                f"Try: (1) disable hyperparameter tuning, "
                f"(2) increase outcome_balance_strength, "
                f"or (3) add more draw-signal features."
            )

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
        if self.config.use_ensemble:
            model_hyperparameters["ensemble_types"] = self.config.ensemble_types
        if self.config.use_stacking_ensemble:
            model_hyperparameters["use_stacking_ensemble"] = True
            model_hyperparameters["stacking_base_models"] = (
                self.config.stacking_base_models
            )
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

        model_version = self.model_registry.save_model(
            model=model_to_save,
            name=self.config.model_name,
            metrics=all_metrics,
            model_type=self.config.model_type,
            feature_schema_version=self.config.feature_schema_version,
            hyperparameters=model_hyperparameters,
            description=self.config.description,
            tags=self.config.tags,
        )
        model_dir = self.models_path / self.config.model_type / model_version
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
            model_type=self.config.model_type,
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

        ensemble = StackingEnsemble(base_predictors=base_predictors)
        ensemble.fit(X_train, y_train, X_val, y_val)
        return ensemble

    def _get_oof_probas(
        self,
        X: Any,
        y: Any,
        class_weights: dict[int, float] | None,
        hyperparameters: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate Out-Of-Fold predictions for calibration."""
        from sklearn.model_selection import StratifiedKFold

        from algobet.predictions.training.acceleration import (
            resolve_training_hyperparameters,
        )
        from algobet.predictions.training.classifiers import (
            ModelConfig,
            create_predictor,
        )

        n_folds = self.config.calibration_cv_folds
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.config.random_seed
        )

        # If X is a DataFrame, convert to values for indexing
        X_values = X.values if hasattr(X, "values") else X
        y_values = np.asarray(y)

        oof_probas = np.zeros((len(y_values), 3))

        for train_idx, val_idx in skf.split(X_values, y_values):
            X_fold_train, X_fold_val = X_values[train_idx], X_values[val_idx]
            y_fold_train, y_fold_val = y_values[train_idx], y_values[val_idx]

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

            oof_probas[val_idx] = fold_predictor.predict_proba(X_fold_val)

        return oof_probas, y_values

    @staticmethod
    def _validation_log_loss(
        y_true: Any,
        probas: Any,
    ) -> float:
        """Return guarded multiclass log loss for calibration decisions."""
        from sklearn.metrics import log_loss

        normalized = np.asarray(probas, dtype=np.float64)
        return float(log_loss(y_true, normalized, labels=[0, 1, 2]))
