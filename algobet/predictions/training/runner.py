"""Top-level training workflow runner."""

from datetime import datetime
from typing import Any

from algobet.predictions.training.acceleration import resolve_training_hyperparameters
from algobet.predictions.training.calibration import (
    CalibratedPredictor,
    ProbabilityCalibrator,
)
from algobet.predictions.training.classifiers import EnsemblePredictor
from algobet.predictions.training.config import TrainingResult
from algobet.predictions.training.ensemble import EnsembleWeightOptimizer
from algobet.predictions.training.split import get_class_weights
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
            class_weights = get_class_weights(y_train)
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

        # Step 6: Calibrate probabilities (optional)
        # For ensembles, split val into calibration + weight-optimization sets
        if self.config.calibrate_probabilities:
            if self.config.use_ensemble:
                # Split val in half: first half for calibration, second for weights
                val_n = len(y_val) // 2
                X_calib, X_weight = X_val[:val_n], X_val[val_n:]
                y_calib, y_weight = y_val[:val_n], y_val[val_n:]

                self._calibrator = ProbabilityCalibrator(
                    method=self.config.calibration_method,
                )
                val_probas = predictor.predict_proba(X_calib)
                self._calibrator.fit(val_probas, y_calib)
            else:
                self._calibrator = ProbabilityCalibrator(
                    method=self.config.calibration_method,
                )
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
            if self._is_prediction_collapsed(calibrated_report):
                self._calibrator = None
                if self._collapse_recovery is None:
                    self._collapse_recovery = {}
                self._collapse_recovery["calibration_disabled"] = True
                self._collapse_recovery["calibrated_validation_predictions"] = (
                    calibrated_report
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
            predictor, X_test, y_test, self._test_df, apply_calibration=True
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
        if ensemble_weights:
            model_hyperparameters["ensemble_weights"] = ensemble_weights
        if ensemble_val_metrics:
            model_hyperparameters["ensemble_validation_metrics"] = ensemble_val_metrics
        if self.config.calibrate_probabilities:
            model_hyperparameters["calibration_method"] = self.config.calibration_method
            model_hyperparameters["calibration_enabled"] = self._calibrator is not None
        if self._collapse_recovery is not None:
            model_hyperparameters["collapse_recovery"] = self._collapse_recovery

        model_version = self.model_registry.save_model(
            model=CalibratedPredictor(predictor, self._calibrator)
            if self._calibrator
            else predictor,
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
