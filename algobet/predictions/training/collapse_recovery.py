"""Prediction-collapse detection and recovery behavior."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.calibration import CalibratedPredictor
from algobet.predictions.training.classifiers import MatchPredictor
from algobet.predictions.training.split import get_class_weights


class CollapseRecoveryMixin:
    def _prediction_class_report(
        self,
        predictor: MatchPredictor | CalibratedPredictor,
        X: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Summarize predicted-class diversity for model quality gates."""
        probas = predictor.predict_proba(X)
        if not isinstance(probas, np.ndarray) or probas.ndim != 2:
            return None

        predictions = np.argmax(probas, axis=1)
        counts = np.bincount(predictions, minlength=probas.shape[1])
        n_samples = int(len(predictions))
        max_share = float(counts.max() / n_samples) if n_samples else 0.0
        return {
            "num_classes": int(np.count_nonzero(counts)),
            "counts": [int(count) for count in counts.tolist()],
            "max_share": max_share,
            "mean_probabilities": [
                float(value) for value in probas.mean(axis=0).tolist()
            ],
        }

    def _is_prediction_collapsed(self, report: dict[str, Any] | None) -> bool:
        """Return True when validation predictions are too class-collapsed."""
        if report is None:
            return False
        return int(report["num_classes"]) < self.config.min_prediction_classes

    def _restore_full_feature_matrices(
        self,
        y_train: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
        """Refit transforms against full raw features after feature pruning."""
        if (
            self._train_raw_features is None
            or self._val_raw_features is None
            or self._test_raw_features is None
            or not hasattr(self.feature_pipeline, "clear_selected_features")
        ):
            return None

        self.feature_pipeline.clear_selected_features()
        self._selected_feature_names = None
        return (
            self.feature_pipeline.fit_transform_raw_features(
                self._train_raw_features,
                y_train,
            ),
            self.feature_pipeline.transform_raw_features(self._val_raw_features),
            self.feature_pipeline.transform_raw_features(self._test_raw_features),
        )

    def _collapse_recovery_hyperparameters(
        self,
        hyperparameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Relax XGBoost parameters enough to avoid class-prior collapse."""
        if self.config.model_type != "xgboost":
            return dict(hyperparameters)

        recovered = dict(hyperparameters)
        recovered["max_depth"] = max(int(recovered.get("max_depth", 3)), 3)
        recovered["learning_rate"] = max(
            float(recovered.get("learning_rate", 0.03)),
            0.03,
        )
        recovered["n_estimators"] = max(int(recovered.get("n_estimators", 600)), 600)
        recovered["min_child_weight"] = min(
            float(recovered.get("min_child_weight", 10)),
            5.0,
        )
        recovered["gamma"] = min(float(recovered.get("gamma", 1.0)), 0.5)
        recovered["reg_alpha"] = min(float(recovered.get("reg_alpha", 2.0)), 1.0)
        recovered["reg_lambda"] = min(float(recovered.get("reg_lambda", 10.0)), 5.0)
        recovered["subsample"] = max(float(recovered.get("subsample", 0.7)), 0.8)
        recovered["colsample_bytree"] = max(
            float(recovered.get("colsample_bytree", 0.5)),
            0.8,
        )
        return recovered

    def _train_with_collapse_recovery(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        X_test: NDArray[np.float64],
        hyperparameters: dict[str, Any],
        class_weights: dict[int, float] | None,
    ) -> tuple[
        MatchPredictor,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        dict[int, float] | None,
        dict[str, Any],
    ]:
        """Train the final model and recover from one-class collapse."""
        predictor = self._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            hyperparameters,
            class_weights,
        )
        initial_report = self._prediction_class_report(predictor, X_val)
        if not self._is_prediction_collapsed(initial_report):
            self._collapse_recovery = {
                "enabled": self.config.collapse_recovery,
                "triggered": False,
                "validation_predictions": initial_report,
            }
            return predictor, X_train, X_val, X_test, class_weights, hyperparameters

        if not self.config.collapse_recovery or self.config.model_type != "xgboost":
            raise ValueError(
                "Training produced a degenerate model: validation predictions "
                f"covered {initial_report['num_classes']} classes with counts "
                f"{initial_report['counts']}. Enable collapse recovery or adjust "
                "the model configuration."
            )

        recovery_notes: list[dict[str, Any]] = [
            {"stage": "initial", "validation_predictions": initial_report}
        ]
        recovered_weights = class_weights or get_class_weights(y_train)
        recovered_hyperparameters = dict(hyperparameters)

        restored = self._restore_full_feature_matrices(y_train)
        if restored is not None:
            X_train, X_val, X_test = restored
            recovery_notes.append(
                {
                    "stage": "restore_full_feature_set",
                    "num_features": len(self.feature_pipeline.feature_names),
                }
            )

        predictor = self._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            recovered_hyperparameters,
            recovered_weights,
        )
        recovered_report = self._prediction_class_report(predictor, X_val)
        if not self._is_prediction_collapsed(recovered_report):
            self._collapse_recovery = {
                "enabled": True,
                "triggered": True,
                "strategy": "class_weighted_full_features",
                "validation_predictions": recovered_report,
                "notes": recovery_notes,
            }
            return (
                predictor,
                X_train,
                X_val,
                X_test,
                recovered_weights,
                recovered_hyperparameters,
            )

        recovery_notes.append(
            {
                "stage": "class_weighted_full_features",
                "validation_predictions": recovered_report,
            }
        )
        recovered_hyperparameters = self._collapse_recovery_hyperparameters(
            hyperparameters
        )
        predictor = self._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            recovered_hyperparameters,
            recovered_weights,
        )
        relaxed_report = self._prediction_class_report(predictor, X_val)
        if not self._is_prediction_collapsed(relaxed_report):
            self._collapse_recovery = {
                "enabled": True,
                "triggered": True,
                "strategy": "class_weighted_relaxed_xgboost",
                "validation_predictions": relaxed_report,
                "notes": recovery_notes,
            }
            return (
                predictor,
                X_train,
                X_val,
                X_test,
                recovered_weights,
                recovered_hyperparameters,
            )

        raise ValueError(
            "Training produced a degenerate model even after recovery: validation "
            f"predictions covered {relaxed_report['num_classes']} classes with "
            f"counts {relaxed_report['counts']}. Refusing to save or activate it."
        )
