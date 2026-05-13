"""Hyperparameter tuning behavior for training pipelines."""

import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.config import (
    ALLOWED_FEATURE_GROUPS,
    LIGHTGBM_SEARCH_SPACE,
    XGBOOST_SEARCH_SPACE,
)
from algobet.predictions.training.tuner import (
    HyperparameterTuner,
    TuningConfig,
    TuningResult,
)


class TuningPipelineMixin:
    def _tune_hyperparameters(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        class_weights: dict[int, float] | None,
    ) -> TuningResult | None:
        """Run hyperparameter tuning for the primary model type.

        Uses time-series CV inside the training set so the val set remains
        uncontaminated for early stopping and calibration.
        """
        from algobet.predictions.training.tuner import DEFAULT_SEARCH_SPACES

        model_type = self.config.model_type
        n_trials = self.config.tuning_trials

        # Choose search space
        use_epl_space = set(self.config.feature_groups or []) == set(
            ALLOWED_FEATURE_GROUPS
        )
        if use_epl_space and model_type in ("xgboost", "lightgbm"):
            search_space = (
                XGBOOST_SEARCH_SPACE
                if model_type == "xgboost"
                else LIGHTGBM_SEARCH_SPACE
            )
        else:
            search_space = DEFAULT_SEARCH_SPACES.get(model_type, {})

        tuning_config = TuningConfig(
            model_type=model_type,
            n_trials=n_trials,
            search_space=search_space,
        )

        tuner = HyperparameterTuner(
            model_type=model_type,
            config=tuning_config,
        )

        # Pass X_val=None so the tuner uses time-series CV inside X_train,
        # keeping val free for early stopping and calibration.
        return tuner.tune(
            X_train, y_train, X_val=None, y_val=None, class_weights=class_weights
        )

    def _tune_per_model(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        class_weights: dict[int, float] | None,
    ) -> dict[str, TuningResult]:
        """Tune each ensemble model independently using CV inside the training set."""
        from algobet.predictions.training.tuner import DEFAULT_SEARCH_SPACES

        results: dict[str, TuningResult] = {}
        model_types = self.config.ensemble_types or ["xgboost", "lightgbm"]

        for model_type in model_types:
            use_epl_space = set(self.config.feature_groups or []) == set(
                ALLOWED_FEATURE_GROUPS
            )
            if use_epl_space and model_type in ("xgboost", "lightgbm"):
                search_space = (
                    XGBOOST_SEARCH_SPACE
                    if model_type == "xgboost"
                    else LIGHTGBM_SEARCH_SPACE
                )
            else:
                search_space = DEFAULT_SEARCH_SPACES.get(model_type, {})

            tuning_config = TuningConfig(
                model_type=model_type,
                n_trials=self.config.tuning_trials,
                search_space=search_space,
            )

            tuner = HyperparameterTuner(
                model_type=model_type,
                config=tuning_config,
            )
            # CV inside train; val stays clean for early stopping + calibration.
            result = tuner.tune(
                X_train, y_train, X_val=None, y_val=None, class_weights=class_weights
            )
            results[model_type] = result

        return results
