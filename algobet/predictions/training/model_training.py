"""Model fitting behavior for single models and ensembles."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.acceleration import resolve_training_hyperparameters
from algobet.predictions.training.classifiers import (
    EnsemblePredictor,
    MatchPredictor,
    ModelConfig,
    create_predictor,
)


class ModelTrainingMixin:
    def _train_model(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        hyperparameters: dict[str, Any],
        class_weights: dict[int, float] | None,
    ) -> MatchPredictor:
        """Train the prediction model."""
        if self.config.use_ensemble:
            # Train ensemble of multiple model types
            predictors = []
            for model_type in self.config.ensemble_types:
                model_params = hyperparameters.get(model_type, {})
                if not model_params:
                    model_params = (
                        hyperparameters if self.config.model_type == model_type else {}
                    )
                if not model_params:
                    model_params = self.config.hyperparameters.copy()
                model_hyperparameters = resolve_training_hyperparameters(
                    model_type=model_type,
                    hyperparameters=model_params,
                )
                config = ModelConfig(
                    model_type=model_type,
                    hyperparameters=model_hyperparameters,
                    class_weights=class_weights,
                    random_seed=self.config.random_seed,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                )
                predictor = create_predictor(model_type, config)
                predictor.set_feature_names(self.feature_pipeline.feature_names)
                predictor.fit(X_train, y_train, X_val, y_val)
                predictors.append(predictor)

            return EnsemblePredictor(predictors=predictors)
        else:
            # Train single model
            model_hyperparameters = resolve_training_hyperparameters(
                model_type=self.config.model_type,
                hyperparameters=hyperparameters,
            )
            config = ModelConfig(
                model_type=self.config.model_type,
                hyperparameters=model_hyperparameters,
                class_weights=class_weights,
                random_seed=self.config.random_seed,
                early_stopping_rounds=self.config.early_stopping_rounds,
            )

            predictor = create_predictor(self.config.model_type, config)
            predictor.set_feature_names(self.feature_pipeline.feature_names)
            predictor.fit(X_train, y_train, X_val, y_val)

            return predictor

    def _get_ensemble_probas(
        self,
        predictor: EnsemblePredictor,
        model_type: str,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Extract probabilities from a specific base model in the ensemble."""
        for p in predictor.predictors:
            if p.model_type == model_type:
                return p.predict_proba(X)
        return None
