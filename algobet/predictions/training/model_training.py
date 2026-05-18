"""Model fitting behavior for single models and ensembles."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.acceleration import resolve_training_hyperparameters
from algobet.predictions.training.classifiers import (
    DixonColesPredictor,
    EnsemblePredictor,
    HybridPoissonPredictor,
    MatchPredictor,
    ModelConfig,
    create_predictor,
)
from algobet.predictions.training.market_mediation import MarketMediationPredictor


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

                if model_type == "dixon_coles":
                    predictor = DixonColesPredictor(config)
                    predictor.set_feature_names(self.feature_pipeline.feature_names)
                    home_goals = self._train_df["home_score"].values.astype(np.float64)
                    away_goals = self._train_df["away_score"].values.astype(np.float64)
                    predictor.fit_with_scores(
                        X_train, y_train, home_goals, away_goals, X_val, y_val
                    )
                elif model_type == "hybrid_poisson":
                    predictor = HybridPoissonPredictor(config)
                    predictor.set_feature_names(self.feature_pipeline.feature_names)
                    home_goals = self._train_df["home_score"].values.astype(np.float64)
                    away_goals = self._train_df["away_score"].values.astype(np.float64)
                    predictor.fit_with_scores(
                        X_train, y_train, home_goals, away_goals, X_val, y_val
                    )
                else:
                    predictor = create_predictor(model_type, config)
                    predictor.set_feature_names(self.feature_pipeline.feature_names)
                    predictor.fit(X_train, y_train, X_val, y_val)
                predictors.append(predictor)

            ensemble = EnsemblePredictor(predictors=predictors)
            ensemble._is_fitted = True
            return ensemble
        else:
            if self.config.model_type == "dixon_coles":
                predictor = DixonColesPredictor(
                    ModelConfig(
                        model_type="dixon_coles",
                        hyperparameters=hyperparameters,
                        class_weights=class_weights,
                        random_seed=self.config.random_seed,
                        early_stopping_rounds=self.config.early_stopping_rounds,
                    )
                )
                predictor.set_feature_names(self.feature_pipeline.feature_names)
                home_goals = self._train_df["home_score"].values.astype(np.float64)
                away_goals = self._train_df["away_score"].values.astype(np.float64)
                predictor.fit_with_scores(
                    X_train, y_train, home_goals, away_goals, X_val, y_val
                )
                return predictor

            if self.config.model_type == "hybrid_poisson":
                predictor = HybridPoissonPredictor(
                    ModelConfig(
                        model_type="hybrid_poisson",
                        hyperparameters=hyperparameters,
                        class_weights=class_weights,
                        random_seed=self.config.random_seed,
                        early_stopping_rounds=self.config.early_stopping_rounds,
                    )
                )
                predictor.set_feature_names(self.feature_pipeline.feature_names)
                home_goals = self._train_df["home_score"].values.astype(np.float64)
                away_goals = self._train_df["away_score"].values.astype(np.float64)
                predictor.fit_with_scores(
                    X_train, y_train, home_goals, away_goals, X_val, y_val
                )
                return predictor

            if self.config.model_type == "market_mediation":
                mediation_params = {
                    **hyperparameters,
                    "min_expected_clv": self.config.min_expected_clv,
                    "min_positive_clv_probability": (
                        self.config.min_positive_clv_probability
                    ),
                    "closing_odds_required": self.config.closing_odds_required,
                }
                predictor = MarketMediationPredictor(
                    ModelConfig(
                        model_type="market_mediation",
                        hyperparameters=mediation_params,
                        class_weights=class_weights,
                        random_seed=self.config.random_seed,
                        early_stopping_rounds=self.config.early_stopping_rounds,
                    )
                )
                predictor.set_feature_names(self.feature_pipeline.feature_names)
                predictor.fit_with_market_data(
                    X_train,
                    y_train,
                    self._train_df,
                    X_val,
                    y_val,
                    self._val_df,
                )
                return predictor

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
