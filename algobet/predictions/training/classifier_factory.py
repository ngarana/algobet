"""Predictor factory separated from concrete classifier implementations."""

from algobet.predictions.training.classifiers import MatchPredictor, ModelConfig


def create_predictor(
    model_type: str,
    config: ModelConfig | None = None,
) -> MatchPredictor:
    """Create a predictor by model type."""
    from algobet.predictions.training.classifiers import (
        LightGBMPredictor,
        RandomForestPredictor,
        XGBoostPredictor,
    )

    predictors = {
        "xgboost": XGBoostPredictor,
        "lightgbm": LightGBMPredictor,
        "random_forest": RandomForestPredictor,
    }

    if model_type not in predictors:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(predictors.keys())}"
        )

    return predictors[model_type](config=config)
