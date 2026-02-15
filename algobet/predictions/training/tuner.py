"""Hyperparameter tuning for match prediction models.

Provides Bayesian optimization and grid search capabilities for finding
optimal hyperparameters. Uses time-series cross-validation to prevent
data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import log_loss

from algobet.predictions.training.classifiers import (
    ModelConfig,
    create_predictor,
)
from algobet.predictions.training.split import (
    ExpandingWindowSplitter,
)

# Try importing optuna
try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import optuna as optuna_types


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    model_type: str
    n_trials: int = 50
    timeout: int | None = None  # Seconds
    n_jobs: int = 1
    cv_splits: int = 3
    metric: str = "log_loss"
    direction: str = "minimize"

    # Search space constraints
    search_space: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    best_params: dict[str, Any]
    best_score: float
    best_trial: int
    n_trials: int
    study_name: str
    model_type: str
    all_trials: list[dict[str, Any]] = field(default_factory=list)


# Default search spaces for each model type
DEFAULT_SEARCH_SPACES: dict[str, dict[str, tuple[Any, Any]]] = {
    "xgboost": {
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (100, 1000),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_weight": (1, 10),
        "gamma": (0.0, 0.5),
        "reg_alpha": (0.0, 2.0),
        "reg_lambda": (0.5, 5.0),
    },
    "lightgbm": {
        "num_leaves": (15, 127),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (100, 1000),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_samples": (5, 50),
        "reg_alpha": (0.0, 2.0),
        "reg_lambda": (0.0, 5.0),
    },
    "random_forest": {
        "n_estimators": (100, 1000),
        "max_depth": (5, 20),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": ("sqrt", "log2", None),
    },
}


class HyperparameterTuner:
    """Hyperparameter tuner using Optuna Bayesian optimization.

    Uses time-series cross-validation to evaluate hyperparameters
    while preventing data leakage.

    Example:
        >>> tuner = HyperparameterTuner(model_type="xgboost")
        >>> result = tuner.tune(X, y)
        >>> print(f"Best params: {result.best_params}")
        >>> print(f"Best score: {result.best_score}")
    """

    def __init__(
        self,
        model_type: str,
        config: TuningConfig | None = None,
        search_space: dict[str, tuple[Any, Any]] | None = None,
    ) -> None:
        """Initialize hyperparameter tuner.

        Args:
            model_type: Type of model to tune
            config: Tuning configuration
            search_space: Custom search space (overrides defaults)
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        self.model_type = model_type
        self.config = config or TuningConfig(model_type=model_type)
        self.search_space = search_space or DEFAULT_SEARCH_SPACES.get(model_type, {})

    def tune(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
        class_weights: dict[int, float] | None = None,
    ) -> TuningResult:
        """Run hyperparameter optimization.

        Args:
            X: Training features
            y: Training labels
            X_val: Optional validation set (if provided, uses single split)
            y_val: Optional validation labels
            class_weights: Optional class weights for imbalanced data

        Returns:
            TuningResult with best parameters and score
        """

        # Create objective function
        def objective(trial: optuna_types.Trial) -> float:
            return self._objective(
                trial=trial,
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                class_weights=class_weights,
            )

        # Create study
        study = optuna.create_study(
            direction=self.config.direction,
            study_name=f"{self.model_type}_tuning",
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        # Compile results
        all_trials = []
        for trial in study.trials:
            all_trials.append(
                {
                    "trial_number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": trial.state.name,
                }
            )

        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial=study.best_trial.number,
            n_trials=len(study.trials),
            study_name=study.study_name,
            model_type=self.model_type,
            all_trials=all_trials,
        )

    def _objective(
        self,
        trial: optuna.Trial,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None,
        y_val: NDArray[np.int64] | None,
        class_weights: dict[int, float] | None,
    ) -> float:
        """Objective function for Optuna optimization."""
        # Sample hyperparameters
        params = self._sample_params(trial)

        # Create model config
        config = ModelConfig(
            model_type=self.model_type,
            hyperparameters=params,
            class_weights=class_weights,
        )

        # Evaluate with cross-validation or single split
        if X_val is not None and y_val is not None:
            score = self._evaluate_single_split(
                config=config,
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
            )
        else:
            score = self._evaluate_cv(
                config=config,
                X=X,
                y=y,
            )

        return score

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}

        for param_name, param_range in self.search_space.items():
            low, high = param_range

            if isinstance(low, int) and isinstance(high, int):
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif isinstance(low, float) or isinstance(high, float):
                params[param_name] = trial.suggest_float(
                    param_name, low, high, log=param_name == "learning_rate"
                )
            elif isinstance(low, str):
                # Categorical
                choices = [low, high] if high is not None else [low]
                params[param_name] = trial.suggest_categorical(param_name, choices)
            elif low is None and isinstance(high, str):
                params[param_name] = trial.suggest_categorical(param_name, [None, high])

        return params

    def _evaluate_single_split(
        self,
        config: ModelConfig,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
    ) -> float:
        """Evaluate model with single train/val split."""
        predictor = create_predictor(self.model_type, config)
        predictor.fit(X, y, X_val, y_val)

        probas = predictor.predict_proba(X_val)

        if self.config.metric == "log_loss":
            return log_loss(y_val, probas)
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}")

    def _evaluate_cv(
        self,
        config: ModelConfig,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> float:
        """Evaluate model with time-series cross-validation."""
        splitter = ExpandingWindowSplitter(
            min_train_size=max(100, len(X) // 4),
            val_size=max(50, len(X) // 8),
            test_size=max(50, len(X) // 8),
            step_size=max(50, len(X) // 8),
        )

        # Create dummy DataFrame for splitter
        import pandas as pd

        dummy_df = pd.DataFrame({"match_date": range(len(X))})

        scores = []
        for split in splitter.split(dummy_df):
            X_train = X[split.train_indices]
            y_train = y[split.train_indices]
            X_val = X[split.val_indices]
            y_val = y[split.val_indices]

            predictor = create_predictor(self.model_type, config)
            predictor.fit(X_train, y_train, X_val, y_val)

            probas = predictor.predict_proba(X_val)

            if self.config.metric == "log_loss":
                scores.append(log_loss(y_val, probas))
            else:
                raise ValueError(f"Unknown metric: {self.config.metric}")

        return np.mean(scores)


class GridSearchTuner:
    """Grid search for hyperparameter tuning.

    Simpler alternative to Bayesian optimization that exhaustively
    searches a predefined parameter grid.
    """

    def __init__(
        self,
        model_type: str,
        param_grid: dict[str, list[Any]],
        metric: str = "log_loss",
    ) -> None:
        """Initialize grid search tuner.

        Args:
            model_type: Type of model to tune
            param_grid: Dictionary mapping parameter names to values to try
            metric: Metric to optimize
        """
        self.model_type = model_type
        self.param_grid = param_grid
        self.metric = metric

    def tune(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        class_weights: dict[int, float] | None = None,
    ) -> TuningResult:
        """Run grid search.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Optional class weights

        Returns:
            TuningResult with best parameters
        """
        import itertools

        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))

        best_score = float("inf")
        best_params = {}
        all_trials = []

        for i, values in enumerate(combinations):
            params = dict(zip(param_names, values, strict=False))

            config = ModelConfig(
                model_type=self.model_type,
                hyperparameters=params,
                class_weights=class_weights,
            )

            predictor = create_predictor(self.model_type, config)
            predictor.fit(X, y, X_val, y_val)

            probas = predictor.predict_proba(X_val)
            score = log_loss(y_val, probas)

            all_trials.append(
                {
                    "trial_number": i,
                    "params": params,
                    "value": score,
                    "state": "COMPLETE",
                }
            )

            if score < best_score:
                best_score = score
                best_params = params

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_trial=0,
            n_trials=len(combinations),
            study_name=f"{self.model_type}_grid_search",
            model_type=self.model_type,
            all_trials=all_trials,
        )


def get_default_search_space(model_type: str) -> dict[str, tuple[Any, Any]]:
    """Get default search space for a model type.

    Args:
        model_type: Type of model

    Returns:
        Dictionary mapping parameter names to (low, high) tuples
    """
    return DEFAULT_SEARCH_SPACES.get(model_type, {}).copy()
