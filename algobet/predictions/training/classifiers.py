"""Match prediction classifiers.

Provides classifier implementations for football match outcome prediction.
All classifiers follow a common interface for easy swapping and ensemble use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from algobet.predictions.training.split import decode_targets

# Try importing optional dependencies
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None


@dataclass
class ModelConfig:
    """Configuration for a prediction model."""

    model_type: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    class_weights: dict[int, float] | None = None

    # Training settings
    early_stopping_rounds: int = 50
    eval_metric: str = "mlogloss"


class MatchPredictor(ABC):
    """Abstract base class for match prediction models.

    All prediction models must implement this interface for compatibility
    with the training pipeline and model registry.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize predictor with configuration.

        Args:
            config: Model configuration
        """
        self.config = config
        self._model: Any = None
        self._label_encoder = LabelEncoder()
        self._is_fitted = False
        self._feature_names: list[str] | None = None

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier."""

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "MatchPredictor":
        """Fit the model on training data.

        Args:
            X: Training features
            y: Training labels (encoded as 0, 1, 2)
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels

        Returns:
            self
        """

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        """Predict match outcomes.

        Args:
            X: Feature matrix

        Returns:
            List of predicted outcomes ('H', 'D', 'A')
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        encoded_preds = self._model.predict(X)
        return decode_targets(encoded_preds.astype(int))

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict outcome probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 3) with probabilities for
            [Home Win, Draw, Away Win]
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self._model.predict_proba(X)

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Return feature importance if available."""
        if not self._is_fitted or self._feature_names is None:
            return None

        importance = None

        # Try different importance attributes
        for attr in ["feature_importances_", "coef_"]:
            if hasattr(self._model, attr):
                values = getattr(self._model, attr)
                if attr == "coef_" and values.ndim > 1:
                    values = np.abs(values).mean(axis=0)
                importance = values
                break

        if importance is not None:
            return dict(zip(self._feature_names, importance.tolist()))

        return None

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self._model,
                "config": self.config,
                "label_encoder": self._label_encoder,
                "is_fitted": self._is_fitted,
                "feature_names": self._feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MatchPredictor":
        """Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded predictor instance
        """
        data = joblib.load(path)
        predictor = cls(config=data["config"])
        predictor._model = data["model"]
        predictor._label_encoder = data["label_encoder"]
        predictor._is_fitted = data["is_fitted"]
        predictor._feature_names = data.get("feature_names")
        return predictor

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for importance tracking."""
        self._feature_names = names


class XGBoostPredictor(MatchPredictor):
    """XGBoost classifier for match prediction.

    Primary model choice due to:
    - Excellent handling of tabular data
    - Built-in missing value handling
    - Feature importance extraction
    - Strong performance on structured data
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        if config is None:
            config = ModelConfig(
                model_type="xgboost",
                hyperparameters={
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 500,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "gamma": 0.1,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                },
            )
        super().__init__(config)

    @property
    def model_type(self) -> str:
        return "xgboost"

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "XGBoostPredictor":
        """Fit XGBoost model."""
        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y)

        # Default hyperparameters
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": self.config.eval_metric,
            "random_state": self.config.random_seed,
            "use_label_encoder": False,
            **self.config.hyperparameters,
        }

        # Add class weights if provided
        if self.config.class_weights:
            # Convert to sample weights
            sample_weights = np.array(
                [self.config.class_weights.get(int(yi), 1.0) for yi in y]
            )
        else:
            sample_weights = None

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y_encoded, weight=sample_weights)

        # Validation set for early stopping
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            y_val_encoded = self._label_encoder.transform(y_val)
            dval = xgb.DMatrix(X_val, label=y_val_encoded)
            evals.append((dval, "val"))

        # Train model
        n_estimators = params.pop("n_estimators", 500)

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds if X_val is not None else None,
            verbose_eval=False,
        )

        self._is_fitted = True
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities using DMatrix."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        dmatrix = xgb.DMatrix(X)
        return self._model.predict(dmatrix)

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        """Predict outcomes."""
        proba = self.predict_proba(X)
        encoded_preds = np.argmax(proba, axis=1)
        return decode_targets(encoded_preds)

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from XGBoost."""
        if not self._is_fitted or self._feature_names is None:
            return None

        # Get importance scores
        importance = self._model.get_score(importance_type="gain")

        # Map to feature names
        result = {}
        for i, name in enumerate(self._feature_names):
            key = f"f{i}"
            result[name] = importance.get(key, 0.0)

        return result


class LightGBMPredictor(MatchPredictor):
    """LightGBM classifier for match prediction.

    Advantages:
    - Faster training than XGBoost
    - Lower memory usage
    - Native categorical feature support
    - Good for large datasets
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        if config is None:
            config = ModelConfig(
                model_type="lightgbm",
                hyperparameters={
                    "num_leaves": 31,
                    "learning_rate": 0.1,
                    "n_estimators": 500,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_samples": 20,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                },
            )
        super().__init__(config)

    @property
    def model_type(self) -> str:
        return "lightgbm"

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "LightGBMPredictor":
        """Fit LightGBM model."""
        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y)

        # Default hyperparameters
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "random_state": self.config.random_seed,
            "verbose": -1,
            **self.config.hyperparameters,
        }

        # Add class weights if provided
        if self.config.class_weights:
            params["class_weight"] = self.config.class_weights

        # Create dataset
        train_data = lgb.Dataset(X, label=y_encoded)

        # Validation set
        valid_sets = [train_data]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            y_val_encoded = self._label_encoder.transform(y_val)
            val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("val")

        # Train model
        n_estimators = params.pop("n_estimators", 500)

        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds) if X_val is not None else None,
            ],
        )

        self._is_fitted = True
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self._model.predict(X)

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        """Predict outcomes."""
        proba = self.predict_proba(X)
        encoded_preds = np.argmax(proba, axis=1)
        return decode_targets(encoded_preds)

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from LightGBM."""
        if not self._is_fitted or self._feature_names is None:
            return None

        importance = self._model.feature_importance(importance_type="gain")
        return dict(zip(self._feature_names, importance.tolist()))


class RandomForestPredictor(MatchPredictor):
    """Random Forest classifier for match prediction.

    Advantages:
    - Robust to outliers
    - Less prone to overfitting
    - Good ensemble diversity with GBM models
    - No hyperparameter sensitivity
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        if config is None:
            config = ModelConfig(
                model_type="random_forest",
                hyperparameters={
                    "n_estimators": 500,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "n_jobs": -1,
                },
            )
        super().__init__(config)

    @property
    def model_type(self) -> str:
        return "random_forest"

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "RandomForestPredictor":
        """Fit Random Forest model."""
        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y)

        # Create model
        self._model = RandomForestClassifier(
            random_state=self.config.random_seed,
            **self.config.hyperparameters,
        )

        # Set class weights
        if self.config.class_weights:
            self._model.class_weight = self.config.class_weights

        # Fit
        self._model.fit(X, y_encoded)

        self._is_fitted = True
        return self


class EnsemblePredictor(MatchPredictor):
    """Ensemble of multiple prediction models.

    Combines predictions from multiple base models using weighted
    averaging of probabilities.
    """

    def __init__(
        self,
        predictors: list[MatchPredictor],
        weights: list[float] | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        """Initialize ensemble.

        Args:
            predictors: List of fitted predictor instances
            weights: Optional weights for each predictor (default: equal)
            config: Optional configuration
        """
        if config is None:
            config = ModelConfig(model_type="ensemble")

        super().__init__(config)
        self.predictors = predictors
        self.weights = weights or [1.0 / len(predictors)] * len(predictors)

        if len(self.weights) != len(predictors):
            raise ValueError("Number of weights must match number of predictors")

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    @property
    def model_type(self) -> str:
        return "ensemble"

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "EnsemblePredictor":
        """Fit all base predictors."""
        for predictor in self.predictors:
            predictor.fit(X, y, X_val, y_val)

        self._is_fitted = True
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities as weighted average of base models."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get predictions from each model
        all_probas = []
        for predictor in self.predictors:
            probas = predictor.predict_proba(X)
            all_probas.append(probas)

        # Weighted average
        weighted_proba = np.zeros_like(all_probas[0])
        for proba, weight in zip(all_probas, self.weights):
            weighted_proba += weight * proba

        return weighted_proba

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Average feature importance across base models."""
        if not self._is_fitted:
            return None

        all_importance = []
        for predictor in self.predictors:
            imp = predictor.feature_importance
            if imp is not None:
                all_importance.append(imp)

        if not all_importance:
            return None

        # Average importance
        result = {}
        for key in all_importance[0].keys():
            result[key] = np.mean([imp[key] for imp in all_importance])

        return result

    def save(self, path: Path) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save each predictor
        for i, predictor in enumerate(self.predictors):
            predictor.save(path / f"predictor_{i}.joblib")

        # Save ensemble metadata
        joblib.dump(
            {
                "config": self.config,
                "weights": self.weights,
                "n_predictors": len(self.predictors),
                "predictor_types": [p.model_type for p in self.predictors],
            },
            path / "ensemble_meta.joblib",
        )

    @classmethod
    def load(cls, path: Path) -> "EnsemblePredictor":
        """Load ensemble from disk."""
        path = Path(path)

        meta = joblib.load(path / "ensemble_meta.joblib")

        predictors = []
        for i in range(meta["n_predictors"]):
            predictor_path = path / f"predictor_{i}.joblib"
            # Load based on type (simplified - would need type registry)
            data = joblib.load(predictor_path)
            config = data["config"]
            predictor_type = config.model_type

            if predictor_type == "xgboost":
                predictor = XGBoostPredictor.load(predictor_path)
            elif predictor_type == "lightgbm":
                predictor = LightGBMPredictor.load(predictor_path)
            elif predictor_type == "random_forest":
                predictor = RandomForestPredictor.load(predictor_path)
            else:
                raise ValueError(f"Unknown predictor type: {predictor_type}")

            predictors.append(predictor)

        return cls(
            predictors=predictors,
            weights=meta["weights"],
            config=meta["config"],
        )


def create_predictor(
    model_type: str,
    config: ModelConfig | None = None,
) -> MatchPredictor:
    """Factory function to create predictors.

    Args:
        model_type: Type of predictor ('xgboost', 'lightgbm', 'random_forest')
        config: Optional model configuration

    Returns:
        MatchPredictor instance

    Raises:
        ValueError: If model type is not recognized
    """
    predictors = {
        "xgboost": XGBoostPredictor,
        "lightgbm": LightGBMPredictor,
        "random_forest": RandomForestPredictor,
    }

    if model_type not in predictors:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(predictors.keys())}"
        )

    predictor_class = predictors[model_type]
    return predictor_class(config=config)
