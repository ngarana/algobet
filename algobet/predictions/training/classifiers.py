"""Match prediction classifiers.

Provides classifier implementations for football match outcome prediction.
All classifiers follow a common interface for easy swapping and ensemble use.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
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
    xgb = None  # type: ignore[assignment]
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None  # type: ignore[assignment]


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
        self.config = replace(
            config,
            hyperparameters=dict(config.hyperparameters),
        )
        self._model: Any = None
        self._label_encoder = LabelEncoder()
        self._is_fitted = False
        self._feature_names: list[str] | None = None
        self._effective_hyperparameters: dict[str, Any] = dict(
            self.config.hyperparameters
        )

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier."""

    @property
    def default_hyperparameters(self) -> dict[str, Any]:
        """Return the predictor's built-in hyperparameter defaults."""
        return {}

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
            return dict(zip(self._feature_names, importance.tolist(), strict=False))

        return None

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Fix SYCL inference incompatibility: reset XGBoost device to cpu
        # so it doesn't fail when loaded on the main API (which uses CPU).
        if hasattr(self, "_model") and self._model is not None:
            if isinstance(self._model, list):
                for b in self._model:
                    if hasattr(b, "set_param"):
                        b.set_param({"device": "cpu"})
            elif hasattr(self._model, "set_param"):
                self._model.set_param({"device": "cpu"})

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

    @property
    def effective_hyperparameters(self) -> dict[str, Any]:
        """Return the effective hyperparameters used for the current model."""
        return dict(self._effective_hyperparameters)

    def _resolve_effective_hyperparameters(self) -> dict[str, Any]:
        """Merge built-in defaults with config overrides and persist the result."""
        resolved = {
            **self.default_hyperparameters,
            **self.config.hyperparameters,
        }
        self._effective_hyperparameters = dict(resolved)
        self.config = replace(self.config, hyperparameters=dict(resolved))
        return dict(resolved)


def compute_adaptive_regularization(
    params: dict[str, Any],
    n_samples: int,
    n_features: int,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Boost regularization when the samples-per-feature ratio is low.

    When too few samples are available relative to features, tree-based
    models overfit: they memorize noise in the training set and generalize
    poorly.  This function progressively increases XGBoost regularization
    as the ratio deteriorates.

    Thresholds:
        ratio >= 40  → no adjustment (comfortable)
        20 <= ratio < 40 → moderate boost
        ratio < 20  → aggressive boost

    Args:
        params: Original XGBoost hyperparameters.
        n_samples: Number of training samples.
        n_features: Number of features.
        defaults: Fallback defaults for missing keys (should match the
            predictor's ``default_hyperparameters``).

    Returns:
        Adjusted hyperparameters dict (copy).
    """
    ratio = n_samples / max(n_features, 1)
    if ratio >= 40:
        return dict(params)

    base = dict(defaults) if defaults else {}
    adjusted = dict(params)

    if ratio < 20:
        adjusted["max_depth"] = min(
            params.get("max_depth", base.get("max_depth", 6)), 3
        )
        adjusted["min_child_weight"] = max(
            params.get("min_child_weight", base.get("min_child_weight", 3)), 10
        )
        adjusted["gamma"] = max(params.get("gamma", base.get("gamma", 0.1)), 1.0)
        adjusted["reg_alpha"] = max(
            params.get("reg_alpha", base.get("reg_alpha", 0.1)), 5.0
        )
        adjusted["reg_lambda"] = max(
            params.get("reg_lambda", base.get("reg_lambda", 1.0)), 10.0
        )
        adjusted["colsample_bytree"] = min(
            params.get("colsample_bytree", base.get("colsample_bytree", 0.8)), 0.4
        )
        adjusted["subsample"] = min(
            params.get("subsample", base.get("subsample", 0.8)), 0.6
        )
        adjusted["learning_rate"] = min(
            params.get("learning_rate", base.get("learning_rate", 0.1)), 0.03
        )
    else:
        adjusted["max_depth"] = min(
            params.get("max_depth", base.get("max_depth", 6)), 4
        )
        adjusted["min_child_weight"] = max(
            params.get("min_child_weight", base.get("min_child_weight", 3)), 5
        )
        adjusted["gamma"] = max(params.get("gamma", base.get("gamma", 0.1)), 0.5)
        adjusted["reg_alpha"] = max(
            params.get("reg_alpha", base.get("reg_alpha", 0.1)), 1.0
        )
        adjusted["reg_lambda"] = max(
            params.get("reg_lambda", base.get("reg_lambda", 1.0)), 5.0
        )
        adjusted["colsample_bytree"] = min(
            params.get("colsample_bytree", base.get("colsample_bytree", 0.8)), 0.5
        )
        adjusted["subsample"] = min(
            params.get("subsample", base.get("subsample", 0.8)), 0.7
        )
        adjusted["learning_rate"] = min(
            params.get("learning_rate", base.get("learning_rate", 0.1)), 0.05
        )

    return adjusted


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
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        if config is None:
            config = ModelConfig(model_type="xgboost")
        super().__init__(config)

    @property
    def model_type(self) -> str:
        return "xgboost"

    @property
    def default_hyperparameters(self) -> dict[str, Any]:
        """Return conservative odds-free probability defaults for XGBoost."""
        return {
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 1200,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "min_child_weight": 10,
            "gamma": 1.0,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
        }

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

        # Merge defaults with config overrides
        resolved = self._resolve_effective_hyperparameters()

        # Apply adaptive regularization when sample-per-feature ratio is low
        n_samples, n_features = X.shape
        resolved = compute_adaptive_regularization(
            resolved, n_samples, n_features, defaults=self.default_hyperparameters
        )
        self._effective_hyperparameters = dict(resolved)
        self.config = replace(self.config, hyperparameters=dict(resolved))

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": self.config.eval_metric,
            "random_state": self.config.random_seed,
            **resolved,
        }
        uses_sycl_device = (
            str(params.get("device", "")).strip().lower().startswith("sycl")
        )

        # Add class weights if provided
        if self.config.class_weights:
            # Convert to sample weights
            sample_weights = np.array(
                [self.config.class_weights.get(int(yi), 1.0) for yi in y]
            )
        else:
            sample_weights = None

        # The upstream multiclass SYCL path currently crashes on Intel iGPU
        # runtimes, so we train one binary booster per class instead.
        if uses_sycl_device:
            if X_val is not None and y_val is not None:
                warnings.warn(
                    "XGBoost SYCL multiclass training uses one-vs-rest binary "
                    "boosters and disables in-training eval sets / early stopping "
                    "because the upstream multiclass SYCL path currently crashes "
                    "on Intel iGPU runtimes.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._model = self._fit_sycl_one_vs_rest(
                X=X,
                y_encoded=y_encoded,
                params=params,
                sample_weights=sample_weights,
            )
            self._is_fitted = True
            return self

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y_encoded, weight=sample_weights)

        # Validation set for early stopping
        train_kwargs: dict[str, Any] = {
            "evals": None,
            "early_stopping_rounds": None,
        }
        if X_val is not None and y_val is not None:
            y_val_encoded = self._label_encoder.transform(y_val)
            dval = xgb.DMatrix(X_val, label=y_val_encoded)
            train_kwargs["evals"] = [(dtrain, "train"), (dval, "val")]
            train_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds

        # Train model
        n_estimators = params.pop("n_estimators", 500)

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=train_kwargs["evals"],
            early_stopping_rounds=train_kwargs["early_stopping_rounds"],
            verbose_eval=False,
        )

        self._is_fitted = True
        return self

    def _fit_sycl_one_vs_rest(
        self,
        X: NDArray[np.float64],
        y_encoded: NDArray[np.int64],
        params: dict[str, Any],
        sample_weights: NDArray[np.float64] | None,
    ) -> list[Any]:
        """Train one binary XGBoost booster per class on the SYCL GPU path."""
        binary_params = dict(params)
        n_estimators = binary_params.pop("n_estimators", 500)
        binary_params.pop("num_class", None)
        binary_params["objective"] = "binary:logistic"
        if binary_params.get("eval_metric") == "mlogloss":
            binary_params["eval_metric"] = "logloss"

        boosters = []
        for class_index in range(len(self._label_encoder.classes_)):
            binary_targets = (y_encoded == class_index).astype(np.int64)
            dtrain = xgb.DMatrix(X, label=binary_targets, weight=sample_weights)
            booster = xgb.train(
                binary_params,
                dtrain,
                num_boost_round=n_estimators,
                evals=None,
                early_stopping_rounds=None,
                verbose_eval=False,
            )
            boosters.append(booster)
        return boosters

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities using DMatrix."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        dmatrix = xgb.DMatrix(X)
        if isinstance(self._model, list):
            probas = np.column_stack(
                [booster.predict(dmatrix) for booster in self._model]
            ).astype(np.float64)
            row_sums = probas.sum(axis=1, keepdims=True)
            zero_rows = row_sums.squeeze(axis=1) <= 0
            row_sums[zero_rows] = 1.0
            probas = probas / row_sums
            if np.any(zero_rows):
                probas[zero_rows] = 1.0 / probas.shape[1]
            return probas
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
        if isinstance(self._model, list):
            importance: dict[str, float] = {}
            for booster in self._model:
                booster_scores = booster.get_score(importance_type="gain")
                for feature_key, score in booster_scores.items():
                    importance[feature_key] = importance.get(feature_key, 0.0) + score
            if self._model:
                importance = {
                    feature_key: score / len(self._model)
                    for feature_key, score in importance.items()
                }
        else:
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
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm"
            )

        if config is None:
            config = ModelConfig(model_type="lightgbm")
        super().__init__(config)

    @property
    def model_type(self) -> str:
        return "lightgbm"

    @property
    def default_hyperparameters(self) -> dict[str, Any]:
        """Return the default LightGBM training settings."""
        return {
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "num_threads": -1,
        }

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
            **self._resolve_effective_hyperparameters(),
        }

        # Convert class weights to per-sample weights for lgb.Dataset
        sample_weight = None
        if self.config.class_weights:
            sample_weight = [
                self.config.class_weights.get(int(yi), 1.0) for yi in y_encoded
            ]

        # Create dataset
        train_data = lgb.Dataset(X, label=y_encoded, weight=sample_weight)

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
                lgb.early_stopping(self.config.early_stopping_rounds)
                if X_val is not None
                else None,
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
        return dict(zip(self._feature_names, importance.tolist(), strict=False))


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
            config = ModelConfig(model_type="random_forest")
        super().__init__(config)

    @property
    def model_type(self) -> str:
        return "random_forest"

    @property
    def default_hyperparameters(self) -> dict[str, Any]:
        """Return the default Random Forest training settings."""
        return {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
        }

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
            **self._resolve_effective_hyperparameters(),
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
        for proba, weight in zip(all_probas, self.weights, strict=False):
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
        for key in all_importance[0]:
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
    from algobet.predictions.training.classifier_factory import (
        create_predictor as factory_create_predictor,
    )

    return factory_create_predictor(model_type, config=config)
