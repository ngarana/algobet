"""Stacking ensemble meta-learner for match prediction.

Combines base model outputs through a lightweight calibrated classifier
(LogisticRegression) for more expressive blending than weighted averaging.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.calibration import ProbabilityCalibrator
from algobet.predictions.training.classifiers import MatchPredictor, ModelConfig
from algobet.predictions.training.split import decode_targets


class StackingEnsemble(MatchPredictor):
    """Stacking ensemble with logistic meta-learner and isotonic calibration.

    Base models train on X_train and predict on X_val. The meta-learner
    trains on the concatenated base probabilities from X_val. Isotonic
    calibration is applied to the meta-learner output for well-calibrated
    final probabilities.
    """

    def __init__(
        self,
        base_predictors: list[MatchPredictor],
        config: ModelConfig | None = None,
    ) -> None:
        """Initialize stacking ensemble.

        Args:
            base_predictors: List of UNFITTED base predictor instances
            config: Optional model configuration
        """
        if config is None:
            config = ModelConfig(model_type="stacking_ensemble")

        super().__init__(config)
        self.base_predictors = base_predictors
        self._meta_learner: Any = None
        self._meta_calibrator: ProbabilityCalibrator | None = None

    @property
    def model_type(self) -> str:
        return "stacking_ensemble"

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "StackingEnsemble":
        """Fit base models and meta-learner.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features for meta-learner training
            y_val: Validation labels

        Returns:
            self
        """
        if X_val is None or y_val is None:
            raise ValueError(
                "StackingEnsemble requires X_val and y_val for meta-learner training"
            )

        from sklearn.linear_model import LogisticRegression

        # Step 1: Fit base models on training data
        for predictor in self.base_predictors:
            predictor.fit(X_train, y_train, X_val, y_val)

        # Step 2: Collect base predictions on validation set
        val_probas = self._collect_base_probas(X_val)

        # Step 3: Train meta-learner on validation base predictions
        self._meta_learner = LogisticRegression(
            C=1.0,
            multi_class="multinomial",
            max_iter=1000,
            random_state=self.config.random_seed,
        )
        self._meta_learner.fit(val_probas, y_val)

        # Step 4: Isotonic calibration on meta-learner validation output
        meta_val_probas = self._meta_learner.predict_proba(val_probas)
        self._meta_calibrator = ProbabilityCalibrator(method="isotonic")
        self._meta_calibrator.fit(meta_val_probas, y_val)

        self._is_fitted = True
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities through the full stacking pipeline."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        base_probas = self._collect_base_probas(X)
        meta_probas = self._meta_learner.predict_proba(base_probas)
        calibrated = self._meta_calibrator.calibrate(meta_probas)

        # Collapse guard: ensure at least 2% per class floor
        calibrated = np.maximum(calibrated, 0.02)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / np.maximum(row_sums, 1e-10)

        return calibrated

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        """Predict outcomes."""
        proba = self.predict_proba(X)
        encoded_preds = np.argmax(proba, axis=1)
        return decode_targets(encoded_preds)

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Return meta-learner coefficient importance by base model."""
        if not self._is_fitted or self._meta_learner is None:
            return None

        # LogisticRegression with multinomial has coef_ of shape (n_classes, n_features)
        coeffs = np.abs(self._meta_learner.coef_).mean(axis=0)
        labels = []
        for predictor in self.base_predictors:
            for cls in range(3):
                labels.append(f"{predictor.model_type}_class_{cls}")
        return dict(zip(labels, coeffs.tolist(), strict=False))

    def _collect_base_probas(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Concatenate base model probabilities."""
        probas = [predictor.predict_proba(X) for predictor in self.base_predictors]
        return np.hstack(probas)

    def save(self, path: Path) -> None:
        """Save stacking ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save base predictors
        for i, predictor in enumerate(self.base_predictors):
            predictor.save(path / f"base_{i}_{predictor.model_type}.joblib")

        joblib.dump(
            {
                "config": self.config,
                "meta_learner": self._meta_learner,
                "meta_calibrator": self._meta_calibrator,
                "base_predictor_types": [p.model_type for p in self.base_predictors],
                "is_fitted": self._is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "StackingEnsemble":
        """Load stacking ensemble from disk."""
        path = Path(path)

        meta = joblib.load(path)
        predictor_types = meta["base_predictor_types"]

        from algobet.predictions.training.classifiers import (
            DixonColesPredictor,
            LightGBMPredictor,
            XGBoostPredictor,
        )

        type_map: dict[str, type[MatchPredictor]] = {
            "xgboost": XGBoostPredictor,
            "lightgbm": LightGBMPredictor,
            "dixon_coles": DixonColesPredictor,
        }

        base_predictors = []
        for i, pt in enumerate(predictor_types):
            predictor_class = type_map.get(pt)
            if predictor_class is None:
                raise ValueError(f"Unknown base predictor type: {pt}")
            base_predictors.append(predictor_class.load(path / f"base_{i}_{pt}.joblib"))

        ensemble = cls(base_predictors=base_predictors, config=meta["config"])
        ensemble._meta_learner = meta["meta_learner"]
        ensemble._meta_calibrator = meta["meta_calibrator"]
        ensemble._is_fitted = meta["is_fitted"]
        return ensemble
