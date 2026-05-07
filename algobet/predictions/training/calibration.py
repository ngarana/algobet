"""Probability calibration for match prediction models.

Ensures that predicted probabilities are well-calibrated, meaning
a 70% predicted probability corresponds to a 70% actual frequency.
This is critical for betting applications where probability accuracy
directly impacts expected value calculations.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import IsotonicRegression


def _normalize_probability_rows(
    probas: NDArray[np.float64],
    n_classes: int,
) -> NDArray[np.float64]:
    """Clip invalid values and renormalize rows to sum to one."""
    if probas.ndim != 2:
        raise ValueError(f"Expected 2D probability matrix, got shape {probas.shape}")

    sanitized = np.asarray(probas, dtype=np.float64)
    sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)
    sanitized = np.clip(sanitized, 0.0, None)

    row_sums = sanitized.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) <= 0
    row_sums[zero_rows] = 1.0
    sanitized = sanitized / row_sums

    if np.any(zero_rows):
        sanitized[zero_rows] = 1.0 / n_classes

    return np.clip(sanitized, 1e-12, 1.0)


class ProbabilityCalibrator:
    """Calibrates predicted probabilities using isotonic or sigmoid regression.

    Provides per-class calibration for multiclass (H/D/A) predictions.
    Critical for value betting where accurate probability estimates matter.

    Example:
        >>> calibrator = ProbabilityCalibrator(method="isotonic")
        >>> calibrator.fit(probas, y_true)
        >>> calibrated = calibrator.calibrate(new_probas)
    """

    def __init__(
        self,
        method: str = "isotonic",
        n_classes: int = 3,
    ) -> None:
        """Initialize probability calibrator.

        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
            n_classes: Number of classes (default 3 for H/D/A)
        """
        self.method = method
        self.n_classes = n_classes
        self._calibrators: list[Any] = []
        self._is_fitted = False

    def fit(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> "ProbabilityCalibrator":
        """Fit calibrators for each class.

        Uses one-vs-rest approach for multiclass calibration.

        Args:
            probas: Raw probabilities from model, shape (n_samples, n_classes)
            y_true: True labels, shape (n_samples,)

        Returns:
            self
        """
        probas = _normalize_probability_rows(probas, self.n_classes)
        n_samples, n_classes = probas.shape

        if n_classes != self.n_classes:
            raise ValueError(f"Expected {self.n_classes} classes, got {n_classes}")

        self._calibrators = []

        for cls in range(self.n_classes):
            # Binary target: 1 if this class, 0 otherwise
            y_binary = (y_true == cls).astype(int)

            # Get probabilities for this class
            cls_probas = probas[:, cls]

            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
            else:
                # Sigmoid (Platt scaling)
                calibrator = _SigmoidCalibration()

            # Fit calibrator
            calibrator.fit(cls_probas, y_binary)
            self._calibrators.append(calibrator)

        self._is_fitted = True
        return self

    def calibrate(
        self,
        probas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calibrate probabilities.

        Args:
            probas: Raw probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        probas = _normalize_probability_rows(probas, self.n_classes)
        n_samples, n_classes = probas.shape

        if n_classes != self.n_classes:
            raise ValueError(f"Expected {self.n_classes} classes, got {n_classes}")

        calibrated = np.zeros_like(probas)

        for cls in range(self.n_classes):
            calibrated[:, cls] = self._calibrators[cls].transform(probas[:, cls])

        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / np.maximum(row_sums, 1e-10)

        return calibrated

    def fit_calibrate(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Fit and calibrate in one step.

        Args:
            probas: Raw probabilities
            y_true: True labels

        Returns:
            Calibrated probabilities
        """
        self.fit(probas, y_true)
        return self.calibrate(probas)

    def save(self, path: Path) -> None:
        """Save calibrator to disk.

        Args:
            path: Path to save calibrator
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "method": self.method,
                "n_classes": self.n_classes,
                "calibrators": self._calibrators,
                "is_fitted": self._is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """Load calibrator from disk.

        Args:
            path: Path to load calibrator from

        Returns:
            Loaded ProbabilityCalibrator instance
        """
        data = joblib.load(path)

        calibrator = cls(
            method=data["method"],
            n_classes=data["n_classes"],
        )
        calibrator._calibrators = data["calibrators"]
        calibrator._is_fitted = data["is_fitted"]

        return calibrator


class _SigmoidCalibration:
    """Sigmoid (Platt scaling) calibration.

    Fits a logistic regression to convert raw scores to probabilities.
    """

    def __init__(self) -> None:
        self._a: float = 1.0
        self._b: float = 0.0

    def fit(
        self,
        scores: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> "_SigmoidCalibration":
        """Fit sigmoid calibration parameters."""
        # Use logistic regression-like fitting
        # Simplified: use sklearn's approach
        from sklearn.linear_model import LogisticRegression

        # Reshape for sklearn
        X = scores.reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(X, y)

        self._a = lr.coef_[0, 0]
        self._b = lr.intercept_[0]

        return self

    def transform(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply sigmoid calibration."""
        return 1.0 / (1.0 + np.exp(-(self._a * scores + self._b)))


def calculate_calibration_metrics(
    y_true: NDArray[np.int64],
    probas: NDArray[np.float64],
    n_bins: int = 10,
) -> dict[str, float]:
    """Calculate calibration quality metrics.

    Args:
        y_true: True labels
        probas: Predicted probabilities
        n_bins: Number of bins for calibration calculation

    Returns:
        Dictionary with calibration metrics
    """
    n_samples, n_classes = probas.shape

    # Expected Calibration Error (ECE)
    ece = 0.0
    mce = 0.0  # Maximum Calibration Error

    for cls in range(n_classes):
        cls_probas = probas[:, cls]
        cls_true = (y_true == cls).astype(int)

        # Bin samples by predicted probability
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        for i in range(n_bins):
            in_bin = (cls_probas > bin_boundaries[i]) & (
                cls_probas <= bin_boundaries[i + 1]
            )
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = cls_true[in_bin].mean()
                confidence_in_bin = cls_probas[in_bin].mean()
                calibration_error = abs(confidence_in_bin - accuracy_in_bin)

                ece += calibration_error * prop_in_bin
                mce = max(mce, calibration_error)

    # Brier score
    brier_score = 0.0
    for cls in range(n_classes):
        cls_true = (y_true == cls).astype(float)
        brier_score += np.mean((probas[:, cls] - cls_true) ** 2)
    brier_score /= n_classes

    # Log loss
    eps = 1e-10
    log_loss_val = 0.0
    for i in range(n_samples):
        true_cls = y_true[i]
        log_loss_val -= np.log(probas[i, true_cls] + eps)
    log_loss_val /= n_samples

    return {
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "brier_score": brier_score,
        "log_loss": log_loss_val,
    }


def calibration_curve(
    y_true: NDArray[np.int64],
    probas: NDArray[np.float64],
    n_bins: int = 10,
    cls: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute calibration curve for a single class.

    Args:
        y_true: True labels
        probas: Predicted probabilities
        n_bins: Number of bins
        cls: Class index to compute curve for

    Returns:
        Tuple of (mean_predicted, mean_actual) for each bin
    """
    cls_probas = probas[:, cls]
    cls_true = (y_true == cls).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    mean_predicted = []
    mean_actual = []

    for i in range(n_bins):
        in_bin = (cls_probas > bin_boundaries[i]) & (
            cls_probas <= bin_boundaries[i + 1]
        )

        if in_bin.sum() > 0:
            mean_predicted.append(cls_probas[in_bin].mean())
            mean_actual.append(cls_true[in_bin].mean())

    return np.array(mean_predicted), np.array(mean_actual)


class CalibratedPredictor:
    """Wrapper that adds calibration to any predictor.

    Combines a base predictor with probability calibration.
    """

    def __init__(
        self,
        predictor: Any,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> None:
        """Initialize calibrated predictor.

        Args:
            predictor: Base predictor with predict_proba method
            calibrator: Optional fitted calibrator
        """
        self.predictor = predictor
        self.calibrator = calibrator

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        """Predict outcomes."""
        return self.predictor.predict(X)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict calibrated probabilities.

        Args:
            X: Feature matrix

        Returns:
            Calibrated probability matrix
        """
        raw_probas = self.predictor.predict_proba(X)

        if self.calibrator is not None:
            return self.calibrator.calibrate(raw_probas)

        return raw_probas

    def fit_calibrator(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        method: str = "isotonic",
    ) -> None:
        """Fit calibrator on validation data.

        Args:
            X: Validation features
            y: Validation labels
            method: Calibration method
        """
        raw_probas = self.predictor.predict_proba(X)

        self.calibrator = ProbabilityCalibrator(method=method)
        self.calibrator.fit(raw_probas, y)

    def save(self, path: Path) -> None:
        """Save calibrated predictor."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.predictor.save(path / "predictor.joblib")

        if self.calibrator is not None:
            self.calibrator.save(path / "calibrator.joblib")

    @classmethod
    def load(
        cls,
        path: Path,
        predictor_class: type,
    ) -> "CalibratedPredictor":
        """Load calibrated predictor.

        Args:
            path: Path to load from
            predictor_class: Class of the base predictor

        Returns:
            Loaded CalibratedPredictor instance
        """
        path = Path(path)

        predictor = predictor_class.load(path / "predictor.joblib")

        calibrator_path = path / "calibrator.joblib"
        calibrator = None
        if calibrator_path.exists():
            calibrator = ProbabilityCalibrator.load(calibrator_path)

        return cls(predictor=predictor, calibrator=calibrator)


class OddsAnchoredBlender:
    """Blend model predictions with market-implied odds probabilities.

    When enriched models have too many features relative to training
    samples, they tend to overpower the market signal — flipping strong
    away favourites into home favourites, for example.  This blender
    preserves the directional information from odds while allowing the
    model to make adjustments based on enriched features.

    The blend weight is optimised on a validation set to minimise log
    loss.  A weight of 0.0 means pure odds; 1.0 means pure model.
    """

    def __init__(self, blend_weight: float = 0.5) -> None:
        self.blend_weight = blend_weight

    def blend(
        self,
        model_probas: NDArray[np.float64],
        odds_probas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Blend model probabilities with odds-implied probabilities.

        Args:
            model_probas: Model predictions, shape (n_samples, 3).
            odds_probas: Market-implied probabilities, shape (n_samples, 3).

        Returns:
            Blended probabilities, shape (n_samples, 3).
        """
        w = self.blend_weight
        model_probas = _normalize_probability_rows(model_probas, 3)
        odds_probas = _normalize_probability_rows(odds_probas, 3)
        blended = w * model_probas + (1 - w) * odds_probas
        return _normalize_probability_rows(blended, 3)

    @classmethod
    def optimize_weight(
        cls,
        model_probas: NDArray[np.float64],
        odds_probas: NDArray[np.float64],
        y_true: NDArray[np.int64],
        n_steps: int = 101,
        max_weight: float = 0.8,
    ) -> tuple[float, float]:
        """Find the blend weight that minimises log loss on validation data.

        Args:
            model_probas: Model predictions on validation set.
            odds_probas: Market-implied probabilities on validation set.
            y_true: True labels on validation set.
            n_steps: Number of weight values to evaluate.
            max_weight: Maximum allowed weight for the model.

        Returns:
            Tuple of (best_weight, best_log_loss).
        """
        from sklearn.metrics import log_loss

        best_weight = 0.5
        best_loss = float("inf")

        for i in range(n_steps):
            w = i / (n_steps - 1)
            if w > max_weight:
                continue
            blender = cls(blend_weight=w)
            blended = blender.blend(model_probas, odds_probas)
            loss = log_loss(y_true, blended, labels=[0, 1, 2])
            if loss < best_loss:
                best_loss = loss
                best_weight = w

        return best_weight, best_loss
