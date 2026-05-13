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


class TemperatureScaling:
    """Per-class temperature calibration for multiclass probabilities.

    Finds one temperature scalar per class that minimises NLL on the
    validation set, then applies softmax(log(p) / T[cls]).  Unlike a
    single global T, per-class temperatures can correct asymmetric
    overconfidence (e.g. a model that is overconfident on HOME but
    underconfident on DRAW).

    T[cls] > 1  →  pushes that class toward uniform (fixes overconfidence)
    T[cls] < 1  →  sharpens that class           (fixes underconfidence)

    Because all temperatures divide log-probabilities before a shared
    softmax, no class can ever be collapsed to zero and rankings can
    change — the calibrator can correct skewed argmax distributions.

    Example:
        >>> ts = TemperatureScaling()
        >>> ts.fit(val_probas, y_val)
        >>> cal = ts.calibrate(test_probas)
    """

    def __init__(self) -> None:
        self._temperatures: NDArray[np.float64] = np.ones(3, dtype=np.float64)
        self._is_fitted = False

    @property
    def temperatures(self) -> NDArray[np.float64]:
        return self._temperatures

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> "TemperatureScaling":
        """Find optimal per-class T by minimising NLL on validation data.

        Args:
            probas: Predicted probabilities (n_samples, n_classes)
            y_true: True integer labels (n_samples,)

        Returns:
            self
        """
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss

        n_classes = probas.shape[1]
        eps = 1e-8
        log_p = np.log(np.clip(probas, eps, 1.0))

        def nll(T_vec: NDArray[np.float64]) -> float:
            # Guard against degenerate T values
            T_vec = np.maximum(T_vec, 0.01)
            scaled = log_p / T_vec[np.newaxis, :]
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp_s = np.exp(shifted)
            cal = exp_s / exp_s.sum(axis=1, keepdims=True)
            return float(log_loss(y_true, cal, labels=list(range(n_classes))))

        result = minimize(
            nll,
            x0=np.ones(n_classes),
            bounds=[(0.1, 10.0)] * n_classes,
            method="L-BFGS-B",
        )
        self._temperatures = np.asarray(result.x, dtype=np.float64)
        self._is_fitted = True
        return self

    def calibrate(
        self,
        probas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply per-class temperature scaling.

        Args:
            probas: Raw probabilities (n_samples, n_classes)

        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise ValueError("TemperatureScaling not fitted. Call fit() first.")
        eps = 1e-8
        log_p = np.log(np.clip(probas, eps, 1.0))
        scaled = log_p / self._temperatures[np.newaxis, :]
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    # Allow duck-typing alongside sklearn-style calibrators
    def transform(self, *_args: Any, **_kwargs: Any) -> NDArray[np.float64]:
        raise NotImplementedError("Use calibrate() for TemperatureScaling")

    def save(self, path: Path) -> None:
        joblib.dump(
            {"temperatures": self._temperatures, "is_fitted": self._is_fitted},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaling":
        data = joblib.load(path)
        obj = cls()
        temps = data.get("temperatures", data.get("temperature"))
        if temps is not None:
            obj._temperatures = np.asarray(temps, dtype=np.float64).flatten()
        obj._is_fitted = data["is_fitted"]
        return obj


class ProbabilityCalibrator:
    """Calibrates probabilities using isotonic, sigmoid, or temperature scaling.

    Provides per-class calibration for multiclass (H/D/A) predictions.

    Temperature scaling is the recommended default: it fits one temperature
    scalar per class by minimising NLL, correcting asymmetric overconfidence
    without risking per-class collapse to zero.

    Example:
        >>> calibrator = ProbabilityCalibrator(method="temperature")
        >>> calibrator.fit(probas, y_true)
        >>> calibrated = calibrator.calibrate(new_probas)
    """

    def __init__(
        self,
        method: str = "temperature",
        n_classes: int = 3,
    ) -> None:
        """Initialize probability calibrator.

        Args:
            method: Calibration method ('temperature', 'isotonic', or 'sigmoid')
            n_classes: Number of classes (default 3 for H/D/A)
        """
        self.method = method
        self.n_classes = n_classes
        self._calibrators: list[Any] = []
        self._temperature_scaler: TemperatureScaling | None = None
        self._is_fitted = False

    def fit(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> "ProbabilityCalibrator":
        """Fit calibrator on validation data.

        Args:
            probas: Raw probabilities from model, shape (n_samples, n_classes)
            y_true: True labels, shape (n_samples,)

        Returns:
            self
        """
        probas = _normalize_probability_rows(probas, self.n_classes)
        _, n_classes = probas.shape

        if n_classes != self.n_classes:
            raise ValueError(f"Expected {self.n_classes} classes, got {n_classes}")

        if self.method == "temperature":
            self._temperature_scaler = TemperatureScaling()
            self._temperature_scaler.fit(probas, y_true)
        else:
            self._calibrators = []
            for cls in range(self.n_classes):
                y_binary = (y_true == cls).astype(int)
                cls_probas = probas[:, cls]
                if self.method == "isotonic":
                    cal = IsotonicRegression(out_of_bounds="clip")
                else:
                    cal = _SigmoidCalibration()
                cal.fit(cls_probas, y_binary)
                self._calibrators.append(cal)

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
        _, n_classes = probas.shape

        if n_classes != self.n_classes:
            raise ValueError(f"Expected {self.n_classes} classes, got {n_classes}")

        if self.method == "temperature":
            # Temperature scaling cannot collapse classes; no floor needed.
            assert self._temperature_scaler is not None
            return self._temperature_scaler.calibrate(probas)

        # Per-class isotonic / sigmoid path
        calibrated = np.zeros_like(probas)
        for cls in range(self.n_classes):
            calibrated[:, cls] = self._calibrators[cls].transform(probas[:, cls])

        # Apply minimum floor (2%) to prevent per-class collapse
        calibrated = np.maximum(calibrated, 0.02)
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
        """Save calibrator to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "method": self.method,
                "n_classes": self.n_classes,
                "calibrators": self._calibrators,
                "temperature_scaler": self._temperature_scaler,
                "is_fitted": self._is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """Load calibrator from disk."""
        data = joblib.load(path)

        calibrator = cls(
            method=data["method"],
            n_classes=data["n_classes"],
        )
        calibrator._calibrators = data.get("calibrators", [])
        calibrator._temperature_scaler = data.get("temperature_scaler")
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
        draw_boost_calibrator: "DrawBoostCalibrator | None" = None,
    ) -> None:
        """Initialize calibrated predictor.

        Args:
            predictor: Base predictor with predict_proba method
            calibrator: Optional fitted calibrator
            draw_boost_calibrator: Optional draw boost post-processor
        """
        self.predictor = predictor
        self.calibrator = calibrator
        self.draw_boost_calibrator = draw_boost_calibrator

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
            raw_probas = self.calibrator.calibrate(raw_probas)

        if self.draw_boost_calibrator is not None:
            raw_probas = self.draw_boost_calibrator.calibrate(raw_probas)

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

        if self.draw_boost_calibrator is not None:
            self.draw_boost_calibrator.save(path / "draw_boost.joblib")

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

        draw_boost_path = path / "draw_boost.joblib"
        draw_boost_calibrator = None
        if draw_boost_path.exists():
            draw_boost_calibrator = DrawBoostCalibrator.load(draw_boost_path)

        return cls(
            predictor=predictor,
            calibrator=calibrator,
            draw_boost_calibrator=draw_boost_calibrator,
        )


class DrawAwareCalibrator:
    """Fitted blend calibrator for XGBoost + Dixon-Coles.

    Learns optimal blend weight α on validation data to improve draw predictions
    while maintaining overall calibration. Replaces blind multiplier approaches.

    Example:
        >>> calibrator = DrawAwareCalibrator()
        >>> calibrator.fit(xgb_probs, dc_probs, y_val)
        >>> blended = calibrator.calibrate(xgb_test_probs, dc_test_probs)
    """

    def __init__(self) -> None:
        """Initialize draw-aware calibrator."""
        self._alpha: float | None = None
        self._is_fitted = False

    @property
    def alpha(self) -> float:
        """Return fitted blend weight."""
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted")
        return self._alpha  # type: ignore[return-value]

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        xgb_probas: NDArray[np.float64],
        dc_probas: NDArray[np.float64],
        y_val: NDArray[np.int64],
    ) -> "DrawAwareCalibrator":
        """Fit blend weight α on validation data.

        Grid-searches α ∈ [0.0, 0.5] to minimize log loss while ensuring:
        - Draw share ≥ 15%
        - All 3 classes predicted

        Args:
            xgb_probas: XGBoost probabilities, shape (n, 3)
            dc_probas: Dixon-Coles probabilities, shape (n, 3)
            y_val: True labels, shape (n,)

        Returns:
            self
        """
        from sklearn.metrics import log_loss

        best_alpha = 0.0
        best_ll = float("inf")

        for alpha in np.arange(0.0, 0.81, 0.02):
            blended = (1 - alpha) * xgb_probas + alpha * dc_probas

            # Renormalize
            row_sums = blended.sum(axis=1, keepdims=True)
            blended = blended / np.maximum(row_sums, 1e-12)

            # Check quality gates
            preds = np.argmax(blended, axis=1)
            draw_share = (preds == 1).mean()
            n_classes = len(np.unique(preds))

            if draw_share < 0.15 or n_classes < 3:
                continue

            ll = log_loss(y_val, blended, labels=[0, 1, 2])
            if ll < best_ll:
                best_ll = ll
                best_alpha = float(alpha)

        self._alpha = best_alpha
        self._is_fitted = True
        return self

    def calibrate(
        self,
        xgb_probas: NDArray[np.float64],
        dc_probas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply fitted blend to new probabilities.

        Args:
            xgb_probas: XGBoost probabilities, shape (n, 3)
            dc_probas: Dixon-Coles probabilities, shape (n, 3)

        Returns:
            Blended and renormalized probabilities, shape (n, 3)
        """
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        blended = (1 - self._alpha) * xgb_probas + self._alpha * dc_probas

        # Renormalize
        row_sums = blended.sum(axis=1, keepdims=True)
        blended = blended / np.maximum(row_sums, 1e-12)

        return np.clip(blended, 1e-12, 1.0)

    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        import joblib

        joblib.dump(
            {"alpha": self._alpha, "is_fitted": self._is_fitted},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "DrawAwareCalibrator":
        """Load calibrator from disk."""
        import joblib

        data = joblib.load(path)
        calibrator = cls()
        calibrator._alpha = data["alpha"]
        calibrator._is_fitted = data["is_fitted"]
        return calibrator


class DrawBoostCalibrator:
    """Post-processing calibrator that boosts draw probabilities.

    Many models under-predict draws (common in football prediction).
    This simple calibrator multiplies the draw probability by a boost factor
    and renormalizes so probabilities sum to 1.

    Example:
        >>> calibrator = DrawBoostCalibrator(boost_factor=1.5)
        >>> calibrated = calibrator.calibrate(raw_probs)
        >>> # Draws (column 1) are boosted, H/A renormalized
    """

    def __init__(
        self,
        boost_factor: float = 1.0,
    ) -> None:
        """Initialize draw boost calibrator.

        Args:
            boost_factor: Multiplier for draw probability.
                1.0 = no change
                1.5 = boost draws by 50%
                2.0 = double draw probability
        """
        if boost_factor <= 0:
            raise ValueError("boost_factor must be positive")
        self.boost_factor = boost_factor
        self._is_fitted = True  # No fitting needed for this simple calibrator

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def calibrate(
        self,
        probas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply draw boost calibration.

        Args:
            probas: Raw probabilities, shape (n_samples, 3)
                Expected to be [Home, Draw, Away] probabilities

        Returns:
            Calibrated probabilities with boosted draws, shape (n_samples, 3)
        """
        if probas.shape[1] != 3:
            raise ValueError(f"Expected 3 classes (H/D/A), got shape {probas.shape}")

        # Boost draw probability (column index 1)
        calibrated = probas.copy()
        calibrated[:, 1] = calibrated[:, 1] * self.boost_factor

        # Renormalize so rows sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        calibrated = calibrated / row_sums

        return np.clip(calibrated, 1e-12, 1.0)

    def __repr__(self) -> str:
        return f"DrawBoostCalibrator(boost_factor={self.boost_factor})"

    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "DrawBoostCalibrator":
        """Load calibrator from disk."""
        import joblib

        return joblib.load(path)


class DrawAwarePredictor:
    """Wrapper that applies DrawAwareCalibrator on top of a base predictor.

    Chains predictions through: base predictor → DC model → DrawAwareCalibrator.
    This is needed because DrawAwareCalibrator requires *two* probability
    vectors (base + DC) to produce its blended output.
    """

    def __init__(
        self,
        base_predictor: Any,
        dc_model: Any,
        draw_aware_calibrator: DrawAwareCalibrator,
    ) -> None:
        """Initialize draw-aware predictor.

        Args:
            base_predictor: Primary model (e.g., XGBoost or StackingEnsemble)
            dc_model: Fitted DixonColesPredictor
            draw_aware_calibrator: Fitted DrawAwareCalibrator
        """
        self.base_predictor = base_predictor
        self.dc_model = dc_model
        self.draw_aware_calibrator = draw_aware_calibrator

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        """Predict outcomes from blended probabilities."""
        probas = self.predict_proba(X)
        outcomes = ["HOME", "DRAW", "AWAY"]
        return [outcomes[int(np.argmax(p))] for p in probas]

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict blended probabilities.

        Args:
            X: Feature matrix

        Returns:
            Blended probability matrix
        """
        base_probas = self.base_predictor.predict_proba(X)
        dc_probas = self.dc_model.predict_proba(X)
        return self.draw_aware_calibrator.calibrate(base_probas, dc_probas)

    def save(self, path: Path) -> None:
        """Save draw-aware predictor components."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.base_predictor.save(path / "base_predictor.joblib")
        self.dc_model.save(path / "dc_model.joblib")
        self.draw_aware_calibrator.save(path / "draw_aware_calibrator.joblib")

    @classmethod
    def load(cls, path: Path) -> "DrawAwarePredictor":
        """Load draw-aware predictor from disk."""
        from algobet.predictions.training.classifiers import (
            DixonColesPredictor,
            LightGBMPredictor,
            RandomForestPredictor,
            XGBoostPredictor,
        )

        path = Path(path)

        # Try to determine the base predictor class
        base_path = path / "base_predictor.joblib"

        # Default to XGBoost if we can't determine, but check if it's a calibrated dir
        is_calibrated = (base_path / "predictor.joblib").exists()

        # In a real scenario, we'd store the class name in metadata.
        # For now, we try to load as Calibrated, then fallback to raw.
        if is_calibrated:
            # We assume XGBoost as the primary engine for now
            base_predictor = CalibratedPredictor.load(
                base_path, predictor_class=XGBoostPredictor
            )
        else:
            # Try loading as raw XGBoost
            try:
                base_predictor = XGBoostPredictor.load(base_path)
            except Exception:
                # Fallback to LightGBM if XGBoost fails
                try:
                    base_predictor = LightGBMPredictor.load(base_path)
                except Exception:
                    base_predictor = RandomForestPredictor.load(base_path)

        dc_model = DixonColesPredictor.load(path / "dc_model.joblib")
        draw_aware_calibrator = DrawAwareCalibrator.load(
            path / "draw_aware_calibrator.joblib"
        )

        return cls(
            base_predictor=base_predictor,
            dc_model=dc_model,
            draw_aware_calibrator=draw_aware_calibrator,
        )
