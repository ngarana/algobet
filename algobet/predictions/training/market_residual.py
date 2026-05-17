"""Market-residual prediction model.

Instead of predicting outcomes directly, this model predicts the residual
between the true outcome probability and the market-implied probability.
This reframes the problem from "predict football matches" (hard) to
"find where the market is wrong" (still hard, but better-defined).

When odds are not available, the model falls back to the base predictor.
"""

from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.classifiers import MatchPredictor, ModelConfig


class MarketResidualPredictor(MatchPredictor):
    """Predicts match outcomes by modeling residual from market odds.

    Architecture:
        1. Convert odds to implied probabilities (anchor)
        2. Train a base model to predict outcome probabilities
        3. Compute residual = model_prob - market_prob
        4. Bet when residual > threshold (positive edge)

    The model combines market-implied probabilities with learned residuals,
    producing calibrated probabilities anchored to the efficient market.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        if config is None:
            config = ModelConfig(model_type="market_residual")
        super().__init__(config)
        self._base_predictor: MatchPredictor | None = None
        self._blend_alpha: float = 0.5

    @property
    def model_type(self) -> str:
        return "market_residual"

    def set_base_predictor(self, predictor: MatchPredictor) -> None:
        """Set the base predictor model."""
        self._base_predictor = predictor

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "MarketResidualPredictor":
        """Fit is a no-op; blending happens at inference time."""
        self._is_fitted = True
        return self

    def fit_blend_weight(
        self,
        y: NDArray[np.int64],
        model_probas: NDArray[np.float64],
        market_probas: NDArray[np.float64],
    ) -> float:
        """Find optimal blend weight between model and market probabilities.

        Searches for alpha that minimizes log-loss on validation data:
            final_prob = alpha * model_prob + (1 - alpha) * market_prob

        Args:
            y: True labels
            model_probas: Model predicted probabilities
            market_probas: Market-implied probabilities

        Returns:
            Optimal blend weight alpha
        """
        from sklearn.metrics import log_loss

        best_alpha = 0.5
        best_ll = float("inf")

        for alpha in np.linspace(0.05, 0.95, 19):
            blended = alpha * model_probas + (1 - alpha) * market_probas
            blended = np.clip(blended, 1e-12, 1.0)
            blended = blended / blended.sum(axis=1, keepdims=True)
            ll = log_loss(y, blended, labels=[0, 1, 2])
            if ll < best_ll:
                best_ll = ll
                best_alpha = float(alpha)

        self._blend_alpha = best_alpha
        return best_alpha

    def predict_proba(
        self,
        X: NDArray[np.float64],
        odds: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Predict probabilities blending model and market.

        Args:
            X: Feature matrix
            odds: Optional odds matrix (n_samples, 3). If provided,
                market-implied probabilities are blended with model output.

        Returns:
            Array of shape (n_samples, 3) with blended probabilities
        """
        if not self._is_fitted or self._base_predictor is None:
            raise ValueError("Model not fitted. Call set_base_predictor() first.")

        model_probas = self._base_predictor.predict_proba(X)

        if odds is None:
            return model_probas

        valid_mask = np.isfinite(odds).all(axis=1) & (odds > 0).all(axis=1)
        result = model_probas.copy()

        if valid_mask.any():
            valid_odds = odds[valid_mask]
            market_probas = 1.0 / valid_odds
            market_probas = market_probas / market_probas.sum(axis=1, keepdims=True)

            blended = (
                self._blend_alpha * model_probas[valid_mask]
                + (1 - self._blend_alpha) * market_probas
            )
            blended = np.clip(blended, 1e-12, 1.0)
            blended = blended / blended.sum(axis=1, keepdims=True)
            result[valid_mask] = blended

        return result

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        """Predict outcomes (without odds blending)."""
        probas = self.predict_proba(X, odds=None)
        outcomes = ["HOME", "DRAW", "AWAY"]
        return [outcomes[int(np.argmax(p))] for p in probas]

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Delegate to base predictor."""
        if self._base_predictor is not None:
            return self._base_predictor.feature_importance
        return None

    def save(self, path: Path) -> None:
        """Save to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "blend_alpha": self._blend_alpha,
                "is_fitted": self._is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MarketResidualPredictor":
        """Load from disk."""
        data = joblib.load(path)
        predictor = cls(data["config"])
        predictor._blend_alpha = data["blend_alpha"]
        predictor._is_fitted = data["is_fitted"]
        return predictor
