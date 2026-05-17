"""Weighted ensemble optimizer for combining XGBoost and LightGBM predictions.

Uses deterministic grid search over xgboost weights [0.05, 0.95] and
selects weights that minimize validation log loss while rejecting
class-collapsed candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class EnsembleWeightResult:
    """Result of ensemble weight optimization."""

    xgboost_weight: float
    lightgbm_weight: float
    validation_log_loss: float
    validation_metrics: dict[str, Any]
    num_classes: int
    max_prediction_share: float


class EnsembleWeightOptimizer:
    """Optimize weights for combining XGBoost and LightGBM predictions.

    Uses deterministic grid search over possible weight splits,
    evaluating each combination on validation data.

    Example:
        >>> optimizer = EnsembleWeightOptimizer(
        ...     xgboost_proba=xgb_val_proba,
        ...     lightgbm_proba=lgbm_val_proba,
        ...     y_val=y_val,
        ... )
        >>> result = optimizer.optimize()
        >>> print(f"Best weight: XGB={result.xgboost_weight:.2f}")
    """

    def __init__(
        self,
        xgboost_proba: NDArray[np.float64],
        lightgbm_proba: NDArray[np.float64],
        y_val: NDArray[np.int64],
        weight_step: float = 0.01,
        min_weight: float = 0.05,
        max_weight: float = 0.95,
    ) -> None:
        """Initialize optimizer with predictions and labels."""
        self.xgboost_proba = xgboost_proba
        self.lightgbm_proba = lightgbm_proba
        self.y_val = y_val
        self.weight_step = weight_step
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _compute_blended_proba(
        self, xgb_weight: float, lgbm_weight: float
    ) -> NDArray[np.float64]:
        """Compute weighted average of probabilities."""
        return xgb_weight * self.xgboost_proba + lgbm_weight * self.lightgbm_proba

    def _evaluate_blend(
        self, xgb_weight: float, lgbm_weight: float
    ) -> EnsembleWeightResult | None:
        """Evaluate a specific weight combination.

        Returns None if the blend produces class collapse or other
        quality issues.
        """
        from sklearn.metrics import log_loss

        blended = self._compute_blended_proba(xgb_weight, lgbm_weight)

        # Normalize rows to ensure valid probability distribution
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        blended = blended / row_sums

        preds = np.argmax(blended, axis=1)
        unique_classes = len(np.unique(preds))
        max_share = float(np.bincount(preds, minlength=3).max() / len(preds))

        # Reject class-collapsed candidates
        if unique_classes < 3:
            return None

        # Reject if one class dominates too heavily
        if max_share > 0.70:
            return None

        val_ll = log_loss(self.y_val, blended)

        return EnsembleWeightResult(
            xgboost_weight=xgb_weight,
            lightgbm_weight=lgbm_weight,
            validation_log_loss=val_ll,
            validation_metrics={
                "num_classes": unique_classes,
                "max_prediction_share": max_share,
            },
            num_classes=unique_classes,
            max_prediction_share=max_share,
        )

    def optimize(self) -> EnsembleWeightResult:
        """Run grid search to find optimal weights.

        Iterates over weights from min_weight to max_weight in steps
        of weight_step. Returns the weight combination with the lowest
        validation log loss that passes quality gates.

        Raises:
            ValueError: If no valid weight combination is found.
        """
        best_result: EnsembleWeightResult | None = None

        xgb_weights = np.arange(
            self.min_weight,
            self.max_weight + self.weight_step / 2,
            self.weight_step,
        )

        for xgb_w in xgb_weights:
            lgbm_w = round(1.0 - xgb_w, 4)
            if lgbm_w < self.min_weight:
                continue

            result = self._evaluate_blend(xgb_w, lgbm_w)
            if result is None:
                continue

            if (
                best_result is None
                or result.validation_log_loss < best_result.validation_log_loss
            ):
                best_result = result

        if best_result is None:
            raise ValueError(
                "No valid ensemble weight found: all combinations either "
                "collapsed to fewer than 3 classes or exceeded the maximum "
                "prediction share threshold (0.70)."
            )

        return best_result
