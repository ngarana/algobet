"""Model evaluation and market diagnostic behavior."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from algobet.predictions.training.calibration import calculate_calibration_metrics
from algobet.predictions.training.classifiers import MatchPredictor


class EvaluationPipelineMixin:
    def _evaluate(
        self,
        predictor: MatchPredictor,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        matches_df: pd.DataFrame | None = None,
        apply_calibration: bool = True,
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            predictor: Fitted predictor
            X: Feature matrix
            y: True labels
            matches_df: Optional match metadata for market diagnostics
            apply_calibration: If False, report raw probabilities even when
                a calibrator exists. Used for train/val to avoid calibrator
                contamination (the calibrator is fit on val data).
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
        )

        # Get predictions
        probas = predictor.predict_proba(X)

        # Apply calibration only when explicitly requested.
        # We skip calibration for train/val because the calibrator was fit on
        # the validation set; applying it back to val would make calibration
        # metrics look artificially perfect (data leakage).
        if apply_calibration and self._calibrator is not None:
            probas = self._calibrator.calibrate(probas)

        y_pred = np.argmax(probas, axis=1)
        prediction_counts = np.bincount(y_pred, minlength=probas.shape[1])
        max_prediction_share = (
            float(prediction_counts.max() / len(y_pred)) if len(y_pred) else 0.0
        )

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, probas, labels=list(range(probas.shape[1]))),
            "precision_macro": precision_score(
                y, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "predicted_classes": float(np.count_nonzero(prediction_counts)),
            "max_prediction_share": max_prediction_share,
        }

        # Add calibration metrics
        cal_metrics = calculate_calibration_metrics(y, probas)
        metrics.update(cal_metrics)
        metrics.update(self._calculate_market_diagnostics(y, probas, matches_df))

        return metrics

    def _calculate_market_diagnostics(
        self,
        y: NDArray[np.int64],
        probas: NDArray[np.float64],
        matches_df: pd.DataFrame | None,
    ) -> dict[str, float]:
        """Compare model probabilities to implied odds for diagnostics only."""
        if matches_df is None:
            return {}

        required_columns = ["odds_home", "odds_draw", "odds_away"]
        if not all(column in matches_df.columns for column in required_columns):
            return {}

        odds = matches_df[required_columns].astype(float).to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(odds).all(axis=1) & (odds > 0).all(axis=1)
        if not np.any(valid_mask):
            return {"market_samples": 0.0}

        valid_odds = odds[valid_mask]
        implied = 1.0 / valid_odds
        implied = implied / implied.sum(axis=1, keepdims=True)

        y_valid = y[valid_mask]
        model_valid = probas[valid_mask]
        row_indices = np.arange(len(y_valid))

        market_log_loss = -np.log(
            np.clip(implied[row_indices, y_valid], 1e-12, 1.0)
        ).mean()
        market_favorite = np.argmax(implied, axis=1)
        model_favorite = np.argmax(model_valid, axis=1)

        return {
            "market_samples": float(len(y_valid)),
            "market_log_loss": float(market_log_loss),
            "market_favorite_accuracy": float(np.mean(market_favorite == y_valid)),
            "market_model_probability_mae": float(np.abs(model_valid - implied).mean()),
            "market_favorite_agreement": float(
                np.mean(model_favorite == market_favorite)
            ),
        }
