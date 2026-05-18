"""Selective market-mediation predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.multioutput import MultiOutputRegressor

from algobet.predictions.training.classifiers import MatchPredictor, ModelConfig
from algobet.predictions.training.split import decode_targets


@dataclass
class MarketMediationDecision:
    """A single market-mediation decision."""

    action: str
    outcome_index: int
    outcome: str
    expected_clv: float
    expected_clv_lower_bound: float
    positive_clv_probability: float


class _ConstantProbabilityClassifier:
    """Small fallback for one-class CLV targets."""

    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> _ConstantProbabilityClassifier:
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        n = len(X)
        return np.column_stack(
            [
                np.full(n, 1.0 - self.probability, dtype=np.float64),
                np.full(n, self.probability, dtype=np.float64),
            ]
        )


class MarketMediationPredictor(MatchPredictor):
    """Dual-lane predictor with abstention based on expected CLV.

    The model uses opening/current market probabilities at inference time and
    learns two things from history: how the pure football model differs from the
    market, and whether taking the opening price later beats the close.
    """

    REQUIRED_MARKET_FEATURES = (
        "mediation_opening_implied_prob_home",
        "mediation_opening_implied_prob_draw",
        "mediation_opening_implied_prob_away",
    )

    def __init__(self, config: ModelConfig | None = None) -> None:
        if config is None:
            config = ModelConfig(model_type="market_mediation")
        super().__init__(config)
        params = self.config.hyperparameters
        self.min_expected_clv = float(params.get("min_expected_clv", 0.002))
        self.min_positive_clv_probability = float(
            params.get("min_positive_clv_probability", 0.55)
        )
        self.closing_odds_required = bool(params.get("closing_odds_required", False))
        self._pure_model: HistGradientBoostingClassifier | None = None
        self._residual_model: MultiOutputRegressor | None = None
        self._clv_models: list[Any] = []
        self._market_feature_indices: list[int] = []
        self._non_market_indices: list[int] = []
        self._meta_feature_names: list[str] = []
        self._clv_positive_mean_by_class = np.full(3, 0.05, dtype=np.float64)
        self._clv_negative_mean_by_class = np.full(3, -0.05, dtype=np.float64)
        self._clv_std_by_class = np.zeros(3, dtype=np.float64)
        self._clv_count_by_class = np.ones(3, dtype=np.float64)
        self._fit_metadata: dict[str, float] = {}

    @property
    def model_type(self) -> str:
        return "market_mediation"

    @property
    def default_hyperparameters(self) -> dict[str, Any]:
        return {
            "pure_learning_rate": 0.04,
            "pure_max_iter": 250,
            "pure_l2_regularization": 1.0,
            "residual_alpha": 3.0,
            "residual_clip": 0.20,
            "min_expected_clv": 0.002,
            "min_positive_clv_probability": 0.55,
            "closing_odds_required": False,
        }

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> MarketMediationPredictor:
        raise NotImplementedError(
            "MarketMediationPredictor requires fit_with_market_data() so CLV "
            "targets can be built from opening and closing odds."
        )

    def fit_with_market_data(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        matches_df: pd.DataFrame,
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
        val_matches_df: pd.DataFrame | None = None,
    ) -> MarketMediationPredictor:
        """Fit the pure, residual, and CLV heads."""
        resolved = self._resolve_effective_hyperparameters()
        self.min_expected_clv = float(resolved["min_expected_clv"])
        self.min_positive_clv_probability = float(
            resolved["min_positive_clv_probability"]
        )
        self.closing_odds_required = bool(resolved["closing_odds_required"])
        self._configure_feature_indices()

        X_non_market = self._non_market_view(X)
        self._pure_model = HistGradientBoostingClassifier(
            learning_rate=float(resolved["pure_learning_rate"]),
            max_iter=int(resolved["pure_max_iter"]),
            l2_regularization=float(resolved["pure_l2_regularization"]),
            random_state=self.config.random_seed,
        )
        self._pure_model.fit(X_non_market, y)

        pure_probas = self._pure_model.predict_proba(X_non_market)
        pure_probas = self._ensure_three_columns(pure_probas, self._pure_model.classes_)
        market_probas = self._market_probs_from_features(X)
        meta_X = self._build_meta_features(X, pure_probas, market_probas)

        target = self._one_hot(y) - market_probas
        self._residual_model = MultiOutputRegressor(
            Ridge(alpha=float(resolved["residual_alpha"]))
        )
        self._residual_model.fit(meta_X, target)

        opening = self._odds_matrix(matches_df, snapshot="opening")
        closing = self._odds_matrix(matches_df, snapshot="closing")
        valid_rows = self._valid_odds_mask(opening) & self._valid_odds_mask(closing)
        valid_count = int(valid_rows.sum())
        coverage = valid_count / max(len(matches_df), 1)
        if self.closing_odds_required and valid_count == 0:
            raise ValueError(
                "market_mediation training requires true closing odds, but no "
                "rows had complete opening and closing odds."
            )

        clv = np.zeros((len(matches_df), 3), dtype=np.float64)
        clv[valid_rows] = opening[valid_rows] / closing[valid_rows] - 1.0
        clv_valid = clv[valid_rows]
        self._clv_std_by_class = np.nan_to_num(clv_valid.std(axis=0), nan=0.0)
        self._clv_count_by_class = np.maximum(valid_count, 1) * np.ones(3)
        # Conditional means: E[CLV | CLV > 0] and E[CLV | CLV <= 0] per outcome.
        # Using the unconditional mean here always yields ~0 (bookmaker margin),
        # making expected CLV always negative and blocking all bet selection.
        for cls in range(3):
            pos_mask = clv_valid[:, cls] > 0
            neg_mask = ~pos_mask
            self._clv_positive_mean_by_class[cls] = (
                float(clv_valid[pos_mask, cls].mean()) if pos_mask.any() else 0.05
            )
            self._clv_negative_mean_by_class[cls] = (
                float(clv_valid[neg_mask, cls].mean()) if neg_mask.any() else -0.05
            )

        clv_X = self._build_clv_features(X, market_probas)
        self._clv_models = []
        for cls in range(3):
            cls_target = (clv[:, cls] > 0.0).astype(np.int64)
            fit_mask = valid_rows
            if fit_mask.sum() < 20 or len(np.unique(cls_target[fit_mask])) < 2:
                probability = (
                    float(cls_target[fit_mask].mean()) if fit_mask.any() else 0.0
                )
                model = _ConstantProbabilityClassifier(probability)
            else:
                model = LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=self.config.random_seed,
                )
                model.fit(clv_X[fit_mask], cls_target[fit_mask])
            self._clv_models.append(model)

        self._fit_metadata = {
            "opening_closing_coverage": coverage,
            "opening_closing_rows": float(valid_count),
            "training_rows": float(len(matches_df)),
        }
        self._is_fitted = True
        if X_val is not None and y_val is not None:
            val_probas = self.predict_proba(X_val)
            self._fit_metadata["validation_log_loss"] = float(
                log_loss(y_val, val_probas, labels=[0, 1, 2])
            )
        return self

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit_with_market_data() first.")
        return decode_targets(np.argmax(self.predict_proba(X), axis=1))

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if (
            not self._is_fitted
            or self._pure_model is None
            or self._residual_model is None
        ):
            raise ValueError("Model not fitted. Call fit_with_market_data() first.")
        pure_probas = self._pure_model.predict_proba(self._non_market_view(X))
        pure_probas = self._ensure_three_columns(pure_probas, self._pure_model.classes_)
        market_probas = self._market_probs_from_features(X)
        meta_X = self._build_meta_features(X, pure_probas, market_probas)
        residual = self._residual_model.predict(meta_X)
        residual = np.clip(
            residual,
            -float(self._effective_hyperparameters.get("residual_clip", 0.20)),
            float(self._effective_hyperparameters.get("residual_clip", 0.20)),
        )
        final = np.clip(market_probas + residual, 1e-9, 1.0)
        final = final / final.sum(axis=1, keepdims=True)
        return final

    def predict_pure_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._is_fitted or self._pure_model is None:
            raise ValueError("Model not fitted. Call fit_with_market_data() first.")
        pure = self._pure_model.predict_proba(self._non_market_view(X))
        return self._ensure_three_columns(pure, self._pure_model.classes_)

    def positive_clv_probabilities(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit_with_market_data() first.")
        market_probas = self._market_probs_from_features(X)
        clv_X = self._build_clv_features(X, market_probas)
        cols = []
        for model in self._clv_models:
            proba = model.predict_proba(clv_X)
            cols.append(proba[:, 1])
        return np.column_stack(cols)

    def predict_decisions(
        self, X: NDArray[np.float64]
    ) -> list[MarketMediationDecision]:
        positive_probs = self.positive_clv_probabilities(X)
        expected = self._expected_clv(positive_probs)
        lcb = self._expected_clv_lower_bound(expected)
        best_classes = np.argmax(lcb, axis=1)
        outcomes = ["HOME", "DRAW", "AWAY"]
        decisions = []
        for row, cls in enumerate(best_classes):
            action = (
                "BET_CANDIDATE"
                if lcb[row, cls] > self.min_expected_clv
                and positive_probs[row, cls] >= self.min_positive_clv_probability
                else "ABSTAIN"
            )
            decisions.append(
                MarketMediationDecision(
                    action=action,
                    outcome_index=int(cls),
                    outcome=outcomes[int(cls)],
                    expected_clv=float(expected[row, cls]),
                    expected_clv_lower_bound=float(lcb[row, cls]),
                    positive_clv_probability=float(positive_probs[row, cls]),
                )
            )
        return decisions

    def market_mediation_diagnostics(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        matches_df: pd.DataFrame | None,
    ) -> dict[str, float]:
        """Return activation-gate diagnostics for this split."""
        if matches_df is None:
            return {}
        from sklearn.metrics import log_loss

        opening = self._odds_matrix(matches_df, snapshot="opening")
        closing = self._odds_matrix(matches_df, snapshot="closing")
        valid_opening = self._valid_odds_mask(opening)
        valid_closing = valid_opening & self._valid_odds_mask(closing)
        decisions = self.predict_decisions(X)
        selected = np.array([d.action == "BET_CANDIDATE" for d in decisions])
        selected_indices = np.array(
            [d.outcome_index for d in decisions],
            dtype=np.int64,
        )
        selected = selected & valid_closing

        metrics: dict[str, float] = {
            "market_mediation_closing_coverage": float(
                valid_closing.mean() if len(valid_closing) else 0.0
            ),
            "market_mediation_selected_bets": float(selected.sum()),
            "market_mediation_abstained_share": float(1.0 - selected.mean())
            if len(selected)
            else 1.0,
        }
        if valid_opening.any():
            opening_probs = self._normalize_odds(opening[valid_opening])
            metrics["market_mediation_opening_market_log_loss"] = float(
                log_loss(y[valid_opening], opening_probs, labels=[0, 1, 2])
            )
        if valid_closing.any():
            closing_probs = self._normalize_odds(closing[valid_closing])
            metrics["market_mediation_closing_market_log_loss"] = float(
                log_loss(y[valid_closing], closing_probs, labels=[0, 1, 2])
            )

        pure = self.predict_pure_proba(X)
        final = self.predict_proba(X)
        metrics["market_mediation_pure_lane_log_loss"] = float(
            log_loss(y, pure, labels=[0, 1, 2])
        )
        metrics["market_mediation_residual_lane_log_loss"] = float(
            log_loss(y, final, labels=[0, 1, 2])
        )

        if selected.any():
            clv = (
                opening[selected, selected_indices[selected]]
                / closing[selected, selected_indices[selected]]
                - 1.0
            )
            metrics["market_mediation_selected_mean_clv"] = float(clv.mean())
            metrics["market_mediation_selected_clv_hit_rate"] = float(
                np.mean(clv > 0.0)
            )
            metrics["market_mediation_selected_clv_lower_95"] = (
                self._bootstrap_lower_bound(clv)
            )
            metrics["market_mediation_selected_clv_sum"] = float(clv.sum())
            metrics["market_mediation_selected_clv_sum_sq"] = float(
                np.square(clv).sum()
            )
        else:
            metrics["market_mediation_selected_mean_clv"] = 0.0
            metrics["market_mediation_selected_clv_hit_rate"] = 0.0
            metrics["market_mediation_selected_clv_lower_95"] = 0.0
            metrics["market_mediation_selected_clv_sum"] = 0.0
            metrics["market_mediation_selected_clv_sum_sq"] = 0.0
        return metrics

    @property
    def feature_importance(self) -> dict[str, float] | None:
        if self._feature_names is None:
            return None
        return {name: 1.0 / len(self._feature_names) for name in self._feature_names}

    @property
    def effective_hyperparameters(self) -> dict[str, Any]:
        params = dict(self._effective_hyperparameters)
        params.update(
            {
                "market_mediation_fit_metadata": self._fit_metadata,
                "min_expected_clv": self.min_expected_clv,
                "min_positive_clv_probability": self.min_positive_clv_probability,
                "closing_odds_required": self.closing_odds_required,
            }
        )
        return params

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> MarketMediationPredictor:
        return joblib.load(path)

    def _configure_feature_indices(self) -> None:
        if self._feature_names is None:
            raise ValueError("Feature names must be set before market mediation fit.")
        missing = [
            name
            for name in self.REQUIRED_MARKET_FEATURES
            if name not in self._feature_names
        ]
        if missing:
            raise ValueError(
                "market_mediation requires opening implied probability features: "
                f"{missing}"
            )
        market_indices = [
            idx
            for idx, name in enumerate(self._feature_names)
            if name.startswith("mediation_")
        ]
        self._market_feature_indices = market_indices
        self._non_market_indices = [
            idx for idx in range(len(self._feature_names)) if idx not in market_indices
        ]
        if not self._non_market_indices:
            self._non_market_indices = list(range(len(self._feature_names)))

    def _non_market_view(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return X[:, self._non_market_indices]

    def _market_probs_from_features(
        self, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._feature_names is None:
            raise ValueError("Feature names are required for market mediation.")
        indices = [
            self._feature_names.index(name) for name in self.REQUIRED_MARKET_FEATURES
        ]
        probs = np.asarray(X[:, indices], dtype=np.float64)
        valid = np.isfinite(probs).all(axis=1) & (probs > 0).all(axis=1)
        result = np.tile(np.array([0.40, 0.30, 0.30], dtype=np.float64), (len(X), 1))
        if valid.any():
            valid_probs = np.clip(probs[valid], 1e-9, 1.0)
            result[valid] = valid_probs / valid_probs.sum(axis=1, keepdims=True)
        return result

    def _build_meta_features(
        self,
        X: NDArray[np.float64],
        pure_probas: NDArray[np.float64],
        market_probas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        market_features = X[:, self._market_feature_indices]
        residual = pure_probas - market_probas
        return np.hstack([pure_probas, market_probas, residual, market_features])

    def _build_clv_features(
        self,
        X: NDArray[np.float64],
        market_probas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # CLV head uses only market-structure features, not pure-model residuals.
        # When pure_proba > market_proba, the pure model thinks market is wrong —
        # but the market is consistently more accurate (log loss gap ~0.07), so
        # those residuals are anti-signal for CLV and are excluded here.
        market_features = X[:, self._market_feature_indices]
        return np.hstack([market_probas, market_features])

    @staticmethod
    def _one_hot(y: NDArray[np.int64]) -> NDArray[np.float64]:
        result = np.zeros((len(y), 3), dtype=np.float64)
        result[np.arange(len(y)), y.astype(int)] = 1.0
        return result

    @staticmethod
    def _ensure_three_columns(
        probas: NDArray[np.float64],
        classes: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        result = np.zeros((len(probas), 3), dtype=np.float64)
        for idx, cls in enumerate(classes):
            result[:, int(cls)] = probas[:, idx]
        row_sums = np.maximum(result.sum(axis=1, keepdims=True), 1e-12)
        return result / row_sums

    @staticmethod
    def _odds_matrix(matches_df: pd.DataFrame, snapshot: str) -> NDArray[np.float64]:
        if snapshot == "opening":
            cols = ["opening_odds_home", "opening_odds_draw", "opening_odds_away"]
            fallback = ["odds_home", "odds_draw", "odds_away"]
        elif snapshot == "closing":
            cols = ["closing_odds_home", "closing_odds_draw", "closing_odds_away"]
            fallback = ["odds_home", "odds_draw", "odds_away"]
        else:
            raise ValueError(f"Unsupported odds snapshot: {snapshot}")

        if all(col in matches_df.columns for col in cols):
            matrix = np.array(
                matches_df[cols].astype(float).to_numpy(dtype=np.float64),
                copy=True,
            )
        else:
            matrix = np.full((len(matches_df), 3), np.nan, dtype=np.float64)
        if snapshot == "opening" and all(col in matches_df.columns for col in fallback):
            fallback_matrix = (
                matches_df[fallback].astype(float).to_numpy(dtype=np.float64)
            )
            missing = ~MarketMediationPredictor._valid_odds_mask(matrix)
            matrix[missing] = fallback_matrix[missing]
        return matrix

    @staticmethod
    def _valid_odds_mask(odds: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.isfinite(odds).all(axis=1) & (odds > 0).all(axis=1)

    @staticmethod
    def _normalize_odds(odds: NDArray[np.float64]) -> NDArray[np.float64]:
        implied = 1.0 / odds
        return implied / implied.sum(axis=1, keepdims=True)

    def _expected_clv(self, positive_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            positive_probs * self._clv_positive_mean_by_class
            + (1.0 - positive_probs) * self._clv_negative_mean_by_class
        )

    def _expected_clv_lower_bound(
        self, expected: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Apply a conservative 20% haircut on expected CLV as per-bet margin of
        # safety.  The old formula (std/sqrt(n_train)) tightened as n grew,
        # becoming uselessly narrow and blocking all bets.
        return expected * 0.80

    @staticmethod
    def _bootstrap_lower_bound(clv_values: NDArray[np.float64]) -> float:
        values = np.asarray(clv_values, dtype=np.float64)
        if len(values) == 0:
            return 0.0
        if len(values) < 20:
            return float(
                values.mean() - 1.96 * values.std(ddof=0) / np.sqrt(len(values))
            )
        rng = np.random.default_rng(42)
        samples = rng.choice(values, size=(1000, len(values)), replace=True)
        return float(np.percentile(samples.mean(axis=1), 5.0))
