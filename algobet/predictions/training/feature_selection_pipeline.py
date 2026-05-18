"""Feature-pipeline construction and feature-selection behavior."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from algobet.predictions.features.composite import create_generators_by_names
from algobet.predictions.features.pipeline import FeaturePipeline
from algobet.predictions.training.config import ALLOWED_FEATURE_GROUPS
from algobet.predictions.training.feature_selection import (
    FeatureSelectionReport,
    group_aware_selection,
)


class FeatureSelectionPipelineMixin:
    def _create_feature_pipeline_with_groups(
        self, feature_groups: list[str]
    ) -> FeaturePipeline:
        """Create a feature pipeline with only the specified feature groups.

        Args:
            feature_groups: List of feature group names to include

        Returns:
            FeaturePipeline configured with selected generators
        """
        unsupported_groups = sorted(set(feature_groups) - set(ALLOWED_FEATURE_GROUPS))
        if unsupported_groups:
            supported = ", ".join(ALLOWED_FEATURE_GROUPS)
            unsupported = ", ".join(unsupported_groups)
            raise ValueError(
                "Unsupported feature groups: "
                f"{unsupported}. Supported groups: {supported}"
            )

        if not feature_groups:
            return FeaturePipeline.create_default()

        return FeaturePipeline(generators=create_generators_by_names(feature_groups))

    def _choose_selected_feature_names(
        self,
        feature_names: list[str],
        feature_importance: dict[str, float] | None,
        n_samples: int,
        X_val: NDArray[np.float64] | None = None,
    ) -> tuple[list[str], FeatureSelectionReport | None]:
        """Choose a stable feature subset with group-aware protection."""
        if not feature_names:
            raise ValueError("Cannot select features from an empty feature list")

        # If group-aware guards are enabled, use group_aware_selection
        if (
            self.config.min_draw_features > 0
            or self.config.min_away_features > 0
            or self.config.min_enriched_or_coverage > 0
            or self.config.min_low_scoring_features > 0
            or self.config.min_market_mediation_features > 0
        ):
            # Use val features for correlation computation if available
            X_for_corr = (
                X_val if X_val is not None else np.zeros((1, len(feature_names)))
            )
            # MarketMediationPredictor returns uniform feature importance (1/N)
            # because it is a dual-head model with no intrinsic tree-based
            # ranking. With N≈200 features, each gets 1/N ≈ 0.005, which is
            # exactly at the typical threshold and causes ALL features to be
            # rejected. Use threshold=0 so all correlation-pruned features
            # survive into the family guards; selection for market_mediation
            # is driven entirely by correlation pruning and family minimums.
            effective_threshold = (
                0.0
                if self.config.model_type == "market_mediation"
                else self.config.feature_selection_threshold
            )

            selected, report = group_aware_selection(
                feature_names=feature_names,
                X=X_for_corr,
                importance=feature_importance,
                n_samples=n_samples,
                threshold=effective_threshold,
                min_samples_per_feature=self.config.min_samples_per_feature,
                max_correlation=self.config.max_feature_correlation,
                min_draw_features=self.config.min_draw_features,
                min_away_features=self.config.min_away_features,
                min_enriched_or_coverage=self.config.min_enriched_or_coverage,
                min_low_scoring_features=self.config.min_low_scoring_features,
                min_market_mediation_features=self.config.min_market_mediation_features,
            )
            return selected, report

        # Fallback: importance-based selection (legacy)
        importance = {
            name: max(float((feature_importance or {}).get(name, 0.0)), 0.0)
            for name in feature_names
        }
        total_importance = sum(importance.values())
        if total_importance <= 0.0:
            selected = list(feature_names)
        else:
            normalized = {
                name: value / total_importance for name, value in importance.items()
            }
            selected = [
                name
                for name in feature_names
                if normalized[name] >= self.config.feature_selection_threshold
            ]
            if not selected:
                selected = [max(feature_names, key=lambda name: normalized[name])]

        if self.config.min_samples_per_feature:
            max_features = max(1, n_samples // self.config.min_samples_per_feature)
            top_selected = sorted(
                selected,
                key=lambda name: importance.get(name, 0.0),
                reverse=True,
            )[:max_features]
            top_selected_set = set(top_selected)
            selected = [name for name in feature_names if name in top_selected_set]

        return selected, None

    def _apply_feature_selection(
        self,
        X_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        X_test: NDArray[np.float64],
        y_train: NDArray[np.int64],
        y_val: NDArray[np.int64],
        class_weights: dict[int, float] | None,
        hyperparameters: dict[str, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Train a probe model, select important features, then refit transforms."""
        original_feature_names = list(self.feature_pipeline.feature_names)
        probe = self._train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters,
            class_weights=class_weights,
        )
        selected_feature_names, feature_report = self._choose_selected_feature_names(
            feature_names=original_feature_names,
            feature_importance=probe.feature_importance,
            n_samples=len(y_train),
            X_val=X_val,
        )
        if self.config.model_type == "market_mediation":
            required = [
                "mediation_opening_implied_prob_home",
                "mediation_opening_implied_prob_draw",
                "mediation_opening_implied_prob_away",
            ]
            for name in required:
                if (
                    name in original_feature_names
                    and name not in selected_feature_names
                ):
                    selected_feature_names.append(name)
        self._selected_feature_names = selected_feature_names
        self._feature_selection_report = feature_report

        # Verify no odds-derived features leaked into the selected subset
        # unless odds features were explicitly requested via feature_groups.
        odds_groups = {"odds", "odds_residual", "detailed_odds", "market_mediation"}
        requested_groups = set(self.config.feature_groups or [])
        allow_odds = bool(requested_groups & odds_groups)

        if not allow_odds:
            forbidden_terms = (
                "odds",
                "implied_prob",
                "bookmaker",
                "favorite",
                "market",
            )
            leaked = [
                name
                for name in selected_feature_names
                if any(term in name for term in forbidden_terms)
            ]
            if leaked:
                raise ValueError(
                    f"Odds-derived features detected in selected subset: {leaked}"
                )

        if hasattr(self.feature_pipeline, "set_selected_features"):
            self.feature_pipeline.set_selected_features(selected_feature_names)

        if (
            self._train_raw_features is not None
            and self._val_raw_features is not None
            and self._test_raw_features is not None
            and hasattr(self.feature_pipeline, "fit_transform_raw_features")
            and hasattr(self.feature_pipeline, "transform_raw_features")
        ):
            X_train_selected = self.feature_pipeline.fit_transform_raw_features(
                self._train_raw_features,
                y_train,
            )
            X_val_selected = self.feature_pipeline.transform_raw_features(
                self._val_raw_features
            )
            X_test_selected = self.feature_pipeline.transform_raw_features(
                self._test_raw_features
            )
            return X_train_selected, X_val_selected, X_test_selected

        if self._train_df is None or self._val_df is None or self._test_df is None:
            selected_indices = [
                original_feature_names.index(name) for name in selected_feature_names
            ]
            return (
                X_train[:, selected_indices],
                X_val[:, selected_indices],
                X_test[:, selected_indices],
            )

        X_train_selected = self.feature_pipeline.fit_transform(
            self._train_df,
            self.repo,
            y_train,
        )
        X_val_selected = self.feature_pipeline.transform(self._val_df, self.repo)
        X_test_selected = self.feature_pipeline.transform(self._test_df, self.repo)

        return X_train_selected, X_val_selected, X_test_selected
