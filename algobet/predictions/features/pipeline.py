"""Feature pipeline orchestrator for match prediction.

This module provides the main FeaturePipeline class that orchestrates
feature generation and transformation for ML model training and inference.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.generators import (
    FeatureGenerator,
    FeatureSchema,
    create_default_generators,
    create_generators_by_names,
)
from algobet.predictions.features.transformers import (
    TransformerPipeline,
    create_default_transformer_pipeline,
)


@dataclass
class PipelineConfig:
    """Configuration for a feature pipeline."""

    schema_version: str = "v1.0"
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    generator_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    transformer_config: dict[str, Any] = field(default_factory=dict)
    selected_feature_names: list[str] | None = None


class FeaturePipeline:
    """Orchestrates feature generation and transformation.

    The FeaturePipeline is the main entry point for feature engineering.
    It combines feature generators (which create features from raw data)
    with transformers (which preprocess features for ML models).

    Usage:
        # Create pipeline with default generators
        pipeline = FeaturePipeline.create_default()

        # Fit on training data
        features = pipeline.fit_transform(matches, repository)

        # Transform new data for inference
        new_features = pipeline.transform(new_matches, repository)

        # Save for later use
        pipeline.save("data/pipelines/v1.0")

        # Load saved pipeline
        pipeline = FeaturePipeline.load("data/pipelines/v1.0")
    """

    def __init__(
        self,
        generators: FeatureGenerator,
        transformers: TransformerPipeline | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize feature pipeline.

        Args:
            generators: Feature generator (single or composite)
            transformers: Optional transformer pipeline for preprocessing
            config: Optional pipeline configuration
        """
        self.generators = generators
        self.transformers = transformers or create_default_transformer_pipeline()
        self.config = config or PipelineConfig()
        self._fitted = False
        self._feature_schema: FeatureSchema | None = None
        self._feature_names: list[str] | None = (
            list(self.config.selected_feature_names)
            if self.config.selected_feature_names
            else None
        )
        self._selected_feature_names: list[str] | None = (
            list(self.config.selected_feature_names)
            if self.config.selected_feature_names
            else None
        )
        self._last_raw_features: pd.DataFrame | None = None

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names produced by this pipeline."""
        if self._selected_feature_names is not None:
            return list(self._selected_feature_names)

        if self._feature_names is not None:
            return self._feature_names

        names = self.generators.feature_names
        self._feature_names = names
        return names

    @property
    def selected_feature_names(self) -> list[str] | None:
        """Return the selected feature subset when feature pruning is active."""
        if self._selected_feature_names is None:
            return None
        return list(self._selected_feature_names)

    def set_selected_features(self, feature_names: list[str]) -> None:
        """Constrain the pipeline to a selected feature subset."""
        unique_names = list(dict.fromkeys(feature_names))
        if not unique_names:
            raise ValueError("Selected feature list cannot be empty")

        available = set(self.generators.feature_names)
        missing = [name for name in unique_names if name not in available]
        if missing:
            raise ValueError(f"Selected features are unavailable: {missing}")

        self._selected_feature_names = unique_names
        self._feature_names = unique_names
        self.config.selected_feature_names = unique_names
        self._feature_schema = None

    def clear_selected_features(self) -> None:
        """Restore the pipeline to the full generator feature set."""
        self._selected_feature_names = None
        self._feature_names = list(self.generators.feature_names)
        self.config.selected_feature_names = None
        self._feature_schema = None

    @property
    def is_fitted(self) -> bool:
        """Check if pipeline has been fitted."""
        return self._fitted

    def fit(
        self,
        matches: pd.DataFrame,
        repository: MatchRepository,
        y: Any = None,
    ) -> "FeaturePipeline":
        """Fit the pipeline on training data.

        This generates features from matches and fits the transformers.

        Args:
            matches: DataFrame with match records
            repository: Repository for historical data queries
            y: Ignored, present for API compatibility

        Returns:
            self
        """
        raw_features = self.generators.generate(matches, repository)
        self.fit_raw_features(raw_features, y)
        return self

    def transform(
        self,
        matches: pd.DataFrame,
        repository: MatchRepository,
    ) -> np.ndarray:
        """Transform matches to feature matrix.

        Args:
            matches: DataFrame with match records
            repository: Repository for historical data queries

        Returns:
            Transformed feature matrix (samples x features)
        """
        if not self._fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        raw_features = self.generators.generate(matches, repository)
        return self.transform_raw_features(raw_features)

    def fit_transform(
        self,
        matches: pd.DataFrame,
        repository: MatchRepository,
        y: Any = None,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            matches: DataFrame with match records
            repository: Repository for historical data queries
            y: Ignored

        Returns:
            Transformed feature matrix
        """
        raw_features = self.generators.generate(matches, repository)
        return self.fit_transform_raw_features(raw_features, y)

    def fit_raw_features(
        self,
        raw_features: pd.DataFrame,
        y: Any = None,
    ) -> "FeaturePipeline":
        """Fit transformers from precomputed raw features."""
        ensured_features = self._ensure_features(raw_features.copy())
        self.transformers.fit(ensured_features, y)

        self._feature_schema = None
        self._feature_schema = self.get_schema()
        self._last_raw_features = ensured_features
        self._fitted = True

        return self

    def transform_raw_features(self, raw_features: pd.DataFrame) -> np.ndarray:
        """Transform precomputed raw features with fitted transformers."""
        if not self._fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        ensured_features = self._ensure_features(raw_features.copy())
        self._last_raw_features = ensured_features
        return self.transformers.transform(ensured_features)

    def fit_transform_raw_features(
        self,
        raw_features: pd.DataFrame,
        y: Any = None,
    ) -> np.ndarray:
        """Fit and transform precomputed raw features in one step."""
        self.fit_raw_features(raw_features, y)
        if self._last_raw_features is None:
            raise ValueError("Raw features were not captured during fit")
        return self.transformers.transform(self._last_raw_features)

    @property
    def last_raw_features(self) -> pd.DataFrame | None:
        """Most recent ensured raw feature frame produced by fit/transform."""
        if self._last_raw_features is None:
            return None
        return self._last_raw_features.copy()

    def generate_raw(
        self,
        matches: pd.DataFrame,
        repository: MatchRepository,
    ) -> pd.DataFrame:
        """Generate raw features without transformation.

        Useful for inspection and debugging.

        Args:
            matches: DataFrame with match records
            repository: Repository for historical data queries

        Returns:
            DataFrame with raw generated features
        """
        return self.generators.generate(matches, repository)

    def _ensure_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure all expected features are present with defaults.

        Args:
            features: Generated features DataFrame

        Returns:
            DataFrame with all expected features
        """
        import warnings

        expected = set(self.feature_names)
        actual = set(features.columns)
        missing = expected - actual

        if missing:
            warnings.warn(
                "FeaturePipeline is filling missing features with zeros: "
                f"{sorted(missing)}. This usually means the generator setup changed "
                "since the pipeline was saved.",
                RuntimeWarning,
                stacklevel=3,
            )
            for name in missing:
                features[name] = 0.0

        # Reorder to match expected order
        return features[[c for c in self.feature_names if c in features.columns]]

    def get_schema(self) -> FeatureSchema:
        """Get the feature schema for this pipeline."""
        if self._feature_schema is None:
            schema = self.generators.get_schema()
            if self._selected_feature_names is not None:
                schema = FeatureSchema(
                    version=schema.version,
                    features={
                        name: schema.features.get(name, float)
                        for name in self._selected_feature_names
                    },
                    description=schema.description,
                )
            self._feature_schema = schema
        return self._feature_schema

    def _generator_names(self) -> list[str]:
        """Return stable generator names for serialization."""
        if hasattr(self.generators, "generators"):
            return [gen.name for gen in self.generators.generators]
        return [self.generators.name]

    def get_feature_importance(
        self,
        model: BaseEstimator | None = None,
    ) -> dict[str, float] | None:
        """Get feature importance if available.

        Args:
            model: Optional model to extract importance from

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if model is None:
            return None

        importance = None

        # Try different importance attribute names
        for attr in ["feature_importances_", "coef_"]:
            if hasattr(model, attr):
                values = getattr(model, attr)
                if attr == "coef_" and values.ndim > 1:
                    values = values.mean(axis=0)
                importance = values
                break

        if importance is not None:
            return dict(zip(self.feature_names, importance.tolist(), strict=False))

        return None

    def save(self, path: str | Path) -> None:
        """Save pipeline to disk.

        Saves both the configuration and fitted transformers.

        Args:
            path: Directory path to save pipeline
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        # Store the FULL generator schema in feature_names so load() can
        # validate selected features against the reconstructed generators.
        config_data = {
            "schema_version": self.config.schema_version,
            "created_at": self.config.created_at.isoformat(),
            "description": self.config.description,
            "feature_names": self.generators.feature_names,
            "selected_feature_names": self._selected_feature_names,
            "generator_names": self._generator_names(),
            "fitted": self._fitted,
        }

        with open(path / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        # Save transformers
        if self._fitted:
            self.transformers.save(path / "transformers.joblib")

        # Save generator config (simplified)
        generator_config = {
            "type": type(self.generators).__name__,
            "generator_names": self._generator_names(),
            "feature_names": self.generators.feature_names,
        }

        with open(path / "generators.json", "w") as f:
            json.dump(generator_config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Load pipeline from disk.

        Args:
            path: Directory path to load pipeline from

        Returns:
            Loaded FeaturePipeline instance
        """
        path = Path(path)

        # Load configuration
        with open(path / "config.json") as f:
            config_data = json.load(f)

        # Load generator config (stored for future use)
        with open(path / "generators.json") as f:
            generator_config = json.load(f)

        generator_names = generator_config.get("generator_names")
        if isinstance(generator_names, list) and generator_names:
            generators = create_generators_by_names(generator_names)
        else:
            generators = create_default_generators()

        # Create pipeline
        config = PipelineConfig(
            schema_version=config_data["schema_version"],
            created_at=datetime.fromisoformat(config_data["created_at"]),
            description=config_data.get("description", ""),
            selected_feature_names=config_data.get("selected_feature_names"),
        )

        pipeline = cls(
            generators=generators,
            transformers=create_default_transformer_pipeline(),
            config=config,
        )

        # Load fitted transformers if available
        transformers_path = path / "transformers.joblib"
        if transformers_path.exists() and config_data.get("fitted", False):
            pipeline.transformers = TransformerPipeline.load(transformers_path)
            pipeline._fitted = True

        selected_feature_names = config_data.get("selected_feature_names")
        if selected_feature_names:
            # Use the public setter to validate against reconstructed generators
            # and reset the feature schema.
            try:
                pipeline.set_selected_features(list(selected_feature_names))
            except ValueError as exc:
                raise ValueError(
                    f"Saved selected features cannot be restored: {exc}"
                ) from exc
        else:
            pipeline._feature_names = config_data.get(
                "feature_names", generators.feature_names
            )

        return pipeline

    @classmethod
    def create_default(cls) -> "FeaturePipeline":
        """Create pipeline with default configuration.

        Returns:
            FeaturePipeline with default generators and transformers
        """
        return cls(
            generators=create_default_generators(),
            transformers=create_default_transformer_pipeline(),
        )

    def describe(self) -> dict[str, Any]:
        """Get description of the pipeline.

        Returns:
            Dictionary with pipeline information
        """
        return {
            "schema_version": self.config.schema_version,
            "is_fitted": self._fitted,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names[:10]
            + (["..."] if len(self.feature_names) > 10 else []),
            "generator_type": type(self.generators).__name__,
        }


class TrainingDataBuilder:
    """Build training datasets from match history.

    Provides utilities for creating train/test splits and
    preparing data for model training.
    """

    def __init__(self, pipeline: FeaturePipeline) -> None:
        """Initialize data builder.

        Args:
            pipeline: FeaturePipeline to use for feature generation
        """
        self.pipeline = pipeline

    def build_training_data(
        self,
        matches: pd.DataFrame,
        repository: MatchRepository,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        temporal_split: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build training, validation, and test datasets.

        Args:
            matches: DataFrame with match records
            repository: Repository for historical queries
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            temporal_split: If True, split by date (no data leakage)

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Ensure matches have target
        matches = matches.copy()
        matches["result"] = self._calculate_result(matches)

        # Sort by date for temporal split
        if temporal_split:
            matches = matches.sort_values("match_date")

        # Split data
        n = len(matches)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_matches = matches.iloc[:train_end]
        val_matches = matches.iloc[train_end:val_end]
        test_matches = matches.iloc[val_end:]

        # Fit pipeline on training data only
        X_train = self.pipeline.fit_transform(train_matches, repository)
        X_val = self.pipeline.transform(val_matches, repository)
        X_test = self.pipeline.transform(test_matches, repository)

        # Extract targets
        y_train = self._encode_targets(train_matches["result"])
        y_val = self._encode_targets(val_matches["result"])
        y_test = self._encode_targets(test_matches["result"])

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _calculate_result(self, matches: pd.DataFrame) -> pd.Series:
        """Calculate match result from scores.

        Args:
            matches: DataFrame with home_score and away_score columns

        Returns:
            Series with 'H', 'D', 'A' values
        """
        results = []

        for _, match in matches.iterrows():
            home_score = match.get("home_score")
            away_score = match.get("away_score")

            if pd.isna(home_score) or pd.isna(away_score):
                results.append(None)
            elif home_score > away_score:
                results.append("H")
            elif home_score < away_score:
                results.append("A")
            else:
                results.append("D")

        return pd.Series(results, index=matches.index)

    def _encode_targets(self, results: pd.Series) -> np.ndarray:
        """Encode results as integers.

        Args:
            results: Series with 'H', 'D', 'A' values

        Returns:
            Array of integers (0=H, 1=D, 2=A)
        """
        mapping = {"H": 0, "D": 1, "A": 2}
        return results.map(mapping).values


def prepare_match_dataframe(matches: list[Any]) -> pd.DataFrame:
    """Convert match ORM objects to DataFrame for feature generation.

    Args:
        matches: List of Match ORM objects

    Returns:
        DataFrame with match data
    """
    records = []

    for match in matches:
        record = {
            "id": match.id,
            "tournament_id": match.tournament_id,
            "season_id": match.season_id,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "match_date": match.match_date,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "status": match.status,
            "odds_home": match.odds_home,
            "odds_draw": match.odds_draw,
            "odds_away": match.odds_away,
            "num_bookmakers": match.num_bookmakers,
        }
        records.append(record)

    return pd.DataFrame(records)
