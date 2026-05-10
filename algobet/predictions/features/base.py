"""Base contracts for prediction feature generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from algobet.predictions.data.queries import MatchRepository


@dataclass
class FeatureSchema:
    """Schema definition for a set of features."""

    version: str
    features: dict[str, type]
    description: str | None = None

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Validate that dataframe contains expected features.

        Args:
            df: DataFrame to validate

        Returns:
            List of missing feature names
        """
        missing = []
        for name, _dtype in self.features.items():
            if name not in df.columns:
                missing.append(name)
        return missing

    def get_feature_names(self) -> list[str]:
        """Return list of feature names."""
        return list(self.features.keys())


class FeatureGenerator(ABC):
    """Abstract base class for feature generators.

    Feature generators transform raw match data into feature vectors.
    Each generator produces a specific category of features.
    """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names this generator produces."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return generator name for identification."""

    @abstractmethod
    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate features for matches.

        Args:
            matches: DataFrame with match records, must include:
                - id: match identifier
                - home_team_id: home team ID
                - away_team_id: away team ID
                - match_date: match datetime
                - home_score, away_score: scores (may be None for upcoming)
                - odds_home, odds_draw, odds_away: betting odds (may be None)
            repository: MatchRepository for historical queries

        Returns:
            DataFrame indexed by match_id with generated features
        """

    def get_schema(self) -> FeatureSchema:
        """Return the feature schema for this generator."""
        return FeatureSchema(
            version="v1.0",
            features=dict.fromkeys(self.feature_names, float),
        )
