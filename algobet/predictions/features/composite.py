"""Composite feature generator and generator factories."""

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator, FeatureSchema
from algobet.predictions.features.enriched_stats_generator import (
    EnrichedStatsFeatureGenerator,
)
from algobet.predictions.features.head_to_head_generator import HeadToHeadGenerator
from algobet.predictions.features.standings_generator import StandingsFeatureGenerator
from algobet.predictions.features.team_form_generator import TeamFormGenerator
from algobet.predictions.features.temporal_generator import TemporalFeatureGenerator


class CompositeFeatureGenerator(FeatureGenerator):
    """Combines multiple feature generators into a single generator.

    Allows composition of feature generators to create comprehensive
    feature sets.
    """

    def __init__(self, generators: list[FeatureGenerator]) -> None:
        """Initialize composite generator.

        Args:
            generators: List of feature generators to combine
        """
        self.generators = generators

    @property
    def name(self) -> str:
        return "composite"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for gen in self.generators:
            names.extend(gen.feature_names)
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate all features from combined generators.

        Args:
            matches: Match data
            repository: Repository for historical queries

        Returns:
            DataFrame with all combined features indexed by match id
        """
        feature_dfs = []

        for gen in self.generators:
            df = gen.generate(matches, repository)
            feature_dfs.append(df)

        # Merge all feature DataFrames
        if not feature_dfs:
            return pd.DataFrame(index=matches["id"])

        result = feature_dfs[0]
        for df in feature_dfs[1:]:
            result = result.join(df, how="outer")

        return result

    def get_schema(self) -> FeatureSchema:
        """Return combined schema from all generators."""
        all_features: dict[str, type] = {}
        for gen in self.generators:
            schema = gen.get_schema()
            all_features.update(schema.features)

        return FeatureSchema(
            version="v1.0",
            features=all_features,
        )


def create_default_generators() -> CompositeFeatureGenerator:
    """Create the default set of feature generators.

    Returns:
        CompositeFeatureGenerator with all standard generators
    """
    return create_generators_by_names(
        [
            "team_form",
            "head_to_head",
            "temporal",
            "standings",
            "enriched_stats",
        ]
    )


def create_generators_by_names(generator_names: list[str]) -> CompositeFeatureGenerator:
    """Create feature generators by their stable group names."""
    generator_factories = {
        "team_form": lambda: TeamFormGenerator(
            window_sizes=[3, 5, 10],
            include_venue_specific=True,
        ),
        "head_to_head": lambda: HeadToHeadGenerator(
            max_h2h_matches=5,
            max_years_back=3,
        ),
        "temporal": lambda: TemporalFeatureGenerator(
            include_rest_days=True,
            include_fixture_density=True,
            include_season_period=True,
        ),
        "standings": lambda: StandingsFeatureGenerator(
            relegation_threshold=3,
            euro_spot_start=4,
            euro_spot_end=7,
        ),
        "enriched_stats": lambda: EnrichedStatsFeatureGenerator(
            window_sizes=[3, 5],
            include_diffs=False,
        ),
    }

    unsupported = sorted(set(generator_names) - set(generator_factories))
    if unsupported:
        raise ValueError(f"Unsupported feature generators: {', '.join(unsupported)}")

    return CompositeFeatureGenerator(
        generators=[generator_factories[name]() for name in generator_names]
    )
