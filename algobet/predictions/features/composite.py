"""Composite feature generator and generator factories."""

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator, FeatureSchema
from algobet.predictions.features.detailed_odds_generator import (
    DetailedOddsFeatureGenerator,
)
from algobet.predictions.features.draw_signal_generator import (
    DrawSignalFeatureGenerator,
)
from algobet.predictions.features.elo_rating_generator import EloRatingGenerator
from algobet.predictions.features.enriched_stats_generator import (
    EnrichedStatsFeatureGenerator,
)
from algobet.predictions.features.expected_points_generator import (
    ExpectedPointsGenerator,
)
from algobet.predictions.features.head_to_head_generator import HeadToHeadGenerator
from algobet.predictions.features.market_mediation_generator import (
    MarketMediationFeatureGenerator,
)
from algobet.predictions.features.matchup_interaction_generator import (
    MatchupInteractionGenerator,
)
from algobet.predictions.features.odds_generator import OddsFeatureGenerator
from algobet.predictions.features.odds_residual_generator import (
    OddsResidualFeatureGenerator,
)
from algobet.predictions.features.player_quality_generator import (
    PlayerQualityGenerator,
)
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
            "draw_signals",
            "matchup_interaction",
        ]
    )


def create_generators_by_names(generator_names: list[str]) -> CompositeFeatureGenerator:
    """Create feature generators by their stable group names."""
    generator_factories = {
        "team_form": lambda: TeamFormGenerator(
            window_sizes=[5, 10],
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
            include_basic_stats=False,
            include_player_stats=False,
        ),
        "elo_rating": lambda: EloRatingGenerator(
            k_factor=32.0,
            home_advantage=65.0,
            initial_rating=1500.0,
            window_sizes=[5, 10, 20],
        ),
        "expected_points": lambda: ExpectedPointsGenerator(
            window_sizes=[3, 5, 10],
        ),
        "draw_signals": lambda: DrawSignalFeatureGenerator(
            window_sizes=[3, 5, 10],
            xg_window_sizes=[3, 5],
        ),
        "matchup_interaction": lambda: MatchupInteractionGenerator(
            window_sizes=[5, 10],
        ),
        "player_quality": lambda: PlayerQualityGenerator(
            window_sizes=[3, 5],
        ),
        "odds": lambda: OddsFeatureGenerator(),
        "odds_residual": lambda: OddsResidualFeatureGenerator(),
        "detailed_odds": lambda: DetailedOddsFeatureGenerator(),
        "market_mediation": lambda: MarketMediationFeatureGenerator(),
    }

    unsupported = sorted(set(generator_names) - set(generator_factories))
    if unsupported:
        raise ValueError(f"Unsupported feature generators: {', '.join(unsupported)}")

    return CompositeFeatureGenerator(
        generators=[generator_factories[name]() for name in generator_names]
    )
