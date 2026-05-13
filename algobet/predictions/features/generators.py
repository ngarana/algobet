"""Compatibility exports for prediction feature generators.

Feature generators live in focused modules under ``algobet.predictions.features``.
This module preserves the original import path for existing callers.
"""

from algobet.predictions.features.base import FeatureGenerator, FeatureSchema
from algobet.predictions.features.composite import (
    CompositeFeatureGenerator,
    create_default_generators,
    create_generators_by_names,
)
from algobet.predictions.features.elo_rating_generator import EloRatingGenerator
from algobet.predictions.features.enriched_stats_generator import (
    EnrichedStatsFeatureGenerator,
)
from algobet.predictions.features.expected_points_generator import (
    ExpectedPointsGenerator,
)
from algobet.predictions.features.head_to_head_generator import HeadToHeadGenerator
from algobet.predictions.features.odds_generator import OddsFeatureGenerator
from algobet.predictions.features.odds_residual_generator import (
    OddsResidualFeatureGenerator,
)
from algobet.predictions.features.standings_generator import StandingsFeatureGenerator
from algobet.predictions.features.team_form_generator import TeamFormGenerator
from algobet.predictions.features.temporal_generator import TemporalFeatureGenerator

__all__ = [
    "FeatureGenerator",
    "FeatureSchema",
    "TeamFormGenerator",
    "HeadToHeadGenerator",
    "OddsFeatureGenerator",
    "OddsResidualFeatureGenerator",
    "EnrichedStatsFeatureGenerator",
    "StandingsFeatureGenerator",
    "TemporalFeatureGenerator",
    "EloRatingGenerator",
    "ExpectedPointsGenerator",
    "CompositeFeatureGenerator",
    "create_default_generators",
    "create_generators_by_names",
]
