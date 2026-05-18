"""Feature engineering module for match predictions."""

from algobet.predictions.features.base import FeatureGenerator, FeatureSchema
from algobet.predictions.features.composite import (
    CompositeFeatureGenerator,
    create_default_generators,
    create_generators_by_names,
)
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
from algobet.predictions.features.form_features import FormCalculator
from algobet.predictions.features.head_to_head_generator import HeadToHeadGenerator
from algobet.predictions.features.market_mediation_generator import (
    MarketMediationFeatureGenerator,
)
from algobet.predictions.features.odds_generator import OddsFeatureGenerator
from algobet.predictions.features.odds_residual_generator import (
    OddsResidualFeatureGenerator,
)
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    PipelineConfig,
    TrainingDataBuilder,
    prepare_match_dataframe,
)
from algobet.predictions.features.standings_generator import StandingsFeatureGenerator
from algobet.predictions.features.store import (
    FeatureStore,
    features_to_store_format,
)
from algobet.predictions.features.team_form_generator import TeamFormGenerator
from algobet.predictions.features.temporal_generator import TemporalFeatureGenerator
from algobet.predictions.features.transformers import (
    FeatureScaler,
    FeatureSelector,
    MissingValueHandler,
    OddsTransformer,
    PreserveMissingValues,
    TransformerPipeline,
    create_default_transformer_pipeline,
    create_tree_model_transformer_pipeline,
)

__all__ = [
    # Legacy
    "FormCalculator",
    # Generators
    "FeatureGenerator",
    "FeatureSchema",
    "TeamFormGenerator",
    "HeadToHeadGenerator",
    "OddsFeatureGenerator",
    "OddsResidualFeatureGenerator",
    "DetailedOddsFeatureGenerator",
    "MarketMediationFeatureGenerator",
    "EnrichedStatsFeatureGenerator",
    "StandingsFeatureGenerator",
    "TemporalFeatureGenerator",
    "EloRatingGenerator",
    "ExpectedPointsGenerator",
    "DrawSignalFeatureGenerator",
    "CompositeFeatureGenerator",
    "create_default_generators",
    "create_generators_by_names",
    # Transformers
    "FeatureScaler",
    "MissingValueHandler",
    "FeatureSelector",
    "OddsTransformer",
    "PreserveMissingValues",
    "TransformerPipeline",
    "create_default_transformer_pipeline",
    "create_tree_model_transformer_pipeline",
    # Pipeline
    "FeaturePipeline",
    "PipelineConfig",
    "TrainingDataBuilder",
    "prepare_match_dataframe",
    # Store
    "FeatureStore",
    "features_to_store_format",
]
