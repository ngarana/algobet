"""Feature engineering module for match predictions."""

from algobet.predictions.features.form_features import FormCalculator
from algobet.predictions.features.generators import (
    CompositeFeatureGenerator,
    FeatureGenerator,
    FeatureSchema,
    HeadToHeadGenerator,
    OddsFeatureGenerator,
    TeamFormGenerator,
    TemporalFeatureGenerator,
    create_default_generators,
)
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    PipelineConfig,
    TrainingDataBuilder,
    prepare_match_dataframe,
)
from algobet.predictions.features.store import (
    FeatureStore,
    features_to_store_format,
)
from algobet.predictions.features.transformers import (
    FeatureScaler,
    FeatureSelector,
    MissingValueHandler,
    OddsTransformer,
    TransformerPipeline,
    create_default_transformer_pipeline,
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
    "TemporalFeatureGenerator",
    "CompositeFeatureGenerator",
    "create_default_generators",
    # Transformers
    "FeatureScaler",
    "MissingValueHandler",
    "FeatureSelector",
    "OddsTransformer",
    "TransformerPipeline",
    "create_default_transformer_pipeline",
    # Pipeline
    "FeaturePipeline",
    "PipelineConfig",
    "TrainingDataBuilder",
    "prepare_match_dataframe",
    # Store
    "FeatureStore",
    "features_to_store_format",
]
