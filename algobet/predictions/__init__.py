"""AlgoBet Prediction Engine - Football match outcome prediction using ML."""

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.evaluation import (
    BettingMetrics,
    CalibrationAnalysisResult,
    ClassificationMetrics,
    EvaluationResult,
    ReportGenerator,
    analyze_calibration,
    compare_models,
    evaluate_predictions,
    generate_evaluation_report,
)
from algobet.predictions.features import (
    FeaturePipeline,
    FeatureSchema,
    FeatureStore,
    FormCalculator,
    create_default_generators,
)
from algobet.predictions.features.pipeline import TrainingDataBuilder
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training import (
    EnsemblePredictor,
    LightGBMPredictor,
    MatchPredictor,
    ModelConfig,
    ProbabilityCalibrator,
    RandomForestPredictor,
    TemporalSplitter,
    TrainingConfig,
    TrainingPipeline,
    TrainingResult,
    XGBoostPredictor,
    create_predictor,
    train_model,
)

__all__ = [
    # Data
    "MatchRepository",
    # Features
    "FormCalculator",
    "FeaturePipeline",
    "FeatureSchema",
    "FeatureStore",
    "TrainingDataBuilder",
    "create_default_generators",
    # Models
    "ModelRegistry",
    # Training - Classifiers
    "MatchPredictor",
    "ModelConfig",
    "XGBoostPredictor",
    "LightGBMPredictor",
    "RandomForestPredictor",
    "EnsemblePredictor",
    "create_predictor",
    # Training - Split
    "TemporalSplitter",
    # Training - Calibration
    "ProbabilityCalibrator",
    # Training - Pipeline
    "TrainingPipeline",
    "TrainingConfig",
    "TrainingResult",
    "train_model",
    # Evaluation - Metrics
    "ClassificationMetrics",
    "BettingMetrics",
    "EvaluationResult",
    "evaluate_predictions",
    "compare_models",
    # Evaluation - Calibration
    "CalibrationAnalysisResult",
    "analyze_calibration",
    # Evaluation - Reports
    "ReportGenerator",
    "generate_evaluation_report",
]
