"""Training pipeline for match prediction models."""

from algobet.predictions.training.calibration import (
    CalibratedPredictor,
    ProbabilityCalibrator,
    calculate_calibration_metrics,
    calibration_curve,
)
from algobet.predictions.training.classifiers import (
    EnsemblePredictor,
    LightGBMPredictor,
    MatchPredictor,
    ModelConfig,
    RandomForestPredictor,
    XGBoostPredictor,
    create_predictor,
)
from algobet.predictions.training.pipeline import (
    TrainingConfig,
    TrainingPipeline,
    TrainingResult,
    train_model,
)
from algobet.predictions.training.split import (
    ExpandingWindowSplitter,
    SeasonAwareSplitter,
    TemporalSplit,
    TemporalSplitter,
    decode_targets,
    encode_targets,
    get_class_weights,
)
from algobet.predictions.training.tuner import (
    HAS_OPTUNA,
    GridSearchTuner,
    HyperparameterTuner,
    TuningConfig,
    TuningResult,
    get_default_search_space,
)

__all__ = [
    # Classifiers
    "MatchPredictor",
    "ModelConfig",
    "XGBoostPredictor",
    "LightGBMPredictor",
    "RandomForestPredictor",
    "EnsemblePredictor",
    "create_predictor",
    # Split
    "TemporalSplit",
    "TemporalSplitter",
    "ExpandingWindowSplitter",
    "SeasonAwareSplitter",
    "encode_targets",
    "decode_targets",
    "get_class_weights",
    # Tuning
    "HyperparameterTuner",
    "GridSearchTuner",
    "TuningConfig",
    "TuningResult",
    "HAS_OPTUNA",
    "get_default_search_space",
    # Calibration
    "ProbabilityCalibrator",
    "CalibratedPredictor",
    "calculate_calibration_metrics",
    "calibration_curve",
    # Pipeline
    "TrainingPipeline",
    "TrainingConfig",
    "TrainingResult",
    "train_model",
]
