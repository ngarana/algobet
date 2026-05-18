"""Training pipeline for match prediction models."""

from algobet.predictions.training.calibration import (
    CalibratedPredictor,
    ProbabilityCalibrator,
    calculate_calibration_metrics,
    calibration_curve,
)
from algobet.predictions.training.classifiers import (
    CatBoostPredictor,
    DixonColesPredictor,
    EnsemblePredictor,
    HybridPoissonPredictor,
    LightGBMPredictor,
    MatchPredictor,
    ModelConfig,
    RandomForestPredictor,
    XGBoostPredictor,
    compute_adaptive_regularization,
    create_predictor,
)
from algobet.predictions.training.ensemble import EnsembleWeightOptimizer
from algobet.predictions.training.market_mediation import (
    MarketMediationPredictor,
)
from algobet.predictions.training.pipeline import (
    TrainingConfig,
    TrainingPipeline,
    TrainingResult,
    train_model,
)
from algobet.predictions.training.split import (
    ExpandingWindowSplitter,
    OOFTimeAwareSplitter,
    SeasonAwareSplitter,
    TemporalSplit,
    TemporalSplitter,
    WalkForwardSplitter,
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
    "CatBoostPredictor",
    "EnsemblePredictor",
    "DixonColesPredictor",
    "HybridPoissonPredictor",
    "MarketMediationPredictor",
    "create_predictor",
    "compute_adaptive_regularization",
    # Split
    "TemporalSplit",
    "TemporalSplitter",
    "ExpandingWindowSplitter",
    "SeasonAwareSplitter",
    "WalkForwardSplitter",
    "OOFTimeAwareSplitter",
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
    # Ensemble
    "EnsembleWeightOptimizer",
    # Pipeline
    "TrainingPipeline",
    "TrainingConfig",
    "TrainingResult",
    "train_model",
]
