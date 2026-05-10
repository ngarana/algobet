"""Training pipeline configuration contracts and constants."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from algobet.predictions.training.feature_selection import FeatureSelectionReport
from algobet.predictions.training.tuner import TuningResult

MODEL_FEATURE_SCHEMA_VERSION = "v3.0_epl_feature_tuning"
ALLOWED_FEATURE_GROUPS = (
    "team_form",
    "head_to_head",
    "temporal",
    "standings",
    "enriched_stats",
)

# Search spaces for per-model tuning
XGBOOST_SEARCH_SPACE = {
    "max_depth": (2, 5),
    "learning_rate": (0.01, 0.08),
    "n_estimators": (400, 2000),
    "min_child_weight": (3, 30),
    "gamma": (0.0, 3.0),
    "reg_alpha": (0.0, 8.0),
    "reg_lambda": (2.0, 30.0),
    "subsample": (0.55, 0.90),
    "colsample_bytree": (0.40, 0.85),
}
LIGHTGBM_SEARCH_SPACE = {
    "num_leaves": (7, 63),
    "max_depth": (2, 6),
    "learning_rate": (0.01, 0.08),
    "n_estimators": (400, 2500),
    "min_child_samples": (20, 150),
    "min_split_gain": (0.0, 2.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (2.0, 40.0),
    "subsample": (0.55, 0.90),
    "colsample_bytree": (0.40, 0.85),
}


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model settings
    model_type: str = "xgboost"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    use_ensemble: bool = False
    ensemble_types: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])

    # Data range settings
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_matches: int | None = None

    # Filtering: tournament and team selection
    tournament_ids: list[int] | None = None
    team_ids: list[int] | None = None
    venue_filter: str | None = None  # "home", "away", "both"

    # Match quality filters
    min_total_goals: float | None = None
    max_total_goals: float | None = None

    # Feature settings
    feature_schema_version: str = MODEL_FEATURE_SCHEMA_VERSION
    use_feature_cache: bool = True

    # Feature group selection
    feature_groups: list[str] | None = None

    # Split settings
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_strategy: str = "temporal"  # "temporal", "expanding_window", "season_aware"
    gap_days: int = 0
    # Expanding window params
    min_train_size: int = 100
    ew_val_size: int = 50
    ew_test_size: int = 50
    step_size: int = 50
    # Season-aware params
    train_seasons: int = 3
    val_seasons: int = 1
    test_seasons: int = 1

    # Tuning settings
    tune_hyperparameters: bool = False
    tuning_trials: int = 50
    per_model_hyperparameters: bool = True

    # Calibration settings
    calibrate_probabilities: bool = True
    calibration_method: str = "sigmoid"

    # Training settings
    early_stopping_rounds: int = 50
    random_seed: int = 42

    # Outcome balancing control (opt-in; disabled by default to preserve
    # calibrated probabilities)
    outcome_balance: bool = False

    # Degenerate model protection
    collapse_recovery: bool = True
    min_prediction_classes: int = 3

    # Feature selection settings
    feature_selection: bool = False
    feature_selection_threshold: float = 0.002
    min_samples_per_feature: int | None = None
    max_feature_correlation: float = 0.94

    # Family retention guards (minimum features to keep per family)
    min_draw_features: int = 3
    min_away_features: int = 3
    min_enriched_or_coverage: int = 5
    min_low_scoring_features: int = 2

    # Output settings
    model_name: str = "match_predictor"
    description: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of a training run."""

    # Model info
    model_version: str
    model_type: str

    # Metrics
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]

    # Feature info
    feature_schema_version: str
    num_features: int
    feature_importance: dict[str, float] | None

    # Training info
    trained_at: datetime
    training_duration_seconds: float
    config: TrainingConfig

    # Optional per-model tuning results
    per_model_tuning_results: dict[str, TuningResult] | None = None

    # Ensemble info
    ensemble_weights: dict[str, float] | None = None
    ensemble_validation_metrics: dict[str, float] | None = None

    # Feature selection
    feature_selection_report: FeatureSelectionReport | None = None

    # Path
    model_path: Path | None = None
