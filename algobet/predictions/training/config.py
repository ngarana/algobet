"""Training pipeline configuration contracts and constants."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from algobet.predictions.training.feature_selection import FeatureSelectionReport
from algobet.predictions.training.tuner import TuningResult

MODEL_FEATURE_SCHEMA_VERSION = "v4.0_attack_defense_clash"
ALLOWED_FEATURE_GROUPS = (
    "team_form",
    "head_to_head",
    "temporal",
    "standings",
    "enriched_stats",
    "elo_rating",
    "expected_points",
    "draw_signals",
    "matchup_interaction",
    "player_quality",
    "odds",
    "odds_residual",
)

# Search spaces for per-model tuning (~1900 training rows, ~200 features)
XGBOOST_SEARCH_SPACE = {
    "max_depth": (2, 4),
    "learning_rate": (0.01, 0.08),
    "n_estimators": (200, 800),
    "min_child_weight": (8, 50),
    "gamma": (0.0, 3.0),
    "reg_alpha": (0.5, 10.0),
    "reg_lambda": (5.0, 40.0),
    "subsample": (0.55, 0.85),
    "colsample_bytree": (0.35, 0.75),
}
LIGHTGBM_SEARCH_SPACE = {
    "num_leaves": (7, 31),
    "max_depth": (2, 5),
    "learning_rate": (0.01, 0.08),
    "n_estimators": (200, 1000),
    "min_child_samples": (30, 200),
    "min_split_gain": (0.0, 2.0),
    "reg_alpha": (0.5, 10.0),
    "reg_lambda": (5.0, 50.0),
    "subsample": (0.55, 0.85),
    "colsample_bytree": (0.35, 0.75),
}


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model settings
    model_type: str = "xgboost"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    use_ensemble: bool = False
    ensemble_types: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])

    # Stacking ensemble settings
    use_stacking_ensemble: bool = False
    stacking_base_models: list[str] = field(
        default_factory=lambda: ["xgboost", "dixon_coles"]
    )
    stacking_meta_learner: str = "logistic"  # "logistic" or "mlp"
    stacking_n_folds: int = 5

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
    split_strategy: str = (
        "season_aware"  # "temporal", "expanding_window", "season_aware", "walk_forward"
    )
    gap_days: int = 14
    # Expanding window params
    min_train_size: int = 100
    ew_val_size: int = 50
    ew_test_size: int = 50
    step_size: int = 50
    # Season-aware params
    train_seasons: int = 8
    val_seasons: int = 1
    test_seasons: int = 1

    # Tuning settings
    tune_hyperparameters: bool = False
    tuning_trials: int = 50
    per_model_hyperparameters: bool = True

    # Calibration settings
    calibrate_probabilities: bool = True
    calibration_method: str = "temperature"
    use_cv_calibration: bool = True
    calibration_cv_folds: int = 5

    # Draw-aware calibration (Dixon-Coles blend)
    fit_draw_aware_calibrator: bool = False
    dc_model_path: str | None = None

    # Simple post-hoc draw boost (multiplies draw probability)
    draw_boost_factor: float = 1.0

    # Training settings
    early_stopping_rounds: int = 50
    random_seed: int = 42

    # Outcome balancing control (opt-in; disabled by default to preserve
    # calibrated probabilities)
    outcome_balance: bool = False
    outcome_balance_strength: float = 0.5

    # Degenerate model protection
    collapse_recovery: bool = True
    min_prediction_classes: int = 3

    # Feature selection settings
    feature_selection: bool = True
    feature_selection_threshold: float = 0.005
    min_samples_per_feature: int | None = None
    max_feature_correlation: float = 0.90

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
