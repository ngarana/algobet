"""Request and response schemas for ML operation endpoints."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_TRAIN_MODEL_TYPE = os.getenv("ALGOBET_DEFAULT_MODEL_TYPE", "xgboost")


class CalibrateResultResponse(BaseModel):
    """Response schema for calibrate results."""

    base_model_version: str
    calibrated_model_version: str
    method: str
    samples_used: int
    before_metrics: CalibrationMetricsResponse
    after_metrics: CalibrationMetricsResponse
    improvement: dict[str, float]
    is_active: bool


class BacktestHistoryItem(BaseModel):
    """Backtest history item for list response."""

    id: int
    model_version_id: int
    model_name: str | None = None
    model_version: str | None = None
    num_samples: int
    date_range_start: str | None
    date_range_end: str | None
    accuracy: float
    log_loss: float
    f1_macro: float
    roi_percent: float | None
    win_rate: float | None
    evaluated_at: str


class CalibrationMetricsResponse(BaseModel):
    """Calibration metrics response."""

    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    log_loss: float


class BacktestRequest(BaseModel):
    """Request schema for backtest operation."""

    model_version: str | None = None
    tournament_id: int | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_matches: int = Field(default=100, ge=10, le=10000)
    min_edge: float = Field(default=0.0, ge=0.0, le=1.0)


class BacktestHistoryListResponse(BaseModel):
    """Response for backtest history list."""

    items: list[BacktestHistoryItem]
    total: int


class BettingMetricsResponse(BaseModel):
    """Betting metrics response."""

    total_bets: int
    winning_bets: int
    losing_bets: int
    total_stake: float
    total_return: float
    profit_loss: float
    roi_percent: float
    yield_percent: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    average_winning_odds: float
    average_losing_odds: float
    average_kelly_fraction: float
    optimal_kelly_fraction: float
    mean_clv: float = 0.0
    clv_hit_rate: float = 0.0
    clv_weighted_roi: float = 0.0


class BacktestResultResponse(BaseModel):
    """Response schema for backtest results."""

    model_version: str
    evaluated_at: str
    num_samples: int
    date_range: tuple[str, str] | None
    classification: ClassificationMetricsResponse
    betting: BettingMetricsResponse | None = None
    expected_calibration_error: float
    maximum_calibration_error: float
    outcome_accuracy: dict[str, float]


class CalibrateRequest(BaseModel):
    """Request schema for calibrate operation."""

    model_version: str | None = None
    method: str = Field(
        default="isotonic", pattern="^(isotonic|sigmoid|temperature|venn_abers)$"
    )
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    activate: bool = True


class ClassificationMetricsResponse(BaseModel):
    """Classification metrics response."""

    accuracy: float
    log_loss: float
    brier_score: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    top_2_accuracy: float
    cohen_kappa: float


class TrainModelRequest(BaseModel):
    """Request schema for model training."""

    model_type: str = Field(
        default=DEFAULT_TRAIN_MODEL_TYPE,
        pattern="^(xgboost|lightgbm|random_forest|dixon_coles|hybrid_poisson|catboost)$",
    )
    tune_hyperparameters: bool = False
    description: str | None = None
    activate: bool = True

    # Data range settings
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_matches: int = Field(default=100, ge=10, le=100000)

    # Filtering: tournament and team selection
    tournament_ids: list[int] | None = None
    team_ids: list[int] | None = None
    venue_filter: str | None = None  # "home", "away", "both"

    # Match quality filters
    min_total_goals: float | None = None
    max_total_goals: float | None = None

    # Train/val/test split ratios
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.45)
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.45)

    # Training settings
    random_seed: int = Field(default=42, ge=0, le=999999)
    early_stopping_rounds: int = Field(default=50, ge=10, le=500)
    tuning_trials: int = Field(default=50, ge=10, le=500)

    # Calibration settings
    calibrate_probabilities: bool = True
    calibration_method: str = Field(
        default="temperature", pattern="^(isotonic|sigmoid|temperature|venn_abers)$"
    )

    # Outcome balancing control
    outcome_balance: bool | None = None
    outcome_balance_strength: float = Field(default=0.5, ge=0.0, le=1.0)

    # Feature importance pruning
    feature_selection: bool = True
    feature_selection_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    min_samples_per_feature: int | None = Field(default=None, ge=1)

    # Feature groups selection (subset of available generators)
    feature_groups: list[str] | None = None

    # Ensemble training
    use_ensemble: bool = False
    ensemble_types: list[str] | None = None  # e.g. ["xgboost", "lightgbm"]

    # Stacking ensemble
    use_stacking_ensemble: bool = False
    stacking_base_models: list[str] = Field(
        default_factory=lambda: ["xgboost", "dixon_coles"]
    )
    stacking_meta_learner: str = Field(default="logistic", pattern="^(logistic|mlp)$")
    stacking_n_folds: int = Field(default=5, ge=3, le=10)

    # Split strategy
    split_strategy: str = Field(
        default="walk_forward",
        pattern="^(temporal|expanding_window|season_aware|walk_forward)$",
    )
    gap_days: int = Field(default=0, ge=0, le=30)
    # Expanding window params (used when split_strategy = expanding_window)
    min_train_size: int = Field(default=100, ge=50, le=5000)
    ew_val_size: int = Field(default=50, ge=10, le=1000)
    ew_test_size: int = Field(default=50, ge=10, le=1000)
    step_size: int = Field(default=50, ge=10, le=500)
    # Season-aware params (used when split_strategy = season_aware)
    train_seasons: int = Field(default=8, ge=1, le=10)
    val_seasons: int = Field(default=1, ge=1, le=5)
    test_seasons: int = Field(default=1, ge=1, le=5)

    # Model tags
    tags: dict[str, str] = Field(default_factory=dict)

    # Custom hyperparameters (optional)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainModelResponse(BaseModel):
    """Response schema for model training."""

    model_id: int | None = None
    model_version: str
    model_type: str
    is_active: bool
    feature_schema_version: str
    num_features: int
    trained_at: str
    training_duration_seconds: float
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importance: dict[str, float] | None = None
    # Ensemble metadata
    ensemble_weights: dict[str, float] | None = None
    ensemble_validation_metrics: dict[str, float] | None = None
    ensemble_types: list[str] | None = None
    # Stacking metadata
    stacking_metadata: dict[str, Any] | None = None
    # Calibration metadata
    calibration_metadata: dict[str, Any] | None = None
