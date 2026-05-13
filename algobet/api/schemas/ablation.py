"""Request and response schemas for ablation / permutation endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PermutationFamilyResultSchema(BaseModel):
    """Per-family result from a permutation importance run."""

    family: str
    features_in_family: list[str]
    features_found: list[str]
    baseline_log_loss: float
    permuted_log_loss: float
    log_loss_increase: float
    baseline_accuracy: float
    permuted_accuracy: float
    accuracy_decrease: float
    importance_score: float
    importance_rank: int


class AblationModelConfig(BaseModel):
    """Model configuration for ablation retraining runs."""

    model_type: str = Field(
        default="xgboost",
        pattern="^(xgboost|lightgbm|random_forest)$",
    )
    tune_hyperparameters: bool = False
    early_stopping_rounds: int = Field(default=50, ge=10, le=500)
    calibrate_probabilities: bool = True
    calibration_method: str = Field(default="sigmoid", pattern="^(isotonic|sigmoid)$")
    random_seed: int = Field(default=42, ge=0, le=999999)


class AblationRequest(BaseModel):
    """Request schema for ablation / permutation importance analysis."""

    method: str = Field(
        default="permutation",
        pattern="^(permutation|ablation)$",
        description=(
            "'permutation' shuffles feature columns on a trained model (fast). "
            "'ablation' retrains excluding each feature group (slow)."
        ),
    )

    model_version: str | None = Field(
        default=None,
        description="Model version to analyse. None uses the active model.",
    )

    # Permutation-specific settings
    n_repeats: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of shuffle repeats (permutation only).",
    )
    random_state: int = Field(default=42, ge=0, le=999999)

    # Feature families / groups to analyse
    feature_families: list[str] | None = Field(
        default=None,
        description=(
            "Feature families to evaluate. "
            "For permutation: sub-groups like 'draw', 'away', 'form', etc. "
            "For ablation: generator groups like 'team_form', 'head_to_head', etc. "
            "None = evaluate all available families/groups."
        ),
    )
    group_by: str = Field(
        default="family",
        pattern="^(family|generator)$",
        description=(
            "'family' groups by sub-family patterns (draw, away, form, …). "
            "'generator' groups by feature generator (team_form, h2h, …)."
        ),
    )

    # Data filters (shared with training / backtest)
    start_date: datetime | None = None
    end_date: datetime | None = None
    tournament_ids: list[int] | None = None
    min_matches: int = Field(default=100, ge=10, le=10000)

    # Ablation-specific training overrides
    ablation_model_config: AblationModelConfig | None = Field(
        default=None,
        description="Model training config used for each ablation run.",
    )

    # Train/test split (used by both methods for evaluation data)
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.45)
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.45)
    gap_days: int = Field(default=0, ge=0, le=30)


class AblationFamilyResult(BaseModel):
    """Per-family result from a leave-one-out ablation study."""

    family: str
    features_excluded: list[str]
    num_features_used: int
    model_version: str
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    log_loss_delta: float
    accuracy_delta: float


class PermutationImportanceResponse(BaseModel):
    """Response schema for permutation importance analysis."""

    method: str = "permutation"
    model_version: str
    num_samples: int
    n_repeats: int
    baseline_log_loss: float
    baseline_accuracy: float
    families: list[PermutationFamilyResultSchema]
    raw_feature_importance: dict[str, float] | None = None


class AblationStudyResponse(BaseModel):
    """Response schema for leave-one-out ablation study."""

    method: str = "ablation"
    baseline_model_version: str
    baseline_num_features: int
    baseline_train_metrics: dict[str, float]
    baseline_val_metrics: dict[str, float]
    baseline_test_metrics: dict[str, float]
    families: list[AblationFamilyResult]
