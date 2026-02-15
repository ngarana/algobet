"""Data Transfer Objects (DTOs) for the service layer.

This module defines all DTOs used for communication between layers.
DTOs are immutable dataclasses that encapsulate request and response data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

# =============================================================================
# Database DTOs
# =============================================================================


@dataclass(frozen=True)
class DatabaseStatsRequest:
    """Request for database statistics."""

    pass


@dataclass(frozen=True)
class DatabaseStatsResponse:
    """Response with database statistics."""

    tournaments_count: int
    seasons_count: int
    teams_count: int
    matches_count: int
    finished_matches_count: int
    scheduled_matches_count: int
    model_versions_count: int
    scheduled_tasks_count: int


@dataclass(frozen=True)
class DatabaseInitRequest:
    """Request for database initialization."""

    drop_existing: bool = False


@dataclass(frozen=True)
class DatabaseInitResponse:
    """Response from database initialization."""

    success: bool
    tables_created: int
    message: str


@dataclass(frozen=True)
class DatabaseResetRequest:
    """Request for database reset."""

    confirm: bool = False


@dataclass(frozen=True)
class DatabaseResetResponse:
    """Response from database reset."""

    success: bool
    tables_dropped: int
    tables_created: int
    message: str


# =============================================================================
# Query DTOs
# =============================================================================


@dataclass(frozen=True)
class TournamentFilter:
    """Filter criteria for tournaments."""

    name: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class SeasonFilter:
    """Filter criteria for seasons."""

    tournament_name: str | None = None
    season_name: str | None = None


@dataclass(frozen=True)
class TeamFilter:
    """Filter criteria for teams."""

    name: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class MatchFilter:
    """Filter criteria for matches."""

    tournament_name: str | None = None
    season_name: str | None = None
    team_name: str | None = None
    status: str | None = None  # SCHEDULED, FINISHED, LIVE
    limit: int = 50


@dataclass(frozen=True)
class TournamentInfo:
    """Information about a tournament."""

    id: int
    name: str
    url_slug: str
    seasons_count: int


@dataclass(frozen=True)
class TournamentListResponse:
    """Response with list of tournaments."""

    tournaments: list[TournamentInfo]


@dataclass(frozen=True)
class SeasonInfo:
    """Information about a season."""

    id: int
    name: str
    tournament_name: str
    matches_count: int


@dataclass(frozen=True)
class SeasonListResponse:
    """Response with list of seasons."""

    seasons: list[SeasonInfo]


@dataclass(frozen=True)
class TeamInfo:
    """Information about a team."""

    id: int
    name: str
    matches_played: int


@dataclass(frozen=True)
class TeamListResponse:
    """Response with list of teams."""

    teams: list[TeamInfo]


@dataclass(frozen=True)
class MatchInfo:
    """Information about a match."""

    id: int
    home_team: str
    away_team: str
    match_date: datetime | None
    status: str
    home_score: int | None
    away_score: int | None
    tournament_name: str
    season_name: str


@dataclass(frozen=True)
class MatchListResponse:
    """Response with list of matches."""

    matches: list[MatchInfo]
    total_count: int


# =============================================================================
# Model Management DTOs
# =============================================================================


@dataclass(frozen=True)
class ModelListRequest:
    """Request for listing models."""

    include_inactive: bool = False


@dataclass(frozen=True)
class ModelInfo:
    """Information about a model version."""

    version: str
    created_at: datetime
    metrics: dict[str, float] | None
    is_active: bool
    model_type: str
    features_count: int


@dataclass(frozen=True)
class ModelListResponse:
    """Response with list of models."""

    models: list[ModelInfo]
    active_model_version: str | None


@dataclass(frozen=True)
class ModelActivateRequest:
    """Request to activate a model."""

    version: str


@dataclass(frozen=True)
class ModelActivateResponse:
    """Response from model activation."""

    success: bool
    previous_active_version: str | None
    new_active_version: str
    message: str


@dataclass(frozen=True)
class ModelInfoRequest:
    """Request for detailed model information."""

    version: str


@dataclass(frozen=True)
class ModelDetailResponse:
    """Response with detailed model information."""

    version: str
    created_at: datetime
    model_type: str
    features: list[str]
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    training_data_size: int
    is_active: bool


# =============================================================================
# Analysis DTOs
# =============================================================================


@dataclass(frozen=True)
class BacktestRequest:
    """Request for running backtest."""

    min_matches: int = 100
    validation_split: float = 0.2
    model_version: str | None = None


@dataclass(frozen=True)
class BacktestResponse:
    """Response from backtest run."""

    success: bool
    total_matches: int
    training_matches: int
    validation_matches: int
    metrics: dict[str, float]
    model_version: str
    execution_time_seconds: float


@dataclass(frozen=True)
class ValueBetsRequest:
    """Request for finding value bets."""

    min_edge: float = 0.05
    model_version: str | None = None
    limit: int = 20


@dataclass(frozen=True)
class ValueBetInfo:
    """Information about a value bet."""

    match_id: int
    home_team: str
    away_team: str
    match_date: datetime
    bet_type: str  # HOME_WIN, DRAW, AWAY_WIN
    model_probability: float
    market_odds: float | None
    edge: float
    expected_value: float


@dataclass(frozen=True)
class ValueBetsResponse:
    """Response with value bets."""

    value_bets: list[ValueBetInfo]
    model_version: str
    generated_at: datetime


@dataclass(frozen=True)
class CalibrateRequest:
    """Request for model calibration."""

    model_version: str | None = None
    method: str = "isotonic"  # isotonic, sigmoid


@dataclass(frozen=True)
class CalibrateResponse:
    """Response from calibration."""

    success: bool
    model_version: str
    calibration_method: str
    before_calibration_score: float
    after_calibration_score: float
    improvement: float
