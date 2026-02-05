"""Pydantic schemas for match-related API responses."""

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

from .team import TeamResponse
from .tournament import SeasonResponse, TournamentResponse

if TYPE_CHECKING:
    from .prediction import PredictionResponse


class MatchStatus:
    """Match status constants."""

    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    LIVE = "LIVE"


class PredictedOutcome:
    """Predicted outcome constants."""

    HOME = "H"
    DRAW = "D"
    AWAY = "A"


class MatchResponse(BaseModel):
    """Match response schema."""

    id: int
    tournament_id: int
    season_id: int
    home_team_id: int
    away_team_id: int
    match_date: datetime
    home_score: int | None = None
    away_score: int | None = None
    status: str
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None
    num_bookmakers: int | None = None
    created_at: datetime
    updated_at: datetime

    # Computed field - derived from scores if match is finished
    result: str | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is a valid value."""
        valid_statuses = {
            MatchStatus.SCHEDULED,
            MatchStatus.FINISHED,
            MatchStatus.LIVE,
        }
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v

    @field_validator("result")
    @classmethod
    def validate_result(cls, v: str | None) -> str | None:
        """Validate result is a valid value if provided."""
        if v is None:
            return None
        valid_outcomes = {
            PredictedOutcome.HOME,
            PredictedOutcome.DRAW,
            PredictedOutcome.AWAY,
        }
        if v not in valid_outcomes:
            raise ValueError(f"Invalid result: {v}. Must be one of {valid_outcomes}")
        return v


class MatchDetailResponse(MatchResponse):
    """Match with related entities and predictions."""

    tournament: TournamentResponse
    season: SeasonResponse
    home_team: TeamResponse
    away_team: TeamResponse
    predictions: list["PredictionResponse"]
    h2h_matches: list[MatchResponse]  # Last 5 head-to-head


class MatchFilters(BaseModel):
    """Query parameters for filtering matches."""

    status: str | None = None
    tournament_id: int | None = None
    season_id: int | None = None
    team_id: int | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    days_ahead: int | None = None
    has_odds: bool | None = None
    limit: int = 50
    offset: int = 0

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Validate limit is within acceptable range."""
        if v < 1 or v > 100:
            raise ValueError("limit must be between 1 and 100")
        return v
