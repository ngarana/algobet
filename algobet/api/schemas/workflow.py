"""Schemas for daily workflow, watchlist, and user predictions."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .match import MatchDetailResponse, MatchResponse
from .prediction import PredictionListItemResponse, ValueBetResponse

VALID_WATCHLIST_TYPES = {"team", "tournament", "match"}
VALID_OUTCOMES = {"H", "D", "A"}
VALID_TOTAL_GOALS_PICKS = {"OVER", "UNDER"}


class ProfilePreferencesRequest(BaseModel):
    """Update request for local workflow preferences."""

    display_name: str | None = None
    default_days_ahead: int | None = Field(default=None, ge=1, le=30)
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    min_ev: float | None = Field(default=None, ge=0)
    favorite_bookie: str | None = None
    followed_tournament_ids: list[int] | None = None


class ProfilePreferencesResponse(BaseModel):
    """Local profile preferences and followed leagues."""

    profile_key: str
    display_name: str
    default_days_ahead: int
    min_confidence: float
    min_ev: float
    favorite_bookie: str | None = None
    followed_tournament_ids: list[int]


class WatchlistEntryRequest(BaseModel):
    """Request to add a watchlist entry."""

    entry_type: str
    entry_id: int

    @field_validator("entry_type")
    @classmethod
    def validate_entry_type(cls, value: str) -> str:
        if value not in VALID_WATCHLIST_TYPES:
            raise ValueError("entry_type must be team, tournament, or match")
        return value


class WatchlistEntryResponse(BaseModel):
    """A watchlist entry with a display label."""

    id: int
    entry_type: str
    entry_id: int
    label: str
    meta: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WatchlistResponse(BaseModel):
    """Grouped watchlist entries."""

    teams: list[WatchlistEntryResponse]
    tournaments: list[WatchlistEntryResponse]
    matches: list[WatchlistEntryResponse]


class UserPredictionRequest(BaseModel):
    """Create or update a user's prediction for a match."""

    match_id: int
    pick_1x2: str | None = None
    home_score: int | None = Field(default=None, ge=0)
    away_score: int | None = Field(default=None, ge=0)
    total_goals_line: float | None = Field(default=None, ge=0)
    total_goals_pick: str | None = None
    notes: str | None = None

    @field_validator("pick_1x2")
    @classmethod
    def validate_pick(cls, value: str | None) -> str | None:
        if value is not None and value not in VALID_OUTCOMES:
            raise ValueError("pick_1x2 must be H, D, or A")
        return value

    @field_validator("total_goals_pick")
    @classmethod
    def validate_total_goals_pick(cls, value: str | None) -> str | None:
        if value is not None and value not in VALID_TOTAL_GOALS_PICKS:
            raise ValueError("total_goals_pick must be OVER or UNDER")
        return value


class UserPredictionResponse(BaseModel):
    """A user's prediction, with model comparison and scoring when available."""

    id: int
    match_id: int
    pick_1x2: str | None
    home_score: int | None
    away_score: int | None
    total_goals_line: float | None
    total_goals_pick: str | None
    notes: str | None
    model_prediction: PredictionListItemResponse | None = None
    is_correct_1x2: bool | None = None
    is_exact_score: bool | None = None
    points: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DailyWorkflowResponse(BaseModel):
    """Dashboard data for the daily workflow."""

    date: str
    today_matches: list[PredictionListItemResponse]
    high_confidence: list[PredictionListItemResponse]
    value_bets: list[ValueBetResponse]
    watched_fixtures: list[MatchDetailResponse]
    results_summary: "ResultsSummaryResponse"
    watchlist: WatchlistResponse


class ResultsSummaryResponse(BaseModel):
    """Accuracy summary for a period."""

    label: str
    start_date: datetime
    end_date: datetime
    model_predictions: int
    model_correct: int
    model_accuracy: float | None
    user_predictions: int
    user_correct: int
    user_accuracy: float | None


class ResultsReviewItemResponse(BaseModel):
    """Finished prediction review row."""

    match: MatchResponse
    model_prediction: PredictionListItemResponse | None = None
    user_prediction: UserPredictionResponse | None = None
    actual_result: str | None = None
    model_correct: bool | None = None
    user_correct: bool | None = None


class ResultsReviewResponse(BaseModel):
    """Results review for day/week/month windows."""

    summaries: list[ResultsSummaryResponse]
    items: list[ResultsReviewItemResponse]


class MatchOddsRowResponse(BaseModel):
    """Bookmaker odds row for match detail."""

    bookmaker: str
    odds_home: float
    odds_draw: float
    odds_away: float
    scraped_at: datetime | None = None
    source: str | None = None


class RecentTeamMatchResponse(BaseModel):
    """Recent match form row for one team."""

    match_id: int
    match_date: datetime
    opponent_name: str
    venue: str
    goals_for: int
    goals_against: int
    result: str


class RecentFormResponse(BaseModel):
    """Recent form split by home and away teams."""

    home: list[RecentTeamMatchResponse]
    away: list[RecentTeamMatchResponse]


class TeamStatsComparisonResponse(BaseModel):
    """Available recent-stat averages for one team."""

    team_id: int
    team_name: str
    matches: int
    avg_goals_for: float
    avg_goals_against: float
    avg_shots: float | None = None
    avg_shots_on_target: float | None = None
    avg_corners: float | None = None


class StatsComparisonResponse(BaseModel):
    """Home/away stats comparison."""

    home: TeamStatsComparisonResponse
    away: TeamStatsComparisonResponse


class ModelFeatureExplanationResponse(BaseModel):
    """Simple feature explanation row for a prediction."""

    feature: str
    label: str
    value: float
    direction: str
    impact: float


class SimilarAccuracyResponse(BaseModel):
    """Historical model accuracy on similar prediction cases."""

    sample_size: int
    correct: int
    accuracy: float | None
    description: str


class MatchWorkflowDetailResponse(BaseModel):
    """Enriched match workflow payload used by the detail page."""

    match: MatchDetailResponse
    odds_comparison: list[MatchOddsRowResponse]
    recent_form: RecentFormResponse
    stats_comparison: StatsComparisonResponse
    model_explanation: list[ModelFeatureExplanationResponse]
    similar_accuracy: SimilarAccuracyResponse
    user_prediction: UserPredictionResponse | None = None
    watched: bool
