"""Pydantic schemas for API request/response validation"""

# Import base/common schemas first
from .common import FormBreakdown, PaginatedResponse

# Import schemas that depend on the above
from .match import (
    MatchDetailResponse,
    MatchFilters,
    MatchResponse,
    MatchStatus,
    PredictedOutcome,
)
from .model import ModelVersionResponse
from .prediction import (
    PredictionFilters,
    PredictionListItemResponse,
    PredictionMatchSummaryResponse,
    PredictionResponse,
    PredictionWithMatchResponse,
    ValueBetResponse,
)
from .team import TeamResponse, TeamWithStatsResponse
from .tournament import SeasonResponse, TournamentResponse
from .workflow import (
    DailyWorkflowResponse,
    MatchWorkflowDetailResponse,
    ProfilePreferencesRequest,
    ProfilePreferencesResponse,
    ResultsReviewResponse,
    ResultsSummaryResponse,
    UserPredictionRequest,
    UserPredictionResponse,
    WatchlistEntryRequest,
    WatchlistEntryResponse,
    WatchlistResponse,
)

# Rebuild models to resolve forward references
MatchDetailResponse.model_rebuild()
PredictionWithMatchResponse.model_rebuild()
PredictionListItemResponse.model_rebuild()
DailyWorkflowResponse.model_rebuild()
MatchWorkflowDetailResponse.model_rebuild()

__all__ = [
    "FormBreakdown",
    "PaginatedResponse",
    "TeamResponse",
    "TeamWithStatsResponse",
    "TournamentResponse",
    "SeasonResponse",
    "ModelVersionResponse",
    "MatchResponse",
    "MatchDetailResponse",
    "MatchFilters",
    "MatchStatus",
    "PredictedOutcome",
    "PredictionResponse",
    "PredictionMatchSummaryResponse",
    "PredictionListItemResponse",
    "PredictionWithMatchResponse",
    "PredictionFilters",
    "ValueBetResponse",
    "ProfilePreferencesRequest",
    "ProfilePreferencesResponse",
    "WatchlistEntryRequest",
    "WatchlistEntryResponse",
    "WatchlistResponse",
    "UserPredictionRequest",
    "UserPredictionResponse",
    "DailyWorkflowResponse",
    "ResultsSummaryResponse",
    "ResultsReviewResponse",
    "MatchWorkflowDetailResponse",
]
