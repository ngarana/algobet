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
    PredictionResponse,
    PredictionWithMatchResponse,
    ValueBetResponse,
)
from .team import TeamResponse, TeamWithStatsResponse
from .tournament import SeasonResponse, TournamentResponse

# Rebuild models to resolve forward references
MatchDetailResponse.model_rebuild()
PredictionWithMatchResponse.model_rebuild()

__all__ = [
    "FormBreakdown",
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
    "PredictionWithMatchResponse",
    "PredictionFilters",
    "ValueBetResponse",
]
