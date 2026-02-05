"""Pydantic schemas for team-related API responses."""

from pydantic import BaseModel, ConfigDict

from .common import FormBreakdown


class TeamResponse(BaseModel):
    """Team response schema."""

    id: int
    name: str

    model_config = ConfigDict(from_attributes=True)


class TeamWithStatsResponse(TeamResponse):
    """Team with computed statistics."""

    total_matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    current_form: FormBreakdown
