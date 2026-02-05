"""Pydantic schemas for tournament-related API responses."""


from pydantic import BaseModel, ConfigDict


class TournamentResponse(BaseModel):
    """Tournament response schema."""

    id: int
    name: str
    country: str
    url_slug: str

    model_config = ConfigDict(from_attributes=True)


class SeasonResponse(BaseModel):
    """Season response schema."""

    id: int
    tournament_id: int
    name: str  # e.g., "2023/2024"
    start_year: int
    end_year: int
    url_suffix: str | None = None

    model_config = ConfigDict(from_attributes=True)
