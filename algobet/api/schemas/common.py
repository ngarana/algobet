from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class FormBreakdown(BaseModel):
    """Team form breakdown with computed statistics."""

    avg_points: float
    win_rate: float
    draw_rate: float
    loss_rate: float
    avg_goals_for: float
    avg_goals_against: float

    model_config = ConfigDict(from_attributes=True)


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response schema."""

    items: list[T]
    total: int
    limit: int
    offset: int

    model_config = ConfigDict(from_attributes=True)
