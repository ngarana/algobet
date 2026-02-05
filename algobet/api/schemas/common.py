"""Common schemas used across multiple API endpoints."""

from pydantic import BaseModel, ConfigDict


class FormBreakdown(BaseModel):
    """Team form breakdown with computed statistics."""

    avg_points: float
    win_rate: float
    draw_rate: float
    loss_rate: float
    avg_goals_for: float
    avg_goals_against: float

    model_config = ConfigDict(from_attributes=True)
