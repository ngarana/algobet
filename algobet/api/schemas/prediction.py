"""Pydantic schemas for prediction-related API responses."""

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

from .match import MatchDetailResponse, MatchResponse, PredictedOutcome

if TYPE_CHECKING:
    from .model import ModelVersionResponse


class ValueBetResponse(BaseModel):
    """Value bet response schema.

    Represents a betting opportunity where the model's predicted probability
    suggests positive expected value compared to market odds.
    """

    match: MatchResponse
    prediction_id: int
    predicted_outcome: str  # 'H', 'D', or 'A'
    predicted_probability: float
    market_odds: float
    expected_value: float  # EV = (predicted_probability * odds) - 1
    kelly_fraction: float  # Kelly criterion recommended stake fraction
    confidence: float

    model_config = ConfigDict(from_attributes=True)

    @field_validator("predicted_outcome")
    @classmethod
    def validate_outcome(cls, v: str) -> str:
        """Validate predicted outcome is a valid value."""
        valid_outcomes = {
            PredictedOutcome.HOME,
            PredictedOutcome.DRAW,
            PredictedOutcome.AWAY,
        }
        if v not in valid_outcomes:
            raise ValueError(
                f"Invalid predicted outcome: {v}. Must be one of {valid_outcomes}"
            )
        return v

    @field_validator("predicted_probability", "confidence")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v

    @field_validator("expected_value")
    @classmethod
    def validate_expected_value(cls, v: float) -> float:
        """Validate expected value is a reasonable number."""
        if v < -1 or v > 100:
            raise ValueError("Expected value seems unreasonable")
        return v


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    id: int
    match_id: int
    model_version_id: int
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_outcome: str
    confidence: float
    predicted_at: datetime
    actual_roi: float | None = None

    # Computed field
    max_probability: float

    model_config = ConfigDict(from_attributes=True)

    @field_validator("predicted_outcome")
    @classmethod
    def validate_outcome(cls, v: str) -> str:
        """Validate predicted outcome is a valid value."""
        valid_outcomes = {
            PredictedOutcome.HOME,
            PredictedOutcome.DRAW,
            PredictedOutcome.AWAY,
        }
        if v not in valid_outcomes:
            raise ValueError(
                f"Invalid predicted outcome: {v}. Must be one of {valid_outcomes}"
            )
        return v

    @field_validator("prob_home", "prob_draw", "prob_away")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class PredictionWithMatchResponse(PredictionResponse):
    """Prediction with full match details."""

    match: MatchDetailResponse
    model_version: "ModelVersionResponse"


class PredictionFilters(BaseModel):
    """Query parameters for filtering predictions."""

    match_id: int | None = None
    model_version_id: int | None = None
    has_result: bool | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    min_confidence: float | None = None

    @field_validator("min_confidence")
    @classmethod
    def validate_min_confidence(cls, v: float | None) -> float | None:
        """Validate min_confidence is between 0 and 1 if provided."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        return v
