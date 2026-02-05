"""Pydantic schemas for model-related API responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class ModelVersionResponse(BaseModel):
    """Model version response schema."""

    id: int
    name: str
    version: str
    algorithm: str  # 'xgboost', 'random_forest', etc.
    accuracy: float | None = None
    file_path: str
    is_active: bool
    created_at: datetime
    metrics: dict[str, Any] | None = None  # JSONB field
    hyperparameters: dict[str, Any] | None = None  # JSONB field
    feature_schema_version: str | None = None
    description: str | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("accuracy")
    @classmethod
    def validate_accuracy(cls, v: float | None) -> float | None:
        """Validate accuracy is between 0 and 1 if provided."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("Accuracy must be between 0 and 1")
        return v
