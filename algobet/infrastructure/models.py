"""Base SQLAlchemy models for AlgoBet.

This module provides the base class and mixins used by all feature models.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )


class MetadataMixin:
    """Mixin that adds metadata storage as JSONB."""

    metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
