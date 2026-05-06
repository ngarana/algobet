"""Persistence models for the single-user daily workflow."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from algobet.infrastructure.models import Base


class UserProfile(Base):
    """A local profile used for personalization without full authentication."""

    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_key: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), default="Local User")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    preferences: Mapped["ProfilePreference"] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        uselist=False,
    )
    watchlist_entries: Mapped[list["WatchlistEntry"]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
    )
    user_predictions: Mapped[list["UserPrediction"]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
    )


class ProfilePreference(Base):
    """Default workflow preferences for the local profile."""

    __tablename__ = "profile_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("user_profiles.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    default_days_ahead: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    min_confidence: Mapped[float] = mapped_column(Float, default=0.55, nullable=False)
    min_ev: Mapped[float] = mapped_column(Float, default=0.05, nullable=False)
    favorite_bookie: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    profile: Mapped[UserProfile] = relationship(back_populates="preferences")


class WatchlistEntry(Base):
    """A watched tournament, team, or match for the local profile."""

    __tablename__ = "watchlist_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=False
    )
    entry_type: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_id: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    profile: Mapped[UserProfile] = relationship(back_populates="watchlist_entries")

    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "entry_type",
            "entry_id",
            name="uq_watchlist_profile_entry",
        ),
    )


class UserPrediction(Base):
    """A user's own prediction for a match."""

    __tablename__ = "user_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=False
    )
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False
    )
    pick_1x2: Mapped[str | None] = mapped_column(String(1), nullable=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_goals_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_goals_pick: Mapped[str | None] = mapped_column(String(10), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    profile: Mapped[UserProfile] = relationship(back_populates="user_predictions")
    match: Mapped["Match"] = relationship("Match")

    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "match_id",
            name="uq_user_prediction_profile_match",
        ),
    )


if TYPE_CHECKING:
    from algobet.models import Match
