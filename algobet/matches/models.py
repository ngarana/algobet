"""Match-related database models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from algobet.infrastructure.models import Base


class Match(Base):
    """A football match with result and betting odds."""

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tournament_id: Mapped[int | None] = mapped_column(
        ForeignKey("tournaments.id"), nullable=True
    )
    season_id: Mapped[int | None] = mapped_column(
        ForeignKey("seasons.id"), nullable=True
    )
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)

    # Match details
    match_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    home_score: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # Nullable for upcoming matches
    away_score: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # Nullable for upcoming matches
    status: Mapped[str] = mapped_column(
        String(50), default="SCHEDULED", nullable=False
    )  # e.g., 'SCHEDULED', 'FINISHED', 'LIVE'

    # Betting odds (decimal format)
    odds_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_bookmakers: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships - will be set up in teams/models.py
    tournament: Mapped[Tournament] = relationship(
        "Tournament", back_populates="matches"
    )
    season: Mapped[Season] = relationship("Season", back_populates="matches")
    home_team: Mapped[Team] = relationship(
        "Team",
        back_populates="home_matches",
        foreign_keys=[home_team_id],
    )
    away_team: Mapped[Team] = relationship(
        "Team",
        back_populates="away_matches",
        foreign_keys=[away_team_id],
    )
    predictions: Mapped[list[Prediction]] = relationship(
        "Prediction", back_populates="match", cascade="all, delete-orphan"
    )
    model_features: Mapped[list[ModelFeature]] = relationship(
        "ModelFeature", back_populates="match", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "tournament_id",
            "season_id",
            "home_team_id",
            "away_team_id",
            "match_date",
            name="uq_match",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Match(id={self.id}, date={self.match_date.date()}, "
            f"home_team={self.home_team_id}, away_team={self.away_team_id}, "
            f"score={self.home_score}-{self.away_score}, status='{self.status}')>"
        )

    @property
    def result(self) -> str | None:
        """Return match result as 'H', 'D', or 'A'.

        Returns None if scores are not available.
        """
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "H"
        elif self.home_score < self.away_score:
            return "A"
        return "D"


class MatchStatistics(Base):
    """Detailed match statistics for feature engineering.

    Stores rich match data from Football-Data.co.uk CSV files including
    shots, fouls, corners, cards, and half-time scores.
    """

    __tablename__ = "match_statistics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    # Half-time scores
    ht_home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ht_away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ht_result: Mapped[str | None] = mapped_column(String(1), nullable=True)  # H/D/A

    # Shots
    home_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Fouls
    home_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Corners
    home_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Cards
    home_yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Expected goals (soccerdata / Understat)
    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Other
    referee: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Relationships
    match: Mapped[Match] = relationship("Match", back_populates="statistics")

    def __repr__(self) -> str:
        return (
            f"<MatchStatistics(id={self.id}, match_id={self.match_id}, "
            f"ht_score={self.ht_home_score}-{self.ht_away_score}, "
            f"shots={self.home_shots}-{self.away_shots})>"
        )


# Set up relationship back-reference
Match.statistics = relationship(
    "MatchStatistics",
    back_populates="match",
    cascade="all, delete-orphan",
    uselist=False,
)
if TYPE_CHECKING:
    from algobet.predictions.models.base import ModelFeature, Prediction
    from algobet.teams.models import Season, Team, Tournament
