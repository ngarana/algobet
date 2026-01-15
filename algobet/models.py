"""SQLAlchemy database models for football match data."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
    Float,
    DateTime,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Tournament(Base):
    """Football tournament/league (e.g., Premier League, La Liga)."""

    __tablename__ = "tournaments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    url_slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    # Relationships
    seasons: Mapped[list["Season"]] = relationship(
        back_populates="tournament", cascade="all, delete-orphan"
    )
    matches: Mapped[list["Match"]] = relationship(
        back_populates="tournament", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Tournament(id={self.id}, name='{self.name}', country='{self.country}')>"
        )


class Season(Base):
    """A season within a tournament (e.g., 2023/2024)."""

    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tournament_id: Mapped[int] = mapped_column(
        ForeignKey("tournaments.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., "2023/2024"
    start_year: Mapped[int] = mapped_column(Integer, nullable=False)
    end_year: Mapped[int] = mapped_column(Integer, nullable=False)
    url_suffix: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # e.g., "2023-2024" or None for current

    # Relationships
    tournament: Mapped["Tournament"] = relationship(back_populates="seasons")
    matches: Mapped[list["Match"]] = relationship(
        back_populates="season", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("tournament_id", "name", name="uq_tournament_season"),
    )

    def __repr__(self) -> str:
        return f"<Season(id={self.id}, name='{self.name}')>"


class Team(Base):
    """A football team."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    # Relationships
    home_matches: Mapped[list["Match"]] = relationship(
        back_populates="home_team",
        foreign_keys="Match.home_team_id",
        cascade="all, delete-orphan",
    )
    away_matches: Mapped[list["Match"]] = relationship(
        back_populates="away_team",
        foreign_keys="Match.away_team_id",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Team(id={self.id}, name='{self.name}')>"


class Match(Base):
    """A football match with result and betting odds."""

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tournament_id: Mapped[int] = mapped_column(
        ForeignKey("tournaments.id"), nullable=False
    )
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)

    # Match details
    match_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    home_score: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Nullable for upcoming matches
    away_score: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Nullable for upcoming matches
    status: Mapped[str] = mapped_column(
        String(50), default="SCHEDULED", nullable=False
    )  # e.g., 'SCHEDULED', 'FINISHED', 'LIVE'

    # Betting odds (decimal format)
    odds_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    num_bookmakers: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    tournament: Mapped["Tournament"] = relationship(back_populates="matches")
    season: Mapped["Season"] = relationship(back_populates="matches")
    home_team: Mapped["Team"] = relationship(
        back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team: Mapped["Team"] = relationship(
        back_populates="away_matches", foreign_keys=[away_team_id]
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
    def result(self) -> Optional[str]:
        """Return match result as 'H', 'D', or 'A'. Returns None if scores are not available."""
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "H"
        elif self.home_score < self.away_score:
            return "A"
        return "D"
