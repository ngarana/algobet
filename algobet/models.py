"""SQLAlchemy database models for football match data."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
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
    url_suffix: Mapped[str | None] = mapped_column(
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


class ModelVersion(Base):
    """Stores trained model metadata and versioning information."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Additional metadata stored as JSONB
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    feature_schema_version: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<ModelVersion(id={self.id}, name='{self.name}', "
            f"version='{self.version}', algorithm='{self.algorithm}', "
            f"is_active={self.is_active})>"
        )


class Prediction(Base):
    """Stores match predictions generated by ML models."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False
    )

    # Probability predictions
    prob_home: Mapped[float] = mapped_column(Float, nullable=False)
    prob_draw: Mapped[float] = mapped_column(Float, nullable=False)
    prob_away: Mapped[float] = mapped_column(Float, nullable=False)

    # Prediction metadata
    predicted_outcome: Mapped[str] = mapped_column(String(1), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Optional actual ROI tracking
    actual_roi: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    match: Mapped["Match"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")

    __table_args__ = (
        UniqueConstraint(
            "match_id", "model_version_id", name="uq_prediction_match_model"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, match_id={self.match_id}, "
            f"predicted='{self.predicted_outcome}', confidence={self.confidence:.3f})>"
        )

    @property
    def max_probability(self) -> float:
        """Return the highest probability among the three outcomes."""
        return max(self.prob_home, self.prob_draw, self.prob_away)


# Add relationship back-reference to Match
Match.predictions = relationship("Prediction", back_populates="match")


class ScheduledTask(Base):
    """Represents a scheduled task for automated scraping or predictions."""

    __tablename__ = "scheduled_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    task_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "scrape_upcoming", "scrape_results", "predict", etc.
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Task parameters stored as JSON
    parameters: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    # Metadata
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    executions: Mapped[list["TaskExecution"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<ScheduledTask(id={self.id}, name='{self.name}', "
            f"type='{self.task_type}', cron='{self.cron_expression}', "
            f"is_active={self.is_active})>"
        )


class TaskExecution(Base):
    """Records the execution history of scheduled tasks."""

    __tablename__ = "task_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("scheduled_tasks.id", ondelete="CASCADE"), nullable=False
    )

    # Execution status
    status: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "pending", "running", "completed", "failed"
    started_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Execution results
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Relationships
    task: Mapped["ScheduledTask"] = relationship(back_populates="executions")

    def __repr__(self) -> str:
        return (
            f"<TaskExecution(id={self.id}, task_id={self.task_id}, "
            f"status='{self.status}', started_at={self.started_at})>"
        )

    @property
    def duration(self) -> float | None:
        """Return execution duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
