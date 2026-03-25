"""Scraping-related database models."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from algobet.infrastructure.models import Base


class ScrapingJob(Base):
    """Represents a web scraping job."""

    __tablename__ = "scraping_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "upcoming", "results", "odds"
    status: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "pending", "running", "completed", "failed", "cancelled"

    # Source information
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    tournament_id: Mapped[int | None] = mapped_column(
        ForeignKey("tournaments.id"), nullable=True
    )
    season_id: Mapped[int | None] = mapped_column(
        ForeignKey("seasons.id"), nullable=True
    )

    # Execution info
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Results
    matches_scraped: Mapped[int | None] = mapped_column(Integer, nullable=True)
    matches_inserted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    matches_updated: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Error information
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Metadata
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # User/initiator
    initiated_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    logs: Mapped[list["ScrapingLog"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<ScrapingJob(id={self.id}, type='{self.job_type}', "
            f"status='{self.status}', matches={self.matches_scraped})>"
        )

    @property
    def duration(self) -> float | None:
        """Return scraping duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ScrapingLog(Base):
    """Detailed logs for scraping jobs."""

    __tablename__ = "scraping_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("scraping_jobs.id", ondelete="CASCADE"), nullable=False
    )

    # Log entry
    level: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "DEBUG", "INFO", "WARNING", "ERROR"
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Additional context
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    job: Mapped["ScrapingJob"] = relationship(back_populates="logs")

    def __repr__(self) -> str:
        return (
            f"<ScrapingLog(id={self.id}, job_id={self.job_id}, level='{self.level}')>"
        )


class ScrapedOdds(Base):
    """Stores scraped betting odds history for analysis."""

    __tablename__ = "scraped_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False
    )

    # Bookmaker info
    bookmaker: Mapped[str] = mapped_column(String(100), nullable=False)

    # Odds
    odds_home: Mapped[float] = mapped_column(Float, nullable=False)
    odds_draw: Mapped[float] = mapped_column(Float, nullable=False)
    odds_away: Mapped[float] = mapped_column(Float, nullable=False)

    # Metadata
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Source
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ScrapedOdds(id={self.id}, match_id={self.match_id}, "
            f"bookmaker='{self.bookmaker}')>"
        )


class ScrapingSource(Base):
    """Configuration for scraping sources."""

    __tablename__ = "scraping_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    url_pattern: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "odds", "results", "schedule"

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Configuration
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Rate limiting
    requests_per_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Metadata
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_scraped: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<ScrapingSource(id={self.id}, name='{self.name}', "
            f"type='{self.source_type}', active={self.is_active})>"
        )
