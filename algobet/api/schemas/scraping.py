"""Pydantic schemas for scraping operations."""

from datetime import date as date_cls, datetime
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ScrapingJobStatus(str, Enum):
    """Scraping job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScrapingType(str, Enum):
    """Type of scraping operation."""

    UPCOMING = "upcoming"
    RESULTS = "results"
    BY_DATE = "by-date"
    IMPORT = "import"


class ScrapeScope(str, Enum):
    """Scope of a scraping request."""

    ALL = "all"
    LEAGUE = "league"


class ScrapingJobBase(BaseModel):
    """Base schema for scraping job."""

    scraping_type: ScrapingType | None = Field(
        None, description="Type of scraping operation"
    )
    tournament_url: HttpUrl | None = Field(
        None, description="URL of tournament to scrape"
    )
    tournament_id: int | None = Field(
        None, description="Tournament ID to resolve URL from database"
    )
    tournament_name: str | None = Field(
        None, description="Name of tournament to scrape"
    )
    team_id: int | None = Field(None, description="Optional team ID to target")
    season: str | None = Field(None, description="Season to scrape (e.g., '2023-2024')")
    start_date: datetime | None = Field(
        None, description="Start date for results scraping"
    )
    end_date: datetime | None = Field(None, description="End date for results scraping")
    scope: ScrapeScope = Field(
        ScrapeScope.ALL, description="Scope of scrape: 'all' or 'league'"
    )
    country: str | None = Field(None, description="Country name")
    league_name: str | None = Field(None, description="League name")
    period: str | None = Field(
        None, description="Period/date (e.g., '2023/2024' or '2023-2024')"
    )
    period_start: str | None = Field(
        None, description="Start period for range scraping (e.g., '2010-2011')"
    )
    period_end: str | None = Field(
        None, description="End period for range scraping (e.g., '2019-2020')"
    )


class ScrapingJobCreate(ScrapingJobBase):
    """Schema for creating a scraping job."""

    pass


class UpcomingScrapeRequest(BaseModel):
    """Request schema for upcoming match scraping."""

    tournament_id: int | None = Field(
        None, description="Tournament ID to resolve from database"
    )
    tournament_url: HttpUrl | None = Field(
        None, description="Optional manual OddsPortal URL override"
    )
    team_id: int | None = Field(
        None, description="Optional Team ID to target specific matches"
    )
    scope: ScrapeScope = Field(
        ScrapeScope.ALL, description="Scrape all leagues or a specific league"
    )

    @model_validator(mode="after")
    def validate_scope(self) -> "UpcomingScrapeRequest":
        """Require a league target when scope is league."""
        if (
            self.scope == ScrapeScope.LEAGUE
            and self.tournament_id is None
            and self.tournament_url is None
        ):
            raise ValueError(
                "tournament_id or tournament_url is required when scope='league'"
            )
        return self


class ResultsScrapeRequest(BaseModel):
    """Request schema for historical results scraping."""

    tournament_id: int | None = Field(
        None, description="Tournament ID to resolve from database"
    )
    tournament_url: HttpUrl | None = Field(
        None, description="Optional manual OddsPortal URL override"
    )
    team_id: int | None = Field(
        None, description="Optional Team ID to target specific matches"
    )
    period: str | None = Field(
        None, description="Season label such as '2023/2024' or '2023-2024'"
    )
    period_start: str | None = Field(
        None, description="Start period for range scraping (e.g., '2010-2011')"
    )
    period_end: str | None = Field(
        None, description="End period for range scraping (e.g., '2019-2020')"
    )
    max_pages: int | None = Field(
        None, ge=1, description="Optional page limit for historical scraping"
    )

    @model_validator(mode="after")
    def validate_target(self) -> "ResultsScrapeRequest":
        """Require a tournament target."""
        if self.tournament_id is None and self.tournament_url is None:
            raise ValueError("tournament_id or tournament_url is required")
        return self


class ByDateScrapeRequest(BaseModel):
    """Request schema for daily match scraping."""

    date: date_cls | None = Field(
        None, description="Date to scrape in YYYY-MM-DD format"
    )
    tournament_id: int | None = Field(
        None, description="Tournament ID to resolve from database"
    )
    tournament_url: HttpUrl | None = Field(
        None, description="Optional manual OddsPortal URL override"
    )
    team_id: int | None = Field(
        None, description="Optional Team ID to target specific matches"
    )
    scope: ScrapeScope = Field(
        ScrapeScope.ALL, description="Scrape all leagues or a specific league"
    )

    @model_validator(mode="after")
    def validate_scope(self) -> "ByDateScrapeRequest":
        """Require a league target when scope is league."""
        if (
            self.scope == ScrapeScope.LEAGUE
            and self.tournament_id is None
            and self.tournament_url is None
        ):
            raise ValueError(
                "tournament_id or tournament_url is required when scope='league'"
            )
        return self


class ScrapingJobResponse(ScrapingJobBase):
    """Schema for scraping job response."""

    id: str = Field(..., description="Unique job identifier")
    status: ScrapingJobStatus = Field(..., description="Current job status")
    progress: float = Field(0.0, description="Progress percentage (0-100)")
    message: str | None = Field(None, description="Status message or error details")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: datetime | None = Field(None, description="Job start timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")
    matches_scraped: int = Field(0, description="Number of matches scraped")
    matches_saved: int = Field(0, description="Number of matches saved")
    errors: list[str] = Field(
        default_factory=list, description="List of error messages"
    )


class ScrapingJobUpdate(BaseModel):
    """Schema for updating scraping job status."""

    status: ScrapingJobStatus | None = Field(None, description="Updated job status")
    progress: float | None = Field(None, description="Updated progress percentage")
    message: str | None = Field(None, description="Updated status message")
    matches_scraped: int | None = Field(None, description="Updated match count")
    matches_saved: int | None = Field(None, description="Updated matches saved count")
    errors: list[str] | None = Field(None, description="Updated error list")
    started_at: datetime | None = Field(None, description="Job start timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")


class ScrapingProgress(BaseModel):
    """Schema for scraping progress updates."""

    job_id: str = Field(..., description="Job identifier")
    status: ScrapingJobStatus | None = Field(None, description="Current job status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    message: str = Field(..., description="Progress message")
    matches_scraped: int = Field(0, description="Number of matches scraped so far")
    matches_saved: int = Field(0, description="Number of matches saved")
    current_page: int | None = Field(None, description="Current page being scraped")
    total_pages: int | None = Field(None, description="Total pages to scrape")
    started_at: datetime | None = Field(None, description="Job start timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")
    error: str | None = Field(None, description="Error message if failed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Progress timestamp"
    )


class ScrapingStats(BaseModel):
    """Schema for scraping statistics."""

    total_jobs: int = Field(..., description="Total number of jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    failed_jobs: int = Field(..., description="Number of failed jobs")
    running_jobs: int = Field(..., description="Number of currently running jobs")
    total_matches_scraped: int = Field(
        ..., description="Total matches scraped across all jobs"
    )
    average_duration_seconds: float | None = Field(
        None, description="Average job duration in seconds"
    )
    success_rate: float = Field(..., description="Success rate percentage")


F = TypeVar("F", bound=ScrapingJobResponse)


class PaginatedResponse(BaseModel, Generic[F]):
    """Paginated response for scraping jobs."""

    items: list[F] = Field(default_factory=list, description="List of items")
    total: int = Field(0, description="Total number of items")
    limit: int = Field(50, description="Page size limit")
    offset: int = Field(0, description="Offset into results")

    model_config = ConfigDict(from_attributes=True)
