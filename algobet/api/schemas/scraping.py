"""Pydantic schemas for scraping operations."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


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


class ScrapingJobBase(BaseModel):
    """Base schema for scraping job."""

    scraping_type: ScrapingType = Field(..., description="Type of scraping operation")
    tournament_url: HttpUrl | None = Field(
        None, description="URL of tournament to scrape"
    )
    tournament_name: str | None = Field(
        None, description="Name of tournament to scrape"
    )
    season: str | None = Field(
        None, description="Season to scrape (e.g., '2023-2024')"
    )
    start_date: datetime | None = Field(
        None, description="Start date for results scraping"
    )
    end_date: datetime | None = Field(
        None, description="End date for results scraping"
    )


class ScrapingJobCreate(ScrapingJobBase):
    """Schema for creating a scraping job."""

    pass


class ScrapingJobResponse(ScrapingJobBase):
    """Schema for scraping job response."""

    id: str = Field(..., description="Unique job identifier")
    status: ScrapingJobStatus = Field(..., description="Current job status")
    progress: float = Field(0.0, description="Progress percentage (0-100)")
    message: str | None = Field(None, description="Status message or error details")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: datetime | None = Field(None, description="Job start timestamp")
    completed_at: datetime | None = Field(
        None, description="Job completion timestamp"
    )
    matches_scraped: int = Field(0, description="Number of matches scraped")
    errors: list[str] = Field(
        default_factory=list, description="List of error messages"
    )


class ScrapingJobUpdate(BaseModel):
    """Schema for updating scraping job status."""

    status: ScrapingJobStatus | None = Field(None, description="Updated job status")
    progress: float | None = Field(None, description="Updated progress percentage")
    message: str | None = Field(None, description="Updated status message")
    matches_scraped: int | None = Field(None, description="Updated match count")
    errors: list[str] | None = Field(None, description="Updated error list")


class ScrapingProgress(BaseModel):
    """Schema for scraping progress updates."""

    job_id: str = Field(..., description="Job identifier")
    progress: float = Field(..., description="Progress percentage (0-100)")
    message: str = Field(..., description="Progress message")
    matches_scraped: int = Field(0, description="Number of matches scraped so far")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Progress timestamp"
    )


class ScrapingJobList(BaseModel):
    """Schema for listing scraping jobs."""

    jobs: list[ScrapingJobResponse] = Field(..., description="List of scraping jobs")
    total: int = Field(..., description="Total number of jobs")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(10, description="Number of jobs per page")


class ScrapingStats(BaseModel):
    """Schema for scraping statistics."""

    total_jobs: int = Field(..., description="Total number of scraping jobs")
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
