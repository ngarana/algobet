"""Tests for scraping API schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl, ValidationError

from algobet.api.schemas.scraping import (
    ScrapingJobCreate,
    ScrapingJobResponse,
    ScrapingJobStatus,
    ScrapingJobUpdate,
    ScrapingProgress,
    ScrapingStats,
    ScrapingType,
)


class TestScrapingJobStatus:
    """Tests for ScrapingJobStatus enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert ScrapingJobStatus.PENDING == "pending"
        assert ScrapingJobStatus.RUNNING == "running"
        assert ScrapingJobStatus.COMPLETED == "completed"
        assert ScrapingJobStatus.FAILED == "failed"
        assert ScrapingJobStatus.CANCELLED == "cancelled"

    def test_enum_iteration(self):
        """Test that enum can be iterated."""
        statuses = list(ScrapingJobStatus)
        assert len(statuses) == 5
        assert ScrapingJobStatus.PENDING in statuses
        assert ScrapingJobStatus.RUNNING in statuses
        assert ScrapingJobStatus.COMPLETED in statuses
        assert ScrapingJobStatus.FAILED in statuses
        assert ScrapingJobStatus.CANCELLED in statuses


class TestScrapingType:
    """Tests for ScrapingType enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert ScrapingType.UPCOMING == "upcoming"
        assert ScrapingType.RESULTS == "results"

    def test_enum_iteration(self):
        """Test that enum can be iterated."""
        types = list(ScrapingType)
        assert len(types) == 2
        assert ScrapingType.UPCOMING in types
        assert ScrapingType.RESULTS in types


class TestScrapingJobCreate:
    """Tests for ScrapingJobCreate schema."""

    def test_valid_creation_minimal(self):
        """Test creation with minimal required fields."""
        job = ScrapingJobCreate(
            scraping_type=ScrapingType.UPCOMING,
        )
        assert job.scraping_type == ScrapingType.UPCOMING
        assert job.tournament_url is None
        assert job.tournament_name is None
        assert job.season is None
        assert job.start_date is None
        assert job.end_date is None

    def test_valid_creation_with_url(self):
        """Test creation with tournament URL."""
        url = "https://www.oddsportal.com/football/england/premier-league/"
        job = ScrapingJobCreate(
            scraping_type=ScrapingType.RESULTS,
            tournament_url=HttpUrl(url),
            season="2023-2024",
        )
        assert job.scraping_type == ScrapingType.RESULTS
        assert str(job.tournament_url) == url
        assert job.season == "2023-2024"

    def test_valid_creation_with_dates(self):
        """Test creation with date range."""
        start_date = datetime(2023, 8, 1)
        end_date = datetime(2024, 5, 31)
        job = ScrapingJobCreate(
            scraping_type=ScrapingType.RESULTS,
            tournament_url=HttpUrl(
                "https://www.oddsportal.com/football/england/premier-league/"
            ),
            start_date=start_date,
            end_date=end_date,
        )
        assert job.start_date == start_date
        assert job.end_date == end_date

    def test_invalid_url(self):
        """Test validation with invalid URL."""
        with pytest.raises(ValidationError):
            ScrapingJobCreate(
                scraping_type=ScrapingType.UPCOMING,
                tournament_url="not-a-valid-url",
            )


class TestScrapingJobResponse:
    """Tests for ScrapingJobResponse schema."""

    def test_valid_response(self):
        """Test creation of valid response."""
        job = ScrapingJobResponse(
            id="test-job-123",
            scraping_type=ScrapingType.UPCOMING,
            status=ScrapingJobStatus.RUNNING,
            progress=50.0,
            message="Scraping in progress",
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            matches_scraped=25,
            errors=["Error 1", "Error 2"],
        )
        assert job.id == "test-job-123"
        assert job.scraping_type == ScrapingType.UPCOMING
        assert job.status == ScrapingJobStatus.RUNNING
        assert job.progress == 50.0
        assert job.message == "Scraping in progress"
        assert job.matches_scraped == 25
        assert len(job.errors) == 2
        assert job.created_at is not None
        assert job.started_at is not None

    def test_response_with_defaults(self):
        """Test response with default values."""
        job = ScrapingJobResponse(
            id="test-job-456",
            scraping_type=ScrapingType.RESULTS,
            status=ScrapingJobStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        assert job.progress == 0.0
        assert job.message is None
        assert job.matches_scraped == 0
        assert job.errors == []
        assert job.started_at is None
        assert job.completed_at is None

    def test_response_with_optional_fields(self):
        """Test response with all optional fields."""
        now = datetime.now(timezone.utc)
        job = ScrapingJobResponse(
            id="test-job-789",
            scraping_type=ScrapingType.UPCOMING,
            status=ScrapingJobStatus.COMPLETED,
            progress=100.0,
            message="Job completed successfully",
            created_at=now,
            started_at=now,
            completed_at=now,
            matches_scraped=100,
            errors=[],
            tournament_url=HttpUrl("https://www.oddsportal.com/"),
            tournament_name="Premier League",
            season="2023-2024",
        )
        assert job.progress == 100.0
        assert job.message == "Job completed successfully"
        assert job.started_at == now
        assert job.completed_at == now
        assert job.matches_scraped == 100
        assert job.tournament_name == "Premier League"
        assert job.season == "2023-2024"


class TestScrapingJobUpdate:
    """Tests for ScrapingJobUpdate schema."""

    def test_update_all_fields(self):
        """Test updating all fields."""
        update = ScrapingJobUpdate(
            status=ScrapingJobStatus.COMPLETED,
            progress=100.0,
            message="Job finished",
            matches_scraped=50,
            errors=["Final error"],
        )
        assert update.status == ScrapingJobStatus.COMPLETED
        assert update.progress == 100.0
        assert update.message == "Job finished"
        assert update.matches_scraped == 50
        assert update.errors == ["Final error"]

    def test_update_partial_fields(self):
        """Test updating only some fields."""
        update = ScrapingJobUpdate(
            progress=75.0,
            message="75% complete",
        )
        assert update.progress == 75.0
        assert update.message == "75% complete"
        assert update.status is None
        assert update.matches_scraped is None
        assert update.errors is None

    def test_update_empty(self):
        """Test empty update."""
        update = ScrapingJobUpdate()
        assert update.status is None
        assert update.progress is None
        assert update.message is None
        assert update.matches_scraped is None
        assert update.errors is None


class TestScrapingProgress:
    """Tests for ScrapingProgress schema."""

    def test_valid_progress(self):
        """Test valid progress update."""
        progress = ScrapingProgress(
            job_id="job-123",
            progress=50.0,
            message="Halfway done",
            matches_scraped=25,
        )
        assert progress.job_id == "job-123"
        assert progress.progress == 50.0
        assert progress.message == "Halfway done"
        assert progress.matches_scraped == 25
        assert progress.timestamp is not None

    def test_progress_defaults(self):
        """Test progress with default values."""
        progress = ScrapingProgress(
            job_id="job-456",
            progress=100.0,
            message="Complete",
        )
        assert progress.matches_scraped == 0
        assert isinstance(progress.timestamp, datetime)


class TestScrapingStats:
    """Tests for ScrapingStats schema."""

    def test_valid_stats(self):
        """Test valid statistics."""
        stats = ScrapingStats(
            total_jobs=100,
            completed_jobs=80,
            failed_jobs=15,
            running_jobs=5,
            total_matches_scraped=5000,
            average_duration_seconds=120.5,
            success_rate=84.21,
        )
        assert stats.total_jobs == 100
        assert stats.completed_jobs == 80
        assert stats.failed_jobs == 15
        assert stats.running_jobs == 5
        assert stats.total_matches_scraped == 5000
        assert stats.average_duration_seconds == 120.5
        assert stats.success_rate == 84.21

    def test_stats_with_zero_jobs(self):
        """Test statistics with no jobs."""
        stats = ScrapingStats(
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            running_jobs=0,
            total_matches_scraped=0,
            success_rate=0.0,
        )
        assert stats.total_jobs == 0
        assert stats.average_duration_seconds is None
        assert stats.success_rate == 0.0

    def test_stats_calculation(self):
        """Test that stats can be calculated correctly."""
        # This would test any calculated fields if they existed
        stats = ScrapingStats(
            total_jobs=100,
            completed_jobs=80,
            failed_jobs=15,
            running_jobs=5,
            total_matches_scraped=5000,
            success_rate=80.0,
        )
        # Verify the success rate calculation would be correct
        # (completed_jobs / (completed_jobs + failed_jobs)) * 100
        expected_rate = (80 / (80 + 15)) * 100
        assert abs(stats.success_rate - expected_rate) < 0.01


class TestSchemaValidation:
    """General validation tests for all schemas."""

    def test_progress_percentage_validation(self):
        """Test that progress percentages are valid."""
        # Valid percentages
        ScrapingJobResponse(
            id="test",
            scraping_type=ScrapingType.UPCOMING,
            status=ScrapingJobStatus.RUNNING,
            progress=0.0,
            created_at=datetime.now(timezone.utc),
        )

        ScrapingJobResponse(
            id="test",
            scraping_type=ScrapingType.UPCOMING,
            status=ScrapingJobStatus.RUNNING,
            progress=50.0,
            created_at=datetime.now(timezone.utc),
        )

        ScrapingJobResponse(
            id="test",
            scraping_type=ScrapingType.UPCOMING,
            status=ScrapingJobStatus.COMPLETED,
            progress=100.0,
            created_at=datetime.now(timezone.utc),
        )

    def test_negative_progress(self):
        """Test that negative progress is handled."""
        # This should work but might be considered invalid by business logic
        job = ScrapingJobResponse(
            id="test",
            scraping_type=ScrapingType.UPCOMING,
            status=ScrapingJobStatus.RUNNING,
            progress=-10.0,
            created_at=datetime.now(timezone.utc),
        )
        assert job.progress == -10.0

    def test_progress_over_100(self):
        """Test that progress over 100% is handled."""
        # This should work but might be considered invalid by business logic
        job = ScrapingJobResponse(
            id="test",
            scraping_type=ScrapingType.UPCOMING,
            status=ScrapingJobStatus.RUNNING,
            progress=150.0,
            created_at=datetime.now(timezone.utc),
        )
        assert job.progress == 150.0
