"""Tests for scraping API router endpoints."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from algobet.api.main import app
from algobet.api.routers.scraping import scraping_jobs
from algobet.api.schemas.scraping import (
    ScrapingJobResponse,
    ScrapingJobStatus,
    ScrapingType,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_job():
    """Create a sample scraping job."""

    job = ScrapingJobResponse(
        id=str(uuid.uuid4()),
        scraping_type=ScrapingType.UPCOMING,
        status=ScrapingJobStatus.COMPLETED,
        progress=100.0,
        message="Job completed successfully",
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        started_at=datetime.now(timezone.utc) - timedelta(minutes=45),
        completed_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        matches_scraped=50,
        errors=[],
    )

    return job


@pytest.fixture
def mock_scraping_service():
    """Create a mock scraping service."""
    with patch("algobet.api.routers.scraping.ScrapingService") as mock:
        service_instance = MagicMock()
        mock.return_value = service_instance

        # Mock the scrape methods
        mock_progress = MagicMock()
        mock_progress.matches_saved = 25
        service_instance.scrape_upcoming.return_value = mock_progress
        service_instance.scrape_results.return_value = mock_progress

        yield service_instance


class TestScrapingUpcomingEndpoint:
    """Tests for POST /api/v1/scraping/upcoming endpoint."""

    def test_scrape_upcoming_without_url(self, client, mock_scraping_service):
        """Test scraping upcoming matches without specific URL."""
        response = client.post("/api/v1/scraping/upcoming")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["scraping_type"] == "upcoming"
        assert data["status"] == "pending"
        assert data["progress"] == 0.0
        assert "id" in data
        assert data["created_at"] is not None

    def test_scrape_upcoming_with_url(self, client, mock_scraping_service):
        """Test scraping upcoming matches with specific tournament URL."""
        url = "https://www.oddsportal.com/football/england/premier-league/"
        response = client.post(f"/api/v1/scraping/upcoming?tournament_url={url}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["scraping_type"] == "upcoming"
        assert data["tournament_url"] == url
        assert data["status"] == "pending"

    def test_scrape_upcoming_invalid_url(self, client):
        """Test scraping with invalid URL format."""
        response = client.post("/api/v1/scraping/upcoming?tournament_url=invalid-url")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_scrape_upcoming_service_error(self, client, mock_scraping_service):
        """Test handling of service errors."""
        mock_scraping_service.scrape_upcoming.side_effect = Exception("Service error")

        response = client.post("/api/v1/scraping/upcoming")

        # The job should still be created, but the background task will handle the error
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "pending"


class TestScrapingResultsEndpoint:
    """Tests for POST /api/v1/scraping/results endpoint."""

    def test_scrape_results_with_required_params(self, client, mock_scraping_service):
        """Test scraping results with required parameters."""
        url = "https://www.oddsportal.com/football/england/premier-league/results/"
        response = client.post(f"/api/v1/scraping/results?tournament_url={url}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["scraping_type"] == "results"
        assert data["tournament_url"] == url
        assert data["status"] == "pending"

    def test_scrape_results_with_optional_params(self, client, mock_scraping_service):
        """Test scraping results with optional parameters."""
        url = "https://www.oddsportal.com/football/england/premier-league/results/"
        season = "2023-2024"
        start_date = "2023-08-01T00:00:00"
        end_date = "2024-05-31T23:59:59"

        response = client.post(
            f"/api/v1/scraping/results?tournament_url={url}&season={season}"
            f"&start_date={start_date}&end_date={end_date}"
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["scraping_type"] == "results"
        assert data["season"] == season
        assert data["start_date"] is not None
        assert data["end_date"] is not None

    def test_scrape_results_missing_url(self, client):
        """Test scraping results without required URL."""
        response = client.post("/api/v1/scraping/results")

        # Should still create the job, but background task will fail
        assert response.status_code == status.HTTP_200_OK

    def test_scrape_results_invalid_dates(self, client):
        """Test scraping results with invalid date format."""
        url = "https://www.oddsportal.com/football/england/premier-league/results/"
        response = client.post(
            f"/api/v1/scraping/results?tournament_url={url}&start_date=invalid-date"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestListJobsEndpoint:
    """Tests for GET /api/v1/scraping/jobs endpoint."""

    def test_list_jobs_empty(self, client):
        """Test listing jobs when no jobs exist."""
        # Clear any existing jobs
        scraping_jobs.clear()

        response = client.get("/api/v1/scraping/jobs")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["jobs"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 50

    def test_list_jobs_with_data(self, client, sample_job):
        """Test listing jobs with existing data."""
        # Clear existing jobs and add sample
        scraping_jobs.clear()
        scraping_jobs[sample_job.id] = sample_job

        response = client.get("/api/v1/scraping/jobs")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert len(data["jobs"]) == 1
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == sample_job.id

    def test_list_jobs_with_status_filter(self, client, sample_job):
        """Test filtering jobs by status."""
        scraping_jobs.clear()
        scraping_jobs[sample_job.id] = sample_job

        # Filter by completed status
        response = client.get("/api/v1/scraping/jobs?status_filter=completed")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["status"] == "completed"

    def test_list_jobs_with_pagination(self, client):
        """Test pagination of job listing."""
        scraping_jobs.clear()

        # Create multiple jobs
        for i in range(10):
            job = ScrapingJobResponse(
                id=f"job-{i}",
                scraping_type=ScrapingType.UPCOMING,
                status=ScrapingJobStatus.COMPLETED,
                progress=100.0,
                created_at=datetime.now(timezone.utc) - timedelta(minutes=i),
            )
            scraping_jobs[job.id] = job

        # Test pagination
        response = client.get("/api/v1/scraping/jobs?limit=5&offset=0")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert len(data["jobs"]) == 5
        assert data["total"] == 10
        assert data["page"] == 1
        assert data["page_size"] == 5

    def test_list_jobs_invalid_status_filter(self, client):
        """Test filtering with invalid status."""
        response = client.get("/api/v1/scraping/jobs?status_filter=invalid-status")

        # Should return empty list for invalid status
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["jobs"] == []


class TestGetJobEndpoint:
    """Tests for GET /api/v1/scraping/jobs/{job_id} endpoint."""

    def test_get_existing_job(self, client, sample_job):
        """Test getting an existing job."""
        scraping_jobs.clear()
        scraping_jobs[sample_job.id] = sample_job

        response = client.get(f"/api/v1/scraping/jobs/{sample_job.id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["id"] == sample_job.id
        assert data["status"] == "completed"
        assert data["progress"] == 100.0

    def test_get_nonexistent_job(self, client):
        """Test getting a non-existent job."""
        nonexistent_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/scraping/jobs/{nonexistent_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_job_with_invalid_id_format(self, client):
        """Test getting a job with invalid ID format."""
        response = client.get("/api/v1/scraping/jobs/invalid-id-format")

        # Should return 404 for any non-existent ID
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestGetStatsEndpoint:
    """Tests for GET /api/v1/scraping/stats endpoint."""

    def test_stats_empty(self, client):
        """Test statistics when no jobs exist."""
        scraping_jobs.clear()

        response = client.get("/api/v1/scraping/stats")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total_jobs"] == 0
        assert data["completed_jobs"] == 0
        assert data["failed_jobs"] == 0
        assert data["running_jobs"] == 0
        assert data["total_matches_scraped"] == 0
        assert data["average_duration_seconds"] is None
        assert data["success_rate"] == 0.0

    def test_stats_with_mixed_jobs(self, client):
        """Test statistics with various job statuses."""
        scraping_jobs.clear()

        # Create jobs with different statuses
        now = datetime.now(timezone.utc)

        # Completed jobs
        for i in range(5):
            job = ScrapingJobResponse(
                id=f"completed-{i}",
                scraping_type=ScrapingType.UPCOMING,
                status=ScrapingJobStatus.COMPLETED,
                progress=100.0,
                created_at=now - timedelta(hours=2),
                started_at=now - timedelta(hours=1, minutes=30),
                completed_at=now - timedelta(hours=1),
                matches_scraped=20,
            )
            scraping_jobs[job.id] = job

        # Failed jobs
        for i in range(2):
            job = ScrapingJobResponse(
                id=f"failed-{i}",
                scraping_type=ScrapingType.RESULTS,
                status=ScrapingJobStatus.FAILED,
                progress=50.0,
                created_at=now - timedelta(hours=3),
                started_at=now - timedelta(hours=2, minutes=45),
                completed_at=now - timedelta(hours=2, minutes=30),
                matches_scraped=10,
                errors=["Network error"],
            )
            scraping_jobs[job.id] = job

        # Running jobs
        job = ScrapingJobResponse(
            id="running-1",
            scraping_type=ScrapingType.UPCOMING,
            status=ScrapingJobStatus.RUNNING,
            progress=75.0,
            created_at=now - timedelta(minutes=30),
            started_at=now - timedelta(minutes=20),
            matches_scraped=15,
        )
        scraping_jobs[job.id] = job

        response = client.get("/api/v1/scraping/stats")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total_jobs"] == 8  # 5 completed + 2 failed + 1 running
        assert data["completed_jobs"] == 5
        assert data["failed_jobs"] == 2
        assert data["running_jobs"] == 1
        assert data["total_matches_scraped"] == 135  # 5*20 + 2*10 + 15
        assert data["average_duration_seconds"] is not None  # Should calculate average
        assert data["success_rate"] == 71.43  # 5/(5+2)*100


class TestBackgroundTaskIntegration:
    """Tests for background task integration."""

    @pytest.mark.asyncio
    async def test_background_task_execution(self, client, mock_scraping_service):
        """Test that background tasks are properly scheduled."""
        # Clear existing jobs
        scraping_jobs.clear()

        response = client.post("/api/v1/scraping/upcoming")

        assert response.status_code == status.HTTP_200_OK
        job_data = response.json()
        job_id = job_data["id"]

        # Verify job was created
        assert job_id in scraping_jobs
        assert scraping_jobs[job_id].status == ScrapingJobStatus.PENDING

    def test_job_status_updates_during_execution(self, client, mock_scraping_service):
        """Test that job status is updated during background execution."""
        # This would require more complex mocking of the background task
        # For now, just verify the job creation works
        response = client.post("/api/v1/scraping/upcoming")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "pending"


class TestErrorHandling:
    """Tests for error handling in scraping endpoints."""

    def test_database_connection_error(self, client):
        """Test handling of database connection errors."""
        # This would require mocking the database dependency
        # For now, just verify basic error handling works
        response = client.post("/api/v1/scraping/upcoming")

        # Should not crash the application
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]

    def test_invalid_json_response(self, client):
        """Test handling of invalid JSON in responses."""
        # The endpoints should always return valid JSON
        response = client.get("/api/v1/scraping/jobs")

        assert response.status_code == status.HTTP_200_OK

        # Verify response is valid JSON
        try:
            data = response.json()
            assert isinstance(data, dict)
        except ValueError:
            pytest.fail("Response is not valid JSON")

    def test_large_response_handling(self, client):
        """Test handling of large responses."""
        # Create many jobs to test pagination and large responses
        scraping_jobs.clear()

        for i in range(100):
            job = ScrapingJobResponse(
                id=f"large-job-{i}",
                scraping_type=ScrapingType.UPCOMING,
                status=ScrapingJobStatus.COMPLETED,
                progress=100.0,
                created_at=datetime.now(timezone.utc) - timedelta(minutes=i),
                matches_scraped=i,
            )
            scraping_jobs[job.id] = job

        response = client.get("/api/v1/scraping/jobs?limit=50")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert len(data["jobs"]) == 50
        assert data["total"] == 100


class TestConcurrentRequests:
    """Tests for handling concurrent requests."""

    def test_concurrent_job_creation(self, client, mock_scraping_service):
        """Test creating multiple jobs concurrently."""
        scraping_jobs.clear()

        # Create multiple jobs in sequence (simulating concurrent requests)
        job_ids = []
        for _i in range(5):
            response = client.post("/api/v1/scraping/upcoming")
            assert response.status_code == status.HTTP_200_OK
            job_ids.append(response.json()["id"])

        # Verify all jobs were created
        assert len(job_ids) == 5
        assert len(set(job_ids)) == 5  # All IDs should be unique

        for job_id in job_ids:
            assert job_id in scraping_jobs

    def test_concurrent_stats_requests(self, client):
        """Test concurrent stats requests."""
        # Multiple concurrent stats requests should be handled safely
        responses = []
        for _i in range(3):
            response = client.get("/api/v1/scraping/stats")
            responses.append(response)

        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "total_jobs" in data
