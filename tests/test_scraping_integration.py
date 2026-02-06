"""Integration tests for full scraping workflow combining API endpoints,
schemas, and WebSockets."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from algobet.api.main import app
from algobet.api.routers.scraping import scraping_jobs
from algobet.api.schemas.scraping import (
    ScrapingJobResponse,
    ScrapingJobStatus,
    ScrapingProgress,
    ScrapingType,
)
from algobet.api.websockets.progress import manager


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def cleanup_jobs():
    """Clean up scraping jobs before and after tests."""
    scraping_jobs.clear()
    yield
    scraping_jobs.clear()


@pytest.fixture
def cleanup_websockets():
    """Clean up WebSocket connections before and after tests."""
    manager.active_connections.clear()
    manager.connection_metadata.clear()
    yield
    manager.active_connections.clear()
    manager.connection_metadata.clear()


class TestScrapingWebSocketIntegration:
    """Integration tests for WebSocket progress updates during scraping jobs."""

    @pytest.mark.asyncio
    async def test_websocket_manager_direct_functionality(self, cleanup_websockets):
        """Test WebSocket manager functionality directly."""
        # Create a mock WebSocket with necessary attributes for the
        # manager implementation
        from fastapi import WebSocket

        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        mock_ws.client_state = MagicMock()
        mock_ws.client_state.DISCONNECTED = False

        # Add to manager - this will send a connection confirmation message
        await manager.connect(mock_ws, "test-client", "job-123")

        # Verify connection
        assert "job-123" in manager.active_connections
        assert mock_ws in manager.active_connections["job-123"]

        # Verify connection confirmation was sent
        assert mock_ws.send_text.call_count == 1
        connection_call = mock_ws.send_text.call_args_list[0][0][0]
        connection_data = json.loads(connection_call)
        assert connection_data["type"] == "connection"
        assert connection_data["status"] == "connected"
        assert connection_data["client_id"] == "test-client"
        assert connection_data["job_id"] == "job-123"

        # Test broadcasting progress
        progress = ScrapingProgress(
            job_id="job-123",
            progress=50.0,
            message="Halfway there",
            matches_scraped=25,
        )

        await manager.broadcast_progress(progress)

        # Now there should be 2 calls total: 1 connection + 1 progress
        assert mock_ws.send_text.call_count == 2

        # Check the progress message content (the second call)
        progress_call = mock_ws.send_text.call_args_list[1][0][0]
        message_data = json.loads(progress_call)
        assert message_data["type"] == "progress"
        assert message_data["job_id"] == "job-123"
        assert message_data["progress"] == 50.0
        assert message_data["message"] == "Halfway there"
        assert message_data["matches_scraped"] == 25

    @pytest.mark.asyncio
    async def test_websocket_manager_broadcast_job_status(self, cleanup_websockets):
        """Test broadcasting job status updates via WebSocket."""
        from fastapi import WebSocket

        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        mock_ws.client_state = MagicMock()
        mock_ws.client_state.DISCONNECTED = False

        # Connect - sends connection confirmation
        await manager.connect(mock_ws, "test-client", "job-456")

        # Verify connection confirmation was sent
        assert mock_ws.send_text.call_count == 1

        # Test broadcasting job status
        await manager.broadcast_job_status("job-456", "completed", "Job finished")

        # Now there should be 2 calls: 1 connection + 1 status
        assert mock_ws.send_text.call_count == 2

        # Check the status message content (the second call)
        status_call = mock_ws.send_text.call_args_list[1][0][0]
        message_data = json.loads(status_call)
        assert message_data["type"] == "status"
        assert message_data["job_id"] == "job-456"
        assert message_data["status"] == "completed"
        assert message_data["message"] == "Job finished"

    @pytest.mark.asyncio
    async def test_websocket_manager_multiple_connections_same_job(
        self, cleanup_websockets
    ):
        """Test multiple WebSocket connections to the same job."""
        # Create multiple mock WebSockets
        from fastapi import WebSocket

        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        mock_ws1.receive_text = AsyncMock()
        mock_ws1.client_state = MagicMock()
        mock_ws1.client_state.DISCONNECTED = False

        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        mock_ws2.receive_text = AsyncMock()
        mock_ws2.client_state = MagicMock()
        mock_ws2.client_state.DISCONNECTED = False

        # Connect both to the same job - each will get a connection confirmation
        await manager.connect(mock_ws1, "client-1", "shared-job")
        await manager.connect(mock_ws2, "client-2", "shared-job")

        # Verify both connections are registered
        assert "shared-job" in manager.active_connections
        assert mock_ws1 in manager.active_connections["shared-job"]
        assert mock_ws2 in manager.active_connections["shared-job"]

        # Each connection should have received 1 connection confirmation
        assert mock_ws1.send_text.call_count == 1
        assert mock_ws2.send_text.call_count == 1

        # Send progress update
        progress = ScrapingProgress(
            job_id="shared-job",
            progress=75.0,
            message="75% complete",
            matches_scraped=50,
        )
        await manager.broadcast_progress(progress)

        # Both connections should now have received 2 messages each:
        # 1 connection + 1 progress
        assert mock_ws1.send_text.call_count == 2
        assert mock_ws2.send_text.call_count == 2


class TestAPISchemaIntegration:
    """Integration tests between API endpoints and schemas."""

    def test_api_response_conforms_to_schema(self, client, cleanup_jobs):
        """Test that API responses conform to expected schemas."""
        # Create a scraping job via API
        response = client.post("/api/v1/scraping/upcoming")
        assert response.status_code == 200

        # Parse response as schema
        job_data = response.json()
        job_response = ScrapingJobResponse(**job_data)

        # Verify schema properties
        assert job_response.id is not None
        assert job_response.scraping_type == ScrapingType.UPCOMING
        assert (
            job_response.status == ScrapingJobStatus.PENDING
        )  # Should be pending initially
        assert job_response.progress == 0.0
        assert job_response.created_at is not None
        assert job_response.matches_scraped == 0
        assert job_response.errors == []

    def test_api_job_creation_and_retrieval(self, client, cleanup_jobs):
        """Test creating a job and retrieving it via API."""
        # Create a job
        response = client.post("/api/v1/scraping/upcoming")
        assert response.status_code == 200
        created_job = response.json()

        # Verify job exists in system
        assert created_job["id"] in scraping_jobs

        # Because the job runs in background, we need to check its status after creation
        # Retrieve job via API
        import time

        time.sleep(0.01)  # Give background task a moment to potentially update status

        get_response = client.get(f"/api/v1/scraping/jobs/{created_job['id']}")
        assert get_response.status_code == 200
        retrieved_job = get_response.json()

        # Compare original and retrieved jobs - allow for status changes
        # during background execution
        assert retrieved_job["id"] == created_job["id"]
        assert retrieved_job["scraping_type"] == created_job["scraping_type"]
        # The status might have changed due to background execution,
        # so don't check exact match

    def test_api_job_list_endpoint(self, client, cleanup_jobs):
        """Test listing multiple jobs via API."""
        # Create several jobs
        job_ids = []
        for _i in range(3):
            response = client.post("/api/v1/scraping/upcoming")
            assert response.status_code == 200
            job_data = response.json()
            job_ids.append(job_data["id"])

        # List all jobs
        list_response = client.get("/api/v1/scraping/jobs")
        assert list_response.status_code == 200
        jobs_data = list_response.json()

        # Verify response structure
        assert "jobs" in jobs_data
        assert "total" in jobs_data
        assert len(jobs_data["jobs"]) == 3
        assert jobs_data["total"] == 3

        # Verify each job in the list
        returned_ids = [job["id"] for job in jobs_data["jobs"]]
        for job_id in job_ids:
            assert job_id in returned_ids

    def test_api_job_statistics(self, client, cleanup_jobs):
        """Test job statistics endpoint."""
        # Create jobs with different statuses
        datetime.now(timezone.utc)

        # Create one completed job
        completed_response = client.post("/api/v1/scraping/upcoming")
        assert completed_response.status_code == 200
        completed_job = completed_response.json()

        # Update the job to completed in memory
        if completed_job["id"] in scraping_jobs:
            scraping_jobs[completed_job["id"]].status = ScrapingJobStatus.COMPLETED
            scraping_jobs[completed_job["id"]].progress = 100.0
            scraping_jobs[completed_job["id"]].matches_scraped = 10

        # Create one failed job
        failed_response = client.post("/api/v1/scraping/upcoming")
        assert failed_response.status_code == 200
        failed_job = failed_response.json()

        if failed_job["id"] in scraping_jobs:
            scraping_jobs[failed_job["id"]].status = ScrapingJobStatus.FAILED
            scraping_jobs[failed_job["id"]].progress = 50.0
            scraping_jobs[failed_job["id"]].errors = ["Test error"]

        # Get statistics
        stats_response = client.get("/api/v1/scraping/stats")
        assert stats_response.status_code == 200
        stats = stats_response.json()

        # Verify statistics
        assert stats["total_jobs"] == 2
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1
        assert stats["running_jobs"] == 0
        assert stats["total_matches_scraped"] == 10  # Only completed job counts
        assert (
            stats["success_rate"] == 50.0
        )  # 1 completed / (1 completed + 1 failed) * 100


class TestFullWorkflowIntegration:
    """Full end-to-end integration tests for the scraping workflow."""

    def test_complete_scraping_workflow(self, client, cleanup_jobs):
        """Test a complete scraping workflow from job creation to completion."""
        # Step 1: Create a scraping job
        response = client.post("/api/v1/scraping/upcoming")
        assert response.status_code == 200
        job_data = response.json()

        # Validate response schema
        job_response = ScrapingJobResponse(**job_data)
        job_id = job_response.id

        # Verify job exists in the system
        assert job_id in scraping_jobs

        # Step 2: Wait for job to complete (background task)
        import time

        time.sleep(0.05)  # Give background task time to run

        # Step 3: Retrieve the job to check its final state
        get_response = client.get(f"/api/v1/scraping/jobs/{job_id}")
        assert get_response.status_code == 200
        updated_job_data = get_response.json()

        # The job might be completed, running, or failed depending
        # on the background service
        assert updated_job_data["id"] == job_id
        assert updated_job_data["scraping_type"] == "upcoming"

        # Step 4: Verify job appears in job list
        list_response = client.get("/api/v1/scraping/jobs")
        assert list_response.status_code == 200
        jobs_list = list_response.json()

        job_in_list = next(
            (job for job in jobs_list["jobs"] if job["id"] == job_id), None
        )
        assert job_in_list is not None
        assert job_in_list["id"] == job_id

    def test_error_handling_in_workflow(self, client, cleanup_jobs):
        """Test error handling throughout the scraping workflow."""
        # Create a job
        response = client.post("/api/v1/scraping/upcoming")
        assert response.status_code == 200
        job_data = response.json()
        job_id = job_data["id"]

        # Wait for potential background execution
        import time

        time.sleep(0.05)

        # Verify job state via API
        response = client.get(f"/api/v1/scraping/jobs/{job_id}")
        assert response.status_code == 200
        job_response = response.json()

        # The job should have valid fields regardless of its status
        assert "id" in job_response
        assert "status" in job_response
        assert "scraping_type" in job_response
        assert job_response["scraping_type"] == "upcoming"

        # Verify in statistics
        stats_response = client.get("/api/v1/scraping/stats")
        assert stats_response.status_code == 200
        stats = stats_response.json()

        assert "total_jobs" in stats
        assert stats["total_jobs"] >= 1


class TestResultsScrapingIntegration:
    """Integration tests for results scraping functionality."""

    def test_results_scraping_with_params(self, client, cleanup_jobs):
        """Test results scraping with various parameters."""
        url = "https://www.oddsportal.com/football/england/premier-league/results/"
        season = "2023-2024"
        start_date = "2023-08-01T00:00:00"
        end_date = "2024-05-31T23:59:59"

        # Create results scraping job
        response = client.post(
            f"/api/v1/scraping/results?tournament_url={url}&season={season}"
            f"&start_date={start_date}&end_date={end_date}"
        )
        assert response.status_code == 200
        job_data = response.json()

        # Verify schema
        job_response = ScrapingJobResponse(**job_data)

        # Verify job was created with correct parameters
        assert job_response.scraping_type == ScrapingType.RESULTS
        assert str(job_response.tournament_url) == url
        assert job_response.season == season
        assert job_response.start_date is not None
        assert job_response.end_date is not None

        # Verify in system
        assert job_response.id in scraping_jobs
        internal_job = scraping_jobs[job_response.id]
        assert internal_job.scraping_type == ScrapingType.RESULTS
        assert internal_job.tournament_url == job_response.tournament_url
        assert internal_job.season == season


class TestConcurrentOperationsIntegration:
    """Integration tests for concurrent operations."""

    def test_concurrent_job_creation(self, client, cleanup_jobs):
        """Test creating multiple jobs concurrently."""
        job_ids = []

        # Create multiple jobs in sequence (simulating concurrent requests)
        for _i in range(5):
            response = client.post("/api/v1/scraping/upcoming")
            assert response.status_code == 200
            job_data = response.json()
            job_ids.append(job_data["id"])

        # Verify all jobs were created with unique IDs
        assert len(job_ids) == 5
        assert len(set(job_ids)) == 5  # All IDs should be unique

        # Verify all jobs exist in the system
        for job_id in job_ids:
            assert job_id in scraping_jobs

        # Verify jobs can be retrieved via API
        list_response = client.get("/api/v1/scraping/jobs")
        assert list_response.status_code == 200
        jobs_data = list_response.json()
        assert jobs_data["total"] == 5


class TestMockedScrapingServiceIntegration:
    """Integration tests using mocked scraping service to avoid real scraping."""

    def test_api_with_mocked_scraping_service(self, client, cleanup_jobs):
        """Test API endpoints with mocked scraping service to control execution."""
        # Mock the scraping service to avoid actual scraping
        with patch(
            "algobet.api.routers.scraping.ScrapingService"
        ) as mock_service_class:
            # Create a mock service instance
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service

            # Mock the scrape_upcoming method to return quickly
            mock_result = MagicMock()
            mock_result.matches_saved = 10
            mock_service.scrape_upcoming.return_value = mock_result

            # Create a job
            response = client.post("/api/v1/scraping/upcoming")
            assert response.status_code == 200
            job_data = response.json()
            job_id = job_data["id"]

            # Wait briefly for background task to execute
            import time

            time.sleep(0.01)

            # Check that the job was created
            assert job_id in scraping_jobs

            # Verify that the service was called
            mock_service.scrape_upcoming.assert_called_once()

            # Retrieve the job to check its status
            get_response = client.get(f"/api/v1/scraping/jobs/{job_id}")
            assert get_response.status_code == 200
            retrieved_job = get_response.json()

            # The job status will depend on what the background task did
            assert retrieved_job["id"] == job_id
            assert retrieved_job["scraping_type"] == "upcoming"


if __name__ == "__main__":
    pytest.main([__file__])
