"""Focused tests for router-side scraping progress wiring."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from algobet.api.main import app
from algobet.api.routers.scraping import scraping_jobs
from algobet.services.scraping_service import JobStatus, ScrapingProgress


def setup_function() -> None:
    scraping_jobs.clear()


def teardown_function() -> None:
    scraping_jobs.clear()


def test_results_endpoint_forwards_max_pages() -> None:
    client = TestClient(app)
    tournament_url = (
        "https://www.oddsportal.com/football/england/premier-league/results/"
    )

    with (
        patch(
            "algobet.api.routers.scraping.manager.broadcast_progress",
            new=AsyncMock(),
        ),
        patch(
            "algobet.api.routers.scraping.manager.broadcast_job_status",
            new=AsyncMock(),
        ),
        patch("algobet.api.routers.scraping.ScrapingService") as mock_service_class,
    ):
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.scrape_results.return_value = MagicMock(matches_saved=12)

        response = client.post(
            f"/api/v1/scraping/results?tournament_url={tournament_url}&max_pages=7"
        )

        assert response.status_code == 200
        mock_service.scrape_results.assert_called_once_with(
            url=tournament_url,
            max_pages=7,
        )


def test_upcoming_endpoint_sets_started_at_and_broadcasts_incremental_progress() -> (
    None
):
    client = TestClient(app)

    with (
        patch(
            "algobet.api.routers.scraping.manager.broadcast_progress",
            new=AsyncMock(),
        ) as broadcast_progress,
        patch(
            "algobet.api.routers.scraping.manager.broadcast_job_status",
            new=AsyncMock(),
        ),
        patch("algobet.api.routers.scraping.ScrapingService") as mock_service_class,
    ):
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        def fake_scrape_upcoming(
            url: str = "https://www.oddsportal.com/matches/football/",
        ):
            progress_callback = mock_service_class.call_args.kwargs["progress_callback"]
            progress_callback(
                ScrapingProgress(
                    job_id=uuid4(),
                    status=JobStatus.RUNNING,
                    progress=42.0,
                    current_page=2,
                    total_pages=5,
                    matches_scraped=8,
                    matches_saved=0,
                    message="Scraping page 2/5...",
                    started_at=datetime.now(timezone.utc),
                )
            )
            return MagicMock(matches_saved=11)

        mock_service.scrape_upcoming.side_effect = fake_scrape_upcoming

        response = client.post("/api/v1/scraping/upcoming")

        assert response.status_code == 200
        job_id = response.json()["id"]
        job = scraping_jobs[job_id]

        assert job.started_at is not None
        assert broadcast_progress.call_count >= 1

        progress_payload = broadcast_progress.call_args[0][0]
        assert progress_payload.progress == 42.0
        assert progress_payload.matches_scraped == 8
        assert progress_payload.current_page == 2
        assert progress_payload.total_pages == 5


def test_by_date_endpoint_builds_date_url_and_forwards_target_date() -> None:
    client = TestClient(app)

    with (
        patch(
            "algobet.api.routers.scraping.manager.broadcast_progress",
            new=AsyncMock(),
        ),
        patch(
            "algobet.api.routers.scraping.manager.broadcast_job_status",
            new=AsyncMock(),
        ),
        patch("algobet.api.routers.scraping.ScrapingService") as mock_service_class,
    ):
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.scrape_matches_by_date.return_value = MagicMock(
            matches_saved=4,
            matches_scraped=4,
        )

        response = client.post(
            "/api/v1/scraping/by-date",
            json={"date": "2026-04-21", "scope": "all"},
        )

        assert response.status_code == 200
        payload = response.json()

        assert (
            payload["tournament_url"]
            == "https://www.oddsportal.com/matches/football/20260421/"
        )
        assert payload["period"] == "2026-04-21"

        mock_service.scrape_matches_by_date.assert_called_once_with(
            url="https://www.oddsportal.com/matches/football/20260421/",
            target_date=date(2026, 4, 21),
        )
