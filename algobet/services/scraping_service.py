"""Scraping service for orchestrating data collection from API-Football.

This service uses the API-Football client instead of web scraping to fetch
match data, fixtures, and results reliably.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import Match, Team, Tournament
from algobet.infrastructure.api_football_client import (
    APIFootballClient,
    APIFootballFixture,
)
from algobet.services.base import BaseService


class JobStatus(str, Enum):
    """Status of a scraping job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScrapingProgress:
    """Progress update for a scraping job."""

    job_id: UUID
    status: JobStatus
    progress: float = 0.0
    current_page: int = 0
    total_pages: int = 0
    matches_scraped: int = 0
    matches_saved: int = 0
    message: str = ""
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class ScrapingJob:
    """Represents a scraping job."""

    id: UUID = field(default_factory=uuid4)
    job_type: str = ""  # "results" or "upcoming"
    url: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: ScrapingProgress | None = None
    created_at: datetime = field(default_factory=datetime.now)


class ScrapingService(BaseService[Any]):
    """Service for managing scraping operations using API-Football."""

    # In-memory job storage (replace with Redis/DB for production)
    _jobs: dict[UUID, ScrapingJob] = {}

    def __init__(
        self,
        session: Session,
        progress_callback: Callable[[ScrapingProgress], None] | None = None,
    ) -> None:
        """Initialize the scraping service.

        Args:
            session: SQLAlchemy database session
            progress_callback: Optional callback for progress updates
        """
        super().__init__(session)
        self.progress_callback = progress_callback

    def _emit_progress(self, progress: ScrapingProgress) -> None:
        """Emit progress update to callback if registered.

        Args:
            progress: Progress update to emit
        """
        if self.progress_callback:
            self.progress_callback(progress)

    def create_job(self, job_type: str, url: str) -> ScrapingJob:
        """Create a new scraping job.

        Args:
            job_type: Type of job ("results" or "upcoming")
            url: URL to scrape

        Returns:
            Created ScrapingJob
        """
        job = ScrapingJob(job_type=job_type, url=url)
        self._jobs[job.id] = job
        return job

    def get_job(self, job_id: UUID) -> ScrapingJob | None:
        """Get a job by ID.

        Args:
            job_id: UUID of the job

        Returns:
            ScrapingJob or None if not found
        """
        return self._jobs.get(job_id)

    def list_jobs(self, status: JobStatus | None = None) -> list[ScrapingJob]:
        """List all jobs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of ScrapingJob objects sorted by creation date
        """
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def get_or_create_tournament(
        self, country: str, name: str, slug: str, api_football_id: int | None = None
    ) -> Tournament:
        """Get or create a tournament.

        Args:
            country: Country name
            name: Tournament name
            slug: URL slug for the tournament
            api_football_id: API-Football league ID

        Returns:
            Tournament instance
        """
        # Try to find by API-Football ID first, then by slug
        if api_football_id:
            tournament = self.session.execute(
                select(Tournament).where(Tournament.api_football_id == api_football_id)
            ).scalar_one_or_none()
            if tournament:
                return tournament

        tournament = self.session.execute(
            select(Tournament).where(Tournament.url_slug == slug)
        ).scalar_one_or_none()

        if not tournament:
            tournament = Tournament(
                name=name,
                country=country,
                url_slug=slug,
                api_football_id=api_football_id,
            )
            self.session.add(tournament)
            self.session.flush()

        return tournament

    def get_or_create_team(self, name: str, api_football_id: int | None = None) -> Team:
        """Get or create a team.

        Args:
            name: Team name
            api_football_id: API-Football team ID

        Returns:
            Team instance
        """
        # Try to find by API-Football ID first, then by name
        if api_football_id:
            team = self.session.execute(
                select(Team).where(Team.api_football_id == api_football_id)
            ).scalar_one_or_none()
            if team:
                return team

        team = self.session.execute(
            select(Team).where(Team.name == name)
        ).scalar_one_or_none()

        if not team:
            team = Team(name=name, api_football_id=api_football_id)
            self.session.add(team)
            self.session.flush()

        return team

    def scrape_upcoming(
        self,
        url: str = "",
        headless: bool = True,
        league_ids: list[int] | None = None,
    ) -> ScrapingProgress:
        """Fetch upcoming matches from API-Football.

        Args:
            url: Legacy parameter (ignored - uses API-Football)
            headless: Legacy parameter (ignored - no browser needed)
            league_ids: List of league IDs to fetch (defaults to config)

        Returns:
            Final progress update
        """
        job = self.create_job("upcoming", url or "api-football://upcoming")
        progress = ScrapingProgress(
            job_id=job.id,
            status=JobStatus.RUNNING,
            progress=5.0,
            started_at=datetime.now(),
            message="Starting upcoming matches fetch from API-Football...",
        )
        self._emit_progress(progress)

        try:
            client = APIFootballClient()

            progress.progress = 10.0
            progress.message = "Fetching upcoming fixtures from API-Football..."
            self._emit_progress(progress)

            # Fetch upcoming fixtures for all configured leagues
            response = client.get_all_upcoming(league_ids=league_ids, next=10)
            fixtures = response.fixtures

            progress.progress = 50.0
            progress.matches_scraped = len(fixtures)
            progress.message = f"Found {len(fixtures)} upcoming matches. Saving..."
            self._emit_progress(progress)

            # Save matches to database
            saved_count = self._save_api_fixtures(fixtures, is_upcoming=True)
            progress.matches_saved = saved_count

            progress.progress = 100.0
            progress.status = JobStatus.COMPLETED
            progress.completed_at = datetime.now()
            progress.message = (
                f"Completed! Fetched {len(fixtures)} upcoming matches from "
                f"{response.requests_made} API requests, saved {saved_count}."
            )

        except ValueError as e:
            # API key not configured
            progress.status = JobStatus.FAILED
            progress.progress = 0.0
            progress.error = str(e)
            progress.message = f"Configuration error: {e}"
            progress.completed_at = datetime.now()
        except Exception as e:
            progress.status = JobStatus.FAILED
            progress.progress = 0.0
            progress.error = str(e)
            progress.message = f"Failed: {e}"
            progress.completed_at = datetime.now()

        self._emit_progress(progress)
        job.status = progress.status
        job.progress = progress

        return progress

    def scrape_results(
        self,
        url: str = "",
        max_pages: int | None = None,
        headless: bool = True,
        league_id: int | None = None,
        season: int | None = None,
    ) -> ScrapingProgress:
        """Fetch match results from API-Football.

        Args:
            url: Legacy parameter (ignored - uses API-Football)
            max_pages: Number of results to fetch per league
            headless: Legacy parameter (ignored - no browser needed)
            league_id: Specific league ID to fetch results for
            season: Season year to fetch results for

        Returns:
            Final progress update
        """
        job = self.create_job("results", url or "api-football://results")
        progress = ScrapingProgress(
            job_id=job.id,
            status=JobStatus.RUNNING,
            progress=5.0,
            started_at=datetime.now(),
            message="Starting results fetch from API-Football...",
        )
        self._emit_progress(progress)

        try:
            client = APIFootballClient()

            progress.progress = 10.0
            progress.message = "Fetching match results from API-Football..."
            self._emit_progress(progress)

            # Determine how many results to fetch
            last_count = max_pages if max_pages else 20

            if league_id:
                # Fetch results for specific league
                response = client.get_results(league_id=league_id, last=last_count)
                fixtures = response.fixtures
                requests_made = response.requests_made
            else:
                # Fetch results for all configured leagues
                from algobet.infrastructure.config import get_config

                config = get_config()
                league_ids = config.scraping.default_league_ids

                fixtures = []
                requests_made = 0
                for lid in league_ids:
                    try:
                        resp = client.get_results(league_id=lid, last=last_count)
                        fixtures.extend(resp.fixtures)
                        requests_made += resp.requests_made
                    except Exception as e:
                        print(f"Warning: Failed to fetch results for league {lid}: {e}")
                        continue

            progress.progress = 50.0
            progress.matches_scraped = len(fixtures)
            progress.message = f"Found {len(fixtures)} match results. Saving..."
            self._emit_progress(progress)

            # Save matches to database
            saved_count = self._save_api_fixtures(fixtures, is_upcoming=False)
            progress.matches_saved = saved_count

            progress.progress = 100.0
            progress.status = JobStatus.COMPLETED
            progress.completed_at = datetime.now()
            progress.message = (
                f"Completed! Fetched {len(fixtures)} results from "
                f"{requests_made} API requests, saved {saved_count}."
            )

        except ValueError as e:
            # API key not configured
            progress.status = JobStatus.FAILED
            progress.progress = 0.0
            progress.error = str(e)
            progress.message = f"Configuration error: {e}"
            progress.completed_at = datetime.now()
        except Exception as e:
            progress.status = JobStatus.FAILED
            progress.progress = 0.0
            progress.error = str(e)
            progress.message = f"Failed: {e}"
            progress.completed_at = datetime.now()

        self._emit_progress(progress)
        job.status = progress.status
        job.progress = progress

        return progress

    def _save_api_fixtures(
        self, fixtures: list[APIFootballFixture], is_upcoming: bool = True
    ) -> int:
        """Save fixtures from API-Football to database.

        Args:
            fixtures: List of APIFootballFixture objects
            is_upcoming: Whether these are upcoming matches or results

        Returns:
            Number of matches saved
        """
        saved = 0
        for fixture in fixtures:
            # Get or create teams
            home_team = self.get_or_create_team(
                fixture.home_team.name,
                api_football_id=fixture.home_team.id,
            )
            away_team = self.get_or_create_team(
                fixture.away_team.name,
                api_football_id=fixture.away_team.id,
            )

            # Get or create tournament
            tournament = self.get_or_create_tournament(
                country=fixture.league.country,
                name=fixture.league.name,
                slug=fixture.league.name.lower().replace(" ", "-"),
                api_football_id=fixture.league.id,
            )

            # Check for existing match
            existing = self.session.execute(
                select(Match).where(
                    Match.home_team_id == home_team.id,
                    Match.away_team_id == away_team.id,
                    Match.match_date == fixture.date,
                )
            ).scalar_one_or_none()

            if existing:
                # Update odds if available
                if fixture.odds_home:
                    existing.odds_home = fixture.odds_home
                    existing.odds_draw = fixture.odds_draw
                    existing.odds_away = fixture.odds_away

                # Update scores for finished matches
                if fixture.is_finished:
                    existing.home_score = fixture.goals.home
                    existing.away_score = fixture.goals.away
                    existing.status = "FINISHED"
            else:
                # Determine status
                if fixture.is_finished:
                    status = "FINISHED"
                elif fixture.is_live:
                    status = "LIVE"
                else:
                    status = "SCHEDULED"

                # Create new match
                match = Match(
                    tournament_id=tournament.id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    match_date=fixture.date,
                    status=status,
                    home_score=fixture.goals.home if fixture.is_finished else None,
                    away_score=fixture.goals.away if fixture.is_finished else None,
                    odds_home=fixture.odds_home,
                    odds_draw=fixture.odds_draw,
                    odds_away=fixture.odds_away,
                    api_football_id=fixture.id,
                )
                self.session.add(match)
                saved += 1

        self.session.flush()
        return saved
