"""Scraping service for orchestrating data collection from OddsPortal."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import Match, Team, Tournament
from algobet.scraper import OddsPortalScraper, ScrapedMatch
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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ScrapingService(BaseService[Any]):
    """Service for managing scraping operations."""

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
        self, country: str, name: str, slug: str
    ) -> Tournament:
        """Get or create a tournament.

        Args:
            country: Country name
            name: Tournament name
            slug: URL slug for the tournament

        Returns:
            Tournament instance
        """
        tournament = self.session.execute(
            select(Tournament).where(Tournament.url_slug == slug)
        ).scalar_one_or_none()

        if not tournament:
            tournament = Tournament(name=name, country=country, url_slug=slug)
            self.session.add(tournament)
            self.session.flush()

        return tournament

    def get_or_create_team(self, name: str) -> Team:
        """Get or create a team.

        Args:
            name: Team name

        Returns:
            Team instance
        """
        team = self.session.execute(
            select(Team).where(Team.name == name)
        ).scalar_one_or_none()

        if not team:
            team = Team(name=name)
            self.session.add(team)
            self.session.flush()

        return team

    def scrape_upcoming(
        self,
        url: str = "https://www.oddsportal.com/matches/football/",
        headless: bool = True,
    ) -> ScrapingProgress:
        """Scrape upcoming matches.

        Args:
            url: URL to scrape upcoming matches from
            headless: Run browser in headless mode

        Returns:
            Final progress update
        """
        job = self.create_job("upcoming", url)
        progress = ScrapingProgress(
            job_id=job.id,
            status=JobStatus.RUNNING,
            progress=5.0,
            started_at=datetime.now(timezone.utc),
            message="Starting upcoming matches scrape...",
        )
        self._emit_progress(progress)

        try:
            with OddsPortalScraper(headless=headless) as scraper:
                scraper.navigate_to_upcoming(url)

                progress.progress = 25.0
                progress.current_page = 1
                progress.total_pages = 1
                progress.message = "Scraping upcoming matches..."
                self._emit_progress(progress)

                matches_data = scraper.scrape_upcoming_matches()
                progress.matches_scraped = len(matches_data)
                progress.progress = 75.0
                progress.message = f"Found {len(matches_data)} matches. Saving..."
                self._emit_progress(progress)

                # Save matches
                saved_count = self._save_upcoming_matches(matches_data)
                progress.matches_saved = saved_count

                progress.status = JobStatus.COMPLETED
                progress.progress = 100.0
                progress.completed_at = datetime.now(timezone.utc)
                progress.message = (
                    f"Completed! Scraped {len(matches_data)} matches, "
                    f"saved {saved_count}."
                )

        except Exception as e:
            progress.status = JobStatus.FAILED
            progress.progress = 100.0
            progress.error = str(e)
            progress.message = f"Failed: {e}"
            progress.completed_at = datetime.now(timezone.utc)

        self._emit_progress(progress)
        job.status = progress.status
        job.progress = progress

        return progress

    def scrape_matches_by_date(
        self,
        target_date: date_cls,
        url: str | None = None,
        headless: bool = True,
    ) -> ScrapingProgress:
        """Scrape matches scheduled for a specific calendar date.

        Args:
            target_date: Calendar date to scrape
            url: Optional pre-resolved OddsPortal URL to scrape
            headless: Run browser in headless mode

        Returns:
            Final progress update
        """
        target_url = url or self._matches_url_for_date(target_date)
        job = self.create_job("by-date", target_url)
        progress = ScrapingProgress(
            job_id=job.id,
            status=JobStatus.RUNNING,
            progress=5.0,
            started_at=datetime.now(timezone.utc),
            message=f"Starting scrape for {target_date.isoformat()}...",
        )
        self._emit_progress(progress)

        try:
            with OddsPortalScraper(headless=headless) as scraper:
                scraper.navigate_to_upcoming(target_url)

                progress.progress = 25.0
                progress.current_page = 1
                progress.total_pages = 1
                progress.message = (
                    f"Loaded matches board for {target_date.isoformat()}. Scraping..."
                )
                self._emit_progress(progress)

                matches_data = scraper.scrape_upcoming_matches(
                    only_future_matches=False,
                    buffer_minutes=0,
                )
                progress.matches_scraped = len(matches_data)

                filtered_matches = self._filter_matches_by_date(
                    matches_data,
                    target_date,
                )
                progress.progress = 75.0
                progress.matches_scraped = len(filtered_matches)
                progress.message = (
                    f"Found {len(filtered_matches)} matches for "
                    f"{target_date.isoformat()}. Saving..."
                )
                self._emit_progress(progress)

                saved_count = self._save_upcoming_matches(filtered_matches)
                progress.matches_saved = saved_count

                progress.status = JobStatus.COMPLETED
                progress.progress = 100.0
                progress.completed_at = datetime.now(timezone.utc)
                progress.message = (
                    f"Completed! Scraped {len(filtered_matches)} matches for "
                    f"{target_date.isoformat()}, saved {saved_count}."
                )

        except Exception as e:
            progress.status = JobStatus.FAILED
            progress.progress = 100.0
            progress.error = str(e)
            progress.message = f"Failed: {e}"
            progress.completed_at = datetime.now(timezone.utc)

        self._emit_progress(progress)
        job.status = progress.status
        job.progress = progress

        return progress

    def scrape_results(
        self,
        url: str,
        max_pages: int | None = None,
        headless: bool = True,
    ) -> ScrapingProgress:
        """Scrape historical results.

        Args:
            url: OddsPortal results URL
            max_pages: Maximum pages to scrape (None for all)
            headless: Run browser in headless mode

        Returns:
            Final progress update
        """
        job = self.create_job("results", url)
        progress = ScrapingProgress(
            job_id=job.id,
            status=JobStatus.RUNNING,
            progress=5.0,
            started_at=datetime.now(timezone.utc),
            message="Starting results scrape...",
        )
        self._emit_progress(progress)

        try:
            with OddsPortalScraper(headless=headless) as scraper:
                scraper.navigate_to_results(url)

                # Get total pages
                total_pages = scraper.get_page_count()
                if max_pages:
                    total_pages = min(total_pages, max_pages)
                progress.total_pages = total_pages
                progress.message = f"Loaded results board with {total_pages} pages."
                self._emit_progress(progress)

                # Parse league info from URL
                country, league_name, slug = self._parse_league_info(url)
                tournament = self.get_or_create_tournament(country, league_name, slug)

                # Scrape each page
                all_matches = []
                for page_num in range(1, total_pages + 1):
                    progress.current_page = page_num
                    progress.message = f"Scraping page {page_num}/{total_pages}..."

                    if page_num > 1:
                        scraper.go_to_page(page_num)

                    matches = scraper.scrape_current_page()
                    all_matches.extend(matches)
                    progress.matches_scraped = len(all_matches)
                    progress.progress = min(
                        90.0, 5.0 + (page_num / max(total_pages, 1)) * 85.0
                    )
                    self._emit_progress(progress)

                # Save all matches
                progress.progress = 95.0
                progress.message = (
                    f"Saving {len(all_matches)} matches from {total_pages} pages..."
                )
                self._emit_progress(progress)
                saved_count = self._save_result_matches(all_matches, tournament)
                progress.matches_saved = saved_count

                progress.status = JobStatus.COMPLETED
                progress.progress = 100.0
                progress.completed_at = datetime.now(timezone.utc)
                progress.message = (
                    f"Completed! Scraped {len(all_matches)} matches from "
                    f"{total_pages} pages, saved {saved_count}."
                )

        except Exception as e:
            progress.status = JobStatus.FAILED
            progress.progress = 100.0
            progress.error = str(e)
            progress.message = f"Failed: {e}"
            progress.completed_at = datetime.now(timezone.utc)

        self._emit_progress(progress)
        job.status = progress.status
        job.progress = progress

        return progress

    def _parse_league_info(self, url: str) -> tuple[str, str, str]:
        """Extract country, league name, and slug from URL.

        Args:
            url: OddsPortal results URL

        Returns:
            Tuple of (country, league_name, slug)
        """
        import re

        match = re.search(r"/football/([^/]+)/([^/]+?)(?:-\d{4}-\d{4})?/results/", url)
        if not match:
            raise ValueError(f"Cannot parse league info from URL: {url}")

        country = match.group(1).replace("-", " ").title()
        slug = match.group(2)
        league_name = slug.replace("-", " ").title()

        return country, league_name, slug

    def _matches_url_for_date(self, target_date: date_cls) -> str:
        """Build the OddsPortal daily matches URL for a specific date."""
        return (
            "https://www.oddsportal.com/matches/football/"
            f"{target_date.strftime('%Y%m%d')}/"
        )

    def _filter_matches_by_date(
        self,
        matches_data: list[dict[str, Any]],
        target_date: date_cls,
    ) -> list[dict[str, Any]]:
        """Keep only matches whose parsed kickoff falls on the target date."""
        return [
            match_data
            for match_data in matches_data
            if match_data.get("match_date")
            and match_data["match_date"].date() == target_date
        ]

    def _save_upcoming_matches(self, matches_data: list[dict[str, Any]]) -> int:
        """Save upcoming matches to database.

        Args:
            matches_data: List of match data dictionaries

        Returns:
            Number of matches saved
        """
        saved = 0
        for match_data in matches_data:
            # Get or create teams
            home_team = self.get_or_create_team(match_data["home_team"])
            away_team = self.get_or_create_team(match_data["away_team"])

            # Get or create tournament (if available)
            tournament = None
            if match_data.get("tournament_slug"):
                tournament = self.get_or_create_tournament(
                    country=match_data.get("country", "Unknown"),
                    name=match_data.get("tournament_name", "Unknown"),
                    slug=match_data["tournament_slug"],
                )

            # Check for existing match
            existing = self.session.execute(
                select(Match).where(
                    Match.home_team_id == home_team.id,
                    Match.away_team_id == away_team.id,
                    Match.match_date == match_data["match_date"],
                )
            ).scalar_one_or_none()

            if existing:
                # Update odds if available
                if match_data.get("odds_home"):
                    existing.odds_home = match_data["odds_home"]
                    existing.odds_draw = match_data.get("odds_draw")
                    existing.odds_away = match_data.get("odds_away")

                # Update tournament if missing
                if existing.tournament_id is None and tournament:
                    existing.tournament_id = tournament.id
            else:
                # Create new match
                match = Match(
                    tournament_id=tournament.id if tournament else None,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    match_date=match_data["match_date"],
                    status="SCHEDULED",
                    odds_home=match_data.get("odds_home"),
                    odds_draw=match_data.get("odds_draw"),
                    odds_away=match_data.get("odds_away"),
                )
                self.session.add(match)
                saved += 1

        self.session.flush()
        return saved

    def _save_result_matches(
        self, matches: list[ScrapedMatch], tournament: Tournament
    ) -> int:
        """Save result matches to database.

        Args:
            matches: List of ScrapedMatch objects
            tournament: Tournament instance

        Returns:
            Number of matches saved
        """
        saved = 0
        for scraped in matches:
            home_team = self.get_or_create_team(scraped.home_team)
            away_team = self.get_or_create_team(scraped.away_team)

            # Check for existing match
            existing = self.session.execute(
                select(Match).where(
                    Match.tournament_id == tournament.id,
                    Match.home_team_id == home_team.id,
                    Match.away_team_id == away_team.id,
                    Match.match_date == scraped.match_date,
                )
            ).scalar_one_or_none()

            if not existing:
                match = Match(
                    tournament_id=tournament.id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    match_date=scraped.match_date,
                    home_score=scraped.home_score,
                    away_score=scraped.away_score,
                    status="FINISHED",
                    odds_home=scraped.odds_home,
                    odds_draw=scraped.odds_draw,
                    odds_away=scraped.odds_away,
                    num_bookmakers=scraped.num_bookmakers,
                )
                self.session.add(match)
                saved += 1

        self.session.flush()
        return saved
