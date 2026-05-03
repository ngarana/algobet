"""Scraping service for orchestrating data collection from OddsPortal."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from algobet.models import Match, Season, Team, Tournament
from algobet.scraper import OddsPortalScraper, ScrapedMatch, is_retryable_network_error
from algobet.services.base import BaseService

logger = logging.getLogger(__name__)


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
        # Make slug unique by including country to avoid collisions
        # e.g., Germany's Bundesliga and Austria's Bundesliga both use "bundesliga" slug
        unique_slug = f"{country.lower()}-{slug}" if slug else slug

        tournament = self.session.execute(
            select(Tournament).where(Tournament.url_slug == unique_slug)
        ).scalar_one_or_none()

        # Fallback: check by name + country if slug-based lookup fails
        if not tournament:
            tournament = self.session.execute(
                select(Tournament).where(
                    Tournament.name == name,
                    Tournament.country == country,
                )
            ).scalar_one_or_none()

        if not tournament:
            tournament = Tournament(name=name, country=country, url_slug=unique_slug)
            self.session.add(tournament)
            self.session.flush()

        return tournament

    def get_or_create_season(
        self, tournament: Tournament, season_label: str | None
    ) -> Season | None:
        """Get or create a season for a tournament from a YYYY/YYYY label."""
        season_info = self._parse_season_label(season_label)
        if season_info is None:
            return None

        season = self.session.execute(
            select(Season).where(
                Season.tournament_id == tournament.id,
                Season.name == str(season_info["name"]),
            )
        ).scalar_one_or_none()

        if not season:
            season = Season(
                tournament_id=tournament.id,
                name=str(season_info["name"]),
                start_year=int(season_info["start_year"]),
                end_year=int(season_info["end_year"]),
                url_suffix=str(season_info["url_suffix"]),
            )
            self.session.add(season)
            self.session.flush()
        elif season.url_suffix != str(season_info["url_suffix"]):
            season.url_suffix = str(season_info["url_suffix"])

        return season

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

    def _parse_season_label(
        self, season_label: str | None
    ) -> dict[str, str | int] | None:
        """Normalize a season label into database-ready fields."""
        if season_label is None:
            return None

        normalized = season_label.strip()
        if not normalized:
            return None

        match = re.fullmatch(r"(\d{4})[-/](\d{4})", normalized)
        if not match:
            logger.warning(f"Ignoring unsupported season label: {season_label}")
            return None

        start_year = int(match.group(1))
        end_year = int(match.group(2))
        if end_year != start_year + 1:
            logger.warning(f"Ignoring invalid season range: {season_label}")
            return None

        return {
            "name": f"{start_year}/{end_year}",
            "start_year": start_year,
            "end_year": end_year,
            "url_suffix": f"{start_year}-{end_year}",
        }

    def _season_label_from_results_url(self, url: str) -> str | None:
        """Infer a YYYY/YYYY season label from a results URL when present."""
        match = re.search(r"-(\d{4})-(\d{4})(?=/results/)", url)
        if not match:
            return None
        return f"{match.group(1)}/{match.group(2)}"

    def _base_results_url(self, url: str) -> str:
        """Strip any season suffix from a results URL."""
        normalized_url = url.rstrip("/") + "/"
        return re.sub(r"-(\d{4}-\d{4})(?=/results/)", "", normalized_url)

    def _build_results_url_for_season(self, url: str, season_label: str) -> str:
        """Build a season-specific results URL from a base tournament results URL."""
        season_info = self._parse_season_label(season_label)
        base_url = self._base_results_url(url)
        if season_info is None:
            return base_url

        pattern = r"/([^/]+)/results/$"
        replacement = f"/\\1-{season_info['url_suffix']}/results/"
        return re.sub(pattern, replacement, base_url)

    def _results_progress_value(
        self,
        page_num: int,
        total_pages: int,
        *,
        season_index: int = 1,
        total_seasons: int = 1,
    ) -> float:
        """Compute progress for page-based historical scraping."""
        season_span = 85.0 / max(total_seasons, 1)
        season_base = 5.0 + (season_index - 1) * season_span
        return min(90.0, season_base + (page_num / max(total_pages, 1)) * season_span)

    def _results_progress_prefix(
        self,
        season_name: str | None,
        *,
        season_index: int = 1,
        total_seasons: int = 1,
    ) -> str:
        """Build a human-readable prefix for progress messages."""
        if total_seasons > 1 and season_name:
            return f"Season {season_name} ({season_index}/{total_seasons})"
        if total_seasons > 1:
            return f"Season {season_index}/{total_seasons}"
        if season_name:
            return f"Season {season_name}"
        return "Results"

    def scrape_upcoming(
        self,
        url: str | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
    ) -> ScrapingProgress:
        """Scrape upcoming matches.

        Args:
            url: URL to scrape upcoming matches from
            headless: Run browser in headless mode

        Returns:
            Final progress update
        """
        if url is None:
            url = self._matches_url_for_date(datetime.now(timezone.utc).date())
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

                matches_data = scraper.scrape_upcoming_matches(
                    only_future_matches=False, buffer_minutes=0
                )
                progress.matches_scraped = len(matches_data)
                progress.progress = 75.0
                progress.message = f"Found {len(matches_data)} matches. Saving..."
                self._emit_progress(progress)

                # Save matches
                saved_count = self._save_upcoming_matches(matches_data, target_team_id)
                progress.matches_saved = saved_count

                progress.status = JobStatus.COMPLETED
                progress.progress = 100.0
                progress.completed_at = datetime.now(timezone.utc)
                progress.message = (
                    f"Completed! Scraped {len(matches_data)} matches, "
                    f"saved {saved_count}."
                )

        except ConnectionError as e:
            progress.status = JobStatus.FAILED
            progress.progress = 100.0
            progress.error = str(e)
            progress.message = f"Network connection failed: {e}"
            progress.completed_at = datetime.now(timezone.utc)
            logger.error(f"Connection error in scrape_upcoming: {e}")
        except Exception as e:
            error_msg = str(e)
            is_network_error = any(
                err in error_msg.lower()
                for err in [
                    "err_name_not_resolved",
                    "err_internet_disconnected",
                    "err_connection_refused",
                    "err_connection_reset",
                    "err_connection_timed_out",
                    "timeout",
                    "net::",
                ]
            )

            if is_network_error:
                progress.status = JobStatus.FAILED
                progress.progress = 100.0
                progress.error = error_msg
                progress.message = f"Network error during scraping: {e}"
            else:
                progress.status = JobStatus.FAILED
                progress.progress = 100.0
                progress.error = error_msg
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
        target_team_id: int | None = None,
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

                saved_count = self._save_upcoming_matches(
                    filtered_matches, target_team_id
                )
                progress.matches_saved = saved_count

                progress.status = JobStatus.COMPLETED
                progress.progress = 100.0
                progress.completed_at = datetime.now(timezone.utc)
                progress.message = (
                    f"Completed! Scraped {len(filtered_matches)} matches for "
                    f"{target_date.isoformat()}, saved {saved_count}."
                )

        except ConnectionError as e:
            progress.status = JobStatus.FAILED
            progress.progress = 100.0
            progress.error = str(e)
            progress.message = f"Network connection failed: {e}"
            progress.completed_at = datetime.now(timezone.utc)
            logger.error(f"Connection error in scrape_matches_by_date: {e}")
        except Exception as e:
            error_msg = str(e)
            is_network_error = any(
                err in error_msg.lower()
                for err in [
                    "err_name_not_resolved",
                    "err_internet_disconnected",
                    "err_connection_refused",
                    "err_connection_reset",
                    "err_connection_timed_out",
                    "timeout",
                    "net::",
                ]
            )

            if is_network_error:
                progress.status = JobStatus.FAILED
                progress.progress = 100.0
                progress.error = error_msg
                progress.message = f"Network error during scraping: {e}"
            else:
                progress.status = JobStatus.FAILED
                progress.progress = 100.0
                progress.error = error_msg
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
        target_team_id: int | None = None,
        season: str | None = None,
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
                (
                    season_matches_scraped,
                    season_matches_saved,
                    total_pages,
                    season_name,
                ) = self._scrape_results_pages(
                    scraper,
                    url,
                    progress,
                    max_pages=max_pages,
                    target_team_id=target_team_id,
                    season=season,
                )

                progress.status = JobStatus.COMPLETED
                progress.progress = 100.0
                progress.completed_at = datetime.now(timezone.utc)
                completed_scope = season_name or "requested results"
                progress.message = (
                    f"Completed! Scraped {season_matches_scraped} matches from "
                    f"{total_pages} pages for {completed_scope}, "
                    f"saved {season_matches_saved}."
                )

        except ConnectionError as e:
            # Network/DNS specific error handling
            progress.status = JobStatus.FAILED
            progress.progress = 100.0
            progress.error = str(e)
            progress.message = (
                f"Network connection failed: {e}. "
                "Please check your network connection and DNS settings. "
                "The target site may be unreachable."
            )
            progress.completed_at = datetime.now(timezone.utc)
            logger.error(f"Connection error in scrape_results: {e}")
        except Exception as e:
            error_msg = str(e)
            if is_retryable_network_error(e):
                progress.status = JobStatus.FAILED
                progress.progress = 100.0
                progress.error = error_msg
                progress.message = (
                    f"Network error during scraping: {e}. "
                    "Please check your network connection and DNS settings."
                )
            else:
                progress.status = JobStatus.FAILED
                progress.progress = 100.0
                progress.error = error_msg
                progress.message = f"Failed: {e}"
            progress.completed_at = datetime.now(timezone.utc)

        self._emit_progress(progress)
        job.status = progress.status
        job.progress = progress

        return progress

    def scrape_results_range(
        self,
        url: str,
        seasons: list[str],
        max_pages: int | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
    ) -> ScrapingProgress:
        """Scrape multiple historical seasons while reusing one browser session."""
        if not seasons:
            raise ValueError(
                "At least one season is required for batch results scraping"
            )

        job = self.create_job("results", url)
        progress = ScrapingProgress(
            job_id=job.id,
            status=JobStatus.RUNNING,
            progress=5.0,
            started_at=datetime.now(timezone.utc),
            message="Starting batched results scrape...",
        )
        self._emit_progress(progress)

        try:
            with OddsPortalScraper(headless=headless) as scraper:
                total_seasons = len(seasons)
                for season_index, season_label in enumerate(seasons, start=1):
                    season_url = self._build_results_url_for_season(url, season_label)
                    progress.current_page = 0
                    progress.total_pages = 0
                    progress.message = (
                        f"Loading season {season_label} "
                        f"({season_index}/{total_seasons})..."
                    )
                    self._emit_progress(progress)
                    self._scrape_results_pages(
                        scraper,
                        season_url,
                        progress,
                        max_pages=max_pages,
                        target_team_id=target_team_id,
                        season=season_label,
                        season_index=season_index,
                        total_seasons=total_seasons,
                    )

                progress.status = JobStatus.COMPLETED
                progress.progress = 100.0
                progress.completed_at = datetime.now(timezone.utc)
                progress.message = (
                    f"Completed! Scraped {progress.matches_scraped} matches across "
                    f"{len(seasons)} seasons, saved {progress.matches_saved}."
                )
        except ConnectionError as e:
            progress.status = JobStatus.FAILED
            progress.progress = 100.0
            progress.error = str(e)
            progress.message = (
                f"Network connection failed: {e}. "
                "Please check your network connection and DNS settings. "
                "The target site may be unreachable."
            )
            progress.completed_at = datetime.now(timezone.utc)
            logger.error(f"Connection error in scrape_results_range: {e}")
        except Exception as e:
            error_msg = str(e)
            if is_retryable_network_error(e):
                progress.status = JobStatus.FAILED
                progress.progress = 100.0
                progress.error = error_msg
                progress.message = (
                    f"Network error during scraping: {e}. "
                    "Please check your network connection and DNS settings."
                )
            else:
                progress.status = JobStatus.FAILED
                progress.progress = 100.0
                progress.error = error_msg
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
        match = re.search(r"/football/([^/]+)/([^/]+?)(?:-\d{4}-\d{4})?/results/", url)
        if not match:
            raise ValueError(f"Cannot parse league info from URL: {url}")

        country = match.group(1).replace("-", " ").title()
        slug = match.group(2)
        league_name = slug.replace("-", " ").title()

        return country, league_name, slug

    # def _matches_url_for_date(self, target_date: date_cls) -> str:
    #     """Build the OddsPortal daily matches URL for a specific date.

    #     Args:
    #         target_date: The date to build the URL for

    #     Returns:
    #         URL for the daily matches page on the given date
    #     """
    #     date_str = target_date.strftime("%Y%m%d")
    #     return f"https://www.oddsportal.com/matches/football/{date_str}/"
    def _matches_url_for_date(self, target_date: date_cls) -> str:
        """Build the OddsPortal URL for upcoming matches.

        IMPORTANT:
        OddsPortal only has ONE upcoming-matches page: https://www.oddsportal.com/matches/
        It always shows "today" + several future days via lazy loading.

        Date-specific scraping is achieved by:
        1. Scraping the full page (with Show-more + infinite scroll)
        2. Filtering matches by their parsed match_date
           (done in _filter_matches_by_date)

        This is the reliable and only supported way.
        """
        # Always return the main upcoming matches page
        # (we could add /soccer/ but the base URL works for all sports
        # and gives football by default)
        return "https://www.oddsportal.com/matches/"

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

    def _save_upcoming_matches(
        self, matches_data: list[dict[str, Any]], target_team_id: int | None = None
    ) -> int:
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

            # Team isolation logic
            if (
                target_team_id is not None
                and home_team.id != target_team_id
                and away_team.id != target_team_id
            ):
                continue

            # Get or create tournament (if available)
            tournament = None
            if match_data.get("tournament_slug"):
                tournament = self.get_or_create_tournament(
                    country=match_data.get("country", "Unknown"),
                    name=match_data.get("tournament_name", "Unknown"),
                    slug=match_data["tournament_slug"],
                )

            # Check for existing match (compare by date and tournament if available)
            # Use defensive check with both team IDs and exact date
            existing_query = select(Match).where(
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
                func.date(Match.match_date) == match_data["match_date"].date(),
            )
            # Only check tournament (to avoid cross-tournament duplicates)
            if tournament:
                existing_query = existing_query.where(
                    Match.tournament_id == tournament.id
                )
            existing = self.session.execute(existing_query).scalar_one_or_none()

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
                # Verify no duplicate exists (defensive check before insert)
                dup_check = self.session.execute(
                    select(Match).where(
                        Match.home_team_id == home_team.id,
                        Match.away_team_id == away_team.id,
                        func.date(Match.match_date)
                        == func.date(match_data["match_date"]),
                    )
                ).scalar_one_or_none()
                if dup_check:
                    match_date_str = match_data["match_date"].date()
                    logger.warning(
                        f"Duplicate upcoming match: {home_team.name} vs "
                        f"{away_team.name} on {match_date_str}, skipping"
                    )
                    continue

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
        self,
        matches: list[ScrapedMatch],
        tournament: Tournament,
        season: Season | None = None,
        target_team_id: int | None = None,
    ) -> int:
        """Save result matches to database.

        Args:
            matches: List of ScrapedMatch objects
            tournament: Tournament instance
            target_team_id: Optional ID of team to filter saves to

        Returns:
            Number of matches saved
        """
        saved = 0
        for scraped in matches:
            home_team = self.get_or_create_team(scraped.home_team)
            away_team = self.get_or_create_team(scraped.away_team)

            # Team isolation logic
            if (
                target_team_id is not None
                and home_team.id != target_team_id
                and away_team.id != target_team_id
            ):
                continue

            match_lookup = select(Match).where(
                Match.tournament_id == tournament.id,
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
                Match.match_date == scraped.match_date,
            )
            if season is not None:
                existing = self.session.execute(
                    match_lookup.where(Match.season_id == season.id)
                ).scalar_one_or_none()
                if existing is None:
                    existing = self.session.execute(
                        match_lookup.where(Match.season_id.is_(None))
                    ).scalar_one_or_none()
            else:
                existing = self.session.execute(
                    match_lookup.where(Match.season_id.is_(None))
                ).scalar_one_or_none()

            if existing:
                if season is not None and existing.season_id is None:
                    existing.season_id = season.id
                existing.home_score = scraped.home_score
                existing.away_score = scraped.away_score
                existing.status = "FINISHED"
                existing.odds_home = scraped.odds_home
                existing.odds_draw = scraped.odds_draw
                existing.odds_away = scraped.odds_away
                existing.num_bookmakers = scraped.num_bookmakers
                continue

            # Verify no duplicate exists (defensive check before insert)
            duplicate_check = self.session.execute(
                select(Match).where(
                    Match.tournament_id == tournament.id,
                    Match.season_id == (season.id if season else None),
                    Match.home_team_id == home_team.id,
                    Match.away_team_id == away_team.id,
                    func.date(Match.match_date) == func.date(scraped.match_date),
                )
            ).scalar_one_or_none()
            if duplicate_check:
                logger.warning(
                    f"Duplicate match detected: {home_team.name} vs {away_team.name} "
                    f"on {scraped.match_date.date()}, skipping"
                )
                continue

            match = Match(
                tournament_id=tournament.id,
                season_id=season.id if season else None,
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

    def _scrape_results_pages(
        self,
        scraper: OddsPortalScraper,
        url: str,
        progress: ScrapingProgress,
        *,
        max_pages: int | None = None,
        target_team_id: int | None = None,
        season: str | None = None,
        season_index: int = 1,
        total_seasons: int = 1,
    ) -> tuple[int, int, int, str | None]:
        """Scrape and persist a results board page by page."""
        scraper.navigate_to_results(url)

        total_pages = scraper.get_page_count()
        if max_pages:
            total_pages = min(total_pages, max_pages)
        progress.total_pages = total_pages

        country, league_name, slug = self._parse_league_info(url)
        tournament = self.get_or_create_tournament(country, league_name, slug)
        season_record = self.get_or_create_season(
            tournament, season or self._season_label_from_results_url(url)
        )
        season_name = season_record.name if season_record else None
        progress_prefix = self._results_progress_prefix(
            season_name,
            season_index=season_index,
            total_seasons=total_seasons,
        )
        progress.message = f"{progress_prefix}: loaded {total_pages} pages."
        self._emit_progress(progress)

        season_matches_scraped = 0
        season_matches_saved = 0

        for page_num in range(1, total_pages + 1):
            progress.current_page = page_num
            if page_num > 1 and not scraper.go_to_page(page_num):
                raise RuntimeError(f"Failed to navigate to page {page_num}")

            matches = scraper.scrape_current_page()
            saved_count = self._save_result_matches(
                matches,
                tournament,
                season=season_record,
                target_team_id=target_team_id,
            )
            self.session.commit()

            season_matches_scraped += len(matches)
            season_matches_saved += saved_count
            progress.matches_scraped += len(matches)
            progress.matches_saved += saved_count
            progress.progress = self._results_progress_value(
                page_num,
                total_pages,
                season_index=season_index,
                total_seasons=total_seasons,
            )
            progress.message = (
                f"{progress_prefix}: page {page_num}/{total_pages}, "
                f"{len(matches)} scraped, {saved_count} saved "
                f"({season_matches_saved} saved this season)."
            )
            self._emit_progress(progress)

        return season_matches_scraped, season_matches_saved, total_pages, season_name
