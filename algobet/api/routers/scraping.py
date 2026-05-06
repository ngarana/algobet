"""API router for scraping operations with background task support."""

import asyncio
import re
import uuid
from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, status
from pydantic import HttpUrl, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas.scraping import (
    ByDateScrapeRequest,
    PaginatedResponse,
    ResultsScrapeRequest,
    ScrapeScope,
    ScrapingJobCreate,
    ScrapingJobResponse,
    ScrapingJobStatus,
    ScrapingJobUpdate,
    ScrapingProgress,
    ScrapingStats,
    ScrapingType,
    UpcomingScrapeRequest,
)
from algobet.api.websockets import manager
from algobet.importers.soccerdata_importer import (
    LEAGUE_MAPPING as FBREF_LEAGUE_MAPPING,
    SoccerDataImporter,
)
from algobet.models import Season, Tournament
from algobet.services.scraping_service import (
    JobStatus,
    ScrapingProgress as ServiceScrapingProgress,
    ScrapingService,
)

router = APIRouter(tags=["scraping"])

# In-memory storage for scraping jobs (replace with Redis/database in production)
scraping_jobs: dict[str, ScrapingJobResponse] = {}

DEFAULT_UPCOMING_URL = "https://www.oddsportal.com/matches/football/"


def _dispatch_async(coro: Any) -> None:
    """Run an async coroutine from sync or async contexts."""
    event_loop = getattr(manager, "event_loop", None)
    if event_loop is not None and event_loop.is_running():
        asyncio.run_coroutine_threadsafe(coro, event_loop)
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)


def update_job_status(job_id: str, update: ScrapingJobUpdate) -> None:
    """Update job status in storage.

    Thread-safe: Can be called from background threads.
    """
    if job_id in scraping_jobs:
        job = scraping_jobs[job_id]
        old_status = job.status
        job_data = job.model_dump()
        for field, value in update.model_dump(exclude_unset=True).items():
            job_data[field] = value
        scraping_jobs[job_id] = ScrapingJobResponse(**job_data)

        # Broadcast status change via WebSocket
        # Only broadcast if status actually changed
        new_status = job_data.get("status", old_status)
        if new_status != old_status and new_status is not None:
            _dispatch_async(
                manager.broadcast_job_status(
                    job_id, new_status, job_data.get("message", "")
                )
            )


def _country_slug(country: str) -> str:
    """Convert a country name to the OddsPortal URL slug."""
    return "-".join(country.lower().strip().split())


def _base_tournament_url(tournament: Tournament) -> str:
    """Build the base OddsPortal URL for a tournament."""
    return (
        "https://www.oddsportal.com/football/"
        f"{_country_slug(tournament.country)}/{tournament.url_slug}/"
    )


def _normalize_period(value: str | None) -> str | None:
    """Normalize a user-provided period value."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _generate_seasons(period_start: str, period_end: str) -> list[str]:
    """Generate a list of seasons between start and end periods.

    Args:
        period_start: Start period (e.g., '2010-2011')
        period_end: End period (e.g., '2019-2020')

    Returns:
        List of season strings (e.g., ['2010-2011', '2011-2012', ...])
    """
    try:
        start_year = int(period_start.split("-")[0])
        end_year = int(period_end.split("-")[0])
        return [f"{year}-{year + 1}" for year in range(start_year, end_year + 1)]
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid period format. Expected 'YYYY-YYYY', "
            f"got '{period_start}' or '{period_end}'"
        ) from e


def _lookup_tournament(
    db: Session, tournament_id: int | None, tournament_url: str | HttpUrl | None
) -> Tournament | None:
    """Resolve a tournament either by ID or URL slug."""
    if tournament_id is not None:
        tournament = db.execute(
            select(Tournament).where(Tournament.id == tournament_id)
        ).scalar_one_or_none()
        if tournament is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tournament {tournament_id} not found",
            )
        return tournament

    if tournament_url is None:
        return None

    url_parts = str(tournament_url).rstrip("/").split("/")
    slug = url_parts[-1]
    country_slug = None

    if slug == "results" and len(url_parts) >= 2:
        slug = url_parts[-2]
        if len(url_parts) >= 3:
            country_slug = url_parts[-3]
    elif len(url_parts) >= 2:
        country_slug = url_parts[-2]

    slug = re.sub(r"-\d{4}-\d{4}$", "", slug)

    query = select(Tournament).where(Tournament.url_slug == slug)
    if country_slug:
        query = query.where(Tournament.country.ilike(country_slug.replace("-", " ")))

    return db.execute(query).scalar_one_or_none()


def _lookup_season(
    db: Session, tournament_id: int, period: str | None
) -> Season | None:
    """Resolve a season record for a tournament and period string."""
    normalized = _normalize_period(period)
    if normalized is None:
        return None

    seasons = db.execute(
        select(Season).where(Season.tournament_id == tournament_id)
    ).scalars()
    candidates = list(seasons)
    for season in candidates:
        if season.name == normalized:
            return season
        if season.name.replace("/", "-") == normalized.replace("/", "-"):
            return season
        if season.url_suffix and season.url_suffix == normalized.replace("/", "-"):
            return season
    return None


def _build_results_url(
    tournament: Tournament, period: str | None, season: Season | None
) -> str:
    """Build the results URL for a tournament and optional season."""
    base_url = f"{_base_tournament_url(tournament)}results/"
    if season and season.url_suffix:
        return (
            "https://www.oddsportal.com/football/"
            f"{_country_slug(tournament.country)}/{tournament.url_slug}-{season.url_suffix}/results/"
        )

    normalized = _normalize_period(period)
    if normalized and normalized.replace("/", "-") != normalized:
        normalized = normalized.replace("/", "-")
    if normalized:
        return (
            "https://www.oddsportal.com/football/"
            f"{_country_slug(tournament.country)}/{tournament.url_slug}-{normalized}/results/"
        )
    return base_url


def _build_matches_url(request_date: date) -> str:
    """Build the OddsPortal daily matches URL for a specific date."""
    return "https://www.oddsportal.com/matches/"


def _ensure_utc_timestamp(value: datetime | None) -> datetime | None:
    """Normalize timestamps so duration math is always timezone-aware."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _job_create_from_tournament(
    scraping_type: ScrapingType,
    tournament: Tournament | None,
    tournament_url: str | HttpUrl | None,
    *,
    scope: ScrapeScope = ScrapeScope.ALL,
    season: str | None = None,
    period: str | None = None,
    request_date: date | None = None,
    team_id: int | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
) -> ScrapingJobCreate:
    """Create a job payload with semantic metadata."""
    return ScrapingJobCreate(
        scraping_type=scraping_type,
        tournament_url=HttpUrl(str(tournament_url)) if tournament_url else None,
        tournament_id=tournament.id if tournament else None,
        tournament_name=tournament.name if tournament else None,
        season=season,
        start_date=None,
        end_date=None,
        scope=scope,
        country=tournament.country if tournament else None,
        league_name=tournament.name if tournament else None,
        period=request_date.isoformat() if request_date else period,
        team_id=team_id,
        period_start=period_start,
        period_end=period_end,
    )


def _map_job_status(status_value: JobStatus) -> ScrapingJobStatus:
    """Map service job statuses to API statuses."""
    return ScrapingJobStatus(status_value.value)


def _progress_callback_for_job(
    job_id: str,
) -> Callable[[ServiceScrapingProgress], None]:
    """Build a service progress callback bound to an API job ID."""

    def callback(progress: ServiceScrapingProgress) -> None:
        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=_map_job_status(progress.status),
                progress=progress.progress,
                message=progress.message,
                matches_scraped=progress.matches_scraped,
                matches_saved=progress.matches_saved,
                errors=[],
                started_at=progress.started_at,
                completed_at=progress.completed_at,
            ),
        )
        _dispatch_async(
            manager.broadcast_progress(
                ScrapingProgress(
                    job_id=job_id,
                    status=_map_job_status(progress.status),
                    progress=progress.progress,
                    message=progress.message,
                    matches_scraped=progress.matches_scraped,
                    matches_saved=progress.matches_saved,
                    current_page=progress.current_page or None,
                    total_pages=progress.total_pages or None,
                    started_at=progress.started_at,
                    completed_at=progress.completed_at,
                    error=progress.error,
                )
            )
        )

    return callback


def run_scraping_job(
    job_id: str,
    job_create: ScrapingJobCreate,
    *,
    max_pages: int | None = None,
) -> None:
    """Execute scraping job with progress updates.

    Note: This function creates its own database session since it runs
    in a background task after the request session has been closed.
    This is a synchronous function because FastAPI BackgroundTasks runs it in a thread.
    """
    import logging

    from algobet.infrastructure.database import session_scope

    logger = logging.getLogger(__name__)
    logger.info(f"[BG TASK] Starting scraping job {job_id}")

    result = None  # Initialize result variable

    try:
        # Update job status to running
        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.RUNNING,
                progress=0.0,
                message="Starting scraping operation...",
                matches_scraped=0,
                matches_saved=0,
                errors=[],
                started_at=datetime.now(timezone.utc),
                completed_at=None,
            ),
        )
        logger.info(f"[BG TASK] Job {job_id} status updated to RUNNING")

        # Create a new session for the background task
        with session_scope() as db:
            logger.info(f"[BG TASK] Database session created for job {job_id}")
            # Initialize scraping service with database session
            service = ScrapingService(
                db, progress_callback=_progress_callback_for_job(job_id)
            )
            logger.info(f"[BG TASK] ScrapingService initialized for job {job_id}")

            # Execute scraping based on type
            logger.info(
                f"[BG TASK] Starting scrape for type {job_create.scraping_type}"
            )
            if job_create.scraping_type == ScrapingType.UPCOMING:
                if not job_create.tournament_url:
                    # Scrape all upcoming matches
                    logger.info("[BG TASK] Calling service.scrape_upcoming()")
                    result = service.scrape_upcoming()
                    logger.info(f"[BG TASK] returned: {result.matches_saved} matches")
                else:
                    # Scrape specific tournament
                    result = service.scrape_upcoming(url=str(job_create.tournament_url))
            elif job_create.scraping_type == ScrapingType.RESULTS:
                if not job_create.tournament_url:
                    raise ValueError("Tournament URL is required for results scraping")

                # Check if period range is specified
                if job_create.period_start and job_create.period_end:
                    # Generate seasons from the range
                    seasons = _generate_seasons(
                        job_create.period_start, job_create.period_end
                    )
                    result = service.scrape_results_range(
                        url=str(job_create.tournament_url),
                        seasons=seasons,
                        max_pages=max_pages,
                        target_team_id=job_create.team_id,
                    )
                else:
                    # No period range, just scrape the provided URL
                    result = service.scrape_results(
                        url=str(job_create.tournament_url),
                        max_pages=max_pages,
                        target_team_id=job_create.team_id,
                        season=job_create.period,
                    )
            elif job_create.scraping_type == ScrapingType.BY_DATE:
                target_date = (
                    date.fromisoformat(job_create.period)
                    if job_create.period
                    else datetime.now(timezone.utc).date()
                )
                result = service.scrape_matches_by_date(
                    url=str(job_create.tournament_url)
                    if job_create.tournament_url
                    else None,
                    target_date=target_date,
                )
            else:
                raise ValueError(
                    f"Unsupported scraping type: {job_create.scraping_type}"
                )

        logger.info(f"[BG TASK] Session closed, result: {result}")

        # Update job status to completed
        matches_saved = result.matches_saved if result else 0
        matches_scraped = result.matches_scraped if result else matches_saved
        job_obj = scraping_jobs.get(job_id)
        started = job_obj.started_at if job_obj else datetime.now(timezone.utc)
        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.COMPLETED,
                progress=100.0,
                message=(
                    f"Scraping completed successfully. "
                    f"{matches_saved} matches processed."
                ),
                matches_scraped=matches_scraped,
                matches_saved=matches_saved,
                errors=[],
                started_at=started,
                completed_at=datetime.now(timezone.utc),
            ),
        )
        logger.info(f"[BG TASK] Job {job_id} completed with {matches_saved} matches")

        # Update the completed_at timestamp directly on the job
        if job_id in scraping_jobs:
            scraping_jobs[job_id].completed_at = datetime.now(timezone.utc)

    except Exception as e:
        import traceback

        logger.error(f"[BG TASK] Job {job_id} failed: {e}")
        logger.error(traceback.format_exc())

        # Update job status to failed
        current_job = scraping_jobs.get(job_id)
        errors = current_job.errors if current_job else []
        errors.append(str(e))
        job_obj_failed = scraping_jobs.get(job_id)
        started_failed = (
            job_obj_failed.started_at if job_obj_failed else datetime.now(timezone.utc)
        )

        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.FAILED,
                progress=0.0,  # or current progress if available
                message=f"Scraping failed: {str(e)}",
                matches_scraped=0,  # or current count if available
                matches_saved=0,
                errors=errors,
                started_at=started_failed,
                completed_at=datetime.now(timezone.utc),
            ),
        )

        # Update the completed_at timestamp directly on the job
        if job_id in scraping_jobs:
            scraping_jobs[job_id].completed_at = datetime.now(timezone.utc)


@router.post("/upcoming", response_model=ScrapingJobResponse)
async def scrape_upcoming(
    background_tasks: BackgroundTasks,
    request: UpcomingScrapeRequest | None = Body(default=None),
    tournament_url: str | None = None,
    tournament_id: int | None = None,
    scope: ScrapeScope | None = None,
    db: Session = Depends(get_db),
) -> ScrapingJobResponse:
    """Start scraping upcoming matches.

    Args:
        tournament_url: Optional URL of specific tournament to scrape
        db: Database session

    Returns:
        Scraping job response with job details

    Raises:
        HTTPException: If scraping job cannot be created
    """
    try:
        request_data = request.model_dump(exclude_unset=True) if request else {}
        if tournament_url is not None:
            request_data["tournament_url"] = tournament_url
        if tournament_id is not None:
            request_data["tournament_id"] = tournament_id
        if scope is not None:
            request_data["scope"] = scope
        resolved_request = UpcomingScrapeRequest(**request_data)
        tournament = _lookup_tournament(
            db, resolved_request.tournament_id, resolved_request.tournament_url
        )
        resolved_url = (
            str(resolved_request.tournament_url)
            if resolved_request.tournament_url
            else _base_tournament_url(tournament)
            if tournament and resolved_request.scope == ScrapeScope.LEAGUE
            else None
        )

        # Create scraping job
        job_create = _job_create_from_tournament(
            ScrapingType.UPCOMING,
            tournament,
            resolved_url,
            scope=resolved_request.scope,
            team_id=resolved_request.team_id,
        )

        job_id = str(uuid.uuid4())
        job = ScrapingJobResponse(
            id=job_id,
            status=ScrapingJobStatus.PENDING,
            progress=0.0,
            message="Job created and queued",
            created_at=datetime.now(timezone.utc),
            **job_create.model_dump(),
        )

        # Store job
        scraping_jobs[job_id] = job

        # Add to background tasks
        background_tasks.add_task(run_scraping_job, job_id, job_create)

        return job

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scraping job: {str(e)}",
        ) from e


@router.post("/results", response_model=ScrapingJobResponse)
async def scrape_results(
    background_tasks: BackgroundTasks,
    request: ResultsScrapeRequest | None = Body(default=None),
    tournament_url: str | None = None,
    tournament_id: int | None = None,
    period: str | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
    max_pages: int | None = None,
    season: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    db: Session = Depends(get_db),
) -> ScrapingJobResponse:
    """Start scraping match results.

    Args:
        tournament_url: URL of tournament to scrape results from
        season: Optional season to scrape (e.g., '2023-2024')
        start_date: Optional start date for results
        end_date: Optional end date for results
        db: Database session

    Returns:
        Scraping job response with job details

    Raises:
        HTTPException: If scraping job cannot be created
    """
    try:
        request_data = request.model_dump(exclude_unset=True) if request else {}
        if tournament_url is not None:
            request_data["tournament_url"] = tournament_url
        if tournament_id is not None:
            request_data["tournament_id"] = tournament_id
        if period is not None:
            request_data["period"] = period
        elif season is not None:
            request_data["period"] = season
        if max_pages is not None:
            request_data["max_pages"] = max_pages
        if period_start is not None:
            request_data["period_start"] = period_start
        if period_end is not None:
            request_data["period_end"] = period_end
        resolved_request = ResultsScrapeRequest(**request_data)
        tournament = _lookup_tournament(
            db, resolved_request.tournament_id, resolved_request.tournament_url
        )
        resolved_period = _normalize_period(resolved_request.period)
        resolved_season = (
            _lookup_season(db, tournament.id, resolved_period) if tournament else None
        )
        resolved_url = (
            str(resolved_request.tournament_url)
            if resolved_request.tournament_url
            else _build_results_url(tournament, resolved_period, resolved_season)
            if tournament
            else None
        )

        # Create scraping job
        job_create = _job_create_from_tournament(
            ScrapingType.RESULTS,
            tournament,
            resolved_url,
            scope=ScrapeScope.LEAGUE if tournament else ScrapeScope.ALL,
            season=resolved_period,
            period=resolved_period,
            team_id=resolved_request.team_id,
        )
        job_create.start_date = start_date
        job_create.end_date = end_date
        job_create.period_start = resolved_request.period_start
        job_create.period_end = resolved_request.period_end

        job_id = str(uuid.uuid4())
        job = ScrapingJobResponse(
            id=job_id,
            status=ScrapingJobStatus.PENDING,
            progress=0.0,
            message="Job created and queued",
            created_at=datetime.now(timezone.utc),
            **job_create.model_dump(),
        )

        # Store job
        scraping_jobs[job_id] = job

        # Add to background tasks
        background_tasks.add_task(
            run_scraping_job,
            job_id,
            job_create,
            max_pages=resolved_request.max_pages,
        )

        return job

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scraping job: {str(e)}",
        ) from e


@router.post("/by-date", response_model=ScrapingJobResponse)
async def scrape_by_date(
    background_tasks: BackgroundTasks,
    request: ByDateScrapeRequest | None = Body(default=None),
    date: str | None = None,
    tournament_url: str | None = None,
    tournament_id: int | None = None,
    scope: ScrapeScope | None = None,
    db: Session = Depends(get_db),
) -> ScrapingJobResponse:
    """Start scraping all matches for a specific date.

    This scrapes the main OddsPortal page for all matches on the given date,
    which is equivalent to scraping all leagues at once.

    Args:
        date: Date in YYYY-MM-DD format (defaults to today)
        tournament_url: Optional URL override
        db: Database session

    Returns:
        Scraping job response with job details

    Raises:
        HTTPException: If scraping job cannot be created
    """
    try:
        request_data = request.model_dump(exclude_unset=True) if request else {}
        if date is not None:
            request_data["date"] = date
        if tournament_url is not None:
            request_data["tournament_url"] = tournament_url
        if tournament_id is not None:
            request_data["tournament_id"] = tournament_id
        if scope is not None:
            request_data["scope"] = scope
        resolved_request = ByDateScrapeRequest(**request_data)
        tournament = _lookup_tournament(
            db, resolved_request.tournament_id, resolved_request.tournament_url
        )
        target_date = resolved_request.date or datetime.now(timezone.utc).date()
        resolved_url = (
            str(resolved_request.tournament_url)
            if resolved_request.tournament_url
            else _base_tournament_url(tournament)
            if tournament and resolved_request.scope == ScrapeScope.LEAGUE
            else _build_matches_url(target_date)
        )

        job_create = _job_create_from_tournament(
            ScrapingType.BY_DATE,
            tournament,
            resolved_url,
            scope=resolved_request.scope,
            request_date=target_date,
            team_id=resolved_request.team_id,
        )

        job_id = str(uuid.uuid4())
        job = ScrapingJobResponse(
            id=job_id,
            status=ScrapingJobStatus.PENDING,
            progress=0.0,
            message=(f"Fetching matches for {target_date.isoformat()}"),
            created_at=datetime.now(timezone.utc),
            **job_create.model_dump(),
        )

        # Store job
        scraping_jobs[job_id] = job

        # Add to background tasks
        background_tasks.add_task(run_scraping_job, job_id, job_create)

        return job

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scraping job: {str(e)}",
        ) from e


@router.get("/jobs", response_model=PaginatedResponse[ScrapingJobResponse])
async def list_jobs(
    status_filter: ScrapingJobStatus | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> PaginatedResponse[ScrapingJobResponse]:
    """List scraping jobs with optional filtering.

    Args:
        status_filter: Optional status filter
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        db: Database session

    Returns:
        List of scraping jobs with pagination
    """
    # Filter jobs by status
    jobs = list(scraping_jobs.values())
    if status_filter:
        jobs = [job for job in jobs if job.status == status_filter]

    # Sort by creation date (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)

    # Apply pagination
    total = len(jobs)
    paginated_jobs = jobs[offset : offset + limit]

    return PaginatedResponse(
        items=paginated_jobs,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/jobs/{job_id}", response_model=ScrapingJobResponse)
async def get_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> ScrapingJobResponse:
    """Get scraping job by ID.

    Args:
        job_id: Unique job identifier
        db: Database session

    Returns:
        Scraping job details

    Raises:
        HTTPException: If job not found
    """
    if job_id not in scraping_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scraping job {job_id} not found",
        )

    return scraping_jobs[job_id]


@router.get("/stats", response_model=ScrapingStats)
async def get_stats(db: Session = Depends(get_db)) -> ScrapingStats:
    """Get scraping statistics.

    Args:
        db: Database session

    Returns:
        Scraping statistics
    """
    jobs = list(scraping_jobs.values())

    if not jobs:
        return ScrapingStats(
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            running_jobs=0,
            total_matches_scraped=0,
            average_duration_seconds=None,
            success_rate=0.0,
        )
    completed_jobs = [job for job in jobs if job.status == ScrapingJobStatus.COMPLETED]
    failed_jobs = [job for job in jobs if job.status == ScrapingJobStatus.FAILED]
    running_jobs = [job for job in jobs if job.status == ScrapingJobStatus.RUNNING]

    total_matches = sum(job.matches_scraped for job in jobs)

    # Calculate average duration
    durations = []
    for job in completed_jobs:
        started_at = _ensure_utc_timestamp(job.started_at)
        completed_at = _ensure_utc_timestamp(job.completed_at)
        if started_at and completed_at:
            duration = (completed_at - started_at).total_seconds()
            durations.append(duration)

    avg_duration = sum(durations) / len(durations) if durations else None

    # Calculate success rate
    completed_count = len(completed_jobs)
    failed_count = len(failed_jobs)
    total_completed = completed_count + failed_count
    success_rate = (
        (completed_count / total_completed * 100) if total_completed > 0 else 0.0
    )

    return ScrapingStats(
        total_jobs=len(jobs),
        completed_jobs=completed_count,
        failed_jobs=failed_count,
        running_jobs=len(running_jobs),
        total_matches_scraped=total_matches,
        average_duration_seconds=avg_duration,
        success_rate=success_rate,
    )


# ---------------------------------------------------------------------------
# Soccerdata stats enrichment (Understat + ESPN)
# ---------------------------------------------------------------------------


@router.post("/import/enrich-stats")
async def enrich_match_stats(
    background_tasks: BackgroundTasks,
    league: str = "ENG-Premier League",
    season: str = "2024",
) -> dict[str, Any]:
    """Enrich existing matches with Understat xG and ESPN player stats."""
    if league not in FBREF_LEAGUE_MAPPING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid league: {league}",
        )

    job_id = str(uuid.uuid4())

    job = ScrapingJobResponse(
        id=job_id,
        status=ScrapingJobStatus.PENDING,
        progress=0.0,
        message=(
            f"Enrichment queued for {FBREF_LEAGUE_MAPPING[league]['name']} {season}"
        ),
        created_at=datetime.now(timezone.utc),
        scraping_type="import",
        tournament_name=FBREF_LEAGUE_MAPPING[league]["name"],
        period=season,
    )

    scraping_jobs[job_id] = job
    background_tasks.add_task(run_enrich_stats, job_id, league, season)

    return {
        "job_id": job_id,
        "message": (
            f"Enrichment started for {FBREF_LEAGUE_MAPPING[league]['name']} {season}"
        ),
    }


def run_enrich_stats(job_id: str, league: str, season: str) -> None:
    """Execute stats enrichment as a background task."""
    import logging

    logger = logging.getLogger("algobet.api.enrich_stats")

    try:
        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.RUNNING,
                progress=10.0,
                message=(
                    f"Enriching {FBREF_LEAGUE_MAPPING[league]['name']} {season}..."
                ),
                started_at=datetime.now(timezone.utc),
            ),
        )

        from algobet.infrastructure.database import session_scope

        with session_scope() as session:
            importer = SoccerDataImporter(session)
            enr = importer.enrich_all(league=league, season=season)

            logger.info(
                "[BG TASK] Enrichment: %d xG, %d players in %d matches",
                enr["understat_enriched"],
                enr["players_added"],
                enr["matches_processed"],
            )

        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.COMPLETED,
                progress=100.0,
                message=(
                    f"Enriched {enr['understat_enriched']} matches with xG "
                    f"and {enr['players_added']} player records"
                ),
                matches_scraped=enr["understat_enriched"],
                matches_saved=enr["players_added"],
                errors=[],
                started_at=(
                    scraping_jobs[job_id].started_at
                    if job_id in scraping_jobs
                    else datetime.now(timezone.utc)
                ),
                completed_at=datetime.now(timezone.utc),
            ),
        )

    except Exception as e:
        import traceback

        logger.error("[BG TASK] Enrichment %s failed: %s", job_id, e)
        logger.error(traceback.format_exc())

        current_job = scraping_jobs.get(job_id)
        start_ts = current_job.started_at if current_job else datetime.now(timezone.utc)

        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.FAILED,
                progress=0.0,
                message=f"Enrichment failed: {str(e)}",
                matches_scraped=0,
                matches_saved=0,
                errors=[str(e)],
                started_at=start_ts,
                completed_at=datetime.now(timezone.utc),
            ),
        )
