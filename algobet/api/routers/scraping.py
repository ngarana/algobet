"""API router for scraping operations with background task support."""

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import HttpUrl
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import PaginatedResponse
from algobet.api.schemas.scraping import (
    ScrapingJobCreate,
    ScrapingJobResponse,
    ScrapingJobStatus,
    ScrapingJobUpdate,
    ScrapingProgress,
    ScrapingStats,
    ScrapingType,
)
from algobet.api.websockets import manager
from algobet.services.scraping_service import (
    ScrapingProgress as ServiceScrapingProgress,
    ScrapingService,
)

router = APIRouter(tags=["scraping"])

# In-memory storage for scraping jobs (replace with Redis/database in production)
scraping_jobs: dict[str, ScrapingJobResponse] = {}


def update_job_status(job_id: str, update: ScrapingJobUpdate) -> None:
    """Update job status in storage."""
    if job_id in scraping_jobs:
        job = scraping_jobs[job_id]
        old_status = job.status
        job_data = job.model_dump()
        for field, value in update.model_dump(exclude_unset=True).items():
            job_data[field] = value
        scraping_jobs[job_id] = ScrapingJobResponse(**job_data)

        # Broadcast status change via WebSocket
        new_status = job_data.get("status", old_status)
        if new_status != old_status:
            asyncio.create_task(
                manager.broadcast_job_status(
                    job_id, new_status, job_data.get("message", "")
                )
            )


async def run_scraping_job(
    job_id: str,
    job_create: ScrapingJobCreate,
    db: Session,
) -> None:
    """Execute scraping job with progress updates using API-Football."""
    try:
        started_at = datetime.now(timezone.utc)

        # Update job status to running
        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.RUNNING,
                progress=0.0,
                message="Starting API-Football fetch...",
                matches_scraped=0,
                errors=[],
                started_at=started_at,
            ),
        )

        # Define progress callback
        def progress_callback(progress: ServiceScrapingProgress) -> None:
            job_status = ScrapingJobStatus(progress.status.value)
            completed_at = (
                progress.completed_at.replace(tzinfo=timezone.utc)
                if progress.completed_at and progress.completed_at.tzinfo is None
                else progress.completed_at
            )
            service_started_at = (
                progress.started_at.replace(tzinfo=timezone.utc)
                if progress.started_at and progress.started_at.tzinfo is None
                else progress.started_at
            )
            error_list = [progress.error] if progress.error else []

            update_job_status(
                job_id,
                ScrapingJobUpdate(
                    status=job_status,
                    progress=progress.progress,
                    message=progress.message,
                    matches_scraped=progress.matches_scraped,
                    errors=error_list,
                    started_at=service_started_at,
                    completed_at=completed_at,
                ),
            )

            # Broadcast progress via WebSocket
            progress_data = ScrapingProgress(
                job_id=job_id,
                status=job_status,
                progress=progress.progress,
                message=progress.message,
                matches_scraped=progress.matches_scraped,
                matches_saved=progress.matches_saved,
                current_page=progress.current_page or None,
                total_pages=progress.total_pages or None,
                started_at=service_started_at,
                completed_at=completed_at,
                error=progress.error,
            )
            asyncio.create_task(manager.broadcast_progress(progress_data))

        # Initialize scraping service with database session
        service = ScrapingService(db, progress_callback=progress_callback)

        # Determine league IDs to use
        league_ids = job_create.league_ids
        if job_create.league_id and not league_ids:
            league_ids = [job_create.league_id]

        # Execute scraping based on type
        if job_create.scraping_type == ScrapingType.UPCOMING:
            result = service.scrape_upcoming(league_ids=league_ids)
        elif job_create.scraping_type == ScrapingType.RESULTS:
            result = service.scrape_results(
                league_id=job_create.league_id,
                max_pages=job_create.max_results,
            )
        else:
            raise ValueError(f"Unsupported scraping type: {job_create.scraping_type}")

        # Update job status to completed
        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.COMPLETED,
                progress=100.0,
                message=result.message,
                matches_scraped=result.matches_saved,
                errors=[],
                completed_at=datetime.now(timezone.utc),
            ),
        )

    except Exception as e:
        # Update job status to failed
        current_job = scraping_jobs.get(job_id)
        errors = current_job.errors if current_job else []
        errors.append(str(e))

        update_job_status(
            job_id,
            ScrapingJobUpdate(
                status=ScrapingJobStatus.FAILED,
                progress=0.0,
                message=f"Scraping failed: {str(e)}",
                matches_scraped=0,
                errors=errors,
                completed_at=datetime.now(timezone.utc),
            ),
        )


@router.post("/upcoming", response_model=ScrapingJobResponse)
async def scrape_upcoming(
    background_tasks: BackgroundTasks,
    tournament_url: str | None = None,
    league_ids: str | None = None,
    db: Session = Depends(get_db),
) -> ScrapingJobResponse:
    """Start scraping upcoming matches using API-Football.

    Args:
        tournament_url: [Legacy] Optional URL (ignored - API-Football uses league IDs)
        league_ids: Comma-separated API-Football league IDs (e.g., "39,140,135")
        db: Database session

    Returns:
        Scraping job response with job details

    Raises:
        HTTPException: If scraping job cannot be created
    """
    try:
        # Parse league IDs from string
        parsed_league_ids: list[int] | None = None
        if league_ids:
            try:
                parsed_league_ids = [int(lid.strip()) for lid in league_ids.split(",")]
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid league_ids format. Expected comma-separated integers.",
                )

        # Create scraping job
        job_create = ScrapingJobCreate(
            scraping_type=ScrapingType.UPCOMING,
            tournament_url=HttpUrl(tournament_url) if tournament_url else None,
            tournament_name=None,
            season=None,
            start_date=None,
            end_date=None,
            league_ids=parsed_league_ids,
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
        background_tasks.add_task(run_scraping_job, job_id, job_create, db)

        return job

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scraping job: {str(e)}",
        ) from e


@router.post("/results", response_model=ScrapingJobResponse)
async def scrape_results(
    background_tasks: BackgroundTasks,
    tournament_url: str | None = None,
    league_id: int | None = None,
    max_results: int | None = None,
    db: Session = Depends(get_db),
) -> ScrapingJobResponse:
    """Start scraping match results using API-Football.

    Args:
        tournament_url: [Legacy] Optional URL (ignored - API-Football uses league ID)
        league_id: API-Football league ID (e.g., 39 for Premier League)
        max_results: Maximum number of results to fetch
        db: Database session

    Returns:
        Scraping job response with job details

    Raises:
        HTTPException: If scraping job cannot be created
    """
    try:
        # Create scraping job
        job_create = ScrapingJobCreate(
            scraping_type=ScrapingType.RESULTS,
            tournament_url=HttpUrl(tournament_url) if tournament_url else None,
            tournament_name=None,
            season=None,
            start_date=None,
            end_date=None,
            league_id=league_id,
            max_results=max_results or 20,
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
        background_tasks.add_task(run_scraping_job, job_id, job_create, db)

        return job

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
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()
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
