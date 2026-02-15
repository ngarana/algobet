# AlgoBet Architecture Refactoring Plan

**Created**: 2026-02-05
**Status**: ✅ Completed (2026-02-14)
**Author**: AI Assistant

> **Note**: This refactoring has been completed successfully. All phases described in this document are now implemented. See [refactor-todo.md](../refactor-todo.md) for detailed completion status.

---

## Executive Summary

This document outlines a comprehensive refactoring plan to unify the API, scraping, and prediction functionalities in AlgoBet. The primary goals are:

1. **Eliminate code duplication** between CLI and API
2. **Move from CLI-based scraping to frontend-controlled operations**
3. **Introduce a service layer** for reusable business logic
4. **Improve testability and maintainability**
5. **Enable real-time progress feedback** for long-running operations

---

## Current Architecture Analysis

### File Structure (As-Is)

```
algobet/
├── api/                         # FastAPI REST API
│   ├── routers/                 # 8 route modules
│   │   ├── matches.py          # Match CRUD (13KB)
│   │   ├── predictions.py      # Prediction endpoints (12KB)
│   │   ├── models.py           # Model management (6KB)
│   │   ├── value_bets.py       # Value bet calculations (7KB)
│   │   ├── seasons.py          # Season CRUD (4KB)
│   │   ├── teams.py            # Team CRUD (5KB)
│   │   └── tournaments.py      # Tournament CRUD (2KB)
│   ├── schemas/                 # Pydantic schemas
│   ├── dependencies.py          # DB session dependency
│   └── main.py                  # FastAPI app setup
├── predictions/                 # ML prediction engine
│   ├── data/queries.py         # MatchRepository
│   ├── features/form_features.py # FormCalculator
│   └── models/registry.py      # ModelRegistry
├── cli.py                       # Click CLI - scraping commands (18KB)
├── predictions_cli.py           # Click CLI - prediction commands (15KB)
├── scraper.py                   # OddsPortalScraper (23KB)
├── database.py                  # DB connection management
└── models.py                    # SQLAlchemy ORM models
```

### Identified Problems

| Problem | Impact | Severity |
|---------|--------|----------|
| **Duplicated prediction logic** | `_generate_features()`, `_get_prediction()` exist in both `predictions_cli.py` and `api/routers/predictions.py` | High |
| **CLI-only scraping** | Users must SSH into server to trigger scrapes; no frontend control | High |
| **Mixed concerns** | Business logic embedded in Click decorators and FastAPI routes | Medium |
| **Inconsistent patterns** | CLI uses `session_scope()`, API uses `Depends(get_db)` differently | Medium |
| **No progress feedback** | Long scraping operations provide no real-time status | Medium |
| **Hard to test** | Business logic tightly coupled to transport layer | High |

### Code Duplication Example

**In `predictions_cli.py` (lines 58-76):**
```python
def _load_model(registry: ModelRegistry, model_version: str | None = None):
    if model_version:
        model = registry.load_model(model_version)
        return model, model_version
    else:
        model, metadata = registry.get_active_model()
        return model, metadata.version
```

**In `api/routers/predictions.py` (similar logic exists):**
```python
# Model loading logic is reimplemented in the generate_predictions endpoint
```

---

## Target Architecture (To-Be)

### Guiding Principles

1. **Service-Oriented Architecture**: All business logic lives in services
2. **API-First Design**: Frontend interacts exclusively through REST/WebSocket API
3. **CLI as Development Tool**: CLI becomes optional, used only for development/debugging
4. **Real-Time Communication**: WebSockets for progress updates on long operations
5. **Background Processing**: Celery/ARQ for async task processing (optional future enhancement)

### Target File Structure

```
algobet/
├── api/
│   ├── routers/                 # Thin wrappers around services
│   │   ├── matches.py
│   │   ├── predictions.py
│   │   ├── models.py
│   │   ├── scraping.py          # NEW: Scraping control endpoints
│   │   ├── jobs.py              # NEW: Background job status
│   │   └── ...
│   ├── schemas/
│   │   ├── scraping.py          # NEW: Scraping request/response schemas
│   │   ├── job.py               # NEW: Job status schemas
│   │   └── ...
│   ├── websockets/              # NEW: WebSocket handlers
│   │   ├── __init__.py
│   │   └── progress.py          # Real-time progress updates
│   ├── dependencies.py
│   └── main.py
├── services/                     # NEW: Business logic layer
│   ├── __init__.py
│   ├── base.py                  # Base service class with common patterns
│   ├── prediction_service.py   # Prediction generation logic
│   ├── scraping_service.py     # Scraping orchestration
│   ├── match_service.py        # Match CRUD + business logic
│   ├── model_service.py        # Model lifecycle management
│   ├── value_bet_service.py    # Value bet calculations
│   └── job_service.py          # Background job management
├── scraper/                      # Refactored scraping module
│   ├── __init__.py
│   ├── base.py                  # Abstract base scraper
│   ├── oddsportal.py           # OddsPortal implementation
│   └── parsers.py              # Extracted parsing logic
├── predictions/                  # ML prediction engine (unchanged)
│   ├── data/queries.py
│   ├── features/form_features.py
│   └── models/registry.py
├── tasks/                        # NEW: Background task definitions
│   ├── __init__.py
│   ├── scraping_tasks.py
│   └── prediction_tasks.py
├── cli/                          # DEPRECATED: Moved to optional dev tools
│   ├── __init__.py
│   └── dev_tools.py             # Database seeding, debugging utilities
├── database.py
└── models.py
```

---

## Phase 1: Service Layer Foundation

**Duration**: 1-2 days
**Priority**: High

### 1.1 Create Base Service Class

**File**: `algobet/services/base.py`

```python
"""Base service class with common patterns."""

from abc import ABC
from typing import TypeVar, Generic

from sqlalchemy.orm import Session

T = TypeVar("T")


class BaseService(ABC, Generic[T]):
    """Base class for all services.

    Provides common functionality like session management and logging.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def commit(self) -> None:
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.session.rollback()
```

### 1.2 Create PredictionService

**File**: `algobet/services/prediction_service.py`

Extract and consolidate from:
- `predictions_cli.py`: lines 58-174 (helper functions)
- `api/routers/predictions.py`: lines 114-229 (generate_predictions)

```python
"""Unified prediction service for generating match predictions."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import Match, Tournament, Prediction, ModelVersion
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.form_features import FormCalculator
from algobet.predictions.models.registry import ModelRegistry
from algobet.services.base import BaseService


@dataclass
class PredictionResult:
    """Result of a match prediction."""

    match_id: int
    match_date: datetime
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    model_version: str
    prob_home: float
    prob_draw: float
    prob_away: float


class PredictionService(BaseService):
    """Service for generating and managing predictions."""

    def __init__(
        self,
        session: Session,
        models_path: Path = Path("data/models")
    ) -> None:
        super().__init__(session)
        self.registry = ModelRegistry(storage_path=models_path, session=session)
        self.repo = MatchRepository(session)
        self.calc = FormCalculator(self.repo)

    def load_model(
        self,
        model_version: str | None = None
    ) -> tuple[Any, str]:
        """Load model from registry.

        Args:
            model_version: Optional specific version ID

        Returns:
            Tuple of (model object, version string)
        """
        if model_version:
            model = self.registry.load_model(model_version)
            return model, model_version
        else:
            model, metadata = self.registry.get_active_model()
            return model, metadata.version

    def generate_features(self, match: Match) -> dict[str, float]:
        """Generate ML features for a match.

        Args:
            match: Match object to generate features for

        Returns:
            Dictionary of feature name to value
        """
        return {
            "home_form": self.calc.calculate_recent_form(
                team_id=match.home_team_id,
                reference_date=match.match_date,
                n_matches=5
            ),
            "away_form": self.calc.calculate_recent_form(
                team_id=match.away_team_id,
                reference_date=match.match_date,
                n_matches=5
            ),
            "home_goals_scored": self.calc.calculate_goals_scored(
                team_id=match.home_team_id,
                reference_date=match.match_date,
                n_matches=5
            ),
            "away_goals_scored": self.calc.calculate_goals_scored(
                team_id=match.away_team_id,
                reference_date=match.match_date,
                n_matches=5
            ),
            "home_goals_conceded": self.calc.calculate_goals_conceded(
                team_id=match.home_team_id,
                reference_date=match.match_date,
                n_matches=5
            ),
            "away_goals_conceded": self.calc.calculate_goals_conceded(
                team_id=match.away_team_id,
                reference_date=match.match_date,
                n_matches=5
            ),
        }

    def get_prediction(
        self,
        model: Any,
        features: dict[str, float]
    ) -> tuple[str, float, dict[str, float]]:
        """Get prediction from model.

        Args:
            model: Loaded model object
            features: Feature dictionary

        Returns:
            Tuple of (predicted_outcome, confidence, probabilities)
        """
        feature_array = np.array([list(features.values())])

        try:
            probs = model.predict_proba(feature_array)[0]
        except AttributeError:
            probs = model.predict(feature_array)[0]

        outcomes = ["HOME", "DRAW", "AWAY"]
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        probabilities = {
            "home": float(probs[0]),
            "draw": float(probs[1]),
            "away": float(probs[2]),
        }

        return outcomes[max_idx], confidence, probabilities

    def predict_match(
        self,
        match: Match,
        model_version: str | None = None,
    ) -> PredictionResult:
        """Generate prediction for a single match.

        Args:
            match: Match object to predict
            model_version: Optional specific model version

        Returns:
            PredictionResult with prediction details
        """
        model, version = self.load_model(model_version)
        features = self.generate_features(match)
        outcome, confidence, probs = self.get_prediction(model, features)

        return PredictionResult(
            match_id=match.id,
            match_date=match.match_date,
            home_team=match.home_team.name,
            away_team=match.away_team.name,
            predicted_outcome=outcome,
            confidence=confidence,
            model_version=version,
            prob_home=probs["home"],
            prob_draw=probs["draw"],
            prob_away=probs["away"],
        )

    def query_matches(
        self,
        match_ids: list[int] | None = None,
        tournament_name: str | None = None,
        days_ahead: int = 7,
        status: str = "SCHEDULED",
    ) -> list[Match]:
        """Query matches based on filters.

        Args:
            match_ids: Optional list of specific match IDs
            tournament_name: Optional tournament name filter
            days_ahead: Number of days ahead to look
            status: Match status filter

        Returns:
            List of Match objects
        """
        if match_ids:
            stmt = select(Match).where(Match.id.in_(match_ids))
        else:
            stmt = select(Match).where(Match.status == status)
            max_date = datetime.now() + timedelta(days=days_ahead)
            stmt = stmt.where(Match.match_date <= max_date)

            if tournament_name:
                stmt = stmt.join(Tournament).where(Tournament.name == tournament_name)

        stmt = stmt.order_by(Match.match_date)
        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def predict_upcoming(
        self,
        days_ahead: int = 7,
        tournament_name: str | None = None,
        min_confidence: float = 0.0,
        model_version: str | None = None,
    ) -> list[PredictionResult]:
        """Generate predictions for upcoming matches.

        Args:
            days_ahead: Number of days ahead to look for matches
            tournament_name: Optional tournament filter
            min_confidence: Minimum confidence threshold
            model_version: Optional specific model version

        Returns:
            List of PredictionResult objects
        """
        matches = self.query_matches(
            tournament_name=tournament_name,
            days_ahead=days_ahead,
            status="SCHEDULED",
        )

        predictions = []
        for match in matches:
            result = self.predict_match(match, model_version)
            if result.confidence >= min_confidence:
                predictions.append(result)

        return predictions

    def save_predictions(
        self,
        predictions: list[PredictionResult]
    ) -> list[Prediction]:
        """Save predictions to database.

        Args:
            predictions: List of PredictionResult to save

        Returns:
            List of created Prediction ORM objects
        """
        # Get model version ID
        model_version = self.session.execute(
            select(ModelVersion).where(
                ModelVersion.version == predictions[0].model_version
            )
        ).scalar_one()

        db_predictions = []
        for pred in predictions:
            db_pred = Prediction(
                match_id=pred.match_id,
                model_version_id=model_version.id,
                prob_home=pred.prob_home,
                prob_draw=pred.prob_draw,
                prob_away=pred.prob_away,
                predicted_outcome=pred.predicted_outcome[0],  # H, D, or A
                confidence=pred.confidence,
            )
            self.session.add(db_pred)
            db_predictions.append(db_pred)

        return db_predictions
```

### 1.3 Create ScrapingService

**File**: `algobet/services/scraping_service.py`

```python
"""Scraping service for orchestrating data collection from OddsPortal."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import Match, Season, Team, Tournament
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


class ScrapingService(BaseService):
    """Service for managing scraping operations."""

    # In-memory job storage (replace with Redis/DB for production)
    _jobs: dict[UUID, ScrapingJob] = {}

    def __init__(
        self,
        session: Session,
        progress_callback: Callable[[ScrapingProgress], None] | None = None,
    ) -> None:
        super().__init__(session)
        self.progress_callback = progress_callback

    def _emit_progress(self, progress: ScrapingProgress) -> None:
        """Emit progress update to callback if registered."""
        if self.progress_callback:
            self.progress_callback(progress)

    def create_job(
        self,
        job_type: str,
        url: str
    ) -> ScrapingJob:
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
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None
    ) -> list[ScrapingJob]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def get_or_create_tournament(
        self,
        country: str,
        name: str,
        slug: str
    ) -> Tournament:
        """Get or create a tournament."""
        tournament = self.session.execute(
            select(Tournament).where(Tournament.url_slug == slug)
        ).scalar_one_or_none()

        if not tournament:
            tournament = Tournament(name=name, country=country, url_slug=slug)
            self.session.add(tournament)
            self.session.flush()

        return tournament

    def get_or_create_team(self, name: str) -> Team:
        """Get or create a team."""
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
            started_at=datetime.now(),
            message="Starting upcoming matches scrape...",
        )
        self._emit_progress(progress)

        try:
            with OddsPortalScraper(headless=headless) as scraper:
                scraper.start()
                scraper.navigate_to_upcoming(url)

                progress.message = "Scraping upcoming matches..."
                self._emit_progress(progress)

                matches_data = scraper.scrape_upcoming_matches()
                progress.matches_scraped = len(matches_data)

                # Save matches
                saved_count = self._save_upcoming_matches(matches_data)
                progress.matches_saved = saved_count

                progress.status = JobStatus.COMPLETED
                progress.completed_at = datetime.now()
                progress.message = f"Completed! Scraped {len(matches_data)} matches, saved {saved_count}."

        except Exception as e:
            progress.status = JobStatus.FAILED
            progress.error = str(e)
            progress.message = f"Failed: {e}"
            progress.completed_at = datetime.now()

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
            started_at=datetime.now(),
            message="Starting results scrape...",
        )
        self._emit_progress(progress)

        try:
            with OddsPortalScraper(headless=headless) as scraper:
                scraper.start()
                scraper.navigate_to_results(url)

                # Get total pages
                total_pages = scraper.get_page_count()
                if max_pages:
                    total_pages = min(total_pages, max_pages)
                progress.total_pages = total_pages

                # Parse league info from URL
                country, league_name, slug = self._parse_league_info(url)
                tournament = self.get_or_create_tournament(country, league_name, slug)

                # Scrape each page
                all_matches = []
                for page_num in range(1, total_pages + 1):
                    progress.current_page = page_num
                    progress.message = f"Scraping page {page_num}/{total_pages}..."
                    self._emit_progress(progress)

                    if page_num > 1:
                        scraper.go_to_page(page_num)

                    matches = scraper.scrape_current_page()
                    all_matches.extend(matches)
                    progress.matches_scraped = len(all_matches)

                # Save all matches
                saved_count = self._save_result_matches(all_matches, tournament)
                progress.matches_saved = saved_count

                progress.status = JobStatus.COMPLETED
                progress.completed_at = datetime.now()
                progress.message = f"Completed! Scraped {len(all_matches)} matches from {total_pages} pages, saved {saved_count}."

        except Exception as e:
            progress.status = JobStatus.FAILED
            progress.error = str(e)
            progress.message = f"Failed: {e}"
            progress.completed_at = datetime.now()

        self._emit_progress(progress)
        job.status = progress.status
        job.progress = progress

        return progress

    def _parse_league_info(self, url: str) -> tuple[str, str, str]:
        """Extract country, league name, and slug from URL."""
        import re
        match = re.search(r"/football/([^/]+)/([^/]+?)(?:-\d{4}-\d{4})?/results/", url)
        if not match:
            raise ValueError(f"Cannot parse league info from URL: {url}")

        country = match.group(1).replace("-", " ").title()
        slug = match.group(2)
        league_name = slug.replace("-", " ").title()

        return country, league_name, slug

    def _save_upcoming_matches(self, matches_data: list[dict]) -> int:
        """Save upcoming matches to database."""
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
        self,
        matches: list[ScrapedMatch],
        tournament: Tournament
    ) -> int:
        """Save result matches to database."""
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
```

---

## Phase 2: API Scraping Endpoints

**Duration**: 1-2 days
**Priority**: High

### 2.1 Create Scraping Schemas

**File**: `algobet/api/schemas/scraping.py`

```python
"""Pydantic schemas for scraping API endpoints."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a scraping job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScrapeUpcomingRequest(BaseModel):
    """Request to scrape upcoming matches."""
    url: str = Field(
        default="https://www.oddsportal.com/matches/football/",
        description="URL to scrape upcoming matches from"
    )


class ScrapeResultsRequest(BaseModel):
    """Request to scrape historical results."""
    url: str = Field(
        ...,
        description="OddsPortal results URL",
        examples=["https://www.oddsportal.com/football/england/premier-league/results/"]
    )
    max_pages: int | None = Field(
        default=None,
        description="Maximum pages to scrape (None for all)",
        ge=1,
        le=100
    )


class ScrapingProgressResponse(BaseModel):
    """Response with scraping progress."""
    job_id: UUID
    status: JobStatus
    current_page: int = 0
    total_pages: int = 0
    matches_scraped: int = 0
    matches_saved: int = 0
    message: str = ""
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


class ScrapingJobResponse(BaseModel):
    """Response with job details."""
    id: UUID
    job_type: str
    url: str
    status: JobStatus
    created_at: datetime
    progress: ScrapingProgressResponse | None = None

    class Config:
        from_attributes = True
```

### 2.2 Create Scraping Router

**File**: `algobet/api/routers/scraping.py`

```python
"""API router for scraping operations."""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas.scraping import (
    JobStatus,
    ScrapeResultsRequest,
    ScrapeUpcomingRequest,
    ScrapingJobResponse,
    ScrapingProgressResponse,
)
from algobet.services.scraping_service import ScrapingService

router = APIRouter()


@router.post("/upcoming", response_model=ScrapingJobResponse)
async def scrape_upcoming_matches(
    request: ScrapeUpcomingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start scraping upcoming matches.

    Runs in the background. Use GET /jobs/{job_id} to check progress.
    """
    service = ScrapingService(db)
    job = service.create_job("upcoming", request.url)

    # Run scraping in background
    background_tasks.add_task(
        _run_upcoming_scrape,
        job.id,
        request.url,
    )

    return ScrapingJobResponse(
        id=job.id,
        job_type=job.job_type,
        url=job.url,
        status=job.status,
        created_at=job.created_at,
    )


@router.post("/results", response_model=ScrapingJobResponse)
async def scrape_results(
    request: ScrapeResultsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start scraping historical results.

    Runs in the background. Use GET /jobs/{job_id} to check progress.
    """
    service = ScrapingService(db)
    job = service.create_job("results", request.url)

    background_tasks.add_task(
        _run_results_scrape,
        job.id,
        request.url,
        request.max_pages,
    )

    return ScrapingJobResponse(
        id=job.id,
        job_type=job.job_type,
        url=job.url,
        status=job.status,
        created_at=job.created_at,
    )


@router.get("/jobs", response_model=list[ScrapingJobResponse])
async def list_scraping_jobs(
    status: JobStatus | None = None,
    db: Session = Depends(get_db),
):
    """List all scraping jobs."""
    service = ScrapingService(db)
    jobs = service.list_jobs(status)
    return [
        ScrapingJobResponse(
            id=j.id,
            job_type=j.job_type,
            url=j.url,
            status=j.status,
            created_at=j.created_at,
            progress=ScrapingProgressResponse(
                job_id=j.progress.job_id,
                status=j.progress.status,
                current_page=j.progress.current_page,
                total_pages=j.progress.total_pages,
                matches_scraped=j.progress.matches_scraped,
                matches_saved=j.progress.matches_saved,
                message=j.progress.message,
                error=j.progress.error,
                started_at=j.progress.started_at,
                completed_at=j.progress.completed_at,
            ) if j.progress else None
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=ScrapingJobResponse)
async def get_scraping_job(
    job_id: UUID,
    db: Session = Depends(get_db),
):
    """Get scraping job by ID."""
    service = ScrapingService(db)
    job = service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return ScrapingJobResponse(
        id=job.id,
        job_type=job.job_type,
        url=job.url,
        status=job.status,
        created_at=job.created_at,
        progress=ScrapingProgressResponse(
            job_id=job.progress.job_id,
            status=job.progress.status,
            current_page=job.progress.current_page,
            total_pages=job.progress.total_pages,
            matches_scraped=job.progress.matches_scraped,
            matches_saved=job.progress.matches_saved,
            message=job.progress.message,
            error=job.progress.error,
            started_at=job.progress.started_at,
            completed_at=job.progress.completed_at,
        ) if job.progress else None
    )


# Background task functions
def _run_upcoming_scrape(job_id: UUID, url: str) -> None:
    """Background task to run upcoming matches scrape."""
    from algobet.database import session_scope

    with session_scope() as session:
        service = ScrapingService(session)
        service._jobs[job_id] = service.get_job(job_id) or service._jobs.get(job_id)
        service.scrape_upcoming(url)


def _run_results_scrape(
    job_id: UUID,
    url: str,
    max_pages: int | None
) -> None:
    """Background task to run results scrape."""
    from algobet.database import session_scope

    with session_scope() as session:
        service = ScrapingService(session)
        service._jobs[job_id] = service.get_job(job_id) or service._jobs.get(job_id)
        service.scrape_results(url, max_pages)
```

### 2.3 Register Router in Main App

**Update**: `algobet/api/main.py`

```python
# Add to imports
from algobet.api.routers.scraping import router as scraping_router

# Add router registration
app.include_router(
    scraping_router,
    prefix="/api/v1/scraping",
    tags=["scraping"],
)
```

---

## Phase 3: WebSocket Progress Updates

**Duration**: 1 day
**Priority**: Medium

### 3.1 Create WebSocket Handler

**File**: `algobet/api/websockets/progress.py`

```python
"""WebSocket handler for real-time progress updates."""

from collections.abc import Callable
from typing import Any
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect

from algobet.services.scraping_service import ScrapingProgress


class ConnectionManager:
    """Manages WebSocket connections for progress updates."""

    def __init__(self):
        self.active_connections: dict[UUID, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: UUID) -> None:
        """Connect a WebSocket to receive updates for a job."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: UUID) -> None:
        """Disconnect a WebSocket."""
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: UUID, message: dict[str, Any]) -> None:
        """Broadcast a message to all connections for a job."""
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass  # Connection may have closed


manager = ConnectionManager()


def create_progress_callback(job_id: UUID) -> Callable[[ScrapingProgress], None]:
    """Create a callback function that broadcasts progress updates."""
    import asyncio

    def callback(progress: ScrapingProgress) -> None:
        # Run async broadcast in event loop
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(
                manager.broadcast(
                    job_id,
                    {
                        "job_id": str(job_id),
                        "status": progress.status.value,
                        "current_page": progress.current_page,
                        "total_pages": progress.total_pages,
                        "matches_scraped": progress.matches_scraped,
                        "matches_saved": progress.matches_saved,
                        "message": progress.message,
                        "error": progress.error,
                    }
                )
            )
        except RuntimeError:
            pass  # No event loop running (CLI mode)

    return callback
```

### 3.2 Add WebSocket Route

**Update**: `algobet/api/main.py`

```python
from fastapi import WebSocket, WebSocketDisconnect
from algobet.api.websockets.progress import manager

@app.websocket("/ws/scraping/{job_id}")
async def websocket_scraping_progress(websocket: WebSocket, job_id: UUID):
    """WebSocket endpoint for real-time scraping progress updates."""
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive, wait for disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
```

---

## Phase 4: Frontend Integration

**Duration**: 2-3 days
**Priority**: High

### 4.1 Scraping Dashboard Component

Create a new frontend page/component for managing scraping operations:

```
frontend/
├── app/
│   └── scraping/
│       └── page.tsx              # Scraping dashboard page
├── components/
│   ├── scraping/
│   │   ├── ScrapingJobCard.tsx   # Individual job card
│   │   ├── ScrapingProgress.tsx  # Progress indicator
│   │   ├── ScrapeForm.tsx        # Form to start new scrape
│   │   └── JobHistoryTable.tsx   # Table of past jobs
│   └── ...
└── hooks/
    └── useScrapingProgress.ts    # WebSocket hook for progress
```

### 4.2 API Client Functions

**File**: `frontend/lib/api/scraping.ts`

```typescript
import { API_BASE_URL } from './config';

export interface ScrapeRequest {
  url: string;
  max_pages?: number;
}

export interface ScrapingJob {
  id: string;
  job_type: 'upcoming' | 'results';
  url: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  progress?: {
    current_page: number;
    total_pages: number;
    matches_scraped: number;
    matches_saved: number;
    message: string;
    error?: string;
  };
}

export async function scrapeUpcoming(url?: string): Promise<ScrapingJob> {
  const response = await fetch(`${API_BASE_URL}/api/v1/scraping/upcoming`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: url || 'https://www.oddsportal.com/matches/football/' }),
  });

  if (!response.ok) throw new Error('Failed to start scraping');
  return response.json();
}

export async function scrapeResults(request: ScrapeRequest): Promise<ScrapingJob> {
  const response = await fetch(`${API_BASE_URL}/api/v1/scraping/results`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) throw new Error('Failed to start scraping');
  return response.json();
}

export async function getScrapingJobs(): Promise<ScrapingJob[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/scraping/jobs`);
  if (!response.ok) throw new Error('Failed to fetch jobs');
  return response.json();
}

export async function getScrapingJob(jobId: string): Promise<ScrapingJob> {
  const response = await fetch(`${API_BASE_URL}/api/v1/scraping/jobs/${jobId}`);
  if (!response.ok) throw new Error('Failed to fetch job');
  return response.json();
}
```

### 4.3 WebSocket Progress Hook

**File**: `frontend/hooks/useScrapingProgress.ts`

```typescript
import { useEffect, useState, useCallback } from 'react';
import { WS_BASE_URL } from '@/lib/api/config';

interface ScrapingProgress {
  job_id: string;
  status: string;
  current_page: number;
  total_pages: number;
  matches_scraped: number;
  matches_saved: number;
  message: string;
  error?: string;
}

export function useScrapingProgress(jobId: string | null) {
  const [progress, setProgress] = useState<ScrapingProgress | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!jobId) return;

    const ws = new WebSocket(`${WS_BASE_URL}/ws/scraping/${jobId}`);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as ScrapingProgress;
      setProgress(data);
    };

    return () => {
      ws.close();
    };
  }, [jobId]);

  return { progress, connected };
}
```

---

## Phase 5: CLI Deprecation

**Duration**: 1 day
**Priority**: Low

### 5.1 Move CLI to Dev Tools

**File**: `algobet/cli/dev_tools.py`

```python
"""Development tools CLI - for debugging and database maintenance only."""

import click

from algobet.database import init_db, session_scope


@click.group()
def dev():
    """Development tools for AlgoBet (not for production use)."""
    pass


@dev.command()
def init():
    """Initialize database tables."""
    init_db()
    click.echo("Database initialized.")


@dev.command()
@click.option("--yes", is_flag=True, help="Skip confirmation")
def reset_db(yes: bool):
    """Reset all database tables (DESTRUCTIVE)."""
    if not yes:
        if not click.confirm("This will delete all data. Continue?"):
            return

    from algobet.models import Base
    from algobet.database import create_db_engine

    engine = create_db_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    click.echo("Database reset complete.")


@dev.command()
def db_stats():
    """Show database statistics."""
    from sqlalchemy import func, select
    from algobet.models import Match, Team, Tournament

    with session_scope() as session:
        stats = {
            "tournaments": session.execute(select(func.count(Tournament.id))).scalar(),
            "teams": session.execute(select(func.count(Team.id))).scalar(),
            "matches": session.execute(select(func.count(Match.id))).scalar(),
            "finished_matches": session.execute(
                select(func.count(Match.id)).where(Match.status == "FINISHED")
            ).scalar(),
            "scheduled_matches": session.execute(
                select(func.count(Match.id)).where(Match.status == "SCHEDULED")
            ).scalar(),
        }

    for key, value in stats.items():
        click.echo(f"{key}: {value}")


if __name__ == "__main__":
    dev()
```

### 5.2 Update pyproject.toml

```toml
[project.scripts]
# Main entry point (optional, for backwards compatibility)
algobet = "algobet.cli.dev_tools:dev"

# Or remove CLI entirely and rely only on API:
# algobet-dev = "algobet.cli.dev_tools:dev"
```

---

## Phase 6: Automated/Scheduled Scraping

**Duration**: 2-3 days
**Priority**: High

This phase implements automated scraping via cron jobs and schedulers, allowing the system to automatically update the database with new matches and results without manual intervention.

### 6.1 Architecture Overview

The scheduled scraping system supports multiple execution methods:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Scheduled Scraping Options                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Option A: System Cron                                               │
│  ┌─────────┐      ┌──────────────┐      ┌─────────────────┐        │
│  │  Cron   │─────>│  CLI Runner  │─────>│ ScrapingService │        │
│  │ (OS)    │      │  (Python)    │      │                 │        │
│  └─────────┘      └──────────────┘      └─────────────────┘        │
│                                                                       │
│  Option B: APScheduler (Built-in)                                    │
│  ┌─────────────────┐      ┌─────────────────┐                       │
│  │   APScheduler   │─────>│ ScrapingService │                       │
│  │ (In-Process)    │      │                 │                       │
│  └─────────────────┘      └─────────────────┘                       │
│                                                                       │
│  Option C: Docker + Cron (Containerized)                             │
│  ┌─────────────────┐      ┌──────────────┐      ┌───────────────┐  │
│  │ Docker Container│─────>│  Supervisor  │─────>│ Scrape Worker │  │
│  │   (Cron)        │      │              │      │               │  │
│  └─────────────────┘      └──────────────┘      └───────────────┘  │
│                                                                       │
│  Option D: API-Triggered (Frontend Scheduler)                        │
│  ┌─────────────────┐      ┌─────────────────┐                       │
│  │ Frontend Timer  │─────>│ POST /scraping  │                       │
│  │  or External    │      │  /scheduled/*   │                       │
│  └─────────────────┘      └─────────────────┘                       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Database Models for Schedule Tracking

**File**: `algobet/models.py` (additions)

```python
class ScheduledTask(Base):
    """Stores scheduled task configurations."""

    __tablename__ = "scheduled_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'scrape_upcoming', 'scrape_results', 'generate_predictions'

    # Cron expression (e.g., "0 6 * * *" for daily at 6 AM)
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)

    # Task configuration as JSON
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # Example config: {"url": "...", "max_pages": 5, "leagues": ["premier-league", "la-liga"]}

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationship to execution history
    executions: Mapped[list["TaskExecution"]] = relationship(
        back_populates="scheduled_task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ScheduledTask(id={self.id}, name='{self.name}', cron='{self.cron_expression}')>"


class TaskExecution(Base):
    """Stores execution history for scheduled tasks."""

    __tablename__ = "task_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scheduled_task_id: Mapped[int] = mapped_column(
        ForeignKey("scheduled_tasks.id", ondelete="CASCADE"), nullable=False
    )

    # Execution details
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # 'running', 'completed', 'failed'
    started_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Results
    matches_scraped: Mapped[int] = mapped_column(Integer, default=0)
    matches_saved: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Detailed log as JSON
    execution_log: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Trigger source
    triggered_by: Mapped[str] = mapped_column(String(50), default="scheduler")  # 'scheduler', 'manual', 'api', 'cron'

    # Relationship
    scheduled_task: Mapped["ScheduledTask"] = relationship(back_populates="executions")

    def __repr__(self) -> str:
        return f"<TaskExecution(id={self.id}, task_id={self.scheduled_task_id}, status='{self.status}')>"
```

### 6.3 Scheduler Service

**File**: `algobet/services/scheduler_service.py`

```python
"""Scheduler service for managing automated scraping tasks."""

from datetime import datetime
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.database import session_scope
from algobet.models import ScheduledTask, TaskExecution
from algobet.services.base import BaseService
from algobet.services.scraping_service import ScrapingService


class SchedulerService(BaseService):
    """Service for managing scheduled scraping tasks."""

    _scheduler: AsyncIOScheduler | None = None
    _instance: "SchedulerService | None" = None

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    @classmethod
    def get_scheduler(cls) -> AsyncIOScheduler:
        """Get or create the APScheduler instance."""
        if cls._scheduler is None:
            cls._scheduler = AsyncIOScheduler()
        return cls._scheduler

    @classmethod
    def start_scheduler(cls) -> None:
        """Start the scheduler if not already running."""
        scheduler = cls.get_scheduler()
        if not scheduler.running:
            scheduler.start()

    @classmethod
    def shutdown_scheduler(cls) -> None:
        """Shutdown the scheduler."""
        if cls._scheduler and cls._scheduler.running:
            cls._scheduler.shutdown()

    def create_task(
        self,
        name: str,
        task_type: str,
        cron_expression: str,
        config: dict[str, Any] | None = None,
        is_active: bool = True,
    ) -> ScheduledTask:
        """Create a new scheduled task.

        Args:
            name: Unique name for the task
            task_type: Type of task ('scrape_upcoming', 'scrape_results', 'generate_predictions')
            cron_expression: Cron expression for scheduling (e.g., "0 6 * * *")
            config: Task-specific configuration
            is_active: Whether the task is active

        Returns:
            Created ScheduledTask
        """
        # Validate cron expression
        try:
            CronTrigger.from_crontab(cron_expression)
        except ValueError as e:
            raise ValueError(f"Invalid cron expression: {e}") from e

        task = ScheduledTask(
            name=name,
            task_type=task_type,
            cron_expression=cron_expression,
            config=config or {},
            is_active=is_active,
        )
        self.session.add(task)
        self.session.flush()

        if is_active:
            self._register_task(task)

        return task

    def get_task(self, task_id: int) -> ScheduledTask | None:
        """Get a scheduled task by ID."""
        return self.session.execute(
            select(ScheduledTask).where(ScheduledTask.id == task_id)
        ).scalar_one_or_none()

    def get_task_by_name(self, name: str) -> ScheduledTask | None:
        """Get a scheduled task by name."""
        return self.session.execute(
            select(ScheduledTask).where(ScheduledTask.name == name)
        ).scalar_one_or_none()

    def list_tasks(self, active_only: bool = False) -> list[ScheduledTask]:
        """List all scheduled tasks."""
        stmt = select(ScheduledTask)
        if active_only:
            stmt = stmt.where(ScheduledTask.is_active == True)
        stmt = stmt.order_by(ScheduledTask.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def update_task(
        self,
        task_id: int,
        cron_expression: str | None = None,
        config: dict[str, Any] | None = None,
        is_active: bool | None = None,
    ) -> ScheduledTask:
        """Update a scheduled task."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task with ID {task_id} not found")

        if cron_expression:
            try:
                CronTrigger.from_crontab(cron_expression)
            except ValueError as e:
                raise ValueError(f"Invalid cron expression: {e}") from e
            task.cron_expression = cron_expression

        if config is not None:
            task.config = config

        if is_active is not None:
            task.is_active = is_active
            if is_active:
                self._register_task(task)
            else:
                self._unregister_task(task)

        self.session.flush()
        return task

    def delete_task(self, task_id: int) -> None:
        """Delete a scheduled task."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task with ID {task_id} not found")

        self._unregister_task(task)
        self.session.delete(task)
        self.session.flush()

    def run_task_now(self, task_id: int, triggered_by: str = "manual") -> TaskExecution:
        """Immediately run a scheduled task."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task with ID {task_id} not found")

        return self._execute_task(task, triggered_by)

    def get_task_history(
        self,
        task_id: int,
        limit: int = 20
    ) -> list[TaskExecution]:
        """Get execution history for a task."""
        stmt = (
            select(TaskExecution)
            .where(TaskExecution.scheduled_task_id == task_id)
            .order_by(TaskExecution.started_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_recent_executions(self, limit: int = 50) -> list[TaskExecution]:
        """Get recent executions across all tasks."""
        stmt = (
            select(TaskExecution)
            .order_by(TaskExecution.started_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def _register_task(self, task: ScheduledTask) -> None:
        """Register a task with APScheduler."""
        scheduler = self.get_scheduler()
        job_id = f"task_{task.id}"

        # Remove existing job if any
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

        # Add new job
        scheduler.add_job(
            self._execute_task_async,
            CronTrigger.from_crontab(task.cron_expression),
            id=job_id,
            args=[task.id],
            replace_existing=True,
        )

    def _unregister_task(self, task: ScheduledTask) -> None:
        """Unregister a task from APScheduler."""
        scheduler = self.get_scheduler()
        job_id = f"task_{task.id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

    async def _execute_task_async(self, task_id: int) -> None:
        """Async wrapper for task execution (called by APScheduler)."""
        with session_scope() as session:
            service = SchedulerService(session)
            task = service.get_task(task_id)
            if task and task.is_active:
                service._execute_task(task, triggered_by="scheduler")

    def _execute_task(self, task: ScheduledTask, triggered_by: str) -> TaskExecution:
        """Execute a scheduled task."""
        # Create execution record
        execution = TaskExecution(
            scheduled_task_id=task.id,
            status="running",
            triggered_by=triggered_by,
        )
        self.session.add(execution)
        self.session.flush()

        try:
            # Execute based on task type
            if task.task_type == "scrape_upcoming":
                result = self._run_scrape_upcoming(task.config)
            elif task.task_type == "scrape_results":
                result = self._run_scrape_results(task.config)
            elif task.task_type == "generate_predictions":
                result = self._run_generate_predictions(task.config)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Update execution with results
            execution.status = "completed"
            execution.completed_at = datetime.now()
            execution.matches_scraped = result.get("matches_scraped", 0)
            execution.matches_saved = result.get("matches_saved", 0)
            execution.execution_log = result

            # Update task last run time
            task.last_run_at = datetime.now()

        except Exception as e:
            execution.status = "failed"
            execution.completed_at = datetime.now()
            execution.error_message = str(e)[:1000]
            execution.execution_log = {"error": str(e)}

        self.session.flush()
        return execution

    def _run_scrape_upcoming(self, config: dict) -> dict:
        """Run upcoming matches scraping task."""
        url = config.get("url", "https://www.oddsportal.com/matches/football/")

        scraping_service = ScrapingService(self.session)
        progress = scraping_service.scrape_upcoming(url=url, headless=True)

        return {
            "matches_scraped": progress.matches_scraped,
            "matches_saved": progress.matches_saved,
            "status": progress.status.value,
            "message": progress.message,
        }

    def _run_scrape_results(self, config: dict) -> dict:
        """Run results scraping task."""
        url = config.get("url")
        max_pages = config.get("max_pages", 5)
        leagues = config.get("leagues", [])

        total_scraped = 0
        total_saved = 0
        results = []

        scraping_service = ScrapingService(self.session)

        if leagues:
            # Scrape multiple leagues
            for league in leagues:
                league_url = f"https://www.oddsportal.com/football/england/{league}/results/"
                if "://" in league:
                    league_url = league
                progress = scraping_service.scrape_results(
                    url=league_url,
                    max_pages=max_pages,
                    headless=True
                )
                total_scraped += progress.matches_scraped
                total_saved += progress.matches_saved
                results.append({
                    "league": league,
                    "scraped": progress.matches_scraped,
                    "saved": progress.matches_saved,
                })
        elif url:
            # Scrape single URL
            progress = scraping_service.scrape_results(
                url=url,
                max_pages=max_pages,
                headless=True
            )
            total_scraped = progress.matches_scraped
            total_saved = progress.matches_saved

        return {
            "matches_scraped": total_scraped,
            "matches_saved": total_saved,
            "details": results,
        }

    def _run_generate_predictions(self, config: dict) -> dict:
        """Run prediction generation task."""
        from algobet.services.prediction_service import PredictionService

        days_ahead = config.get("days_ahead", 7)
        min_confidence = config.get("min_confidence", 0.0)

        prediction_service = PredictionService(self.session)
        predictions = prediction_service.predict_upcoming(
            days_ahead=days_ahead,
            min_confidence=min_confidence,
        )

        if predictions:
            prediction_service.save_predictions(predictions)

        return {
            "predictions_generated": len(predictions),
            "matches_saved": len(predictions),
        }

    def load_all_active_tasks(self) -> None:
        """Load all active tasks into the scheduler (call on startup)."""
        active_tasks = self.list_tasks(active_only=True)
        for task in active_tasks:
            self._register_task(task)
```

### 6.4 CLI Runner for Cron Jobs

**File**: `algobet/cli/scheduled_runner.py`

This provides a CLI entry point for cron jobs to execute scheduled tasks:

```python
"""CLI runner for scheduled tasks - designed for cron job execution."""

import argparse
import logging
import sys
from datetime import datetime

from algobet.database import session_scope
from algobet.services.scheduler_service import SchedulerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("algobet.scheduler")


def run_task(task_name: str) -> int:
    """Run a specific task by name.

    Args:
        task_name: Name of the task to run

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info(f"Starting scheduled task: {task_name}")

    with session_scope() as session:
        service = SchedulerService(session)

        task = service.get_task_by_name(task_name)
        if not task:
            logger.error(f"Task not found: {task_name}")
            return 1

        if not task.is_active:
            logger.warning(f"Task is not active: {task_name}")
            return 0

        try:
            execution = service.run_task_now(task.id, triggered_by="cron")

            if execution.status == "completed":
                logger.info(
                    f"Task completed successfully: {task_name} "
                    f"(scraped: {execution.matches_scraped}, saved: {execution.matches_saved})"
                )
                return 0
            else:
                logger.error(f"Task failed: {task_name} - {execution.error_message}")
                return 1

        except Exception as e:
            logger.exception(f"Unexpected error running task {task_name}: {e}")
            return 1


def run_upcoming_scrape() -> int:
    """Quick runner for scraping upcoming matches (for simple cron)."""
    logger.info("Running upcoming matches scrape...")

    with session_scope() as session:
        from algobet.services.scraping_service import ScrapingService

        service = ScrapingService(session)
        progress = service.scrape_upcoming(headless=True)

        if progress.status.value == "completed":
            logger.info(
                f"Scrape completed: {progress.matches_scraped} scraped, "
                f"{progress.matches_saved} saved"
            )
            return 0
        else:
            logger.error(f"Scrape failed: {progress.error}")
            return 1


def main():
    """Main entry point for scheduled CLI runner."""
    parser = argparse.ArgumentParser(
        description="AlgoBet Scheduled Task Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a specific task by name
  python -m algobet.cli.scheduled_runner --task daily-upcoming-scrape

  # Quick upcoming scrape (no task lookup)
  python -m algobet.cli.scheduled_runner --quick-upcoming

  # List all registered tasks
  python -m algobet.cli.scheduled_runner --list
        """
    )

    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Name of the scheduled task to run"
    )
    parser.add_argument(
        "--quick-upcoming",
        action="store_true",
        help="Quick mode: scrape upcoming matches without task lookup"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all registered scheduled tasks"
    )

    args = parser.parse_args()

    if args.list:
        with session_scope() as session:
            service = SchedulerService(session)
            tasks = service.list_tasks()

            if not tasks:
                print("No scheduled tasks found.")
                return 0

            print(f"{'Name':<30} {'Type':<20} {'Cron':<15} {'Active':<8} {'Last Run':<20}")
            print("-" * 95)
            for task in tasks:
                last_run = task.last_run_at.strftime("%Y-%m-%d %H:%M") if task.last_run_at else "Never"
                print(
                    f"{task.name:<30} {task.task_type:<20} {task.cron_expression:<15} "
                    f"{'Yes' if task.is_active else 'No':<8} {last_run:<20}"
                )
        return 0

    if args.quick_upcoming:
        return run_upcoming_scrape()

    if args.task:
        return run_task(args.task)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### 6.5 API Endpoints for Schedule Management

**File**: `algobet/api/routers/schedules.py`

```python
"""API router for scheduled task management."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.services.scheduler_service import SchedulerService

router = APIRouter()


class ScheduledTaskCreate(BaseModel):
    """Request to create a scheduled task."""
    name: str = Field(..., min_length=1, max_length=100)
    task_type: str = Field(..., pattern="^(scrape_upcoming|scrape_results|generate_predictions)$")
    cron_expression: str = Field(..., description="Cron expression (e.g., '0 6 * * *')")
    config: dict | None = Field(default=None)
    is_active: bool = True


class ScheduledTaskUpdate(BaseModel):
    """Request to update a scheduled task."""
    cron_expression: str | None = None
    config: dict | None = None
    is_active: bool | None = None


class ScheduledTaskResponse(BaseModel):
    """Response for a scheduled task."""
    id: int
    name: str
    task_type: str
    cron_expression: str
    config: dict | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_run_at: datetime | None
    next_run_at: datetime | None

    class Config:
        from_attributes = True


class TaskExecutionResponse(BaseModel):
    """Response for a task execution."""
    id: int
    scheduled_task_id: int
    status: str
    started_at: datetime
    completed_at: datetime | None
    matches_scraped: int
    matches_saved: int
    error_message: str | None
    triggered_by: str

    class Config:
        from_attributes = True


@router.post("/", response_model=ScheduledTaskResponse)
async def create_scheduled_task(
    request: ScheduledTaskCreate,
    db: Session = Depends(get_db),
):
    """Create a new scheduled task."""
    service = SchedulerService(db)

    # Check for duplicate name
    existing = service.get_task_by_name(request.name)
    if existing:
        raise HTTPException(status_code=400, detail="Task with this name already exists")

    try:
        task = service.create_task(
            name=request.name,
            task_type=request.task_type,
            cron_expression=request.cron_expression,
            config=request.config,
            is_active=request.is_active,
        )
        return ScheduledTaskResponse.model_validate(task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=list[ScheduledTaskResponse])
async def list_scheduled_tasks(
    active_only: bool = False,
    db: Session = Depends(get_db),
):
    """List all scheduled tasks."""
    service = SchedulerService(db)
    tasks = service.list_tasks(active_only=active_only)
    return [ScheduledTaskResponse.model_validate(t) for t in tasks]


@router.get("/{task_id}", response_model=ScheduledTaskResponse)
async def get_scheduled_task(
    task_id: int,
    db: Session = Depends(get_db),
):
    """Get a scheduled task by ID."""
    service = SchedulerService(db)
    task = service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return ScheduledTaskResponse.model_validate(task)


@router.patch("/{task_id}", response_model=ScheduledTaskResponse)
async def update_scheduled_task(
    task_id: int,
    request: ScheduledTaskUpdate,
    db: Session = Depends(get_db),
):
    """Update a scheduled task."""
    service = SchedulerService(db)
    try:
        task = service.update_task(
            task_id=task_id,
            cron_expression=request.cron_expression,
            config=request.config,
            is_active=request.is_active,
        )
        return ScheduledTaskResponse.model_validate(task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{task_id}")
async def delete_scheduled_task(
    task_id: int,
    db: Session = Depends(get_db),
):
    """Delete a scheduled task."""
    service = SchedulerService(db)
    try:
        service.delete_task(task_id)
        return {"message": "Task deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{task_id}/run", response_model=TaskExecutionResponse)
async def run_task_now(
    task_id: int,
    db: Session = Depends(get_db),
):
    """Immediately run a scheduled task."""
    service = SchedulerService(db)
    try:
        execution = service.run_task_now(task_id, triggered_by="api")
        return TaskExecutionResponse.model_validate(execution)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{task_id}/history", response_model=list[TaskExecutionResponse])
async def get_task_history(
    task_id: int,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """Get execution history for a task."""
    service = SchedulerService(db)
    task = service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    history = service.get_task_history(task_id, limit=limit)
    return [TaskExecutionResponse.model_validate(e) for e in history]


@router.get("/executions/recent", response_model=list[TaskExecutionResponse])
async def get_recent_executions(
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Get recent task executions across all tasks."""
    service = SchedulerService(db)
    executions = service.get_recent_executions(limit=limit)
    return [TaskExecutionResponse.model_validate(e) for e in executions]
```

### 6.6 Docker Configuration for Scheduled Tasks

**File**: `docker-compose.scheduler.yml`

```yaml
version: '3.8'

services:
  # Main API service (already defined in docker-compose.yml)

  # Scheduler worker - runs scheduled tasks via APScheduler
  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: algobet-scheduler
    command: python -m algobet.scheduler.worker
    environment:
      - POSTGRES_HOST=db
      - POSTGRES_USER=${POSTGRES_USER:-algobet}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-password}
      - POSTGRES_DB=${POSTGRES_DB:-football}
    depends_on:
      - db
    restart: unless-stopped
    networks:
      - algobet-network
    # Playwright needs these for headless browser
    shm_size: '2gb'
    security_opt:
      - seccomp:unconfined

  # Alternative: Cron-based scheduler (lighter weight)
  cron-scheduler:
    build:
      context: .
      dockerfile: Dockerfile.cron
    container_name: algobet-cron
    environment:
      - POSTGRES_HOST=db
      - POSTGRES_USER=${POSTGRES_USER:-algobet}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-password}
      - POSTGRES_DB=${POSTGRES_DB:-football}
    depends_on:
      - db
    restart: unless-stopped
    networks:
      - algobet-network
    shm_size: '2gb'
    profiles:
      - cron  # Only starts with `docker-compose --profile cron up`

networks:
  algobet-network:
    external: true
```

**File**: `Dockerfile.cron`

```dockerfile
FROM python:3.11-slim

# Install cron and dependencies
RUN apt-get update && apt-get install -y \
    cron \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set up Playwright
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Copy requirements and install
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install --system -e .

# Install Playwright browsers
RUN playwright install chromium

# Copy application code
COPY algobet/ ./algobet/

# Copy cron configuration
COPY docker/crontab /etc/cron.d/algobet-cron
RUN chmod 0644 /etc/cron.d/algobet-cron
RUN crontab /etc/cron.d/algobet-cron

# Create log file
RUN touch /var/log/cron.log

# Run cron in foreground
CMD ["cron", "-f"]
```

**File**: `docker/crontab`

```cron
# AlgoBet Scheduled Tasks
# Format: minute hour day month weekday command

# Scrape upcoming matches daily at 6:00 AM
0 6 * * * cd /app && python -m algobet.cli.scheduled_runner --quick-upcoming >> /var/log/cron.log 2>&1

# Scrape upcoming matches again at 6:00 PM (pre-evening matches)
0 18 * * * cd /app && python -m algobet.cli.scheduled_runner --quick-upcoming >> /var/log/cron.log 2>&1

# Generate predictions for upcoming matches daily at 7:00 AM
0 7 * * * cd /app && python -m algobet.cli.scheduled_runner --task daily-predictions >> /var/log/cron.log 2>&1

# Scrape results weekly on Monday at 3:00 AM
0 3 * * 1 cd /app && python -m algobet.cli.scheduled_runner --task weekly-results-scrape >> /var/log/cron.log 2>&1

# Empty line at end is required
```

### 6.7 APScheduler Worker Process

**File**: `algobet/scheduler/worker.py`

```python
"""APScheduler worker process for running scheduled tasks."""

import asyncio
import logging
import signal
import sys

from algobet.database import session_scope
from algobet.services.scheduler_service import SchedulerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("algobet.scheduler.worker")


async def main():
    """Main entry point for the scheduler worker."""
    logger.info("Starting AlgoBet Scheduler Worker...")

    # Start the scheduler
    SchedulerService.start_scheduler()

    # Load all active tasks
    with session_scope() as session:
        service = SchedulerService(session)
        service.load_all_active_tasks()

        active_tasks = service.list_tasks(active_only=True)
        logger.info(f"Loaded {len(active_tasks)} active scheduled tasks")
        for task in active_tasks:
            logger.info(f"  - {task.name}: {task.cron_expression}")

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def shutdown():
        logger.info("Shutting down scheduler...")
        SchedulerService.shutdown_scheduler()
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown)

    logger.info("Scheduler worker is running. Press Ctrl+C to stop.")

    # Keep the worker running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        SchedulerService.shutdown_scheduler()
        logger.info("Scheduler worker stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    sys.exit(0)
```

### 6.8 Integration with FastAPI Lifespan

**Update**: `algobet/api/main.py`

```python
from contextlib import asynccontextmanager

from algobet.services.scheduler_service import SchedulerService

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    print("Starting AlgoBet API...")

    # Optionally start the in-process scheduler
    # (Use this if not running a separate scheduler worker)
    if os.getenv("ENABLE_SCHEDULER", "false").lower() == "true":
        SchedulerService.start_scheduler()
        with session_scope() as session:
            service = SchedulerService(session)
            service.load_all_active_tasks()
        print("Scheduler started with active tasks loaded.")

    yield

    # Shutdown
    SchedulerService.shutdown_scheduler()
    print("Shutting down AlgoBet API...")


# Register the schedules router
from algobet.api.routers.schedules import router as schedules_router

app.include_router(
    schedules_router,
    prefix="/api/v1/schedules",
    tags=["schedules"],
)
```

### 6.9 Frontend Schedule Management

**File Structure**:
```
frontend/
├── app/
│   └── schedules/
│       └── page.tsx              # Schedule management page
├── components/
│   ├── schedules/
│   │   ├── ScheduleCard.tsx     # Individual schedule card
│   │   ├── ScheduleForm.tsx     # Create/edit schedule form
│   │   ├── CronExpressionInput.tsx  # User-friendly cron input
│   │   └── ExecutionHistory.tsx # Execution history table
│   └── ...
```

**API Client** (`frontend/lib/api/schedules.ts`):

```typescript
import { API_BASE_URL } from './config';

export interface ScheduledTask {
  id: number;
  name: string;
  task_type: 'scrape_upcoming' | 'scrape_results' | 'generate_predictions';
  cron_expression: string;
  config: Record<string, unknown> | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_run_at: string | null;
  next_run_at: string | null;
}

export interface TaskExecution {
  id: number;
  scheduled_task_id: number;
  status: 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at: string | null;
  matches_scraped: number;
  matches_saved: number;
  error_message: string | null;
  triggered_by: string;
}

export interface CreateTaskRequest {
  name: string;
  task_type: string;
  cron_expression: string;
  config?: Record<string, unknown>;
  is_active?: boolean;
}

export async function getScheduledTasks(): Promise<ScheduledTask[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/schedules`);
  if (!response.ok) throw new Error('Failed to fetch schedules');
  return response.json();
}

export async function createScheduledTask(request: CreateTaskRequest): Promise<ScheduledTask> {
  const response = await fetch(`${API_BASE_URL}/api/v1/schedules`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error('Failed to create schedule');
  return response.json();
}

export async function updateScheduledTask(
  taskId: number,
  updates: Partial<CreateTaskRequest>
): Promise<ScheduledTask> {
  const response = await fetch(`${API_BASE_URL}/api/v1/schedules/${taskId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  if (!response.ok) throw new Error('Failed to update schedule');
  return response.json();
}

export async function deleteScheduledTask(taskId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/schedules/${taskId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete schedule');
}

export async function runTaskNow(taskId: number): Promise<TaskExecution> {
  const response = await fetch(`${API_BASE_URL}/api/v1/schedules/${taskId}/run`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to run task');
  return response.json();
}

export async function getTaskHistory(taskId: number, limit = 20): Promise<TaskExecution[]> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/schedules/${taskId}/history?limit=${limit}`
  );
  if (!response.ok) throw new Error('Failed to fetch history');
  return response.json();
}

export async function getRecentExecutions(limit = 50): Promise<TaskExecution[]> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/schedules/executions/recent?limit=${limit}`
  );
  if (!response.ok) throw new Error('Failed to fetch executions');
  return response.json();
}
```

### 6.10 Default Scheduled Tasks (Database Seed)

**File**: `algobet/cli/seed_schedules.py`

```python
"""Seed default scheduled tasks into the database."""

from algobet.database import session_scope
from algobet.services.scheduler_service import SchedulerService


DEFAULT_TASKS = [
    {
        "name": "daily-upcoming-scrape",
        "task_type": "scrape_upcoming",
        "cron_expression": "0 6 * * *",  # Daily at 6:00 AM
        "config": {
            "url": "https://www.oddsportal.com/matches/football/"
        },
        "is_active": True,
    },
    {
        "name": "evening-upcoming-scrape",
        "task_type": "scrape_upcoming",
        "cron_expression": "0 18 * * *",  # Daily at 6:00 PM
        "config": {
            "url": "https://www.oddsportal.com/matches/football/"
        },
        "is_active": True,
    },
    {
        "name": "daily-predictions",
        "task_type": "generate_predictions",
        "cron_expression": "0 7 * * *",  # Daily at 7:00 AM
        "config": {
            "days_ahead": 7,
            "min_confidence": 0.0,
        },
        "is_active": True,
    },
    {
        "name": "weekly-results-scrape",
        "task_type": "scrape_results",
        "cron_expression": "0 3 * * 1",  # Monday at 3:00 AM
        "config": {
            "leagues": [
                "https://www.oddsportal.com/football/england/premier-league/results/",
                "https://www.oddsportal.com/football/spain/laliga/results/",
                "https://www.oddsportal.com/football/germany/bundesliga/results/",
            ],
            "max_pages": 3,
        },
        "is_active": False,  # Disabled by default, enable via frontend
    },
]


def seed_schedules():
    """Seed default scheduled tasks."""
    with session_scope() as session:
        service = SchedulerService(session)

        for task_config in DEFAULT_TASKS:
            existing = service.get_task_by_name(task_config["name"])
            if existing:
                print(f"Task already exists: {task_config['name']}")
                continue

            service.create_task(
                name=task_config["name"],
                task_type=task_config["task_type"],
                cron_expression=task_config["cron_expression"],
                config=task_config.get("config"),
                is_active=task_config["is_active"],
            )
            print(f"Created task: {task_config['name']}")


if __name__ == "__main__":
    seed_schedules()
```

### 6.11 Common Cron Expression Reference

For frontend UX, provide a helper component with common patterns:

| Description | Cron Expression |
|-------------|-----------------|
| Every hour | `0 * * * *` |
| Every 6 hours | `0 */6 * * *` |
| Daily at 6 AM | `0 6 * * *` |
| Daily at midnight | `0 0 * * *` |
| Twice daily (6 AM & 6 PM) | `0 6,18 * * *` |
| Every Monday at 3 AM | `0 3 * * 1` |
| First day of month | `0 0 1 * *` |
| Weekdays at 8 AM | `0 8 * * 1-5` |

---

## Migration Checklist

### Pre-Migration
- [ ] Create comprehensive tests for existing CLI and API behavior
- [ ] Document current API contracts
- [ ] Backup production database

### Phase 1: Service Layer
- [ ] Create `algobet/services/` directory
- [ ] Create `base.py` with BaseService
- [ ] Create `prediction_service.py`
- [ ] Create `scraping_service.py`
- [ ] Update existing tests to use services

### Phase 2: API Updates
- [ ] Create `api/schemas/scraping.py`
- [ ] Create `api/routers/scraping.py`
- [ ] Register scraping router in `main.py`
- [ ] Add integration tests for scraping endpoints

### Phase 3: WebSocket Support
- [ ] Create `api/websockets/progress.py`
- [ ] Add WebSocket route to `main.py`
- [ ] Test WebSocket with frontend

### Phase 4: Frontend Integration
- [ ] Create scraping dashboard page
- [ ] Implement API client functions
- [ ] Implement WebSocket hook
- [ ] Add UI components (forms, progress, history)

### Phase 5: CLI Deprecation
- [ ] Move CLI to `cli/dev_tools.py`
- [ ] Update `pyproject.toml` entry points
- [ ] Update documentation
- [ ] Remove deprecated files after validation

### Phase 6: Automated/Scheduled Scraping
- [ ] Add `ScheduledTask` and `TaskExecution` models to `models.py`
- [ ] Run database migration for new tables
- [ ] Create `algobet/services/scheduler_service.py`
- [ ] Create `algobet/api/routers/schedules.py`
- [ ] Create `algobet/api/schemas/schedule.py`
- [ ] Register schedules router in `main.py`
- [ ] Create `algobet/cli/scheduled_runner.py` for cron
- [ ] Create `algobet/scheduler/worker.py` for APScheduler
- [ ] Create `docker-compose.scheduler.yml`
- [ ] Create `Dockerfile.cron`
- [ ] Create `docker/crontab`
- [ ] Create `algobet/cli/seed_schedules.py`
- [ ] Run seed script to create default schedules
- [ ] Test cron-based scheduling
- [ ] Test APScheduler-based scheduling
- [ ] Create frontend schedule management page
- [ ] Add integration tests for schedule endpoints

### Post-Migration
- [ ] Run full test suite
- [ ] Update AGENT.md with new patterns
- [ ] Update README.md
- [ ] Performance testing
- [ ] Verify scheduled tasks run correctly in production

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing API endpoints | Low | High | Comprehensive API tests before refactoring |
| Performance regression | Medium | Medium | Load testing before/after |
| WebSocket connection issues | Medium | Low | Fallback to polling |
| Background task failures | Medium | Medium | Implement retry logic and dead letter queue |
| Cron job failures | Medium | Medium | Logging, email alerts, execution history tracking |
| Scheduler container crashes | Low | High | Docker restart policy, health checks |
| Concurrent scraping conflicts | Low | Medium | Job locking, single-instance constraint |

---

## Success Metrics

1. **Code Quality**
   - Reduction in code duplication (target: 80% reduction)
   - Test coverage increase (target: >85%)
   - Cyclomatic complexity reduction

2. **User Experience**
   - Scraping operations accessible via frontend
   - Real-time progress visibility
   - <3s latency for progress updates
   - Schedule management via frontend dashboard

3. **Maintainability**
   - Single source of truth for business logic
   - Clear separation of concerns
   - Comprehensive documentation

4. **Automation**
   - Daily automated scraping of upcoming matches
   - Automated prediction generation
   - Execution history and audit trail
   - Email/Slack notifications on failures (future enhancement)

---

## Appendix A: File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| **Phase 1: Service Layer** | | |
| `algobet/services/__init__.py` | Create | Services package init |
| `algobet/services/base.py` | Create | Base service class |
| `algobet/services/prediction_service.py` | Create | Prediction business logic |
| `algobet/services/scraping_service.py` | Create | Scraping orchestration |
| **Phase 2: API Updates** | | |
| `algobet/api/schemas/scraping.py` | Create | Scraping schemas |
| `algobet/api/routers/scraping.py` | Create | Scraping endpoints |
| **Phase 3: WebSocket** | | |
| `algobet/api/websockets/__init__.py` | Create | WebSocket package init |
| `algobet/api/websockets/progress.py` | Create | WebSocket handlers |
| **Phase 4: Frontend** | | |
| `frontend/app/scraping/page.tsx` | Create | Scraping dashboard |
| `frontend/components/scraping/*.tsx` | Create | Scraping UI components |
| `frontend/lib/api/scraping.ts` | Create | Scraping API client |
| `frontend/hooks/useScrapingProgress.ts` | Create | WebSocket progress hook |
| **Phase 5: CLI** | | |
| `algobet/cli/__init__.py` | Create | CLI package init |
| `algobet/cli/dev_tools.py` | Create | Development tools CLI |
| `algobet/cli.py` | Deprecate | Move to dev_tools |
| `algobet/predictions_cli.py` | Deprecate | Use PredictionService |
| **Phase 6: Scheduled Scraping** | | |
| `algobet/models.py` | Modify | Add ScheduledTask, TaskExecution |
| `algobet/services/scheduler_service.py` | Create | Schedule management service |
| `algobet/api/schemas/schedule.py` | Create | Schedule API schemas |
| `algobet/api/routers/schedules.py` | Create | Schedule API endpoints |
| `algobet/cli/scheduled_runner.py` | Create | CLI runner for cron jobs |
| `algobet/cli/seed_schedules.py` | Create | Default schedule seeder |
| `algobet/scheduler/__init__.py` | Create | Scheduler package init |
| `algobet/scheduler/worker.py` | Create | APScheduler worker process |
| `docker-compose.scheduler.yml` | Create | Scheduler Docker config |
| `Dockerfile.cron` | Create | Cron container Dockerfile |
| `docker/crontab` | Create | Cron schedule definitions |
| **Configuration Updates** | | |
| `algobet/api/main.py` | Modify | Add routers, WebSocket, scheduler |
| `pyproject.toml` | Modify | Update entry points, add APScheduler |
| `AGENT.md` | Modify | Update architecture docs |
| `README.md` | Modify | Update documentation |

---

## Appendix B: Dependencies to Add

Add the following to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "apscheduler>=3.10.0",  # For scheduled task execution
]
```

---

## Appendix C: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SCHEDULER` | `false` | Enable in-process scheduler (for single-container deployments) |
| `SCHEDULER_TIMEZONE` | `UTC` | Timezone for cron expressions |
| `SCRAPING_HEADLESS` | `true` | Run browser in headless mode |
| `SCRAPING_TIMEOUT` | `30000` | Page load timeout in milliseconds |

---

## Minor Suggestions for Enhancement

While the refactoring plan provides a comprehensive solution, the following enhancements could further improve the system's robustness, security, and scalability:

### **1. Additional Monitoring & Observability**

**Metrics Collection (Prometheus/Grafana Integration):**
- **Implementation**: Add Prometheus client library and expose `/metrics` endpoint
- **Key Metrics**:
  - Scraping operations (duration, success/failure rates, matches scraped/saved)
  - API response times and error rates
  - Database query performance
  - Model prediction accuracy over time
- **Benefits**: Proactive monitoring, performance optimization, alerting capabilities

**Health Check Endpoints:**
- **Implementation**: Add `/health` and `/health/ready` endpoints to all services
- **Checks Include**: Database connectivity, external service availability, disk space
- **Benefits**: Kubernetes/Docker health checks, service discovery, uptime monitoring

**Performance Monitoring:**
- **Implementation**: Add decorators to service methods for timing and profiling
- **Tools**: Use `py-spy` for profiling, `statsd` for metrics collection
- **Benefits**: Identify bottlenecks, optimize scraping performance, capacity planning

### **2. Enhanced Security & Input Validation**

**Rate Limiting for Scraping Endpoints:**
- **Implementation**: Use `slowapi` or custom middleware with Redis backend
- **Configuration**:
  - 10 requests per minute for scraping endpoints
  - 100 requests per minute for general API usage
  - IP-based and user-based limits (when auth is added)
- **Benefits**: Prevent abuse, protect against DoS attacks, ensure fair usage

**Authentication & Authorization:**
- **Implementation**: JWT-based auth with FastAPI dependencies
- **Roles**:
  - `admin`: Full access to scraping and scheduling
  - `user`: Read-only access to matches and predictions
  - `service`: Internal service-to-service communication
- **Benefits**: Secure admin operations, user management, audit trails

**Enhanced Input Validation:**
- **URL Validation**: Verify OddsPortal URLs format and reachability
- **Cron Expression Validation**: Real-time validation with next execution preview
- **Data Sanitization**: SQL injection prevention, XSS protection for frontend
- **Rate Limiting**: Per-IP and per-user limits on sensitive operations
- **Benefits**: Data integrity, security, better user experience

### **3. Scalability & Production Readiness**

**Redis for Job Queue Management:**
- **Implementation**: Replace in-memory `_jobs` dictionary with Redis streams/lists
- **Configuration**:
  - Job status storage with TTL
  - Distributed locking for concurrent operations
  - Pub/sub for real-time progress updates
- **Benefits**: Multiple API instances support, better fault tolerance, persistence

**Horizontal Scaling Support:**
- **Load Balancing**: Sticky sessions for WebSocket connections
- **Database Connection Pooling**: Optimize for multiple API instances
- **Caching Strategy**: Redis caching for frequently accessed data (tournaments, teams)
- **Stateless Services**: Ensure services can run across multiple containers
- **Benefits**: Handle increased load, high availability, zero-downtime deployments

**Circuit Breaker Patterns:**
- **Implementation**: Use `py-breaker` or similar library for external service calls
- **Applications**:
  - OddsPortal scraping (prevent IP blocking)
  - Database operations (handle connection issues)
  - External API calls (if any third-party integrations)
- **Configuration**:
  - 50% failure rate triggers open state
  - 60-second timeout before retry
  - Exponential backoff for retries
- **Benefits**: Resilience, automatic recovery, graceful degradation

### **Implementation Priority & Effort**

**High Priority (Week 1-2):**
- Rate limiting implementation (~2 days)
- Enhanced input validation (~1 day)
- Health check endpoints (~0.5 day)

**Medium Priority (Week 3-4):**
- Basic metrics collection (~2 days)
- Redis job queue setup (~2 days)
- Circuit breaker for scraping (~1 day)

**Low Priority (Future Sprint):**
- Full authentication system (~3-5 days)
- Comprehensive monitoring dashboard (~3-5 days)
- Advanced caching strategies (~2-3 days)

---

## Appendix D: API Endpoints Summary (After Refactor)

### Scraping Endpoints (`/api/v1/scraping`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upcoming` | Start scraping upcoming matches |
| POST | `/results` | Start scraping historical results |
| GET | `/jobs` | List all scraping jobs |
| GET | `/jobs/{job_id}` | Get job status and progress |

### Schedule Endpoints (`/api/v1/schedules`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | List all scheduled tasks |
| POST | `/` | Create a new scheduled task |
| GET | `/{task_id}` | Get task details |
| PATCH | `/{task_id}` | Update task configuration |
| DELETE | `/{task_id}` | Delete a task |
| POST | `/{task_id}/run` | Manually run a task |
| GET | `/{task_id}/history` | Get task execution history |
| GET | `/executions/recent` | Get recent executions across all tasks |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `ws://host/ws/scraping/{job_id}` | Real-time scraping progress updates |
