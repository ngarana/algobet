# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlgoBet is a full-stack football match prediction and betting analytics platform. It fetches match data from OddsPortal, enriches it with advanced statistics (xG, PPDA, player stats), trains ML models (XGBoost, LightGBM, Random Forest) to predict outcomes, detects value bets, and automates daily scraping and predictions via a scheduler.

**Tech Stack:**
- **Backend:** Python 3.10+, FastAPI, SQLAlchemy 2.0, PostgreSQL
- **ML:** scikit-learn, XGBoost, LightGBM, Optuna (hyperparameter tuning)
- **Frontend:** Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, shadcn/ui
- **Scraping:** Playwright (OddsPortal), soccerdata library (Understat/ESPN)
- **Scheduling:** APScheduler for automated tasks
- **DevOps:** Docker, docker-compose, Redis for caching

## High-Level Architecture

```
┌─────────────────────────────────────────┐
│  Frontend (Next.js 15)                  │
│  - Real-time dashboard with TanStack Q  │
│  - WebSocket for live job progress      │
│  - Pages: /matches, /predictions,       │
│    /models, /value-bets, /schedules    │
└─────────┬───────────────────────────────┘
          │ HTTP + WebSocket
          ▼
┌─────────────────────────────────────────┐
│  FastAPI Backend (algobet/api/)         │
│  - Routers: matches, predictions, ML    │
│  - Services layer for business logic    │
│  - WebSocket endpoints for progress     │
└─────────┬───────────────────────────────┘
          │
    ┌─────┴──────┬────────────────┐
    ▼            ▼                ▼
  Services    Database       Scheduler
  ┌────────────────────────────────┐
  │ Core Modules                   │
  ├────────────────────────────────┤
  │ • PredictionService            │
  │   - ML inference for matches   │
  │   - Feature pipeline loading   │
  │                                │
  │ • ScrapingService              │
  │   - Playwright-based scraper   │
  │   - Job tracking via DB        │
  │                                │
  │ • AnalysisService              │
  │   - Backtesting, calibration   │
  │   - Value bet detection        │
  │                                │
  │ • PredictionModule             │
  │   - Feature generation         │
  │   - Model training/registry    │
  │   - Evaluation & reports       │
  └────────────────────────────────┘
    │
    ▼
  ┌────────────────────────────────┐
  │ PostgreSQL + Redis             │
  │ - Matches, teams, tournaments  │
  │ - Predictions, odds, results   │
  │ - Model versions, features     │
  │ - Scheduled tasks & history    │
  └────────────────────────────────┘
```

## Directory Structure

```
algobet/                      # Backend package
├── api/                      # FastAPI app and routers
│   ├── main.py              # FastAPI app initialization, middleware, WebSocket
│   ├── routers/             # API endpoints
│   │   ├── matches.py       # GET /api/v1/matches
│   │   ├── predictions.py   # GET/POST /api/v1/predictions
│   │   ├── scraping.py      # POST /api/v1/scraping/*, WebSocket /ws/*
│   │   ├── ml_operations.py # POST /api/v1/ml/train, /backtest, /calibrate
│   │   ├── models.py        # GET/POST /api/v1/models (registry)
│   │   ├── value_bets.py    # GET /api/v1/value-bets
│   │   ├── schedules.py     # CRUD /api/v1/schedules
│   │   └── workflow.py      # GET /api/v1/workflow/* (dashboard data)
│   ├── schemas/             # Pydantic schemas for requests/responses
│   └── websockets/          # WebSocket connection management
│
├── predictions/             # Machine learning core module
│   ├── features/
│   │   ├── pipeline.py      # FeaturePipeline: orchestrates generation & transformation
│   │   ├── generators.py    # FeatureGenerator subclasses (form, stats, odds, etc.)
│   │   ├── transformers.py  # Data normalization, scaling, selection
│   │   └── store.py         # FeatureStore: caches features to DB
│   ├── training/
│   │   ├── runner.py        # PipelineRunnerMixin: orchestrates full training
│   │   ├── classifiers.py   # XGBoost, LightGBM, RandomForest, Ensemble predictors
│   │   ├── calibration.py   # ProbabilityCalibrator (isotonic/sigmoid)
│   │   ├── tuner.py         # HyperparameterTuner (Optuna-based)
│   │   ├── split.py         # TemporalSplitter, SeasonAwareSplitter
│   │   └── ensemble.py      # EnsembleWeightOptimizer
│   ├── models/
│   │   ├── base.py          # SQLAlchemy: ModelVersion, Prediction, etc.
│   │   └── registry.py      # ModelRegistry: save/load/activate models
│   ├── evaluation/
│   │   ├── metrics.py       # Classification, betting, calibration metrics
│   │   ├── calibration.py   # Reliability diagrams, calibration analysis
│   │   └── reports.py       # HTML/Markdown report generation
│   └── data/
│       ├── queries.py       # MatchRepository: DB queries for training data
│       └── historical_provider.py  # Fetch historical matches
│
├── services/                # Business logic layer
│   ├── prediction_service.py     # PredictionService: production inference
│   ├── scraping_service.py       # ScrapingService: orchestrates OddsPortal scraper
│   ├── analysis_service.py       # AnalysisService: backtest, value bets, calibration
│   ├── scheduler_service.py      # SchedulerService: task CRUD & execution
│   └── ml_ops/                   # ML operations sub-services
│
├── scraping/                # Web scraping module
│   ├── scraper.py          # Playwright scraper for OddsPortal
│   ├── models.py           # ScrapingJob, ScrapingLog, ScrapedOdds models
│   └── service/            # Scraping service classes
│
├── matches/                 # Match domain
│   ├── models.py           # Match, MatchStatistics SQLAlchemy models
│   └── service/            # Match-related business logic
│
├── teams/                   # Teams/tournaments domain
│   ├── models.py           # Tournament, Season, Team, TeamAlias models
│   └── service/            # Team-related queries
│
├── scheduling/             # Task scheduling domain
│   ├── models.py           # ScheduledTask, TaskExecution models
│   └── service/            # Scheduling business logic
│
├── infrastructure/         # Cross-cutting concerns
│   ├── database.py         # DB session management (sync & async)
│   ├── config.py           # Pydantic Settings for env variables
│   ├── exceptions.py       # Custom exception classes
│   └── models.py           # Base class, TimestampMixin
│
├── cli/                    # Command-line interface
│   ├── dev_tools.py        # `algobet init`, `reset-db`, `db-stats`
│   ├── seed_schedules.py   # Populate default scheduled tasks
│   ├── scheduled_runner.py # `algobet-runner --task NAME`
│   └── commands/
│       └── train.py        # `algobet train run` with options
│
├── scheduler/              # APScheduler worker
│   └── worker.py           # `algobet-scheduler` entry point
│
└── models.py              # Central re-export of all SQLAlchemy models

frontend/                   # Next.js frontend (App Router)
├── app/
│   ├── page.tsx           # Dashboard: today's matches, predictions, value bets
│   ├── matches/           # Upcoming & past matches browser
│   ├── predictions/       # Predictions view
│   ├── models/            # Model training workspace & registry
│   ├── value-bets/        # Value bet detection & analysis
│   ├── schedules/         # Scheduled task management
│   └── scraping/          # Job monitoring with WebSocket
├── lib/
│   ├── api/               # HTTP client & WebSocket utilities
│   ├── types/             # TypeScript types & Zod schemas
│   └── queries/           # TanStack Query hooks
└── components/            # Reusable React components
```

## Key Workflows

### 1. **Data Pipeline: Scraping → Feature Engineering → Model Training**

```
OddsPortal (Playwright) → Match (DB)
                             ↓
              Enrich with stats (Understat/ESPN)
                             ↓
              MatchRepository queries historical data
                             ↓
              FeaturePipeline.fit_transform()
              (4 FeatureGenerator classes)
                             ↓
              TrainingPipeline.run()
              (tuning, training, calibration)
                             ↓
              ModelRegistry.save() → data/models/{version}/
                             ↓
              Activate model for predictions
```

### 2. **Daily Scheduled Tasks**

| Time | Task | Handler |
|------|------|---------|
| 6:00 AM | `scrape_upcoming` | Scrapes upcoming matches (7+ days) |
| 7:00 AM | `generate_predictions` | Runs `PredictionService.predict_upcoming()` |
| 6:00 PM | `scrape_upcoming` | Evening odds refresh |
| Mon 3 AM | `scrape_results` | Scrapes weekend results & updates Match.home_score, away_score |
| Mon 5 AM | `enrich_stats` | (Disabled by default) Adds xG/PPDA/player stats via soccerdata |

Tasks are defined in database as `ScheduledTask` records and executed by APScheduler in the scheduler-worker container.

### 3. **Prediction Generation Flow**

```
PredictionService.predict_upcoming()
  → MatchRepository.get_upcoming_matches(days_ahead=7)
  → For each match:
      - Load active ModelVersion
      - Load associated FeaturePipeline
      - Extract features via pipeline.transform()
      - Run model.predict_proba()
      - Create Prediction record (probabilities, confidence)
      - Check if value bet (odds vs. model prob)
```

### 4. **ML Training Workflow**

```
algobet train run --model-type xgboost --tune --ensemble
  ↓
TrainingPipeline initialization
  ├─ FeaturePipeline.fit_transform() on training matches
  ├─ TemporalSplitter.split() → train/val/test
  ├─ (Optional) Hyperparameter tuning via Optuna
  ├─ Fit classifier (XGBoost, LightGBM, etc.)
  ├─ ProbabilityCalibrator.fit() on validation set
  ├─ Evaluate on test set (metrics, calibration plot)
  └─ ModelRegistry.save() + FeaturePipeline.save()
      → data/models/{timestamp}/model.pkl
      → data/pipelines/{timestamp}/config.json
```

## Development Setup & Commands

### Initial Setup

```bash
# Backend setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"  # or: uv pip install -e ".[dev]"

# Initialize database
algobet init

# Frontend setup
cd frontend
npm install  # or: pnpm install
```

### Running Services

```bash
# Backend API (development with auto-reload)
uvicorn algobet.api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (Next.js dev server on :3001)
cd frontend && npm run dev

# With Docker Compose (all services + scheduler)
docker-compose up -d

# Scheduler worker (runs scheduled tasks on cron schedule)
python -m algobet.scheduler.worker
```

### Testing & Linting

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run single test
pytest tests/test_api.py::test_function_name

# Run with coverage
pytest --cov=algobet --cov-report=html

# Frontend tests
cd frontend && npm test

# Lint backend (ruff)
ruff check algobet/

# Fix lint issues
ruff check --fix algobet/

# Type check (mypy)
mypy algobet

# Frontend type check
cd frontend && npm run typecheck

# Lint frontend (eslint)
cd frontend && npm run lint:fix
```

### Database Operations

```bash
# Initialize empty database
algobet init

# Reset database (destructive)
algobet reset-db --yes

# Show database statistics
algobet db-stats

# Seed default scheduled tasks
algobet seed-schedules
```

### Model Training & Analysis

```bash
# Train XGBoost model with hyperparameter tuning
algobet train run --model-type xgboost --tune

# Train ensemble of multiple models
algobet train run --ensemble --ensemble-types xgboost lightgbm

# Run backtest on model
algobet analyze backtest

# Recalibrate model probabilities
algobet analyze calibrate

# Find value bets
algobet analyze value-bets --min-ev 0.05

# List upcoming matches
algobet list upcoming --days 3
```

### Manual Scheduled Task Execution

```bash
# Run a scheduled task immediately
algobet-runner --task daily-upcoming-scrape

# Trigger scraping via API
curl -X POST "http://localhost:8000/api/v1/scraping/upcoming"

# Generate predictions for next 7 days
curl -X POST "http://localhost:8000/api/v1/predictions/generate" \
  -H "Content-Type: application/json" \
  -d '{"days_ahead": 7, "min_confidence": 0.5}'
```

## Critical Implementation Details

### 1. **Feature Engineering Pipeline**

The `FeaturePipeline` (in `algobet/predictions/features/pipeline.py`) orchestrates feature generation:
- **Generators:** Multiple `FeatureGenerator` subclasses create different feature groups (form stats, advanced stats, odds-based)
- **Transformers:** Scaling, imputation, feature selection
- **Storage:** Fitted pipeline saved alongside trained model in `data/pipelines/{timestamp}/`
- **Inference:** `PredictionService` loads the exact pipeline that was used during training to ensure consistent feature schema

**Why this matters:** Prediction generation must use the same feature pipeline that trained the model. Always load the pipeline from the model directory, not recreate it from scratch.

### 2. **Temporal Data Splitting**

Training uses `TemporalSplitter` or `SeasonAwareSplitter` (in `algobet/predictions/training/split.py`):
- Respects time ordering (no future data leakage)
- Season-aware variant groups matches by season
- Prevents overfitting to recent data

**Why this matters:** Standard random CV is invalid for time-series predictions. Always use temporal splits for match prediction models.

### 3. **Model Registry & Versioning**

`ModelRegistry` (in `algobet/predictions/models/registry.py`):
- Stores trained models as pickle files in `data/models/{timestamp}/`
- Tracks metadata (algorithm, accuracy, features, calibration params) in DB (ModelVersion table)
- Supports model activation: only one model is "active" for predictions
- Fallback mechanism: if active model not found, uses most recent

**Why this matters:** Multiple model versions can coexist. Always activate a model before using it for predictions. Version metadata is critical for debugging.

### 4. **Probability Calibration**

After training, models are calibrated using `ProbabilityCalibrator` (in `algobet/predictions/training/calibration.py`):
- Isotonic regression or Sigmoid fitting on validation set
- Critical for value bet detection (need accurate probability estimates)
- Stored alongside model in DB (ModelVersion.calibration_params)

**Why this matters:** Raw model probabilities are often miscalibrated. Always calibrate before using for betting decisions.

### 5. **Scraping Architecture**

`ScrapingService` (in `algobet/services/scraping_service.py`) and `Scraper` (in `algobet/scraper.py`):
- Playwright-based OddsPortal scraper (handles dynamic content)
- Job tracking via `ScrapingJob` DB records for resumability
- Progress tracking via WebSocket to frontend
- Async execution with job queue

**Frontend receives updates:** WebSocket `/ws/scraping/{job_id}` streams progress in real-time (status, match counts, logs).

### 6. **Database Schema Relationships**

Key relationships to understand:
- `Match` → `home_team`, `away_team` (Team)
- `Match` → `tournament`, `season` (Tournament, Season)
- `Match` → `predictions` (Prediction, many-to-one)
- `Match` → `statistics` (MatchStatistics, one-to-one)
- `Prediction` → `model_version` (ModelVersion)
- `ModelVersion` → `model_features` (ModelFeature, tracks which features used)

### 7. **API Request/Response Patterns**

- **Async endpoint pattern:** FastAPI dependency injection for `session: AsyncSession = Depends(...)`
- **Error handling:** Custom exceptions in `algobet/exceptions.py` mapped to HTTP status codes
- **Pagination:** Use `skip` and `limit` query params for large result sets
- **WebSocket:** Connect to `/ws/{path}` for real-time updates (used for scraping jobs)

## Frontend Architecture

- **Framework:** Next.js 15 (App Router, Server Components where possible)
- **State Management:** TanStack Query (server state) + Zustand (client state)
- **API Client:** `lib/api/` utilities for HTTP and WebSocket
- **Forms:** React Hook Form + Zod for validation
- **UI Components:** shadcn/ui (Radix UI primitives) + Tailwind CSS

**Key patterns:**
- Query hooks in `lib/queries/` (e.g., `useMatches()`, `usePredictions()`)
- Type-safe API responses via Zod schemas in `lib/types/`
- WebSocket hooks for real-time job progress
- TanStack Query for automatic caching and refetching

## Important Environment Variables

```bash
# Backend
DATABASE_URL=postgresql://...
POSTGRES_HOST, POSTGRES_PORT, etc.
REDIS_URL=redis://...
API_HOST, API_PORT
CORS_ORIGINS=...
LOG_LEVEL=debug|info|warning

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

See `.env.example` for full reference.

## Testing Strategy

- **Unit tests:** `tests/unit/` for services, models, utilities
- **Integration tests:** `tests/integration/` for API endpoints, DB interactions
- **Fixtures:** `tests/conftest.py` provides test DB session, sample data
- **Async tests:** Use `pytest-asyncio` for async endpoint testing

```bash
# Run integration tests only
pytest tests/integration/

# Run with specific markers
pytest -m "not integration"
```

## Debugging Tips

1. **Database queries:** Set `ALGOBET_DATABASE__ECHO=true` to see SQL logs
2. **Feature engineering:** Add debug prints in `FeaturePipeline.fit_transform()` or use `pipeline._last_raw_features`
3. **Model training:** Check `TrainingResult.metrics` for detailed evaluation stats
4. **Scraping issues:** Check `ScrapingLog` table for error messages; WebSocket progress endpoint shows real-time status
5. **API errors:** Check `algobet/exceptions.py` for custom error types and HTTP mappings

## Common Pitfalls

1. **Forgetting to load FeaturePipeline during inference:** Predictions will use stale/incorrect feature schema
2. **Using random train/test split on time-series data:** Will lead to overfitting; use temporal split
3. **Not calibrating model probabilities:** Value bet detection becomes unreliable
4. **Scraper timeouts with Playwright:** Increase timeout in env var; check OddsPortal page structure
5. **Async session lifecycle:** Always use `async_session_scope()` context manager; don't reuse across requests

## Project Layout Notes

- All SQLAlchemy models inherit from `Base` defined in `algobet/infrastructure/models.py`
- Tests use fixtures from `tests/conftest.py` (test DB, session, factories)
- CLI is the primary entry point for database initialization and training
- Services layer handles business logic; routers delegate to services
- Models are organized by domain (matches, teams, predictions, etc.), not by type (models, services, etc.)
