# AlgoBet - Football Match Prediction & Betting Analytics Platform

A comprehensive full-stack application for fetching, analyzing, and predicting football match outcomes using machine learning. Features a modern React frontend, FastAPI backend, and automated scheduling system.

## Features

### Core Capabilities
- 📊 **Database Management**: PostgreSQL with SQLAlchemy ORM for tournaments, seasons, teams, matches, and predictions
- 🤖 **Machine Learning**: XGBoost/LightGBM ensemble models for match outcome prediction with probability calibration
- ⚽ **OddsPortal Scraping**: Playwright-based web scraping for fixtures, results, and betting odds
- 📈 **Advanced Stats**: soccerdata integration (Understat xG/npxG/PPDA + ESPN player stats)
- 🎯 **Value Bet Detection**: Automated identification of profitable betting opportunities
- 📅 **Automated Scheduling**: APScheduler integration for daily data fetching and predictions
- 🔌 **Real-time Updates**: WebSocket support for live job progress and match updates

### Frontend Features
- Modern React dashboard with Next.js 15 App Router
- Real-time job monitoring with WebSocket updates
- Interactive match analysis with team form visualization
- Prediction confidence badges and value bet indicators
- League selection UI for fetching upcoming matches
- Responsive design with shadcn/ui components

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Next.js 15 Frontend     │  WebSocket Client  │  CLI (Dev Tools)│
│  - React + TypeScript    │  - Real-time       │  - algobet      │
│  - TanStack Query        │    progress        │  - algobet-dev  │
│  - shadcn/ui             │  - Live updates    │                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                      FastAPI Application                         │
├─────────────────────────────────────────────────────────────────┤
│  /api/v1/matches      │  /api/v1/predictions  │  /api/v1/models │
│  /api/v1/tournaments  │  /api/v1/value-bets   │  /api/v1/scraping│
│  /api/v1/teams        │  /api/v1/schedules    │  /ws/progress   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  PredictionService   │  ScrapingService   │  SchedulerService  │
│  - Model inference   │  - OddsPortal      │  - Task CRUD       │
│  - Feature eng.      │    scraper          │  - Cron execution  │
│  - Batch predict     │  - Job tracking    │  - History track   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL Database         │  Model Registry (File System)    │
│  - matches, teams            │  - XGBoost/LightGBM models       │
│  - predictions, tournaments  │  - Feature transformers          │
│  - scheduled_tasks           │  - Version metadata              │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.10+)
- **Database**: PostgreSQL + SQLAlchemy 2.0
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, Optuna
- **Data Sources**: OddsPortal (Playwright), Understat + ESPN (soccerdata library)
- **Scheduling**: APScheduler
- **Testing**: pytest, pytest-asyncio

### Frontend
- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript 5.3+
- **Styling**: Tailwind CSS 3.4+
- **UI Components**: shadcn/ui + Radix UI
- **State Management**: TanStack Query, Zustand
- **Forms**: React Hook Form + Zod

### DevOps
- **Containerization**: Docker + docker-compose
- **Scheduler**: Cron jobs via Docker or system cron
- **Code Quality**: ruff (linting), mypy (type checking)

## Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Node.js 18+ (for frontend)

### Backend Setup

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Database Setup

```bash
# Initialize database tables
algobet init

# Or reset (destructive)
algobet reset-db --yes

# Seed with default scheduled tasks
algobet seed-schedules
```

### Docker (Alternative)

```bash
# Full stack with scheduler
docker-compose up -d

# Database only
docker-compose up -d db
```

## Usage

### Start the API Server

```bash
# Development with auto-reload
uvicorn algobet.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn algobet.api.main:app --host 0.0.0.0 --port 8000

# With scheduler enabled
ENABLE_SCHEDULER=true uvicorn algobet.api.main:app --host 0.0.0.0 --port 8000
```

### Start the Frontend

```bash
cd frontend
npm run dev
```

Access the application at `http://localhost:3000`

### Development CLI Tools

```bash
# Initialize database
algobet init

# Reset database (destructive)
algobet reset-db

# Show database statistics
algobet db-stats

# Run scheduled task manually
algobet-runner --task daily-upcoming-scrape

# Train ML model
algobet train run --model-type xgboost --tune
```

### API Endpoints

#### Scraping Matches (OddsPortal)

```bash
# Scrape upcoming matches
curl -X POST "http://localhost:8000/api/v1/scraping/upcoming"

# Scrape upcoming matches for a specific tournament
curl -X POST "http://localhost:8000/api/v1/scraping/upcoming?tournament_url=football/england/premier-league"

# Scrape match results
curl -X POST "http://localhost:8000/api/v1/scraping/results"

# Check job status
curl "http://localhost:8000/api/v1/scraping/jobs/{job_id}"

# List all jobs
curl "http://localhost:8000/api/v1/scraping/jobs"

# Get scraping statistics
curl "http://localhost:8000/api/v1/scraping/stats"
```

#### Predictions
```bash
# Generate predictions for upcoming matches
curl -X POST "http://localhost:8000/api/v1/predictions/generate" \
  -H "Content-Type: application/json" \
  -d '{"days_ahead": 7, "min_confidence": 0.5}'

# Get predictions
curl "http://localhost:8000/api/v1/predictions?days_ahead=7"

# Get value bets
curl "http://localhost:8000/api/v1/value-bets?min_ev=0.05&days=7"
```

#### Schedule Management
```bash
# Create scheduled task
curl -X POST "http://localhost:8000/api/v1/schedules" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "daily-upcoming",
    "task_type": "scrape_upcoming",
    "cron_expression": "0 6 * * *",
    "config": {"league_ids": [39, 140, 135, 78, 61]}
  }'

# List schedules
curl "http://localhost:8000/api/v1/schedules"

# Run task immediately
curl -X POST "http://localhost:8000/api/v1/schedules/{id}/run"

# Get execution history
curl "http://localhost:8000/api/v1/schedules/{id}/history"
```

### WebSocket Connection

Connect to WebSocket for real-time progress updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/scraping/{job_id}');

ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Status: ${progress.status}`);
  console.log(`Progress: ${progress.progress}%`);
  console.log(`Matches: ${progress.matches_scraped} fetched, ${progress.matches_saved} saved`);
};
```
algobet/                          # Backend Python package (FastAPI)
├── api/
│   ├── main.py                   # FastAPI app, registers all routers
│   ├── dependencies.py           # DB session dependency
│   ├── routers/
│   │   ├── ml_operations.py      # POST /api/v1/ml/train, /backtest, /calibrate
│   │   ├── models.py             # GET/POST/PUT/DELETE /api/v1/models
│   │   ├── predictions.py        # GET/POST /api/v1/predictions
│   │   ├── matches.py, teams.py, tournaments.py, seasons.py, scraping.py, ...
│   │   └── value_bets.py, schedules.py
│   └── schemas/
│       └── model.py              # Pydantic ModelVersionResponse
├── predictions/                  # ** CORE ML MODULE **
│   ├── data/
│   │   └── queries.py            # MatchRepository - DB queries for training
│   ├── features/
│   │   ├── generators.py         # 4 FeatureGenerator classes + composite
│   │   ├── form_features.py      # Legacy FormCalculator (6 features)
│   │   ├── pipeline.py           # FeaturePipeline orchestrator
│   │   ├── transformers.py       # Scaling, imputation, selection
│   │   └── store.py              # FeatureStore - caching to DB
│   ├── models/
│   │   ├── base.py               # SQLAlchemy: ModelVersion, Prediction, ModelFeature, BacktestHistory
│   │   └── registry.py           # ModelRegistry - save/load/activate models
│   ├── training/
│   │   ├── pipeline.py           # TrainingPipeline - end-to-end orchestration
│   │   ├── classifiers.py        # XGBoostPredictor, LightGBMPredictor, RandomForestPredictor, EnsemblePredictor
│   │   ├── calibration.py        # ProbabilityCalibrator (isotonic/sigmoid)
│   │   ├── split.py              # TemporalSplitter, ExpandingWindowSplitter, SeasonAwareSplitter
│   │   ├── tuner.py              # HyperparameterTuner (Optuna) + GridSearchTuner
│   │   └── acceleration.py       # GPU acceleration profiles (Intel iGPU)
│   └── evaluation/
│       ├── metrics.py            # ClassificationMetrics, BettingMetrics, evaluate_predictions()
│       ├── calibration.py        # Calibration analysis, reliability diagrams
│       └── reports.py            # HTML/Markdown report generation
├── services/
│   ├── prediction_service.py     # PredictionService - production inference
│   ├── analysis_service.py       # AnalysisService - backtest, value bets, calibration
│   └── model_management_service.py
├── matches/models.py             # SQLAlchemy Match + MatchStatistics
├── teams/models.py               # SQLAlchemy Tournament, Season, Team
├── scraping/models.py            # SQLAlchemy ScrapingJob, ScrapingLog, ScrapedOdds, ScrapingSource
├── scheduling/models.py          # SQLAlchemy ScheduledTask, TaskExecution
├── infrastructure/
│   ├── models.py                 # Base, TimestampMixin, MetadataMixin
│   └── database.py               # DB connection + session_scope
├── importers/football_data.py    # CSV importer from Football-Data.co.uk
├── cli/commands/train.py         # CLI: `algobet train run`
└── models.py                     # Central re-export of all SQLAlchemy models
frontend/                         # Next.js App Router frontend
├── app/models/page.tsx           # Models page with training workspace + registry
├── lib/
│   ├── api/ml-operations.ts      # Frontend API client: runTrainModel, runBacktest, runCalibrate
│   ├── types/ml-operations.ts    # Zod schemas + TS types for ML ops
│   ├── types/api.ts              # TS types: Match, Prediction, ModelVersion, Team, etc.
│   └── queries/use-ml-operations.ts  # TanStack Query hooks for ML operations
└── components/models/            # 20+ training UI components
    ├── FeatureGroupsSection.tsx   # Toggle 4 feature groups in UI
    ├── GuidedTrainingWorkspace.tsx
    ├── TrainingResultDisplay.tsx
    └── ... (DataSplitSection, HyperparametersSection, EnsembleSection, etc.)
```

## Database Schema

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| tournaments | League/tournament info | id, name, country, url_slug |
| seasons | Season records | id, tournament_id, name, start_year, end_year |
| teams | Team information | id, name |
| matches | Match records | id, home/away_team_id, match_date, scores, odds, status |
| predictions | ML predictions | id, match_id, model_version, probabilities, confidence |
| model_versions | ML model registry | id, version, algorithm, accuracy, is_active |
| scheduled_tasks | Automation config | id, name, cron_expression, is_active |
| task_executions | Automation history | id, task_id, status, started_at, completed_at |

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=algobet --cov-report=html

# Frontend tests
cd frontend
npm test
```

## Daily Workflow

### Automated (runs via scheduler worker or crontab)

| Time | Task | Source | What Happens |
|------|------|--------|--------------|
| 6:00 AM | `scrape_upcoming` | OddsPortal (Playwright) | Scrapes all upcoming matches + odds → creates SCHEDULED `Match` records |
| 7:00 AM | `generate_predictions` | Active ML model | Runs `predict_upcoming()` for next 7 days → creates `Prediction` records |
| 6:00 PM | `scrape_upcoming` | OddsPortal (Playwright) | Second odds refresh, catches late schedule changes |
| Mon 3 AM | `scrape_results` | OddsPortal (Playwright) | Scrapes weekend results → updates scores, sets `status=FINISHED` |

### Soccerdata Enrichment (inactive by default — must be enabled)

After results are in (Mon 3 AM), the enrichment step adds advanced metrics from
Understat and ESPN. This task is **off by default** — enable it once:

```bash
# Enable weekly enrichment (Mon 5 AM)
curl -X PATCH "http://localhost:8010/api/v1/schedules/weekly_stats_enrichment" \
  -H "Content-Type: application/json" \
  -d '{"is_active": true, "parameters": {"season": "2025"}}'
```

| Time | Task | Source | What Happens |
|------|------|--------|--------------|
| Mon 5 AM | `enrich_stats` | Understat + ESPN | Enriches `MatchStatistics` (xG, npxG, PPDA, deep completions) and `player_match_stats` (per-player goals, assists, shots, cards) |

**Manual one-off enrichment:**

```bash
# CLI — all stats (Understat + ESPN)
algobet import-data enrich "ENG-Premier League" --season 2025

# CLI — xG only
algobet import-data enrich-understat "ENG-Premier League" --season 2025

# CLI — player stats only
algobet import-data enrich-players "ENG-Premier League" --season 2025

# API
curl -X POST "http://localhost:8010/api/v1/scraping/import/enrich-stats?league=ENG-Premier%20League&season=2025"
```

### Manual (as needed)

```bash
# After new results are scraped
algobet train run --model-type xgboost          # Retrain model
algobet analyze calibrate                       # Recalibrate probabilities
algobet analyze backtest                        # Evaluate model
algobet analyze value-bets --min-ev 0.05        # Find betting opportunities

# Check upcoming matches
algobet list upcoming --days 3
curl http://localhost:8010/api/v1/predictions/upcoming
```

### Data Flow

```
  OddsPortal ──(scrape)──→ Match (odds, scores)
                                │
  Understat ──(enrich)──→ MatchStatistics (xG, npxG, PPDA, deep)
                                │
  ESPN ───────(enrich)──→ PlayerMatchStats (goals, assists, shots, cards)
                                │
                                ▼
                    Feature Pipeline ──→ Train Model ──→ Predictions
```

## Scheduled Tasks

Default scheduled tasks seeded into the database:

| Task Name | Type | Cron | Active |
|-----------|------|------|--------|
| `daily_upcoming_scrape_morning` | `scrape_upcoming` | `0 6 * * *` | Yes |
| `daily_upcoming_scrape_evening` | `scrape_upcoming` | `0 18 * * *` | Yes |
| `daily_predictions` | `generate_predictions` | `0 7 * * *` | Yes |
| `weekly_results_scrape` | `scrape_results` | `0 3 * * 1` | Yes |
| `weekly_stats_enrichment` | `enrich_stats` | `0 5 * * 1` | **No** |

Seed with: `python -m algobet.cli.seed_schedules`

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/algobet

# API
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Scheduler
ENABLE_SCHEDULER=false

# Model Paths
MODELS_PATH=data/models
```

## CLI Commands

| Command | Module | Purpose |
|---------|--------|---------|
| `algobet` | `algobet.cli.dev_tools` | Development tools (init, reset-db, stats) |
| `algobet-dev` | `algobet.cli.dev_tools` | Development tools alias |
| `algobet-scheduler` | `algobet.scheduler.worker` | APScheduler worker process |
| `algobet-runner` | `algobet.cli.scheduled_runner` | Run scheduled tasks manually |
| `algobet train` | `algobet.cli.commands.train` | ML model training commands |

## Contributing

1. Follow existing code conventions
2. Write comprehensive unit tests for new code
3. Ensure proper error handling and logging
4. Use type hints consistently
5. Run linting: `ruff check .`
6. Run type checking: `mypy algobet`

## License

MIT License - See LICENSE file for details

## Support

For questions or issues:
- Check the documentation in `/docs`
- Review [DEVELOPMENT_TASKS.md](DEVELOPMENT_TASKS.md) for current priorities
- Examine test files for usage examples
