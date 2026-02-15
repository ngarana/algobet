# AlgoBet - Football Match Prediction & Betting Analytics Platform

A comprehensive full-stack application for scraping, analyzing, and predicting football match outcomes using machine learning. Features a modern React frontend, FastAPI backend, and automated scheduling system.

## Features

### Core Capabilities
- 📊 **Database Management**: PostgreSQL with SQLAlchemy ORM for tournaments, seasons, teams, matches, and predictions
- 🤖 **Machine Learning**: XGBoost/LightGBM ensemble models for match outcome prediction with probability calibration
- 🌐 **Web Scraper**: Playwright-based scraper for OddsPortal with real-time progress tracking
- 🎯 **Value Bet Detection**: Automated identification of profitable betting opportunities
- 📅 **Automated Scheduling**: APScheduler integration for daily scraping and predictions
- 🔌 **Real-time Updates**: WebSocket support for live scraping progress and match updates

### Frontend Features
- Modern React dashboard with Next.js 15 App Router
- Real-time scraping job monitoring with WebSocket updates
- Interactive match analysis with team form visualization
- Prediction confidence badges and value bet indicators
- Schedule management UI for automated tasks
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
│  - Feature eng.      │    scraper         │  - Cron execution  │
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
- **Scraping**: Playwright
- **Scheduling**: APScheduler
- **Testing**: pytest, pytest-asyncio (155 tests passing)

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
- Playwright browsers

### Backend Setup

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run playwright install chromium

# Or using pip
pip install -e ".[dev]"
playwright install chromium
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
docker-compose -f docker-compose.yml -f docker-compose.scheduler.yml up -d

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
```

### API Endpoints

#### Scraping Operations
```bash
# Scrape upcoming matches (runs in background)
curl -X POST "http://localhost:8000/api/v1/scraping/upcoming" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.oddsportal.com/matches/football/"}'

# Scrape historical results
curl -X POST "http://localhost:8000/api/v1/scraping/results" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.oddsportal.com/football/england/premier-league/results/", "max_pages": 5}'

# Check scraping job status
curl "http://localhost:8000/api/v1/scraping/jobs/{job_id}"

# List all jobs
curl "http://localhost:8000/api/v1/scraping/jobs"
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
    "name": "daily-scrape",
    "task_type": "scrape_upcoming",
    "cron_expression": "0 6 * * *",
    "config": {"url": "https://www.oddsportal.com/matches/football/"}
  }'

# List schedules
curl "http://localhost:8000/api/v1/schedules"

# Run task immediately
curl -X POST "http://localhost:8000/api/v1/schedules/{id}/run"

# Get execution history
curl "http://localhost:8000/api/v1/schedules/{id}/history"
```

### WebSocket Connection

Connect to WebSocket for real-time scraping progress:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/scraping/{job_id}');

ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Status: ${progress.status}`);
  console.log(`Progress: ${progress.current_page}/${progress.total_pages}`);
  console.log(`Matches: ${progress.matches_scraped} scraped, ${progress.matches_saved} saved`);
};
```

## Project Structure

```
algobet/
├── api/                      # FastAPI application
│   ├── main.py              # FastAPI app entry point
│   ├── dependencies.py      # DB session injection
│   ├── routers/             # API route handlers
│   │   ├── matches.py
│   │   ├── predictions.py
│   │   ├── scraping.py
│   │   ├── schedules.py
│   │   └── ...
│   ├── schemas/             # Pydantic models
│   └── websockets/          # WebSocket handlers
│       └── progress.py
├── services/                 # Business logic layer
│   ├── base.py              # Base service class
│   ├── prediction_service.py
│   ├── scraping_service.py
│   └── scheduler_service.py
├── predictions/              # ML prediction engine
│   ├── data/                # Data queries
│   ├── features/            # Feature engineering
│   ├── models/              # Model registry
│   └── training/            # Training pipeline
├── cli/                      # Development CLI tools
│   └── dev_tools.py
├── scheduler/                # APScheduler worker
│   └── worker.py
├── models.py                 # SQLAlchemy ORM models
├── scraper.py                # OddsPortal scraper
└── database.py               # Database connection

frontend/
├── app/                      # Next.js App Router pages
│   ├── page.tsx             # Dashboard
│   ├── matches/
│   ├── predictions/
│   ├── scraping/
│   └── schedules/
├── components/               # React components
│   ├── ui/                  # shadcn/ui components
│   ├── matches/
│   ├── predictions/
│   ├── scraping/
│   └── schedules/
├── lib/
│   ├── api/                 # API client functions
│   ├── queries/             # TanStack Query hooks
│   ├── types/               # TypeScript types
│   └── utils/
├── hooks/                    # Custom React hooks
└── stores/                   # Zustand stores
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

**Test Coverage**: 155 tests passing, >85% code coverage

## Scheduled Tasks

Default scheduled tasks (configurable via API or database):

| Task | Schedule | Description |
|------|----------|-------------|
| daily-upcoming-scrape | 6:00 AM daily | Scrape upcoming matches |
| evening-upcoming-scrape | 6:00 PM daily | Scrape upcoming matches |
| daily-predictions | 7:00 AM daily | Generate predictions |
| weekly-results-scrape | Monday 3:00 AM | Scrape weekend results |

## Documentation

- [Development Tasks](DEVELOPMENT_TASKS.md) - Sprint planning and task tracking
- [Refactoring Roadmap](refactor-todo.md) - Completed refactoring details
- [Frontend Development Plan](docs/frontend_development_plan.md) - Frontend architecture
- [Prediction Engine Architecture](docs/prediction_engine_architecture.md) - ML system design
- [ML Model Design](docs/ml_model_design.md) - Machine learning specifications

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
