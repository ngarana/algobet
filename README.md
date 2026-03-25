# AlgoBet - Football Match Prediction & Betting Analytics Platform

A comprehensive full-stack application for fetching, analyzing, and predicting football match outcomes using machine learning. Features a modern React frontend, FastAPI backend, and automated scheduling system.

## Features

### Core Capabilities
- 📊 **Database Management**: PostgreSQL with SQLAlchemy ORM for tournaments, seasons, teams, matches, and predictions
- 🤖 **Machine Learning**: XGBoost/LightGBM ensemble models for match outcome prediction with probability calibration
- ⚽ **API-Football Integration**: Reliable JSON API for fixtures, results, and betting odds (no web scraping)
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
│  - Model inference   │  - API-Football    │  - Task CRUD       │
│  - Feature eng.      │    client          │  - Cron execution  │
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
- **Data Source**: [API-Football](https://www.api-football.com/) (free tier: 100 requests/day)
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
- [API-Football API key](https://dashboard.api-football.com/register) (free)

### Backend Setup

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Configure API-Football

1. Register at https://dashboard.api-football.com/register (free)
2. Copy your API key from the dashboard
3. Add to `.env`:

```bash
# API-Football Configuration
ALGOBET_API_FOOTBALL__API_KEY=your_api_key_here
ALGOBET_API_FOOTBALL__BASE_URL=https://v3.football.api-sports.io
ALGOBET_API_FOOTBALL__RATE_LIMIT_PER_DAY=100
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
# Set API key in environment
export ALGOBET_API_FOOTBALL__API_KEY=your_api_key_here

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

#### Fetching Matches (API-Football)

```bash
# Fetch upcoming matches for specific leagues
# League IDs: 39=Premier League, 140=La Liga, 135=Serie A, 78=Bundesliga, 61=Ligue 1
curl -X POST "http://localhost:8000/api/v1/scraping/upcoming?league_ids=39,140,135"

# Fetch match results for a specific league
curl -X POST "http://localhost:8000/api/v1/scraping/results?league_id=39&max_results=20"

# Check job status
curl "http://localhost:8000/api/v1/scraping/jobs/{job_id}"

# List all jobs
curl "http://localhost:8000/api/v1/scraping/jobs"

# Get scraping statistics
curl "http://localhost:8000/api/v1/scraping/stats"
```

#### Popular League IDs

| League | ID |
|--------|-----|
| Premier League | 39 |
| La Liga | 140 |
| Serie A | 135 |
| Bundesliga | 78 |
| Ligue 1 | 61 |
| Champions League | 2 |
| Europa League | 3 |
| Eredivisie | 886 |

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

## Project Structure

```
algobet/
├── api/                          # FastAPI application
│   ├── main.py                  # FastAPI app entry point
│   ├── dependencies.py          # DB session injection
│   ├── routers/                 # API route handlers
│   │   ├── matches.py
│   │   ├── predictions.py
│   │   ├── scraping.py
│   │   ├── schedules.py
│   │   └── ...
│   ├── schemas/                 # Pydantic models
│   └── websockets/              # WebSocket handlers
│       └── progress.py
├── services/                     # Business logic layer
│   ├── base.py                  # Base service class
│   ├── prediction_service.py
│   ├── scraping_service.py      # Uses API-Football client
│   └── scheduler_service.py
├── infrastructure/               # External integrations
│   ├── api_football_client.py   # API-Football client
│   ├── config.py                # Configuration management
│   ├── database.py              # Database connection
│   └── scraper.py               # Legacy web scraper (deprecated)
├── predictions/                  # ML prediction engine
│   ├── data/                    # Data queries
│   ├── features/                # Feature engineering
│   ├── models/                  # Model registry
│   └── training/                # Training pipeline
├── cli/                          # Development CLI tools
│   ├── dev_tools.py
│   └── commands/
│       ├── train.py              # ML training commands
│       └── ...
├── scheduler/                    # APScheduler worker
│   └── worker.py
├── matches/models.py             # Match ORM model
├── teams/models.py               # Team/Tournament ORM models
└── predictions/models.py         # Prediction ORM model

frontend/
├── app/                          # Next.js App Router pages
│   ├── page.tsx                 # Dashboard
│   ├── matches/
│   ├── predictions/
│   ├── scraping/
│   └── schedules/
├── components/                   # React components
│   ├── ui/                      # shadcn/ui components
│   ├── matches/
│   ├── predictions/
│   ├── scraping/
│   └── schedules/
├── lib/
│   ├── api/                     # API client functions
│   │   └── scraping.ts          # API-Football client
│   ├── queries/                 # TanStack Query hooks
│   ├── types/                   # TypeScript types
│   └── utils/
├── hooks/                        # Custom React hooks
└── stores/                       # Zustand stores
```

## Database Schema

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| tournaments | League/tournament info | id, api_football_id, name, country, url_slug |
| seasons | Season records | id, tournament_id, name, start_year, end_year |
| teams | Team information | id, api_football_id, name |
| matches | Match records | id, api_football_id, home/away_team_id, match_date, scores, odds, status |
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

## Scheduled Tasks

Default scheduled tasks (configurable via API or database):

| Task | Schedule | Description |
|------|----------|-------------|
| daily-upcoming-fetch | 6:00 AM daily | Fetch upcoming matches from API-Football |
| evening-upcoming-fetch | 6:00 PM daily | Fetch upcoming matches from API-Football |
| daily-predictions | 7:00 AM daily | Generate predictions |
| weekly-results-fetch | Monday 3:00 AM | Fetch weekend results |

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/algobet

# API
API_HOST=0.0.0.0
API_PORT=8000

# API-Football (required)
ALGOBET_API_FOOTBALL__API_KEY=your_api_key_here
ALGOBET_API_FOOTBALL__BASE_URL=https://v3.football.api-sports.io
ALGOBET_API_FOOTBALL__RATE_LIMIT_PER_DAY=100
ALGOBET_API_FOOTBALL__TIMEOUT=30

# Default league IDs to fetch
ALGOBET_SCRAPING__DEFAULT_LEAGUE_IDS=[39,140,135,78,61]

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

## API-Football Rate Limits

| Plan | Requests/Day | Price |
|------|--------------|-------|
| Free | 100 | $0 |
| Pro | 4,500 | $10/month |
| Ultra | 75,000 | $30/month |

The free tier is sufficient for daily use (fetching upcoming matches + results once or twice per day).

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
- [API-Football Documentation](https://www.api-football.com/documentation-v3)
