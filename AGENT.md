# AlgoBet - AI Agent Guidelines

Football match database and OddsPortal scraper for historical data, betting odds, and team statistics. Full-stack platform with ML prediction capabilities.

## Tech Stack

### Backend
- **Python 3.10+** with type hints (`from __future__ import annotations`)
- **FastAPI** for REST API + WebSocket support
- **SQLAlchemy 2.0+** with PostgreSQL
- **Playwright** for web scraping
- **XGBoost/LightGBM** for ML predictions
- **APScheduler** for background tasks
- **Pydantic v2** for data validation

### Frontend
- **Next.js 15** App Router
- **TypeScript 5.3+**
- **React 19** with hooks
- **Tailwind CSS 3.4+**
- **shadcn/ui** components
- **TanStack Query** for data fetching
- **Zustand** for state management

### DevOps
- **Docker** + docker-compose
- **uv** for Python package management
- **pnpm** for Node.js packages
- **ruff** for linting, **mypy** for type checking
- **pytest** for testing (155+ tests)

## Project Structure

### Architecture Style

The project uses a **hybrid layered-feature architecture**:

- **Backend**: Organized by technical layers (routers, schemas, services) with feature-based modules within each layer
- **Frontend**: Feature-based organization with dedicated folders per domain (matches, predictions, scraping, schedules)

### Backend Structure

```
algobet/
├── api/                          # FastAPI REST API (layer-based)
│   ├── main.py                  # App entry point
│   ├── dependencies.py          # DI & DB sessions
│   ├── routers/                 # Feature-based route handlers
│   │   ├── matches.py           # Match CRUD & operations
│   │   ├── predictions.py       # Prediction generation & retrieval
│   │   ├── scraping.py          # Scraping jobs & control
│   │   ├── schedules.py         # Task scheduling
│   │   ├── ml_operations.py     # Model training & management
│   │   ├── teams.py
│   │   ├── tournaments.py
│   │   └── value_bets.py
│   ├── schemas/                 # Pydantic models (feature-based)
│   │   ├── match.py
│   │   ├── prediction.py
│   │   ├── scraping.py
│   │   ├── model.py
│   │   ├── team.py
│   │   └── tournament.py
│   └── websockets/              # WebSocket handlers
│       └── progress.py
├── cli/                          # CLI tools
│   ├── dev_tools.py             # Main CLI entry
│   └── commands/                # Command modules
│       └── train.py             # ML training commands
├── services/                     # Business logic (feature-based)
│   ├── base.py                  # Service base class
│   ├── prediction_service.py    # Prediction logic
│   ├── scraping_service.py      # Scraping orchestration
│   └── scheduler_service.py     # Task scheduling
├── predictions/                  # ML engine (self-contained feature)
│   ├── data/                    # Data queries
│   ├── features/                # Feature engineering
│   ├── models/                  # Model registry
│   ├── training/                # Training pipeline
│   └── evaluation/              # Model evaluation
├── scheduler/                    # APScheduler worker
├── config.py                     # Pydantic settings
├── models.py                     # SQLAlchemy ORM (all models)
├── scraper.py                    # OddsPortal scraper
└── database.py                   # DB connection
```

### Frontend Structure

```
frontend/
├── app/                          # Next.js pages (feature-based)
│   ├── layout.tsx               # Root layout with providers
│   ├── page.tsx                 # Dashboard
│   ├── matches/
│   │   ├── page.tsx             # Matches list
│   │   ├── loading.tsx
│   │   ├── error.tsx
│   │   └── [id]/                # Dynamic route for match details
│   ├── predictions/
│   ├── scraping/
│   ├── schedules/
│   ├── teams/
│   ├── models/
│   ├── value-bets/
│   ├── backtest/
│   └── calibrate/
├── components/                   # React components (feature-based)
│   ├── ui/                      # shadcn/ui primitives
│   ├── layout/                  # Layout components (Navbar, Sidebar)
│   ├── dashboard/               # Dashboard-specific components
│   ├── matches/                 # Match-related components
│   │   ├── MatchCard.tsx
│   │   ├── MatchFilters.tsx
│   │   └── MatchList.tsx
│   ├── schedules/               # Schedule components
│   ├── scraping/                # Scraping components
│   ├── charts/                  # Shared chart components
│   └── skeletons/               # Loading skeletons
├── lib/
│   ├── api/                     # API client functions
│   │   ├── client.ts            # HTTP client setup
│   │   ├── matches.ts
│   │   ├── predictions.ts
│   │   ├── scraping.ts
│   │   └── schedules.ts
│   ├── queries/                 # TanStack Query hooks
│   │   ├── use-matches.ts
│   │   ├── use-predictions.ts
│   │   ├── use-teams.ts
│   │   └── use-models.ts
│   └── types/                   # TypeScript type definitions
├── hooks/                        # Custom React hooks
└── stores/                       # Zustand state stores
```

### Feature Organization

| Feature | Backend Router | Backend Schema | Backend Service | Frontend Pages | Frontend Components |
|---------|---------------|----------------|-----------------|----------------|---------------------|
| Matches | `routers/matches.py` | `schemas/match.py` | Via repositories | `app/matches/` | `components/matches/` |
| Predictions | `routers/predictions.py` | `schemas/prediction.py` | `services/prediction_service.py` | `app/predictions/` | - |
| Scraping | `routers/scraping.py` | `schemas/scraping.py` | `services/scraping_service.py` | `app/scraping/` | `components/scraping/` |
| Schedules | `routers/schedules.py` | - | `services/scheduler_service.py` | `app/schedules/` | `components/schedules/` |
| ML Models | `routers/ml_operations.py` | `schemas/model.py` | `services/prediction_service.py` | `app/models/` | - |

## Adding New Features

### Backend: Add New Feature Module

When adding a new feature (e.g., `notifications`):

1. **Create ORM Model** (`models.py` or new file):
```python
class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    message: Mapped[str]
    is_read: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
```

2. **Create Pydantic Schema** (`schemas/notification.py`):
```python
from pydantic import BaseModel

class NotificationBase(BaseModel):
    message: str

class NotificationCreate(NotificationBase):
    pass

class NotificationResponse(NotificationBase):
    id: int
    is_read: bool

    model_config = {"from_attributes": True}
```

3. **Create Router** (`routers/notification.py`):
```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..dependencies import get_db

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])

@router.get("", response_model=list[NotificationResponse])
def list_notifications(db: Session = Depends(get_db)):
    ...
```

4. **Register Router** (`api/main.py`):
```python
from .routers import notifications

app.include_router(notifications.router)
```

5. **Add Service Logic** (if needed, `services/notification_service.py`):
```python
from .base import BaseService

class NotificationService(BaseService):
    def send_notification(self, user_id: int, message: str):
        ...
```

### Frontend: Add New Feature Page

1. **Create Page Folder** (`app/notifications/`):
```tsx
// app/notifications/page.tsx
"use client";

export default function NotificationsPage() {
  return (
    <div>
      <h1>Notifications</h1>
      {/* Page content */}
    </div>
  );
}
```

2. **Create API Client** (`lib/api/notifications.ts`):
```typescript
import { apiClient } from './client';

export async function getNotifications() {
  return apiClient.get('/api/v1/notifications');
}
```

3. **Create Query Hook** (`lib/queries/use-notifications.ts`):
```typescript
import { useQuery } from '@tanstack/react-query';
import { getNotifications } from '@/lib/api/notifications';

export function useNotifications() {
  return useQuery({
    queryKey: ['notifications'],
    queryFn: getNotifications,
  });
}
```

4. **Create Components** (`components/notifications/`):
```tsx
// components/notifications/NotificationList.tsx
export function NotificationList() {
  const { data } = useNotifications();
  // Component logic
}
```

5. **Add Navigation** (update sidebar/navbar in `components/layout/`)

### File Naming Conventions

- **Backend**: snake_case (`match_service.py`, `get_matches.py`)
- **Frontend**: PascalCase for components (`MatchCard.tsx`), kebab-case for hooks (`use-matches.ts`)
- **Schemas**: Singular (`match.py`, `prediction.py`)
- **Routers**: Plural (`matches.py`, `predictions.py`)

### Cross-Cutting Concerns

- **Authentication**: Add to `dependencies.py` if needed
- **Logging**: Use `logging_config.py` setup
- **Error Handling**: Use custom exceptions from `exceptions.py`
- **Type Safety**: All public APIs must have type hints

## Code Style & Standards

### Python Guidelines

```python
# Use Python 3.10+ type hints
from __future__ import annotations
from typing import TYPE_CHECKING

# SQLAlchemy 2.0+ patterns
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase

class Base(DeclarativeBase):
    pass

class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
```

### Key Patterns

- **Type hints**: Required on all function signatures
- **f-strings**: For all string formatting
- **Function length**: Keep under 50 lines; extract helpers if longer
- **Docstrings**: Google style on all public classes/functions
- **Click decorators**: `@click.group()`, `@cli.command()`, `@click.option()`
- **Session management**: Use `session_scope()` context manager

### Anti-Patterns to Avoid

**GOD Modules**
- Split files >500 lines into focused modules
- Use composition over inheritance
- Single responsibility per class

**DRY Violations**
- Get-or-create patterns must be in repositories
- No duplicated calculation logic
- Centralize URL parsing logic

**Magic Numbers**
- Name and document all constants
- Use `config.py` with pydantic-settings
- Never hard-code URLs, paths, or timeouts

## Architecture Layers

### Layer Responsibilities

| Layer | Purpose | Rules |
|-------|---------|-------|
| **API** | HTTP request/response | No business logic. Routers delegate to services |
| **CLI** | Command-line interface | No business logic. Commands call services |
| **Service** | Business logic | Uses repositories. Manages transactions |
| **Repository** | Data access | Encapsulates queries. No transaction management |

### Service Layer Principles

1. **Single Responsibility**: One domain area per service
2. **No Direct Queries**: Always use repositories
3. **Transaction Management**: Services manage transactions
4. **Stateless by Default**: No class-level mutable state
5. **Domain Exceptions**: Custom exceptions, not generic ones

## Database Schema

### Core Tables

| Table | Unique Constraints | Key Fields |
|-------|-------------------|------------|
| `tournaments` | `url_slug` | id, name, country, url_slug |
| `seasons` | `(tournament_id, name)` | id, tournament_id, name, start_year, end_year |
| `teams` | `name` | id, name |
| `matches` | Composite key | id, home/away_team_id, match_date, scores, odds, status |
| `predictions` | - | id, match_id, model_version, probabilities, confidence |
| `model_versions` | `version` | id, version, algorithm, accuracy, is_active |
| `scheduled_tasks` | `name` | id, name, cron_expression, is_active |
| `task_executions` | - | id, task_id, status, started_at, completed_at |

### Match Status Values

- `SCHEDULED` - Upcoming match, scores are None
- `FINISHED` - Completed match with scores
- `LIVE` - In progress

### Common Patterns

- **Season naming**: `"YYYY/YYYY+1"` format (e.g., `"2023/2024"`)
- **URL parsing**: `r"/football/([^/]+)/([^/]+?)(?:-\d{4}-\d{4})?/results/"`
- **Deduplication**: Check all key fields before insert

## Environment Variables

```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=algobet
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Scraper
ODDSPORTAL_BASE_URL=https://www.oddsportal.com
SCRAPER_HEADLESS=true
SCRAPER_TIMEOUT_MS=30000

# ML
MODELS_PATH=data/models

# API
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Scheduler
ENABLE_SCHEDULER=false
```

## Development Commands

### Backend Setup

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run playwright install chromium

# Run server
uvicorn algobet.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
# Full stack
docker-compose -f docker-compose.yml -f docker-compose.scheduler.yml up -d

# Backend only
make backend

# View logs
make logs
```

### Testing

```bash
# Backend tests
pytest
pytest --cov=algobet --cov-report=html

# Type checking
mypy algobet

# Linting
ruff check .
ruff format .
```

### CLI Tools

```bash
# Initialize database
algobet init

# Reset database (destructive)
algobet reset-db --yes

# Database statistics
algobet db-stats

# Train ML model
algobet train run --model-type xgboost --tune

# Run scheduled task manually
algobet-runner --task daily-upcoming-scrape
```

## API Endpoints

### Scraping
- `POST /api/v1/scraping/upcoming` - Scrape upcoming matches
- `POST /api/v1/scraping/results` - Scrape historical results
- `GET /api/v1/scraping/jobs/{job_id}` - Get job status
- `GET /api/v1/scraping/jobs` - List all jobs

### Predictions
- `POST /api/v1/predictions/generate` - Generate predictions
- `GET /api/v1/predictions` - Get predictions
- `GET /api/v1/value-bets` - Get value bets

### Schedules
- `POST /api/v1/schedules` - Create scheduled task
- `GET /api/v1/schedules` - List schedules
- `POST /api/v1/schedules/{id}/run` - Run task immediately
- `GET /api/v1/schedules/{id}/history` - Get execution history

### WebSocket
- `WS /ws/scraping/{job_id}` - Real-time scraping progress

## Code Review Checklist

- [ ] No magic numbers - all constants named and documented
- [ ] No hard-coded URLs/paths - use configuration
- [ ] No duplicated logic - extract to shared functions
- [ ] Functions under 50 lines
- [ ] Services use repositories for data access
- [ ] No class-level mutable state in services
- [ ] Domain-specific exceptions, not generic
- [ ] Type hints on all public functions
- [ ] Docstrings on public classes/functions
- [ ] Tests for new functionality
- [ ] Follows existing patterns in codebase

## Scheduled Tasks

Default tasks (configurable via API/database):

| Task | Schedule | Description |
|------|----------|-------------|
| `daily-upcoming-scrape` | 6:00 AM daily | Scrape upcoming matches |
| `evening-upcoming-scrape` | 6:00 PM daily | Evening scrape run |
| `daily-predictions` | 7:00 AM daily | Generate daily predictions |
| `weekly-results-scrape` | Monday 3:00 AM | Scrape weekend results |

## Documentation References

- [Development Tasks](DEVELOPMENT_TASKS.md) - Sprint planning
- [Frontend Plan](docs/frontend_development_plan.md) - Frontend architecture
- [Prediction Engine](docs/prediction_engine_architecture.md) - ML system design
- [IFLOW](IFLOW.md) - Information flow documentation
