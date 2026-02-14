# AlgoBet Development Tasks

This document organizes the development tasks into actionable sprints. The project has been refactored to a service-oriented architecture with API-first design.

## Project Status

- **Backend**: ✅ Service layer, database models, scraper, prediction engine
- **API Layer**: ✅ FastAPI endpoints with full CRUD operations
- **Frontend**: ✅ Next.js 15 with core pages completed
- **Database**: ✅ PostgreSQL with migrations
- **Scheduling**: ✅ APScheduler integration for automated tasks
- **Testing**: ✅ 155 tests passing

### Current Sprint
**Sprint 4: Advanced Features** - ✅ Completed (Tasks 4.1-4.9)

### Next Sprint
**Sprint 5: Polish & Testing** (Tasks 5.1-5.10)

### Recently Completed (Refactoring)
- ✅ Service Layer Foundation (BaseService, PredictionService, ScrapingService, SchedulerService)
- ✅ API Scraping Endpoints with WebSocket progress updates
- ✅ Frontend Scraping Dashboard with real-time progress
- ✅ Frontend Schedules Management page
- ✅ CLI Deprecation (moved to dev tools)
- ✅ Automated/Scheduled Scraping with APScheduler
- ✅ All tests passing (155/155)

---

## Architecture Overview

### Service Layer (`algobet/services/`)
| Service | Purpose | Status |
|---------|---------|--------|
| `base.py` | Abstract base service with session management | ✅ |
| `prediction_service.py` | ML prediction logic | ✅ |
| `scraping_service.py` | OddsPortal scraping orchestration | ✅ |
| `scheduler_service.py` | APScheduler task management | ✅ |

### API Layer (`algobet/api/`)
| Router | Endpoints | Status |
|--------|-----------|--------|
| `/api/v1/tournaments` | Tournament CRUD | ✅ |
| `/api/v1/seasons` | Season management | ✅ |
| `/api/v1/teams` | Team information | ✅ |
| `/api/v1/matches` | Match data and filtering | ✅ |
| `/api/v1/predictions` | Prediction management | ✅ |
| `/api/v1/models` | ML model management | ✅ |
| `/api/v1/value-bets` | Value bet identification | ✅ |
| `/api/v1/scraping` | Scraping operations with WebSocket | ✅ |
| `/api/v1/schedules` | Scheduled task management | ✅ |

### Frontend Structure (`frontend/`)
| Directory | Purpose | Status |
|-----------|---------|--------|
| `app/` | Next.js App Router pages | ✅ |
| `components/ui/` | shadcn/ui components | ✅ |
| `components/layout/` | Navbar, Sidebar, Breadcrumb | ✅ |
| `components/matches/` | Match-related components | ✅ |
| `components/scraping/` | Scraping dashboard components | ✅ |
| `components/schedules/` | Schedule management components | ✅ |
| `lib/api/` | API client functions | ✅ |
| `lib/queries/` | TanStack Query hooks | ✅ |
| `lib/types/` | TypeScript types and Zod schemas | ✅ |
| `stores/` | Zustand stores | ✅ |
| `hooks/` | Custom hooks (useScrapingProgress) | ✅ |

### CLI Entry Points (`pyproject.toml`)
| Command | Module | Purpose |
|---------|--------|---------|
| `algobet` | `algobet.cli.dev_tools` | Development tools |
| `algobet-dev` | `algobet.cli.dev_tools` | Development tools alias |
| `algobet-scheduler` | `algobet.scheduler.worker` | Scheduler worker |
| `algobet-runner` | `algobet.cli.scheduled_runner` | Task runner |

---

## Sprint 1: Foundation ✅ Completed

### Backend API Layer

- [x] **1.1** Add FastAPI dependencies to pyproject.toml
- [x] **1.2** Create algobet/api directory structure
- [x] **1.3** Implement FastAPI main.py with CORS and lifespan
- [x] **1.4** Create dependencies.py with session injection
- [x] **1.5** Create Pydantic schemas for all models
- [x] **1.6** Implement tournaments router
- [x] **1.7** Implement teams router
- [x] **1.8** Implement matches router
- [x] **1.9** Implement predictions router
- [x] **1.10** Implement models router
- [x] **1.11** Implement value-bets endpoint
- [x] **1.12** Add environment variables
- [x] **1.13** Update docker-compose.yml

### Frontend Setup

- [x] **1.14** Initialize Next.js 15 project
- [x] **1.15** Install shadcn/ui
- [x] **1.16** Configure Tailwind CSS
- [x] **1.17** Set up QueryProvider and ThemeProvider
- [x] **1.18** Create base layout (Navbar, Sidebar, Breadcrumb)
- [x] **1.19** Create API client with error handling
- [x] **1.20** Create TypeScript types
- [x] **1.21** Create Zod schemas
- [x] **1.22** Create TanStack Query hooks
- [x] **1.23** Set up Zustand stores

---

## Sprint 2: Core Pages ✅ Completed

- [x] **2.1** Create Dashboard page (/)
- [x] **2.2** Create Matches page (/matches)
- [x] **2.3** Create Match Detail page (/matches/[id])
- [x] **2.4** Create skeleton components
- [x] **2.5** Add error boundaries
- [x] **2.6** Create loading.tsx files

---

## Sprint 3: Predictions & Models ✅ Completed

- [x] **3.1** Create Predictions page (/predictions)
- [x] **3.2** Create Models page (/models)
- [x] **3.3** Create prediction components (PredictionCard, ConfidenceBadge)
- [x] **3.4** Create model components (ModelSelector, MetricsTable)

---

## Sprint 4: Advanced Features ✅ Completed

### Scraping Dashboard

- [x] **4.1** Create Scraping page (/scraping)
  - ScrapeForm for upcoming/results
  - JobHistoryTable
  - Real-time progress via WebSocket

- [x] **4.2** Create scraping components
  - ScrapingJobCard
  - ScrapingProgress
  - ScrapeForm
  - JobHistoryTable

- [x] **4.3** Create WebSocket hook (useScrapingProgress)
  - Real-time progress updates
  - Connection management
  - Auto-reconnection

### Scheduling System

- [x] **4.4** Create Schedules page (/schedules)
  - Schedule list with CRUD
  - Execution history
  - Run now functionality

- [x] **4.5** Create schedule components
  - ScheduleCard
  - ScheduleForm
  - ExecutionHistory

- [x] **4.6** Create schedules API client (`lib/api/schedules.ts`)

### Additional Pages

- [x] **4.7** Create Value Bets page (/value-bets)
- [x] **4.8** Create Teams page (/teams)

---

## Sprint 5: Polish & Testing

### Testing Setup

- [ ] **5.1** Set up Vitest for frontend unit tests
- [ ] **5.2** Write unit tests for utilities and hooks
- [ ] **5.3** Write integration tests for components
- [ ] **5.4** Set up Playwright for E2E tests
- [ ] **5.5** Write E2E tests for critical flows

### Polish & Optimization

- [ ] **5.6** Responsive design fixes
- [ ] **5.7** Performance optimization (React.memo, lazy loading)
- [ ] **5.8** Configure build tools (ESLint, TypeScript)
- [ ] **5.9** Accessibility audit (WCAG 2.1 AA)

### Documentation

- [ ] **5.10** Update README.md with new architecture

---

## Deployment & CI/CD

### Continuous Integration

- [ ] **D1** Create GitHub Actions workflow - Frontend CI
- [ ] **D2** Create GitHub Actions workflow - Backend CI
- [ ] **D3** Create GitHub Actions workflow - Deploy

### Type Generation

- [ ] **D4** Implement OpenAPI type generation

---

## Future Enhancements

- [ ] **F1** User authentication
- [ ] **F2** PWA capabilities
- [ ] **F3** Mobile app (React Native)
- [ ] **F4** PDF report generation
- [ ] **F5** Advanced analytics (A/B testing, ROI tracking)
- [ ] **F6** Social features (share predictions, leaderboards)
- [ ] **F7** Push notifications for value bets
- [ ] **F8** User data import/export
- [ ] **F9** Multi-language support (i18n)
- [ ] **F10** Offline support with service worker

---

## Task Status Legend

- ✅ Complete
- 🚧 In Progress
- ⏳ Blocked
- ❌ Failed
- ⬜ Not Started

---

## Prediction Engine Roadmap

Detailed architecture in [prediction_engine_architecture.md](docs/prediction_engine_architecture.md) and [ml_model_design.md](docs/ml_model_design.md).

### Current State

| Component | Status | Location |
|-----------|--------|----------|
| Query Repository | ✅ | `algobet/predictions/data/queries.py` |
| Form Calculator | ✅ | `algobet/predictions/features/form_features.py` |
| Model Registry | ✅ | `algobet/predictions/models/registry.py` |
| Prediction Service | ✅ | `algobet/services/prediction_service.py` |
| Scheduler Service | ✅ | `algobet/services/scheduler_service.py` |
| API Routers | ✅ | `algobet/api/routers/` |

### Phase PE-1: Feature Engineering Pipeline

**Priority: High | Estimated: 1-2 weeks**

- [ ] **PE-1.1** Create feature generators module
- [ ] **PE-1.2** Create feature transformers
- [ ] **PE-1.3** Create feature pipeline orchestrator
- [ ] **PE-1.4** Create feature store

### Phase PE-2: Model Training Pipeline

**Priority: High | Estimated: 2 weeks**

- [ ] **PE-2.1** Create temporal data splitting
- [ ] **PE-2.2** Create training orchestrator
- [ ] **PE-2.3** Implement XGBoost classifier
- [ ] **PE-2.4** Implement LightGBM classifier
- [ ] **PE-2.5** Implement Random Forest classifier
- [ ] **PE-2.6** Create hyperparameter tuning
- [ ] **PE-2.7** Implement probability calibration

### Phase PE-3: Evaluation Engine

**Priority: Medium | Estimated: 1 week**

- [ ] **PE-3.1** Create metrics module
- [ ] **PE-3.2** Create calibration analysis
- [ ] **PE-3.3** Create report generation

### Phase PE-4: Database Schema Extensions

**Priority: Medium | Estimated: 2-3 days**

- [x] **PE-4.1** predictions table exists
- [x] **PE-4.2** model_features via JSONB in predictions

### Phase PE-5: CLI Enhancements

**Priority: Low | Estimated: 1 week**

- [ ] **PE-5.1** Add backtest command
- [ ] **PE-5.2** Add value-bets command
- [ ] **PE-5.3** Add calibrate command

---

### Performance Targets

| Metric | Target | Baseline (Bookmaker) |
|--------|--------|---------------------|
| Top-1 Accuracy | 50-55% | ~48% |
| Log Loss | < 0.95 | ~1.05 |
| Brier Score | < 0.20 | ~0.21 |
| ROI (Value Bets) | 5-10% | 0% |
| Calibration ECE | < 0.05 | N/A |

---

## Notes

- All API endpoints integrate with existing modules
- Follow existing code conventions and patterns
- Test locally before committing
- Keep frontend and backend types in sync
- Use `algobet` command for dev tools, `algobet-scheduler` for worker