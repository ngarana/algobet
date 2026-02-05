# AlgoBet Development Tasks

This document organizes the frontend development tasks into actionable sprints based on the [Frontend Development Plan](docs/frontend_development_plan.md).

## Project Status

- **Backend**: ✅ CLI, database models, scraper, prediction engine exist
- **API Layer**: ❌ FastAPI endpoints need to be created (Phase 1 - Prerequisite)
- **Frontend**: ❌ Next.js project needs to be initialized
- **Database**: ✅ PostgreSQL configured in docker-compose.yml

---

## Sprint 1: Foundation (Week 1)

### Backend API Layer (Phase 1)

- [ ] **1.1** Add FastAPI dependencies to pyproject.toml
  - fastapi>=0.104.0
  - uvicorn[standard]>=0.24.0
  - pydantic>=2.5.0
  - pydantic-settings>=2.1.0

- [ ] **1.2** Create algobet/api directory structure
  ```
  algobet/api/
  ├── __init__.py
  ├── main.py
  ├── dependencies.py
  ├── routers/
  │   ├── __init__.py
  │   ├── matches.py
  │   ├── tournaments.py
  │   ├── teams.py
  │   ├── predictions.py
  │   ├── models.py
  │   └── metrics.py
  ├── schemas/
  │   ├── __init__.py
  │   ├── match.py
  │   ├── team.py
  │   ├── tournament.py
  │   ├── prediction.py
  │   └── model.py
  └── middleware.py
  ```

- [ ] **1.3** Implement FastAPI main.py
  - FastAPI app entry point
  - CORS middleware configuration
  - Include all routers

- [ ] **1.4** Create dependencies.py
  - Database session dependency injection
  - Integrate with existing session_scope()

- [ ] **1.5** Create Pydantic schemas
  - TournamentResponse, SeasonResponse
  - TeamResponse, TeamWithStatsResponse
  - MatchResponse, MatchDetailResponse
  - PredictionResponse, PredictionWithMatchResponse
  - ModelVersionResponse
  - Align with existing SQLAlchemy models

- [ ] **1.6** Implement tournaments router
  - GET /api/v1/tournaments (list all)
  - GET /api/v1/tournaments/{id} (details)
  - GET /api/v1/tournaments/{id}/seasons

- [ ] **1.7** Implement teams router
  - GET /api/v1/teams (list with search)
  - GET /api/v1/teams/{id} (details)
  - GET /api/v1/teams/{id}/form (FormCalculator integration)
  - GET /api/v1/teams/{id}/matches (MatchRepository integration)

- [ ] **1.8** Implement matches router
  - GET /api/v1/matches (list with filters)
  - GET /api/v1/matches/{id} (details)
  - GET /api/v1/matches/{id}/preview (form analysis)
  - GET /api/v1/matches/{id}/predictions
  - GET /api/v1/matches/{id}/h2h (MatchRepository.get_h2h_matches)

- [ ] **1.9** Implement predictions router
  - GET /api/v1/predictions (list with filters)
  - POST /api/v1/predictions/generate (batch generation)
  - GET /api/v1/predictions/upcoming
  - GET /api/v1/predictions/history

- [ ] **1.10** Implement models router
  - GET /api/v1/models (list all versions)
  - GET /api/v1/models/active (currently active)
  - GET /api/v1/models/{id} (details)
  - POST /api/v1/models/{id}/activate (ModelRegistry integration)
  - DELETE /api/v1/models/{id}
  - GET /api/v1/models/{id}/metrics

- [ ] **1.11** Implement value-bets endpoint
  - GET /api/v1/value-bets (find value opportunities)

- [ ] **1.12** Add environment variables
  - Update .env with API configuration
  - DATABASE_URL, API_HOST, API_PORT, CORS_ORIGINS

- [ ] **1.13** Update docker-compose.yml
  - Add API service with uvicorn
  - Configure dependencies on db service

### Frontend Setup (Phase 2)

- [ ] **1.14** Initialize Next.js 15 project
  ```bash
  npx create-next-app@latest frontend --typescript --tailwind --app
  ```
  - Configure TypeScript strict mode
  - Enable App Router

- [ ] **1.15** Install shadcn/ui
  ```bash
  npx shadcn-ui@latest init
  ```
  - Select components as needed (Button, Card, Badge, Table, etc.)

- [ ] **1.16** Configure Tailwind CSS
  - Add custom dark mode theme colors
  - Set up color scheme (background, foreground, primary, success, warning, danger)

- [ ] **1.17** Set up providers
  - Create QueryProvider (TanStack Query)
  - Create ThemeProvider (next-themes for dark mode)

- [ ] **1.18** Create base layout
  - Root layout.tsx with providers
  - Navbar component
  - Sidebar component
  - Breadcrumb component

- [ ] **1.19** Create API client
  - lib/api/client.ts with error handling
  - Configure queryClient with default options

- [ ] **1.20** Create TypeScript types
  - lib/types/api.ts (Match, Tournament, Team, Prediction, ModelVersion, etc.)
  - Query parameter types (MatchFilters, PredictionFilters)

- [ ] **1.21** Create Zod schemas
  - lib/types/schemas.ts for runtime validation
  - Schemas for all API responses

- [ ] **1.22** Create TanStack Query hooks
  - lib/queries/use-matches.ts
  - lib/queries/use-predictions.ts
  - lib/queries/use-models.ts
  - lib/queries/use-value-bets.ts

- [ ] **1.23** Set up Zustand stores
  - stores/filter-store.ts (global filter state)
  - stores/ui-store.ts (UI state like sidebar)

- [ ] **1.24** Set up MSW for API mocking
  - mocks/handlers.ts
  - mocks/browser.ts
  - mocks/server.ts
  - mocks/data/ (match, prediction, model fixtures)

- [ ] **1.25** Create utility functions
  - lib/utils/cn.ts (Tailwind class merger)
  - lib/utils/format.ts (date formatting)
  - lib/utils/odds.ts (odds calculations)

---

## Sprint 2: Core Pages (Week 2)

### Pages & Components (Phase 3)

- [ ] **2.1** Create Dashboard page (/)
  - Upcoming matches component
  - Active model card
  - Value bets summary
  - Prediction accuracy chart
  - Parallel data fetching with Promise.all

- [ ] **2.2** Create Matches page (/matches)
  - MatchFilters component
  - MatchList component with infinite scroll
  - URL state synchronization for filters
  - Loading skeletons

- [ ] **2.3** Create Match Detail page (/matches/[id])
  - Match info display (teams, date, venue)
  - H2H table (last 5 matches)
  - Team form charts
  - Odds comparison
  - Prediction card with probability breakdown
  - Value bet indicator

- [ ] **2.4** Create skeleton components
  - MatchListSkeleton
  - UpcomingMatchesSkeleton
  - ValueBetsSkeleton
  - ModelCardSkeleton
  - ChartSkeleton
  - FiltersSkeleton

- [ ] **2.5** Add error boundaries
  - Route-level error.tsx files
  - Reusable ErrorBoundary component

- [ ] **2.6** Create loading.tsx files
  - Loading states for all routes

---

## Sprint 3: Predictions & Models (Week 3)

### Predictions & Models Pages

- [ ] **3.1** Create Predictions page (/predictions)
  - Prediction history table
  - Filter by model, date, outcome
  - Accuracy metrics dashboard
  - Export functionality (CSV/JSON)

- [ ] **3.2** Create Upcoming Predictions page (/predictions/upcoming)
  - Model version selector
  - Match filters (tournament, days ahead)
  - Batch prediction generation workflow
  - Results review

- [ ] **3.3** Create Models page (/models)
  - Model registry listing
  - Algorithm type, creation date, metrics display
  - Active model toggle
  - Promote/deactivate actions

- [ ] **3.4** Create Model Detail page (/models/[id])
  - Feature importance chart
  - Confusion matrix heatmap
  - ROC curves
  - Calibration plot
  - Metrics time series
  - ROI curve

### Prediction Components

- [ ] **3.5** Create prediction components
  - PredictionCard
  - ConfidenceBadge
  - ValueBetCard
  - ProbabilityBar

### Model Components

- [ ] **3.6** Create model components
  - ModelSelector
  - MetricsTable
  - ModelActivationToggle

---

## Sprint 4: Advanced Features (Week 4)

### Additional Pages & Features

- [ ] **4.1** Create Value Bets page (/value-bets)
  - Ranked list by expected value
  - Model probability vs market odds comparison
  - Kelly criterion recommended stake
  - Historical ROI display

- [ ] **4.2** Create Teams page (/teams)
  - Team directory with search
  - Filter by tournament
  - Team cards with recent form

- [ ] **4.3** Create Team Detail page (/teams/[id])
  - Team info and season record
  - Form chart (points trend)
  - Goal statistics
  - Home vs away performance
  - Upcoming fixtures with predictions

### Team Components

- [ ] **4.4** Create team components
  - TeamCard
  - TeamFormIndicator
  - TeamSearch

### Match Components

- [ ] **4.5** Create remaining match components
  - MatchCard
  - OddsDisplay
  - H2HTable

### Chart Components

- [ ] **4.6** Create chart components (using Recharts)
  - TeamFormChart
  - PredictionDonut
  - PerformanceLine
  - MetricsBarChart

### Real-time Updates (Phase 2, Section 2.9)

- [ ] **4.7** Implement WebSocket client
  - lib/socket/client.ts with reconnection logic
  - Subscribe/publish pattern

- [ ] **4.8** Create real-time hooks
  - useMatchUpdates for live scores
  - useLiveOddsUpdates for odds changes

- [ ] **4.9** Create LiveMatchCard component
  - Animated LIVE badge
  - Real-time score updates

---

## Sprint 5: Polish & Testing (Week 5)

### Testing Setup (Phase 7)

- [ ] **5.1** Set up Vitest
  - vitest.config.ts
  - tests/setup.ts with MSW
  - Configure coverage reporting

- [ ] **5.2** Write unit tests
  - utils/format.test.ts
  - utils/odds.test.ts
  - hooks/use-matches.test.ts
  - hooks/use-predictions.test.ts
  - hooks/use-debounce.test.ts
  - stores/filter-store.test.ts
  - schemas/validation.test.ts

- [ ] **5.3** Write integration tests
  - api/client.test.ts
  - api/error-handling.test.ts
  - components/match-card.test.tsx
  - components/prediction-card.test.tsx
  - components/filters.test.tsx

- [ ] **5.4** Set up Playwright
  - playwright.config.ts
  - Configure browsers and webServer

- [ ] **5.5** Write E2E tests
  - matches.spec.ts (listing, filtering, navigation)
  - predictions.spec.ts (generation workflow)
  - models.spec.ts (activation flow)
  - value-bets.spec.ts (display)

### Polish & Optimization (Phase 9)

- [ ] **5.6** Responsive design fixes
  - Mobile-first approach
  - Test on different screen sizes

- [ ] **5.7** Performance optimization
  - React.memo for expensive renders
  - Lazy loading for route-heavy components
  - Bundle size analysis with @next/bundle-analyzer

- [ ] **5.8** Configure build tools
  - TypeScript type check script
  - ESLint configuration
  - Next.js build optimization

- [ ] **5.9** Accessibility audit
  - WCAG 2.1 Level AA compliance
  - Semantic HTML
  - ARIA labels
  - Keyboard navigation
  - Screen reader testing

### Documentation

- [ ] **5.10** Update README.md
  - Installation instructions
  - How to run the application
  - API documentation link

---

## Deployment & CI/CD (Phase 8)

### Continuous Integration

- [ ] **D1** Create GitHub Actions workflow - Frontend CI
  - Lint & type check
  - Unit & integration tests
  - E2E tests with Playwright
  - Build step

- [ ] **D2** Create GitHub Actions workflow - Backend CI
  - Lint with Ruff
  - Type check with mypy
  - Pytest with coverage

- [ ] **D3** Create GitHub Actions workflow - Deploy
  - Frontend to Vercel
  - Backend to hosting provider

### Type Generation

- [ ] **D4** Implement OpenAPI type generation
  - scripts/generate-api-types.sh
  - Update package.json scripts
  - CI integration for type sync verification

---

## Future Enhancements (Phase 10)

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

## Dependencies

```
Phase 1 (API Layer) → Phase 2 (Frontend Architecture) → Phase 3 (Pages)
                                                        ↓
Phase 7 (Testing) ← Phase 4 (Advanced Features) ← Phase 5 (Sprint 3)
                                                        ↓
Phase 8 (CI/CD) ← Phase 9 (Optimization) ← Phase 6 (Polish)
```

### Sprint Dependencies

- Sprint 2 depends on Sprint 1 completion
- Sprint 3 depends on Sprint 2 completion
- Sprint 4 depends on Sprint 3 completion
- Sprint 5 depends on Sprint 4 completion

---

## Notes

- All API endpoints should integrate with existing modules (database.py, models.py, predictions/)
- Follow existing code conventions and patterns
- Test locally before committing
- Use MSW for development when backend is not available
- Keep frontend and backend types in sync
