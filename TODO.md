# AlgoBet Frontend Development TODO

**Last Updated:** 2026-02-01
**Progress:** 14 of 53 tasks completed (26.4%)

---

## ✅ Completed Tasks (14/53)

| # | Task | Priority |
|---|------|----------|
| 1 | Add FastAPI dependencies to pyproject.toml | 🔴 High |
| 2 | Create algobet/api directory structure | 🔴 High |
| 3 | Implement FastAPI main.py with app entry point and CORS | 🔴 High |
| 4 | Create dependencies.py for database session injection | 🔴 High |
| 5 | Create Pydantic schemas (tournament, season, team, match, prediction, model) | 🔴 High |
| 6 | Implement matches router with endpoints (list, detail, preview, predictions, h2h) | 🔴 High |
| 7 | Implement teams router with endpoints (list, detail, form, matches) | 🔴 High |
| 8 | Implement tournaments router with endpoints (list, detail, seasons) | 🔴 High |
| 9 | Implement predictions router with endpoints (list, generate, upcoming, history) | 🔴 High |
| 10 | Implement models router with endpoints (list, active, detail, activate, delete, metrics) | 🔴 High |
| 11 | Implement value-bets endpoint (GET /api/v1/value-bets) | 🔴 High |
| 12 | Add environment variables for API configuration | 🔴 High |
| 13 | Update docker-compose.yml with API service | 🔴 High |
| 14 | Implement seasons router with endpoints (matches with filters, pagination) | 🔴 High |

---

## 🚧 High Priority Tasks (11 remaining)

| # | Task |
|---|------|
| 14 | Initialize Next.js 15 project with TypeScript in frontend/ directory |
| 15 | Install and configure shadcn/ui components |
| 16 | Configure Tailwind CSS with custom dark mode theme colors |
| 17 | Set up TanStack Query provider in providers/ |
| 18 | Create base layout with navigation (navbar, sidebar, breadcrumb) |
| 19 | Create API client with error handling (lib/api/client.ts) |
| 20 | Create TypeScript type definitions (lib/types/api.ts) |
| 21 | Create Zod runtime validation schemas (lib/types/schemas.ts) |
| 22 | Create TanStack Query hooks (use-matches, use-predictions, use-models, use-value-bets) |
| 23 | Set up Zustand stores (filter-store, ui-store) |
| 24 | Set up MSW for API mocking (handlers, browser, mock data) |

---

## 🟡 Medium Priority Tasks (19 remaining)

| # | Task |
|---|------|
| 25 | Create utility functions (cn.ts, format.ts, odds.ts) |
| 26 | Create Dashboard page (/) with upcoming matches, active model, value bets, accuracy chart |
| 27 | Create Matches page (/matches) with filters, infinite scroll, match list |
| 28 | Create Match Detail page (/matches/[id]) with H2H, form charts, odds, predictions |
| 29 | Create Predictions page (/predictions) with history table, accuracy metrics |
| 30 | Create Upcoming Predictions page (/predictions/upcoming) with generation workflow |
| 31 | Create Models page (/models) with registry listing, active toggle |
| 32 | Create Model Detail page (/models/[id]) with metrics visualization |
| 33 | Create Value Bets page (/value-bets) with ranked opportunities |
| 34 | Create Teams page (/teams) with search, filtering |
| 35 | Create Team Detail page (/teams/[id]) with performance analysis, form charts |
| 36 | Create chart components (team-form-chart, prediction-donut, performance-line, metrics-bar-chart) |
| 37 | Create match components (match-card, match-list, match-filters, odds-display, h2h-table) |
| 38 | Create prediction components (prediction-card, confidence-badge, value-bet-card, probability-bar) |
| 39 | Create team components (team-card, team-form-indicator, team-search) |
| 40 | Create model components (model-selector, metrics-table, model-activation-toggle) |
| 41 | Implement WebSocket client for real-time updates (lib/socket/client.ts) |
| 42 | Create hooks for real-time updates (use-match-updates, use-live-odds-updates) |
| 43 | Add loading skeleton components |
| 44 | Add error boundaries (route-level and reusable component) |

---

## 🟢 Low Priority Tasks (10 remaining)

| # | Task |
|---|------|
| 45 | Set up Vitest for unit and integration tests |
| 46 | Write unit tests for utils, hooks, stores, schemas |
| 47 | Write integration tests with MSW for API client and components |
| 48 | Set up Playwright for E2E testing |
| 49 | Write E2E tests for key user flows |
| 50 | Configure Next.js build and TypeScript type check |
| 51 | Optimize bundle size (tree-shaking, dynamic imports, bundle analyzer) |
| 52 | Implement OpenAPI type generation script |
| 53 | Create GitHub Actions workflows for CI/CD |

---

## 📊 Summary

- **Total Tasks:** 53
- **Completed:** 14 (26.4%)
- **Pending:** 39 (73.6%)
  - High Priority: 11
  - Medium Priority: 19
  - Low Priority: 10

### API Test Results
- **Total Tests:** 74
- **Passed:** 74 ✅
- **Coverage:** All Phase 1 & Phase 2 API endpoints tested

---

## 🔄 Next Steps

✅ **Phase 1 (API Layer) - COMPLETE** - All 13 tasks finished
✅ **Phase 2 (API Enhancement) - COMPLETE** - Seasons router with full test coverage (74 tests passing)

The backend API foundation is fully complete and tested. The next priority is to initialize the Next.js frontend project and set up the core infrastructure:

1. Initialize Next.js 15 with TypeScript
2. Set up shadcn/ui and Tailwind CSS
3. Create API client and type definitions
4. Build out the core pages and components

---

## 📝 Notes

- See `DEVELOPMENT_TASKS.md` for detailed sprint breakdown
- See `docs/frontend_development_plan.md` for full implementation plan
- API endpoints are ready at `http://localhost:8000` (when running)
- Use `DEVELOPMENT_TASKS.md` for sprint-by-sprint implementation guide
