# AlgoBet Frontend Development Roadmap

## Overview

This document outlines the remaining tasks to complete the frontend development, organized by priority and phase.

---

## Phase 1: Code Quality Cleanup (Priority: HIGH)

### 1.1 Remove Unused Imports and Variables

| File | Issue | Action |
|------|-------|--------|
| `app/backtest/page.tsx` | `modelsData` unused | Remove or use variable |
| `app/calibrate/page.tsx` | `useModels` unused | Remove import |
| `app/predictions/page.tsx` | `TrendingUp`, `Calendar`, `usePredictions` unused | Remove unused imports |
| `app/scraping/page.tsx` | `CardDescription`, `CardHeader`, `CardTitle`, `ScrapingProgressType` unused | Remove unused imports |
| `app/teams/loading.tsx` | `CardHeader` unused | Remove import |
| `app/value-bets/page.tsx` | `ValueBetCard` unused | Remove or use component |
| `components/charts/performance-trends-chart.tsx` | `Card`, `CardContent`, `CardHeader`, `CardTitle`, `data` unused | Remove unused imports |
| `components/dashboard/quick-actions-panel.tsx` | `Card`, `CardContent`, `CardHeader`, `CardTitle`, `Brain`, `Calendar`, `Trophy` unused | Remove unused imports |
| `components/dashboard/recent-activity-feed.tsx` | `Card`, `CardContent`, `CardHeader`, `CardTitle`, `Activity`, `setActivities` unused | Remove unused imports |
| `components/layout/Breadcrumb.tsx` | `cn` unused | Remove import |
| `components/schedules/ScheduleCard.tsx` | `useState`, `MoreVerticalIcon` unused | Remove unused imports |

### 1.2 Fix TypeScript `any` Types

| File | Count | Action |
|------|-------|--------|
| `app/scraping/page.tsx` | 4 occurrences | Replace with proper types |
| `components/charts/performance-trends-chart.tsx` | 1 occurrence | Replace with proper type |
| `lib/queries/use-dashboard-stats.ts` | 2 occurrences | Replace with proper types |

### 1.3 Verification

```bash
cd frontend && pnpm lint
```

---

## Phase 2: Feature Completion (Priority: HIGH)

### 2.1 Dashboard Page (`app/page.tsx`)

- [ ] Implement `useDashboardStats` hook with real API integration
- [ ] Add live match updates via WebSocket
- [ ] Connect quick actions to actual functionality
- [ ] Implement recent activity feed with real data

### 2.2 Matches Pages

#### `app/matches/page.tsx`
- [ ] Connect filters to API query parameters
- [ ] Implement infinite scroll pagination
- [ ] Add match status indicators (live, scheduled, finished)

#### `app/matches/[id]/page.tsx`
- [ ] Display head-to-head history
- [ ] Show prediction probabilities
- [ ] Add odds comparison display

### 2.3 Predictions Pages

#### `app/predictions/page.tsx`
- [ ] Implement prediction history table with sorting
- [ ] Add confidence filtering
- [ ] Display model version information

#### Missing: `app/predictions/upcoming/page.tsx`
- [ ] Create upcoming predictions view
- [ ] Add real-time updates for match odds changes

### 2.4 Models Page (`app/models/page.tsx`)

- [ ] Implement model list with metrics display
- [ ] Add model activation toggle
- [ ] Create model comparison view
- [ ] Add model detail modal/page

### 2.5 Value Bets Page (`app/value-bets/page.tsx`)

- [ ] Connect to value bets API endpoint
- [ ] Implement `ValueBetCard` component
- [ ] Add expected value calculation display
- [ ] Add Kelly criterion stake recommendations

### 2.6 Teams Pages

#### `app/teams/page.tsx`
- [ ] Add team search with autocomplete
- [ ] Connect to teams API

#### `app/teams/[id]/page.tsx` (if missing)
- [ ] Create team profile page
- [ ] Display form chart
- [ ] Show match history

### 2.7 Scraping Page (`app/scraping/page.tsx`)

- [ ] Connect scraping form to API
- [ ] Implement real-time progress updates (WebSocket)
- [ ] Add job history with status indicators

### 2.8 Scheduling Pages

#### `app/schedules/page.tsx`
- [ ] Connect to scheduler API
- [ ] Implement schedule CRUD operations
- [ ] Add execution history view

### 2.9 Backtest & Calibrate Pages

#### `app/backtest/page.tsx`
- [ ] Implement backtest configuration form
- [ ] Add backtest results visualization
- [ ] Connect to backtest API endpoint

#### `app/calibrate/page.tsx`
- [ ] Implement model calibration interface
- [ ] Add calibration curve display
- [ ] Connect to calibration API

---

## Phase 3: API Integration (Priority: HIGH)

### 3.1 Complete API Client Layer

| File | Status | Remaining Work |
|------|--------|----------------|
| `lib/api/client.ts` | ✅ Done | Error handling improvements |
| `lib/api/matches.ts` | ⚠️ Partial | Add H2H, preview endpoints |
| `lib/api/predictions.ts` | ⚠️ Partial | Add generate, upcoming endpoints |
| `lib/api/models.ts` | ⚠️ Partial | Add activate, metrics endpoints |
| `lib/api/teams.ts` | ⚠️ Partial | Add form, stats endpoints |
| `lib/api/value-bets.ts` | ⚠️ Partial | Connect to backend endpoint |
| `lib/api/schedules.ts` | ⚠️ Partial | CRUD operations |
| `lib/api/scraping.ts` | ⚠️ Partial | WebSocket integration |
| `lib/api/ml-operations.ts` | ⚠️ Partial | Backtest, calibrate endpoints |

### 3.2 Complete TanStack Query Hooks

| File | Status | Remaining Work |
|------|--------|----------------|
| `lib/queries/use-matches.ts` | ⚠️ Partial | Add H2H, preview queries |
| `lib/queries/use-predictions.ts` | ⚠️ Partial | Add generate mutation |
| `lib/queries/use-models.ts` | ⚠️ Partial | Add activate mutation |
| `lib/queries/use-teams.ts` | ⚠️ Partial | Add form, stats queries |
| `lib/queries/use-value-bets.ts` | ⚠️ Partial | Connect to API |
| `lib/queries/use-ml-operations.ts` | ⚠️ Partial | Add backtest, calibrate |
| `lib/queries/use-dashboard-stats.ts` | ⚠️ Partial | Fix `any` types, connect to API |

---

## Phase 4: Component Development (Priority: MEDIUM)

### 4.1 Missing Core Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `components/predictions/prediction-card.tsx` | Display prediction with confidence | ❌ Missing |
| `components/predictions/probability-bar.tsx` | Visual probability display | ❌ Missing |
| `components/predictions/confidence-badge.tsx` | Confidence level indicator | ❌ Missing |
| `components/value-bets/value-bet-card.tsx` | Value bet display card | ❌ Missing |
| `components/teams/team-form-chart.tsx` | Team form visualization | ❌ Missing |
| `components/teams/team-card.tsx` | Team listing card | ❌ Missing |
| `components/models/model-activation-toggle.tsx` | Model activation UI | ❌ Missing |
| `components/models/metrics-table.tsx` | Model metrics display | ❌ Missing |

### 4.2 Charts (Recharts Integration)

| Component | Status | Notes |
|-----------|--------|-------|
| `components/charts/performance-trends-chart.tsx` | ⚠️ Needs work | Fix unused imports, connect data |
| `components/charts/team-form-chart.tsx` | ❌ Missing | Need to create |
| `components/charts/prediction-donut.tsx` | ❌ Missing | Need to create |
| `components/charts/calibration-curve.tsx` | ❌ Missing | Need to create |

### 4.3 Existing Components to Fix

| Component | Issue | Action |
|-----------|-------|--------|
| `components/dashboard/quick-actions-panel.tsx` | Stub implementation | Connect to real actions |
| `components/dashboard/recent-activity-feed.tsx` | Mock data | Connect to API |
| `components/scraping/ScrapingProgress.tsx` | Needs WebSocket | Add real-time updates |

---

## Phase 5: Real-time Features (Priority: MEDIUM)

### 5.1 WebSocket Integration

- [ ] Create WebSocket client utility (`lib/socket/client.ts`)
- [ ] Implement `useMatchUpdates` hook for live match data
- [ ] Add scraping progress updates
- [ ] Implement connection status indicator

### 5.2 Live Updates

- [ ] Dashboard live match scores
- [ ] Odds changes for value bets
- [ ] Scraping job progress
- [ ] Prediction generation status

---

## Phase 6: Error Handling & UX (Priority: MEDIUM)

### 6.1 Error Boundaries

| File | Status | Action |
|------|--------|--------|
| `components/error-boundary.tsx` | ✅ Exists | Add fallback UI variants |
| `app/error.tsx` | ✅ Exists | Add retry functionality |
| `app/*/error.tsx` | ✅ Exists | Verify consistent UX |

### 6.2 Loading States

- [ ] Verify all loading.tsx files have appropriate skeletons
- [ ] Add loading indicators for mutations
- [ ] Implement optimistic updates where appropriate

### 6.3 Empty States

- [ ] Add empty state components for each list view
- [ ] Include actionable guidance (e.g., "Run a scrape to populate matches")

---

## Phase 7: Testing (Priority: LOW)

### 7.1 Unit Tests

- [ ] Test API client error handling
- [ ] Test TanStack Query hooks
- [ ] Test utility functions

### 7.2 Integration Tests

- [ ] Test form submissions
- [ ] Test filter interactions
- [ ] Test navigation flows

### 7.3 E2E Tests (Optional)

- [ ] Critical user journeys
- [ ] Prediction workflow
- [ ] Value bet identification

---

## Phase 8: Performance & Polish (Priority: LOW)

### 8.1 Performance Optimizations

- [ ] Implement route-based code splitting
- [ ] Add image optimization for team/tournament logos
- [ ] Implement virtual scrolling for long lists
- [ ] Add service worker for offline support

### 8.2 Accessibility

- [ ] Audit with Lighthouse
- [ ] Add ARIA labels where needed
- [ ] Ensure keyboard navigation
- [ ] Test with screen readers

### 8.3 SEO & Metadata

- [ ] Add metadata to each page
- [ ] Implement Open Graph tags
- [ ] Add structured data for matches

---

## Quick Reference: Current ESLint Warnings

```
/app/backtest/page.tsx
Warning: 'modelsData' is assigned a value but never used.  @typescript-eslint/no-unused-vars

./app/calibrate/page.tsx
Warning: 'useModels' is defined but never used.  @typescript-eslint/no-unused-vars

./app/predictions/page.tsx
Warning: 'TrendingUp' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'Calendar' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'usePredictions' is defined but never used.  @typescript-eslint/no-unused-vars

./app/scraping/page.tsx
Warning: 'CardDescription' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardHeader' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardTitle' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'ScrapingProgressType' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: Unexpected any. Specify a different type.  @typescript-eslint/no-explicit-any (x4)

./app/teams/loading.tsx
Warning: 'CardHeader' is defined but never used.  @typescript-eslint/no-unused-vars

./app/value-bets/page.tsx
Warning: 'ValueBetCard' is defined but never used.  @typescript-eslint/no-unused-vars

./components/charts/performance-trends-chart.tsx
Warning: 'Card' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardContent' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardHeader' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardTitle' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: Unexpected any. Specify a different type.  @typescript-eslint/no-explicit-any
Warning: 'data' is defined but never used. Allowed unused args must match /^_/u.  @typescript-eslint/no-unused-vars

./components/dashboard/quick-actions-panel.tsx
Warning: 'Card' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardContent' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardHeader' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardTitle' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'Brain' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'Calendar' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'Trophy' is defined but never used.  @typescript-eslint/no-unused-vars

./components/dashboard/recent-activity-feed.tsx
Warning: 'Card' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardContent' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardHeader' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'CardTitle' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'Activity' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'setActivities' is assigned a value but never used.  @typescript-eslint/no-unused-vars

./components/layout/Breadcrumb.tsx
Warning: 'cn' is defined but never used.  @typescript-eslint/no-unused-vars

./components/schedules/ScheduleCard.tsx
Warning: 'useState' is defined but never used.  @typescript-eslint/no-unused-vars
Warning: 'MoreVerticalIcon' is defined but never used.  @typescript-eslint/no-unused-vars

./lib/queries/use-dashboard-stats.ts
Warning: Unexpected any. Specify a different type.  @typescript-eslint/no-explicit-any (x2)
```

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Code Quality | 1-2 days | None |
| Phase 2: Feature Completion | 5-7 days | Phase 1 |
| Phase 3: API Integration | 3-4 days | Phase 2 |
| Phase 4: Component Development | 3-4 days | Phase 3 |
| Phase 5: Real-time Features | 2-3 days | Phase 3 |
| Phase 6: Error Handling & UX | 2-3 days | Phase 4 |
| Phase 7: Testing | 3-4 days | Phase 6 |
| Phase 8: Performance & Polish | 2-3 days | Phase 7 |

**Total Estimated Duration: 3-4 weeks**

---

## Getting Started

1. **Start with Phase 1** - Clean up linting warnings for a solid foundation
2. Run `pnpm lint` after each fix to verify
3. Proceed to Phase 2 once all warnings are resolved
4. Work through phases sequentially for best results

## Commands

```bash
# Install dependencies
cd frontend && pnpm install

# Run development server
pnpm dev

# Check linting
pnpm lint

# Fix linting issues
pnpm lint:fix

# Format code
pnpm format

# Build for production
pnpm build
```
