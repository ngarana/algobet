# AlgoBet Frontend Development Plan

## Executive Summary

**Application Type**: Data analytics dashboard for football match predictions and betting insights

**Recommended Stack**: Next.js 15 (App Router) + TypeScript + shadcn/ui + TanStack Query

**Core Capabilities**:
- Real-time match predictions and odds display
- Interactive data visualizations (team form, performance metrics)
- Model management and evaluation dashboards
- Value bet detection with confidence indicators

---

## Phase 1: API Layer Foundation (Prerequisite)

Before building the frontend, we need a REST API layer to make the backend accessible:

### 1.1 FastAPI Implementation
```
algobet/api/
├── __init__.py
├── main.py                 # FastAPI app entry point
├── dependencies.py         # Database sessions, auth
├── routers/
│   ├── __init__.py
│   ├── matches.py          # Match CRUD and queries
│   ├── tournaments.py      # Tournament listing
│   ├── teams.py            # Team stats and history
│   ├── predictions.py      # Prediction endpoints
│   ├── models.py           # Model registry management
│   └── metrics.py          # Performance metrics
├── schemas/
│   ├── __init__.py
│   ├── match.py            # Pydantic models for matches
│   ├── team.py             # Pydantic models for teams
│   ├── tournament.py       # Pydantic models for tournaments
│   ├── prediction.py       # Pydantic models for predictions
│   └── model.py            # Pydantic models for model versions
└── middleware.py           # CORS, logging, error handling
```

### 1.2 API Response Schemas (Aligned with Database Models)

The following schemas align with the existing SQLAlchemy models in [`algobet/models.py`](algobet/models.py):

#### Tournament Schema
```python
class TournamentResponse(BaseModel):
    id: int
    name: str
    country: str
    url_slug: str

    class Config:
        from_attributes = True
```

#### Season Schema
```python
class SeasonResponse(BaseModel):
    id: int
    tournament_id: int
    name: str  # e.g., "2023/2024"
    start_year: int
    end_year: int
    url_suffix: Optional[str]

    class Config:
        from_attributes = True
```

#### Team Schema
```python
class TeamResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

class TeamWithStatsResponse(TeamResponse):
    """Team with computed statistics."""
    total_matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    current_form: FormBreakdown  # From FormCalculator
```

#### Match Schema
```python
class MatchResponse(BaseModel):
    id: int
    tournament_id: int
    season_id: int
    home_team_id: int
    away_team_id: int
    match_date: datetime
    home_score: Optional[int]
    away_score: Optional[int]
    status: str  # 'SCHEDULED', 'FINISHED', 'LIVE'
    odds_home: Optional[float]
    odds_draw: Optional[float]
    odds_away: Optional[float]
    num_bookmakers: Optional[int]
    created_at: datetime
    updated_at: datetime

    # Computed fields
    result: Optional[str]  # 'H', 'D', 'A' or None

    class Config:
        from_attributes = True

class MatchDetailResponse(MatchResponse):
    """Match with related entities and predictions."""
    tournament: TournamentResponse
    season: SeasonResponse
    home_team: TeamResponse
    away_team: TeamResponse
    predictions: list[PredictionResponse]
    h2h_matches: list[MatchResponse]  # Last 5 head-to-head
```

#### ModelVersion Schema
```python
class ModelVersionResponse(BaseModel):
    id: int
    name: str
    version: str
    algorithm: str  # 'xgboost', 'random_forest', etc.
    accuracy: Optional[float]
    file_path: str
    is_active: bool
    created_at: datetime
    metrics: Optional[dict]  # JSONB field
    hyperparameters: Optional[dict]  # JSONB field
    feature_schema_version: Optional[str]
    description: Optional[str]

    class Config:
        from_attributes = True
```

#### Prediction Schema
```python
class PredictionResponse(BaseModel):
    id: int
    match_id: int
    model_version_id: int
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_outcome: str  # 'H', 'D', 'A'
    confidence: float
    predicted_at: datetime
    actual_roi: Optional[float]

    # Computed field
    max_probability: float

    class Config:
        from_attributes = True

class PredictionWithMatchResponse(PredictionResponse):
    """Prediction with full match details."""
    match: MatchDetailResponse
    model_version: ModelVersionResponse
```

### 1.3 Key API Endpoints

#### Tournaments
```
GET    /api/v1/tournaments                    # List all tournaments
GET    /api/v1/tournaments/{id}               # Get tournament details
GET    /api/v1/tournaments/{id}/seasons       # Get seasons for tournament
```

#### Seasons
```
GET    /api/v1/seasons/{id}/matches           # Get matches in season
```

#### Matches
```
GET    /api/v1/matches                        # List matches with filtering
       Query params:
       - status: 'SCHEDULED' | 'FINISHED' | 'LIVE'
       - tournament_id: int
       - season_id: int
       - team_id: int
       - from_date: ISO datetime
       - to_date: ISO datetime
       - days_ahead: int (for upcoming matches)
       - has_odds: bool
       - limit: int (default: 50, max: 100)
       - offset: int
GET    /api/v1/matches/{id}                   # Get match details
GET    /api/v1/matches/{id}/preview           # Get match preview with form
GET    /api/v1/matches/{id}/predictions       # Get predictions for match
GET    /api/v1/matches/{id}/h2h               # Get head-to-head history
```

#### Teams
```
GET    /api/v1/teams                          # List teams with search
       Query params:
       - search: str (name search)
       - tournament_id: int
       - limit: int
       - offset: int
GET    /api/v1/teams/{id}                     # Get team details
GET    /api/v1/teams/{id}/form                # Get team form breakdown
       Query params:
       - n_matches: int (default: 5)
       - reference_date: ISO datetime
GET    /api/v1/teams/{id}/matches             # Get team match history
       Query params:
       - venue: 'home' | 'away' | 'all'
       - limit: int
```

#### Predictions
```
GET    /api/v1/predictions                    # List predictions
       Query params:
       - match_id: int
       - model_version_id: int
       - has_result: bool (filter by actual outcome)
       - from_date: ISO datetime
       - to_date: ISO datetime
       - min_confidence: float
POST   /api/v1/predictions/generate           # Generate predictions
       Body:
       - match_ids: list[int]
       - model_version: str (optional)
       - tournament_id: int (optional)
       - days_ahead: int (optional)
GET    /api/v1/predictions/upcoming           # Get upcoming predictions
GET    /api/v1/predictions/history            # Get prediction accuracy history
```

#### Model Registry
```
GET    /api/v1/models                         # List all model versions
       Query params:
       - algorithm: str
       - active_only: bool
GET    /api/v1/models/active                  # Get currently active model
GET    /api/v1/models/{id}                    # Get model details
POST   /api/v1/models/{id}/activate           # Activate a model version
DELETE /api/v1/models/{id}                    # Delete model version
GET    /api/v1/models/{id}/metrics            # Get detailed metrics
```

#### Value Bets
```
GET    /api/v1/value-bets                     # Find value betting opportunities
       Query params:
       - min_ev: float (minimum expected value, default: 0.05)
       - max_odds: float (default: 10.0)
       - days: int (days ahead, default: 7)
       - model_version: str
       - min_confidence: float
       - max_matches: int (default: 20)
```

---

## Phase 2: Frontend Architecture

### 2.1 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Framework** | Next.js 15 (App Router) | SSR/SSG, React Server Components, built-in routing |
| **Language** | TypeScript 5.3+ | Type safety across stack, strict mode enabled |
| **UI Components** | shadcn/ui + Radix UI | Accessible, customizable, modern design |
| **Styling** | Tailwind CSS 3.4+ | Utility-first, responsive, small bundle |
| **Data Fetching** | TanStack Query (React Query) v5 | Caching, deduplication, optimistic updates |
| **Forms** | React Hook Form + Zod | Type-safe validation, great DX |
| **Runtime Validation** | Zod | API response validation for resilience |
| **Charts** | Recharts | Declarative, responsive, composable |
| **State** | Zustand | Lightweight, no boilerplate, devtools support |
| **Icons** | Lucide React | Consistent icon set |
| **Date Handling** | date-fns | Lightweight date manipulation |
| **HTTP Client** | Native fetch + tanstack-query | No axios needed with modern fetch |
| **API Mocking** | MSW (Mock Service Worker) | API mocking for development and testing |
| **Real-time** | WebSocket | Live match updates and odds streaming |

### 2.2 Project Structure
```
frontend/
├── app/                           # Next.js App Router
│   ├── layout.tsx                 # Root layout with providers
│   ├── page.tsx                   # Dashboard landing
│   ├── error.tsx                  # Error boundary
│   ├── loading.tsx                # Loading UI
│   ├── matches/
│   │   ├── page.tsx               # Match listing with filters
│   │   ├── loading.tsx            # Skeleton loader
│   │   └── [id]/
│   │       ├── page.tsx           # Match detail with prediction
│   │       └── loading.tsx
│   ├── predictions/
│   │   ├── page.tsx               # Prediction history
│   │   ├── upcoming/
│   │   │   └── page.tsx           # Upcoming predictions
│   │   └── layout.tsx
│   ├── models/
│   │   ├── page.tsx               # Model registry
│   │   └── [id]/
│   │       └── page.tsx           # Model detail + metrics
│   ├── value-bets/
│   │   └── page.tsx               # Value bet opportunities
│   ├── teams/
│   │   ├── page.tsx               # Team directory
│   │   └── [id]/
│   │       └── page.tsx           # Team profile with form
│   └── api/                       # Next.js API routes (if needed)
│       └── revalidate/
│           └── route.ts
├── components/
│   ├── ui/                        # shadcn/ui components (auto-generated)
│   ├── providers/                 # Context providers
│   │   ├── query-provider.tsx     # TanStack Query provider
│   │   └── theme-provider.tsx     # Dark mode provider
│   ├── charts/
│   │   ├── team-form-chart.tsx
│   │   ├── prediction-donut.tsx
│   │   ├── performance-line.tsx
│   │   └── metrics-bar-chart.tsx
│   ├── matches/
│   │   ├── match-card.tsx
│   │   ├── match-list.tsx
│   │   ├── match-filters.tsx
│   │   ├── odds-display.tsx
│   │   └── h2h-table.tsx
│   ├── predictions/
│   │   ├── prediction-card.tsx
│   │   ├── confidence-badge.tsx
│   │   ├── value-bet-card.tsx
│   │   └── probability-bar.tsx
│   ├── teams/
│   │   ├── team-card.tsx
│   │   ├── team-form-indicator.tsx
│   │   └── team-search.tsx
│   ├── models/
│   │   ├── model-selector.tsx
│   │   ├── metrics-table.tsx
│   │   └── model-activation-toggle.tsx
│   └── layout/
│       ├── navbar.tsx
│       ├── sidebar.tsx
│       └── breadcrumb.tsx
├── lib/                           # Utility functions and configurations
│   ├── api/                       # API layer
│   │   ├── client.ts              # Base API client with error handling
│   │   ├── tournaments.ts         # Tournament API functions
│   │   ├── matches.ts             # Match API functions
│   │   ├── teams.ts               # Team API functions
│   │   ├── predictions.ts         # Prediction API functions
│   │   └── models.ts              # Model registry API functions
│   ├── queries/                   # TanStack Query hooks
│   │   ├── use-matches.ts
│   │   ├── use-teams.ts
│   │   ├── use-predictions.ts
│   │   ├── use-models.ts
│   │   └── use-value-bets.ts
│   ├── types/                     # TypeScript type definitions
│   │   ├── api.ts                 # API response types
│   │   ├── schemas.ts             # Zod schemas for runtime validation
│   │   └── index.ts
│   ├── utils/
│   │   ├── cn.ts                  # Tailwind class merger
│   │   ├── format.ts              # Date/formatting utilities
│   │   └── odds.ts                # Odds calculations
│   ├── socket/                    # WebSocket utilities
│   │   ├── client.ts              # WebSocket client setup
│   │   └── use-match-updates.ts   # Real-time match updates hook
│   └── config.ts                  # Environment configuration
├── hooks/                         # Custom React hooks
│   ├── use-debounce.ts
│   ├── use-local-storage.ts
│   └── use-media-query.ts
├── stores/                        # Zustand stores
│   ├── filter-store.ts            # Global filter state
│   └── ui-store.ts                # UI state (sidebar, etc.)
├── mocks/                         # MSW API mocking
│   ├── handlers.ts                # API mock handlers
│   ├── browser.ts                 # Browser service worker setup
│   ├── server.ts                  # Node server setup for tests
│   └── data/                      # Mock data fixtures
│       ├── matches.ts
│       ├── predictions.ts
│       └── models.ts
├── public/
│   └── assets/
│       └── logos/                 # Tournament/team logos
└── types/
    └── env.d.ts                   # Environment type declarations
```

### 2.3 TypeScript Type Definitions (Aligned with API)

```typescript
// lib/types/api.ts

export type MatchStatus = 'SCHEDULED' | 'FINISHED' | 'LIVE';

export type PredictedOutcome = 'H' | 'D' | 'A';

export interface Tournament {
  id: number;
  name: string;
  country: string;
  url_slug: string;
}

export interface Season {
  id: number;
  tournament_id: number;
  name: string;
  start_year: number;
  end_year: number;
  url_suffix: string | null;
}

export interface Team {
  id: number;
  name: string;
}

export interface Match {
  id: number;
  tournament_id: number;
  season_id: number;
  home_team_id: number;
  away_team_id: number;
  match_date: string;  // ISO datetime
  home_score: number | null;
  away_score: number | null;
  status: MatchStatus;
  odds_home: number | null;
  odds_draw: number | null;
  odds_away: number | null;
  num_bookmakers: number | null;
  created_at: string;
  updated_at: string;
  result: PredictedOutcome | null;
}

export interface MatchDetail extends Match {
  tournament: Tournament;
  season: Season;
  home_team: Team;
  away_team: Team;
  predictions: Prediction[];
  h2h_matches: Match[];
}

export interface Prediction {
  id: number;
  match_id: number;
  model_version_id: number;
  prob_home: number;
  prob_draw: number;
  prob_away: number;
  predicted_outcome: PredictedOutcome;
  confidence: number;
  predicted_at: string;
  actual_roi: number | null;
  max_probability: number;
}

export interface PredictionWithMatch extends Prediction {
  match: MatchDetail;
  model_version: ModelVersion;
}

export interface ModelVersion {
  id: number;
  name: string;
  version: string;
  algorithm: string;
  accuracy: number | null;
  file_path: string;
  is_active: boolean;
  created_at: string;
  metrics: Record<string, number> | null;
  hyperparameters: Record<string, unknown> | null;
  feature_schema_version: string | null;
  description: string | null;
}

export interface FormBreakdown {
  avg_points: number;
  win_rate: number;
  draw_rate: number;
  loss_rate: number;
  avg_goals_for: number;
  avg_goals_against: number;
}

export interface ValueBet {
  match_id: number;
  match: MatchDetail;
  outcome: PredictedOutcome;
  predicted_probability: number;
  market_odds: number;
  expected_value: number;
  kelly_fraction: number;
}

// Query parameter types
export interface MatchFilters {
  status?: MatchStatus;
  tournament_id?: number;
  season_id?: number;
  team_id?: number;
  from_date?: string;
  to_date?: string;
  days_ahead?: number;
  has_odds?: boolean;
  limit?: number;
  offset?: number;
}

export interface PredictionFilters {
  match_id?: number;
  model_version_id?: number;
  has_result?: boolean;
  from_date?: string;
  to_date?: string;
  min_confidence?: number;
}
```

### 2.4 Zod Runtime Validation Schemas

Runtime validation ensures API responses match expected types, providing resilience against backend changes:

```typescript
// lib/types/schemas.ts

import { z } from 'zod';

// Enum schemas
export const MatchStatusSchema = z.enum(['SCHEDULED', 'FINISHED', 'LIVE']);
export const PredictedOutcomeSchema = z.enum(['H', 'D', 'A']);

// Entity schemas
export const TournamentSchema = z.object({
  id: z.number(),
  name: z.string(),
  country: z.string(),
  url_slug: z.string(),
});

export const TeamSchema = z.object({
  id: z.number(),
  name: z.string(),
});

export const SeasonSchema = z.object({
  id: z.number(),
  tournament_id: z.number(),
  name: z.string(),
  start_year: z.number(),
  end_year: z.number(),
  url_suffix: z.string().nullable(),
});

export const MatchSchema = z.object({
  id: z.number(),
  tournament_id: z.number(),
  season_id: z.number(),
  home_team_id: z.number(),
  away_team_id: z.number(),
  match_date: z.string(),
  home_score: z.number().nullable(),
  away_score: z.number().nullable(),
  status: MatchStatusSchema,
  odds_home: z.number().nullable(),
  odds_draw: z.number().nullable(),
  odds_away: z.number().nullable(),
  num_bookmakers: z.number().nullable(),
  created_at: z.string(),
  updated_at: z.string(),
  result: PredictedOutcomeSchema.nullable(),
});

export const PredictionSchema = z.object({
  id: z.number(),
  match_id: z.number(),
  model_version_id: z.number(),
  prob_home: z.number(),
  prob_draw: z.number(),
  prob_away: z.number(),
  predicted_outcome: PredictedOutcomeSchema,
  confidence: z.number(),
  predicted_at: z.string(),
  actual_roi: z.number().nullable(),
  max_probability: z.number(),
});

export const ModelVersionSchema = z.object({
  id: z.number(),
  name: z.string(),
  version: z.string(),
  algorithm: z.string(),
  accuracy: z.number().nullable(),
  file_path: z.string(),
  is_active: z.boolean(),
  created_at: z.string(),
  metrics: z.record(z.string(), z.number()).nullable(),
  hyperparameters: z.record(z.string(), z.unknown()).nullable(),
  feature_schema_version: z.string().nullable(),
  description: z.string().nullable(),
});

export const ValueBetSchema = z.object({
  match_id: z.number(),
  outcome: PredictedOutcomeSchema,
  predicted_probability: z.number(),
  market_odds: z.number(),
  expected_value: z.number(),
  kelly_fraction: z.number(),
});

// Response array schemas
export const MatchListSchema = z.array(MatchSchema);
export const PredictionListSchema = z.array(PredictionSchema);
export const ModelVersionListSchema = z.array(ModelVersionSchema);
export const ValueBetListSchema = z.array(ValueBetSchema);

// Type inference from schemas
export type MatchValidated = z.infer<typeof MatchSchema>;
export type PredictionValidated = z.infer<typeof PredictionSchema>;
```

### 2.5 MSW API Mocking

Mock Service Worker enables API mocking for development and testing without backend dependency:

```typescript
// mocks/handlers.ts

import { http, HttpResponse } from 'msw';
import { mockMatches, mockMatchDetail } from './data/matches';
import { mockPredictions, mockUpcomingPredictions } from './data/predictions';
import { mockModels, mockActiveModel } from './data/models';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const handlers = [
  // Matches
  http.get(`${API_BASE}/api/v1/matches`, ({ request }) => {
    const url = new URL(request.url);
    const status = url.searchParams.get('status');
    const limit = parseInt(url.searchParams.get('limit') || '50');

    let results = mockMatches;
    if (status) {
      results = results.filter(m => m.status === status);
    }
    return HttpResponse.json(results.slice(0, limit));
  }),

  http.get(`${API_BASE}/api/v1/matches/:id`, ({ params }) => {
    const match = mockMatchDetail(Number(params.id));
    if (!match) {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json(match);
  }),

  // Predictions
  http.get(`${API_BASE}/api/v1/predictions`, () => {
    return HttpResponse.json(mockPredictions);
  }),

  http.get(`${API_BASE}/api/v1/predictions/upcoming`, () => {
    return HttpResponse.json(mockUpcomingPredictions);
  }),

  http.post(`${API_BASE}/api/v1/predictions/generate`, async ({ request }) => {
    const body = await request.json();
    // Simulate prediction generation
    return HttpResponse.json({
      generated: body.match_ids?.length || 0,
      predictions: mockPredictions.slice(0, 5),
    });
  }),

  // Models
  http.get(`${API_BASE}/api/v1/models`, () => {
    return HttpResponse.json(mockModels);
  }),

  http.get(`${API_BASE}/api/v1/models/active`, () => {
    return HttpResponse.json(mockActiveModel);
  }),

  http.post(`${API_BASE}/api/v1/models/:id/activate`, ({ params }) => {
    return HttpResponse.json({ activated: params.id });
  }),

  // Value Bets
  http.get(`${API_BASE}/api/v1/value-bets`, () => {
    return HttpResponse.json([
      {
        match_id: 1,
        outcome: 'H',
        predicted_probability: 0.65,
        market_odds: 2.10,
        expected_value: 0.15,
        kelly_fraction: 0.08,
      },
    ]);
  }),
];
```

```typescript
// mocks/browser.ts
import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';

export const worker = setupWorker(...handlers);
```

```typescript
// mocks/server.ts (for tests)
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);
```

```typescript
// mocks/data/matches.ts
import type { Match, MatchDetail } from '@/lib/types/api';

export const mockMatches: Match[] = [
  {
    id: 1,
    tournament_id: 1,
    season_id: 1,
    home_team_id: 1,
    away_team_id: 2,
    match_date: '2026-02-15T15:00:00Z',
    home_score: null,
    away_score: null,
    status: 'SCHEDULED',
    odds_home: 2.10,
    odds_draw: 3.40,
    odds_away: 3.20,
    num_bookmakers: 12,
    created_at: '2026-02-01T10:00:00Z',
    updated_at: '2026-02-01T10:00:00Z',
    result: null,
  },
  // Add more mock matches...
];

export function mockMatchDetail(id: number): MatchDetail | null {
  const match = mockMatches.find(m => m.id === id);
  if (!match) return null;

  return {
    ...match,
    tournament: { id: 1, name: 'Premier League', country: 'England', url_slug: 'premier-league' },
    season: { id: 1, tournament_id: 1, name: '2025/2026', start_year: 2025, end_year: 2026, url_suffix: null },
    home_team: { id: 1, name: 'Manchester United' },
    away_team: { id: 2, name: 'Arsenal' },
    predictions: [],
    h2h_matches: [],
  };
}
```

**MSW Setup Integration:**

```typescript
// app/providers.tsx (add MSW initialization)
'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from 'next-themes';
import { ReactNode, useEffect, useState } from 'react';

async function initMSW() {
  if (process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_ENABLE_MOCKING === 'true') {
    const { worker } = await import('@/mocks/browser');
    await worker.start({ onUnhandledRequest: 'bypass' });
  }
}

export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());
  const [mswReady, setMswReady] = useState(false);

  useEffect(() => {
    initMSW().then(() => setMswReady(true));
  }, []);

  // In production or when mocking is disabled, render immediately
  if (process.env.NODE_ENV !== 'development' || process.env.NEXT_PUBLIC_ENABLE_MOCKING !== 'true') {
    return (
      <ThemeProvider attribute="class" defaultTheme="dark">
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      </ThemeProvider>
    );
  }

  // Wait for MSW to initialize in development with mocking enabled
  if (!mswReady) return null;

  return (
    <ThemeProvider attribute="class" defaultTheme="dark">
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </ThemeProvider>
  );
}
```

### 2.6 API Client with Error Handling

```typescript
// lib/api/client.ts

import { QueryClient } from '@tanstack/react-query';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class APIError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public response?: Response
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new APIError(
      errorData.detail || errorData.message || `API Error: ${response.statusText}`,
      response.status,
      response
    );
  }

  return response.json();
}

// TanStack Query client configuration
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,  // 5 minutes
      gcTime: 1000 * 60 * 30,    // 30 minutes (formerly cacheTime)
      retry: (failureCount, error) => {
        if (error instanceof APIError && error.statusCode >= 400 && error.statusCode < 500) {
          return false;  // Don't retry client errors
        }
        return failureCount < 3;
      },
      refetchOnWindowFocus: false,
    },
  },
});
```

### 2.5 TanStack Query Hook Patterns

```typescript
// lib/queries/use-matches.ts

import { useQuery, useInfiniteQuery } from '@tanstack/react-query';
import { fetchAPI } from '@/lib/api/client';
import { Match, MatchDetail, MatchFilters } from '@/lib/types/api';

const MATCHES_KEY = 'matches';

export function useMatches(filters: MatchFilters = {}) {
  const queryString = new URLSearchParams();

  if (filters.status) queryString.set('status', filters.status);
  if (filters.tournament_id) queryString.set('tournament_id', String(filters.tournament_id));
  if (filters.season_id) queryString.set('season_id', String(filters.season_id));
  if (filters.team_id) queryString.set('team_id', String(filters.team_id));
  if (filters.from_date) queryString.set('from_date', filters.from_date);
  if (filters.to_date) queryString.set('to_date', filters.to_date);
  if (filters.days_ahead) queryString.set('days_ahead', String(filters.days_ahead));
  if (filters.has_odds !== undefined) queryString.set('has_odds', String(filters.has_odds));

  return useInfiniteQuery({
    queryKey: [MATCHES_KEY, filters],
    queryFn: async ({ pageParam = 0 }) => {
      queryString.set('offset', String(pageParam));
      queryString.set('limit', String(filters.limit || 50));
      return fetchAPI<Match[]>(`/api/v1/matches?${queryString.toString()}`);
    },
    getNextPageParam: (lastPage, pages) => {
      if (lastPage.length < (filters.limit || 50)) return undefined;
      return pages.length * (filters.limit || 50);
    },
    initialPageParam: 0,
  });
}

export function useMatch(id: number) {
  return useQuery({
    queryKey: [MATCHES_KEY, id],
    queryFn: () => fetchAPI<MatchDetail>(`/api/v1/matches/${id}`),
    enabled: !!id,
  });
}

export function useMatchH2H(id: number) {
  return useQuery({
    queryKey: [MATCHES_KEY, id, 'h2h'],
    queryFn: () => fetchAPI<Match[]>(`/api/v1/matches/${id}/h2h`),
    enabled: !!id,
  });
}
```

```typescript
// lib/queries/use-predictions.ts

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchAPI } from '@/lib/api/client';
import { Prediction, PredictionWithMatch, PredictionFilters, ValueBet } from '@/lib/types/api';

const PREDICTIONS_KEY = 'predictions';

export function usePredictions(filters: PredictionFilters = {}) {
  const queryString = new URLSearchParams();

  if (filters.match_id) queryString.set('match_id', String(filters.match_id));
  if (filters.model_version_id) queryString.set('model_version_id', String(filters.model_version_id));
  if (filters.has_result !== undefined) queryString.set('has_result', String(filters.has_result));
  if (filters.from_date) queryString.set('from_date', filters.from_date);
  if (filters.to_date) queryString.set('to_date', filters.to_date);
  if (filters.min_confidence) queryString.set('min_confidence', String(filters.min_confidence));

  return useQuery({
    queryKey: [PREDICTIONS_KEY, filters],
    queryFn: () => fetchAPI<PredictionWithMatch[]>(`/api/v1/predictions?${queryString.toString()}`),
  });
}

export function useUpcomingPredictions(days: number = 7) {
  return useQuery({
    queryKey: [PREDICTIONS_KEY, 'upcoming', days],
    queryFn: () => fetchAPI<PredictionWithMatch[]>(`/api/v1/predictions/upcoming?days=${days}`),
  });
}

export function useGeneratePredictions() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { match_ids?: number[]; tournament_id?: number; days_ahead?: number }) =>
      fetchAPI<Prediction[]>('/api/v1/predictions/generate', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [PREDICTIONS_KEY] });
    },
  });
}

export function useValueBets(minEV: number = 0.05, days: number = 7) {
  return useQuery({
    queryKey: ['value-bets', minEV, days],
    queryFn: () => fetchAPI<ValueBet[]>(`/api/v1/value-bets?min_ev=${minEV}&days=${days}`),
    staleTime: 1000 * 60 * 2,  // 2 minutes - value bets change frequently
  });
}
```

```typescript
// lib/queries/use-models.ts

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchAPI } from '@/lib/api/client';
import { ModelVersion } from '@/lib/types/api';

const MODELS_KEY = 'models';

export function useModels(activeOnly: boolean = false) {
  const queryString = activeOnly ? '?active_only=true' : '';

  return useQuery({
    queryKey: [MODELS_KEY, { activeOnly }],
    queryFn: () => fetchAPI<ModelVersion[]>(`/api/v1/models${queryString}`),
  });
}

export function useActiveModel() {
  return useQuery({
    queryKey: [MODELS_KEY, 'active'],
    queryFn: () => fetchAPI<ModelVersion>('/api/v1/models/active'),
  });
}

export function useModel(id: number) {
  return useQuery({
    queryKey: [MODELS_KEY, id],
    queryFn: () => fetchAPI<ModelVersion>(`/api/v1/models/${id}`),
    enabled: !!id,
  });
}

export function useActivateModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) =>
      fetchAPI(`/api/v1/models/${id}/activate`, { method: 'POST' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [MODELS_KEY] });
    },
  });
}
```

### 2.9 WebSocket Real-Time Updates

Real-time updates for live matches, odds changes, and prediction notifications:

```typescript
// lib/socket/client.ts

type MessageHandler = (data: unknown) => void;

class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private handlers: Map<string, Set<MessageHandler>> = new Map();

  constructor(private url: string) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onerror = (error) => {
        reject(error);
      };

      this.ws.onclose = () => {
        this.attemptReconnect();
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          const { type, data } = message;

          const typeHandlers = this.handlers.get(type);
          if (typeHandlers) {
            typeHandlers.forEach(handler => handler(data));
          }
        } catch (error) {
          console.error('WebSocket message parse error:', error);
        }
      };
    });
  }

  subscribe(eventType: string, handler: MessageHandler): () => void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);

    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler);
    };
  }

  send(type: string, data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }));
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      setTimeout(() => this.connect(), delay);
    }
  }

  disconnect(): void {
    this.ws?.close();
    this.handlers.clear();
  }
}

export const wsClient = new WebSocketClient(
  process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'
);
```

```typescript
// lib/socket/use-match-updates.ts

import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { wsClient } from './client';
import type { Match } from '@/lib/types/api';

interface MatchUpdate {
  match_id: number;
  status?: string;
  home_score?: number;
  away_score?: number;
  odds_home?: number;
  odds_draw?: number;
  odds_away?: number;
}

export function useMatchUpdates(matchId?: number) {
  const queryClient = useQueryClient();

  useEffect(() => {
    // Subscribe to match updates
    const unsubscribe = wsClient.subscribe('match_update', (data) => {
      const update = data as MatchUpdate;

      // If watching a specific match, only process that match's updates
      if (matchId && update.match_id !== matchId) return;

      // Update the query cache optimistically
      queryClient.setQueryData<Match[]>(['matches'], (oldData) => {
        if (!oldData) return oldData;
        return oldData.map((match) =>
          match.id === update.match_id
            ? { ...match, ...update }
            : match
        );
      });

      // Also update individual match cache
      queryClient.setQueryData(['matches', update.match_id], (oldData: Match | undefined) => {
        if (!oldData) return oldData;
        return { ...oldData, ...update };
      });
    });

    return unsubscribe;
  }, [matchId, queryClient]);
}

export function useLiveOddsUpdates() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const unsubscribe = wsClient.subscribe('odds_update', (data) => {
      const update = data as MatchUpdate;

      // Update match with new odds
      queryClient.setQueryData<Match[]>(['matches'], (oldData) => {
        if (!oldData) return oldData;
        return oldData.map((match) =>
          match.id === update.match_id
            ? {
                ...match,
                odds_home: update.odds_home ?? match.odds_home,
                odds_draw: update.odds_draw ?? match.odds_draw,
                odds_away: update.odds_away ?? match.odds_away,
              }
            : match
        );
      });
    });

    return unsubscribe;
  }, [queryClient]);
}
```

```typescript
// Usage in components
// components/matches/live-match-card.tsx

'use client';

import { useMatch } from '@/lib/queries/use-matches';
import { useMatchUpdates } from '@/lib/socket/use-match-updates';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader } from '@/components/ui/card';

export function LiveMatchCard({ matchId }: { matchId: number }) {
  const { data: match, isLoading } = useMatch(matchId);

  // Subscribe to real-time updates for this match
  useMatchUpdates(matchId);

  if (isLoading || !match) return <MatchCardSkeleton />;

  return (
    <Card className="border-primary/50">
      <CardHeader className="flex flex-row items-center justify-between">
        <span className="text-sm text-muted-foreground">
          {match.tournament.name}
        </span>
        <Badge variant="destructive" className="animate-pulse">
          LIVE
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between text-lg font-semibold">
          <span>{match.home_team.name}</span>
          <span className="text-2xl font-bold text-primary">
            {match.home_score ?? 0} - {match.away_score ?? 0}
          </span>
          <span>{match.away_team.name}</span>
        </div>
      </CardContent>
    </Card>
  );
}
```

### 2.10 Zustand Store Structure

```typescript
// stores/filter-store.ts

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { MatchStatus } from '@/lib/types/api';

interface FilterState {
  // Match filters
  selectedTournament: number | null;
  selectedSeason: number | null;
  selectedStatus: MatchStatus | null;
  dateRange: { from: Date | null; to: Date | null } | null;

  // Prediction filters
  minConfidence: number;
  selectedModelVersion: number | null;

  // Actions
  setTournament: (id: number | null) => void;
  setSeason: (id: number | null) => void;
  setStatus: (status: MatchStatus | null) => void;
  setDateRange: (range: { from: Date | null; to: Date | null } | null) => void;
  setMinConfidence: (confidence: number) => void;
  setModelVersion: (id: number | null) => void;
  resetFilters: () => void;
}

export const useFilterStore = create<FilterState>()(
  persist(
    (set) => ({
      selectedTournament: null,
      selectedSeason: null,
      selectedStatus: 'SCHEDULED',
      dateRange: null,
      minConfidence: 0,
      selectedModelVersion: null,

      setTournament: (id) => set({ selectedTournament: id, selectedSeason: null }),
      setSeason: (id) => set({ selectedSeason: id }),
      setStatus: (status) => set({ selectedStatus: status }),
      setDateRange: (range) => set({ dateRange: range }),
      setMinConfidence: (confidence) => set({ minConfidence: confidence }),
      setModelVersion: (id) => set({ selectedModelVersion: id }),
      resetFilters: () => set({
        selectedTournament: null,
        selectedSeason: null,
        selectedStatus: 'SCHEDULED',
        dateRange: null,
        minConfidence: 0,
      }),
    }),
    {
      name: 'algobet-filters',
    }
  )
);
```

```typescript
// stores/ui-store.ts

import { create } from 'zustand';

interface UIState {
  sidebarOpen: boolean;
  activeModal: string | null;
  modalData: unknown;

  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  openModal: (modal: string, data?: unknown) => void;
  closeModal: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  activeModal: null,
  modalData: null,

  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  openModal: (modal, data) => set({ activeModal: modal, modalData: data }),
  closeModal: () => set({ activeModal: null, modalData: null }),
}));
```

---

## Phase 3: Core Pages & Features

### 3.0 Suspense Boundaries & Error Handling

Leverage React 18+ Suspense for streaming and provide resilient error handling:

#### Suspense Boundaries for Parallel Data Streaming

```typescript
// app/matches/page.tsx

import { Suspense } from 'react';
import { MatchList } from '@/components/matches/match-list';
import { MatchFilters } from '@/components/matches/match-filters';
import { MatchListSkeleton } from '@/components/matches/match-list-skeleton';
import { FiltersSkeleton } from '@/components/matches/filters-skeleton';

export default function MatchesPage() {
  return (
    <div className="container mx-auto py-8 space-y-6">
      <h1 className="text-3xl font-bold">Matches</h1>

      {/* Filters load independently */}
      <Suspense fallback={<FiltersSkeleton />}>
        <MatchFilters />
      </Suspense>

      {/* Match list streams in as data loads */}
      <Suspense fallback={<MatchListSkeleton count={10} />}>
        <MatchList />
      </Suspense>
    </div>
  );
}
```

```typescript
// app/dashboard/page.tsx

import { Suspense } from 'react';
import { UpcomingMatches } from '@/components/dashboard/upcoming-matches';
import { ValueBetsSummary } from '@/components/dashboard/value-bets-summary';
import { ActiveModelCard } from '@/components/dashboard/active-model-card';
import { PredictionAccuracyChart } from '@/components/charts/prediction-accuracy';
import {
  UpcomingMatchesSkeleton,
  ValueBetsSkeleton,
  ModelCardSkeleton,
  ChartSkeleton
} from '@/components/skeletons';

export default function DashboardPage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Each section loads independently for faster perceived performance */}
        <Suspense fallback={<UpcomingMatchesSkeleton />}>
          <UpcomingMatches />
        </Suspense>

        <Suspense fallback={<ValueBetsSkeleton />}>
          <ValueBetsSummary />
        </Suspense>

        <Suspense fallback={<ModelCardSkeleton />}>
          <ActiveModelCard />
        </Suspense>
      </div>

      <div className="mt-8">
        <Suspense fallback={<ChartSkeleton height={300} />}>
          <PredictionAccuracyChart />
        </Suspense>
      </div>
    </div>
  );
}
```

#### Route-Level Error Boundaries

```typescript
// app/matches/error.tsx

'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface ErrorBoundaryProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function MatchError({ error, reset }: ErrorBoundaryProps) {
  useEffect(() => {
    // Log to error monitoring service (e.g., Sentry)
    console.error('Match page error:', error);
  }, [error]);

  return (
    <div className="container mx-auto py-8">
      <Card className="border-destructive/50 bg-destructive/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertCircle className="h-5 w-5" />
            Something went wrong
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">
            {error.message || 'Failed to load matches. Please try again.'}
          </p>
          {error.digest && (
            <p className="text-xs text-muted-foreground">
              Error ID: {error.digest}
            </p>
          )}
          <div className="flex gap-4">
            <Button onClick={reset} variant="outline">
              <RefreshCw className="h-4 w-4 mr-2" />
              Try Again
            </Button>
            <Button variant="ghost" asChild>
              <a href="/">Go to Dashboard</a>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

#### Reusable Error Boundary Component

```typescript
// components/error-boundary.tsx

'use client';

import { Component, ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { AlertTriangle } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error) => void;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error) {
    this.props.onError?.(error);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="p-4 rounded-lg border border-destructive/50 bg-destructive/10">
          <div className="flex items-center gap-2 text-destructive mb-2">
            <AlertTriangle className="h-4 w-4" />
            <span className="font-medium">Something went wrong</span>
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <Button
            size="sm"
            variant="outline"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try Again
          </Button>
        </div>
      );
    }

    return this.props.children;
  }
}
```

#### Skeleton Components Library

```typescript
// components/skeletons/index.tsx

import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardHeader } from '@/components/ui/card';

export function MatchListSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-4">
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="space-y-2">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-6 w-48" />
              </div>
              <div className="space-y-2 text-right">
                <Skeleton className="h-4 w-20 ml-auto" />
                <Skeleton className="h-6 w-16 ml-auto" />
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function UpcomingMatchesSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-6 w-40" />
      </CardHeader>
      <CardContent className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex justify-between items-center">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-6 w-16" />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function ValueBetsSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-6 w-32" />
      </CardHeader>
      <CardContent className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="flex justify-between items-center">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-4 w-20" />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function ModelCardSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-6 w-28" />
      </CardHeader>
      <CardContent className="space-y-3">
        <Skeleton className="h-8 w-full" />
        <div className="flex gap-4">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-20" />
        </div>
      </CardContent>
    </Card>
  );
}

export function ChartSkeleton({ height = 200 }: { height?: number }) {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-6 w-40" />
      </CardHeader>
      <CardContent>
        <Skeleton className="w-full" style={{ height }} />
      </CardContent>
    </Card>
  );
}

export function FiltersSkeleton() {
  return (
    <div className="flex gap-4 flex-wrap">
      <Skeleton className="h-10 w-40" />
      <Skeleton className="h-10 w-32" />
      <Skeleton className="h-10 w-48" />
      <Skeleton className="h-10 w-24" />
    </div>
  );
}
```

### 3.1 Dashboard (`/`)
**Purpose**: At-a-glance view of upcoming matches, top predictions, system health

**Components**:
- Upcoming matches with prediction confidence badges
- Active model indicator with accuracy metrics
- Quick value bets summary
- Recent prediction accuracy chart

**Data Requirements**:
```typescript
// Parallel data fetching for dashboard
const dashboardData = await Promise.all([
  fetchAPI('/api/v1/matches?status=SCHEDULED&days_ahead=3&limit=5'),
  fetchAPI('/api/v1/models/active'),
  fetchAPI('/api/v1/value-bets?max_matches=5'),
  fetchAPI('/api/v1/predictions/history?limit=30'),
]);
```

### 3.2 Matches Page (`/matches`)
**Purpose**: Browse matches with filtering

**Features**:
- Filter by tournament, season, date range, status
- Sort by date, confidence, value
- Infinite scroll with TanStack Query
- Quick prediction preview on hover

**URL State Synchronization**:
```typescript
// Sync filters with URL query params for shareable URLs
const router = useRouter();
const searchParams = useSearchParams();

// Read initial state from URL
const initialFilters = {
  status: searchParams.get('status') as MatchStatus || 'SCHEDULED',
  tournament_id: searchParams.get('tournament_id')
    ? Number(searchParams.get('tournament_id'))
    : null,
  // ...
};
```

### 3.3 Match Detail (`/matches/[id]`)
**Purpose**: Full match analysis

**Display**:
- Teams with logos (placeholder icons initially)
- Match date/time, venue
- Historical head-to-head results table (from [`MatchRepository.get_h2h_matches()`](algobet/predictions/data/queries.py:115))
- Team form charts (last 5, 10 matches via [`FormCalculator`](algobet/predictions/features/form_features.py:10))
- Odds comparison (bookmaker vs implied)
- Prediction card with probability breakdown
- Confidence score with explanation
- Value bet indicator (if applicable)

### 3.4 Predictions Page (`/predictions`)
**Purpose**: View prediction history and accuracy

**Features**:
- Table of past predictions with outcomes
- Filter by model version, date range, outcome
- Accuracy metrics dashboard
- Calibration curve visualization
- Export predictions (CSV/JSON)

### 3.5 Upcoming Predictions (`/predictions/upcoming`)
**Purpose**: Generate predictions for upcoming matches

**Workflow**:
1. Select model version
2. Filter matches (tournament, days ahead)
3. Batch generate predictions (POST `/api/v1/predictions/generate`)
4. Review results
5. Save to database

### 3.6 Models Page (`/models`)
**Purpose**: Model registry and management

**Display**:
- List of all model versions (from [`ModelRegistry.list_models()`](algobet/predictions/models/registry.py:208))
- Algorithm type, creation date, metrics
- Active model toggle
- Promote/deactivate models
- Delete models (with confirmation)

### 3.7 Model Detail (`/models/[id]`)
**Purpose**: Deep dive into model performance

**Visualizations**:
- Feature importance bar chart
- Confusion matrix heatmap
- ROC curves (one-vs-rest)
- Calibration plot
- Per-metric time series
- Betting simulation results (ROI curve)

### 3.8 Value Bets (`/value-bets`)
**Purpose**: Identify profitable betting opportunities

**Display**:
- Ranked list of value bets by expected value
- Model probability vs market odds comparison
- Kelly criterion recommended stake
- Historical ROI of value bets

### 3.9 Teams Page (`/teams`)
**Purpose**: Team directory and search

**Features**:
- Search by name, country
- Filter by tournament
- Team card with recent form (from [`FormCalculator.get_form_breakdown()`](algobet/predictions/features/form_features.py:159))

### 3.10 Team Detail (`/teams/[id]`)
**Purpose**: Team performance analysis

**Display**:
- Team info (name, id)
- Current season record
- Form chart (points trend)
- Goal stats (scored/conceded)
- Home vs away performance comparison
- Upcoming fixtures with predictions

---

## Phase 4: UI/UX Design Principles

### 4.1 Color Scheme (Dark Mode First)
```css
/* Tailwind custom colors in tailwind.config.ts */
export default {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: 'hsl(var(--card))',
        'card-foreground': 'hsl(var(--card-foreground))',
        primary: 'hsl(var(--primary))',
        'primary-foreground': 'hsl(var(--primary-foreground))',
        success: 'hsl(var(--success))',
        warning: 'hsl(var(--warning))',
        danger: 'hsl(var(--danger))',
      },
    },
  },
}

/* globals.css */
@layer base {
  :root {
    --background: 222 47% 11%;      /* Deep blue-black: #0f172a */
    --foreground: 210 40% 98%;      /* Off-white: #f8fafc */
    --card: 217 33% 17%;            /* Dark card: #1e293b */
    --card-foreground: 210 40% 98%;
    --primary: 262 83% 58%;         /* Purple accent: #8b5cf6 */
    --primary-foreground: 210 40% 98%;
    --success: 142 76% 36%;         /* Green for wins: #16a34a */
    --warning: 38 92% 50%;          /* Yellow for draws: #f59e0b */
    --danger: 0 84% 60%;            /* Red for losses: #ef4444 */
    --muted: 215 28% 25%;           /* Muted text background */
    --muted-foreground: 215 20% 65%; /* Muted text: #94a3b8 */
    --border: 217 33% 25%;
    --input: 217 33% 25%;
    --ring: 262 83% 58%;
  }
}
```

### 4.2 Confidence Visualization

Confidence scores should be displayed with consistent visual indicators:

```typescript
// components/predictions/confidence-badge.tsx
interface ConfidenceBadgeProps {
  confidence: number;  // 0.0 to 1.0
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const getVariant = () => {
    if (confidence >= 0.7) return 'success';
    if (confidence >= 0.5) return 'warning';
    return 'destructive';
  };

  const getLabel = () => {
    if (confidence >= 0.7) return 'High';
    if (confidence >= 0.5) return 'Medium';
    return 'Low';
  };

  return (
    <Badge variant={getVariant()}>
      {getLabel()} Confidence ({(confidence * 100).toFixed(0)}%)
    </Badge>
  );
}
```

### 4.3 Design Patterns

**Cards**: Consistent card component with:
- Header (title + action button)
- Body (content)
- Footer (metadata)

**Tables**:
- Sticky header
- Sortable columns
- Row hover effects
- Pagination or infinite scroll

**Charts**:
- Consistent color palette
- Tooltips on hover
- Responsive container

**Forms**:
- Inline validation with Zod
- Clear error messages
- Loading states
- Submit button disabled until valid
- Accessible labels and ARIA attributes

### 4.4 Accessibility (a11y) Requirements

- WCAG 2.1 Level AA compliance
- Semantic HTML structure
- ARIA labels for interactive elements
- Keyboard navigation support
- Focus indicators
- Color contrast ratio >= 4.5:1
- Screen reader friendly data tables
- Reduced motion support

---

## Phase 5: Implementation Phases

### Sprint 1: Foundation (Week 1)
1. Initialize Next.js project with TypeScript
2. Set up shadcn/ui with dark mode
3. Configure Tailwind with custom theme colors
4. Set up TanStack Query provider
5. Create base layout with navigation
6. Implement API client with error handling
7. Set up Zustand stores

### Sprint 2: Core Pages (Week 2)
1. Dashboard page with data fetching
2. Matches listing page with infinite scroll
3. Match detail page with H2H and form
4. Implement basic FastAPI backend endpoints
5. Add loading skeletons and error boundaries

### Sprint 3: Predictions & Models (Week 3)
1. Predictions listing page
2. Upcoming predictions workflow
3. Models registry page
4. Model detail with metrics visualization
5. Model activation flow

### Sprint 4: Advanced Features (Week 4)
1. Value bets page
2. Teams directory and detail pages
3. Team form visualization
4. Chart components for metrics
5. Export functionality (CSV/JSON)

### Sprint 5: Polish & Testing (Week 5)
1. Responsive design fixes (mobile-first)
2. Loading states and error boundaries
3. Performance optimization (React.memo, lazy loading)
4. Unit tests for utils and hooks
5. Integration tests for key flows
6. End-to-end testing (Playwright)
7. Accessibility audit
8. Documentation

---

## Phase 6: Backend API Changes Required

### 6.1 FastAPI Dependencies (add to pyproject.toml)
```toml
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "python-jose[cryptography]>=3.3.0",  # For auth (optional)
    "python-multipart>=0.0.6",  # For form data
]
```

### 6.2 Environment Variables

Existing variables (from [`algobet/database.py`](algobet/database.py)):
```bash
# Existing .env variables (shared with CLI)
POSTGRES_USER=algobet
POSTGRES_PASSWORD=password
POSTGRES_DB=football
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

New API-specific variables:
```bash
# .env
DATABASE_URL=postgresql://user:pass@localhost:5432/algobet
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,https://algobet.example.com
API_ENV=development  # development | production
LOG_LEVEL=info
```

New frontend environment variables (`.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=AlgoBet
NEXT_PUBLIC_APP_VERSION=1.0.0
```

### 6.3 FastAPI Backend Integration with Existing Modules

The FastAPI layer integrates directly with existing prediction engine components:

#### Integration Summary

| Existing Module | FastAPI Usage | Example Endpoint |
|-----------------|---------------|------------------|
| [`database.py`](algobet/database.py) | Session dependency injection | All CRUD endpoints |
| [`models.py`](algobet/models.py) | Pydantic schema mapping | GET `/matches/{id}` |
| [`MatchRepository`](algobet/predictions/data/queries.py:12) | Complex queries | GET `/teams/{id}/form` |
| [`FormCalculator`](algobet/predictions/features/form_features.py:10) | Form metrics computation | GET `/matches/{id}/preview` |
| [`ModelRegistry`](algobet/predictions/models/registry.py:47) | Model lifecycle management | POST `/models/{id}/activate` |
| [`scraper.py`](algobet/scraper.py) | Trigger scraping jobs | POST `/scrape` |

```python
# algobet/api/dependencies.py
from fastapi import Depends
from sqlalchemy.orm import Session
from algobet.database import session_scope

def get_db() -> Session:
    """FastAPI dependency for request-scoped database sessions."""
    with session_scope() as session:
        yield session

# algobet/api/routers/teams.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from algobet.api.dependencies import get_db
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.form_features import FormCalculator

router = APIRouter()

@router.get("/teams/{team_id}/form")
def get_team_form(team_id: int, db: Session = Depends(get_db)):
    repo = MatchRepository(db)
    calc = FormCalculator(repo)
    return calc.get_form_breakdown(
        team_id=team_id,
        reference_date=datetime.now(),
        n_matches=5
    )
```

---

## Phase 7: Testing Strategy

### 7.1 Unit Tests
- **Utils**: Formatting, odds calculations
- **Hooks**: Custom React hooks with React Testing Library
- **Stores**: Zustand store actions
- **Schemas**: Zod validation testing

### 7.2 Integration Tests with MSW
- API client error handling
- Query hook data fetching
- Form validation flows
- WebSocket connection handling

#### MSW Test Setup

```typescript
// tests/setup.ts

import { beforeAll, afterEach, afterAll } from 'vitest';
import { server } from '@/mocks/server';
import '@testing-library/jest-dom/vitest';

// Start server before all tests
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));

// Reset handlers after each test
afterEach(() => server.resetHandlers());

// Close server after all tests
afterAll(() => server.close());
```

```typescript
// vitest.config.ts

import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    globals: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'tests/', 'mocks/'],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './'),
    },
  },
});
```

#### Hook Testing Example

```typescript
// tests/unit/hooks/use-matches.test.ts

import { describe, it, expect } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useMatches } from '@/lib/queries/use-matches';
import { ReactNode } from 'react';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('useMatches', () => {
  it('fetches scheduled matches', async () => {
    const { result } = renderHook(
      () => useMatches({ status: 'SCHEDULED' }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.pages[0]).toHaveLength(1);
    expect(result.current.data?.pages[0][0].status).toBe('SCHEDULED');
  });

  it('handles API errors gracefully', async () => {
    // Override handler for error testing
    server.use(
      http.get('*/api/v1/matches', () => {
        return new HttpResponse(null, { status: 500 });
      })
    );

    const { result } = renderHook(
      () => useMatches(),
      { wrapper: createWrapper() }
    );

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toBeDefined();
  });
});
```

#### Component Testing Example

```typescript
// tests/integration/match-card.test.tsx

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MatchCard } from '@/components/matches/match-card';
import { mockMatches } from '@/mocks/data/matches';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } },
});

describe('MatchCard', () => {
  it('renders match details correctly', () => {
    const match = mockMatches[0];

    render(
      <QueryClientProvider client={queryClient}>
        <MatchCard match={match} />
      </QueryClientProvider>
    );

    expect(screen.getByText('Manchester United')).toBeInTheDocument();
    expect(screen.getByText('Arsenal')).toBeInTheDocument();
    expect(screen.getByText('SCHEDULED')).toBeInTheDocument();
  });

  it('displays odds when available', () => {
    const match = { ...mockMatches[0], odds_home: 2.10 };

    render(
      <QueryClientProvider client={queryClient}>
        <MatchCard match={match} />
      </QueryClientProvider>
    );

    expect(screen.getByText('2.10')).toBeInTheDocument();
  });
});
```

### 7.3 E2E Tests (Playwright)
- Match listing and filtering
- Prediction generation workflow
- Model activation flow
- Value bets display
- Real-time updates (WebSocket)

#### Playwright Configuration

```typescript
// playwright.config.ts

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
```

#### E2E Test Example

```typescript
// tests/e2e/matches.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Matches Page', () => {
  test('displays scheduled matches', async ({ page }) => {
    await page.goto('/matches');

    // Wait for loading to complete
    await expect(page.locator('[data-testid="match-list"]')).toBeVisible();

    // Verify matches are displayed
    const matchCards = page.locator('[data-testid="match-card"]');
    await expect(matchCards).toHaveCount(10);
  });

  test('filters matches by status', async ({ page }) => {
    await page.goto('/matches');

    // Select "Finished" status
    await page.getByRole('combobox', { name: 'Status' }).click();
    await page.getByRole('option', { name: 'Finished' }).click();

    // Wait for filtered results
    await page.waitForResponse('**/api/v1/matches*status=FINISHED*');

    // Verify all displayed matches are finished
    const statuses = page.locator('[data-testid="match-status"]');
    for (const status of await statuses.all()) {
      await expect(status).toHaveText('FINISHED');
    }
  });

  test('navigates to match detail', async ({ page }) => {
    await page.goto('/matches');

    // Click on first match
    await page.locator('[data-testid="match-card"]').first().click();

    // Verify navigation to detail page
    await expect(page).toHaveURL(/\/matches\/\d+/);
    await expect(page.locator('h1')).toContainText('vs');
  });
});
```

### 7.4 Test Structure
```
frontend/
├── tests/
│   ├── setup.ts               # Global test setup with MSW
│   ├── unit/
│   │   ├── utils/
│   │   │   ├── format.test.ts
│   │   │   └── odds.test.ts
│   │   ├── hooks/
│   │   │   ├── use-matches.test.ts
│   │   │   ├── use-predictions.test.ts
│   │   │   └── use-debounce.test.ts
│   │   ├── stores/
│   │   │   ├── filter-store.test.ts
│   │   │   └── ui-store.test.ts
│   │   └── schemas/
│   │       └── validation.test.ts
│   ├── integration/
│   │   ├── api/
│   │   │   ├── client.test.ts
│   │   │   └── error-handling.test.ts
│   │   └── components/
│   │       ├── match-card.test.tsx
│   │       ├── prediction-card.test.tsx
│   │       └── filters.test.tsx
│   └── e2e/
│       ├── matches.spec.ts
│       ├── predictions.spec.ts
│       ├── models.spec.ts
│       └── value-bets.spec.ts
├── vitest.config.ts
└── playwright.config.ts
```

---

## Phase 8: Deployment & CI/CD

### 8.1 Frontend Deployment
- **Vercel**: Recommended for Next.js
- Automatic previews on PRs
- Edge functions for API routes (if needed)
- Environment variables via Vercel dashboard

### 8.2 Backend Deployment
- **Docker**: Containerize FastAPI app
- **Update docker-compose.yml**: Add API service

```yaml
# docker-compose.yml additions
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    command: uvicorn algobet.api.main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    volumes:
      - ./algobet:/app/algobet
      - ./data:/app/data
    depends_on:
      - db
    environment:
      - DATABASE_URL=postgresql://algobet:password@db:5432/football
      - CORS_ORIGINS=http://localhost:3000
```

### 8.3 Infrastructure
- PostgreSQL: Existing in docker-compose
- Optional: Redis for caching query results
- Optional: Cloud storage for model artifacts (S3/GCS)

### 8.4 CI/CD Pipeline with GitHub Actions

#### Frontend CI Workflow

```yaml
# .github/workflows/frontend-ci.yml

name: Frontend CI

on:
  push:
    branches: [main, develop]
    paths:
      - 'frontend/**'
      - '.github/workflows/frontend-ci.yml'
  pull_request:
    branches: [main, develop]
    paths:
      - 'frontend/**'

defaults:
  run:
    working-directory: frontend

jobs:
  lint:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Run ESLint
        run: npm run lint

      - name: Run TypeScript type check
        run: npm run typecheck

  test:
    name: Unit & Integration Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Run Vitest
        run: npm run test:coverage

      - name: Upload coverage report
        uses: codecov/codecov-action@v4
        with:
          file: ./frontend/coverage/coverage-final.json
          flags: frontend
          fail_ci_if_error: false

  e2e:
    name: E2E Tests
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps

      - name: Run Playwright tests
        run: npm run test:e2e
        env:
          NEXT_PUBLIC_API_URL: http://localhost:8000
          NEXT_PUBLIC_ENABLE_MOCKING: 'true'

      - name: Upload Playwright report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-report
          path: frontend/playwright-report/
          retention-days: 30

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Build Next.js app
        run: npm run build
        env:
          NEXT_PUBLIC_API_URL: ${{ secrets.API_URL }}

      - name: Upload build artifact
        uses: actions/upload-artifact@v4
        with:
          name: nextjs-build
          path: frontend/.next/
          retention-days: 7
```

#### Backend CI Workflow

```yaml
# .github/workflows/backend-ci.yml

name: Backend CI

on:
  push:
    branches: [main, develop]
    paths:
      - 'algobet/**'
      - 'tests/**'
      - 'pyproject.toml'
      - '.github/workflows/backend-ci.yml'
  pull_request:
    branches: [main, develop]
    paths:
      - 'algobet/**'
      - 'tests/**'
      - 'pyproject.toml'

jobs:
  lint:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[dev]" --system

      - name: Run Ruff
        run: ruff check algobet/

      - name: Run mypy
        run: mypy algobet/

  test:
    name: Tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: algobet
          POSTGRES_PASSWORD: password
          POSTGRES_DB: football_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[dev]" --system

      - name: Run pytest
        run: pytest tests/ --cov=algobet --cov-report=xml
        env:
          POSTGRES_HOST: localhost
          POSTGRES_PORT: 5432
          POSTGRES_USER: algobet
          POSTGRES_PASSWORD: password
          POSTGRES_DB: football_test

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: backend
```

#### CD Workflow (Deploy)

```yaml
# .github/workflows/deploy.yml

name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-frontend:
    name: Deploy Frontend to Vercel
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod'
          working-directory: frontend

  deploy-backend:
    name: Deploy Backend
    runs-on: ubuntu-latest
    needs: [deploy-frontend]
    steps:
      - uses: actions/checkout@v4

      - name: Deploy API to production
        # Customize based on your hosting provider
        run: |
          echo "Deploy to your preferred hosting (Railway, Fly.io, AWS, etc.)"
```

### 8.5 OpenAPI Type Generation

Automatically generate TypeScript types from FastAPI's OpenAPI schema to ensure type consistency:

#### Generation Script

```bash
# scripts/generate-api-types.sh
#!/bin/bash

# Fetch OpenAPI schema from running API
API_URL="${API_URL:-http://localhost:8000}"
OUTPUT_DIR="frontend/lib/types"

echo "Fetching OpenAPI schema from $API_URL..."

# Generate TypeScript types
npx openapi-typescript "$API_URL/openapi.json" \
  --output "$OUTPUT_DIR/api-generated.ts" \
  --export-type \
  --immutable-types

echo "Types generated at $OUTPUT_DIR/api-generated.ts"
```

#### Package.json Script

```json
{
  "scripts": {
    "generate:types": "bash ../scripts/generate-api-types.sh",
    "generate:types:watch": "nodemon --watch ../algobet/api --ext py --exec 'npm run generate:types'"
  }
}
```

#### Usage with Generated Types

```typescript
// lib/api/client.ts (updated to use generated types)

import type { paths, components } from '@/lib/types/api-generated';

// Extract response types from generated schema
type MatchResponse = components['schemas']['MatchResponse'];
type PredictionResponse = components['schemas']['PredictionResponse'];

// Type-safe API calls
export async function getMatches(
  params?: paths['/api/v1/matches']['get']['parameters']['query']
): Promise<MatchResponse[]> {
  const queryString = new URLSearchParams(params as Record<string, string>);
  return fetchAPI(`/api/v1/matches?${queryString}`);
}

export async function getMatch(
  id: number
): Promise<paths['/api/v1/matches/{id}']['get']['responses']['200']['content']['application/json']> {
  return fetchAPI(`/api/v1/matches/${id}`);
}
```

#### CI Integration for Type Generation

```yaml
# Add to .github/workflows/frontend-ci.yml

  type-sync:
    name: Verify API Types Sync
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: algobet
          POSTGRES_PASSWORD: password
          POSTGRES_DB: football
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install backend dependencies
        run: pip install -e ".[dev]"

      - name: Start API server
        run: |
          uvicorn algobet.api.main:app --host 0.0.0.0 --port 8000 &
          sleep 5
        env:
          POSTGRES_HOST: localhost
          POSTGRES_DB: football

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Generate types
        working-directory: frontend
        run: |
          npm ci
          npm run generate:types

      - name: Check for type changes
        run: |
          if [[ -n $(git status --porcelain frontend/lib/types/api-generated.ts) ]]; then
            echo "::error::API types are out of sync. Run 'npm run generate:types' and commit the changes."
            exit 1
          fi
```

## Phase 9: Performance Optimization Guidelines

### 9.1 Data Fetching
- Use TanStack Query caching strategically
- Implement pagination for large lists
- Use `staleTime` to prevent excessive refetching
- Parallel fetching with `Promise.all` for independent queries

### 9.2 Rendering
- Use React Server Components where possible
- Implement virtualization for long lists (react-window)
- Lazy load heavy chart components
- Use `React.memo` for expensive renders

### 9.3 Bundle Size
- Tree-shake unused components
- Dynamic imports for route-heavy components
- Analyze bundle with @next/bundle-analyzer

### 9.4 Images
- Use Next.js Image component for optimization
- Provide blur placeholders for better UX

---

## Phase 10: Future Enhancements

1. **Authentication**: User accounts for saved predictions, custom models
2. ~~**WebSocket**: Real-time odds updates and live match status~~ ✅ *Implemented in Section 2.9*
3. **Mobile App**: React Native wrapper or PWA
4. **Export**: PDF reports, scheduled email digests
5. **Advanced Analytics**: A/B testing of models, ROI tracking per strategy
6. **Social Features**: Share predictions, community leaderboards
7. **Notifications**: Alert when value bets are detected (push notifications)
8. **Import/Export**: User data import/export (CSV, JSON)
9. **Multi-language Support**: i18n with next-intl
10. **Offline Support**: Service worker for offline access to cached data

---

## Appendix A: Quick Reference

### API Endpoint Summary

| Method | Endpoint | Description | Query Params |
|--------|----------|-------------|--------------|
| GET | `/tournaments` | List tournaments | - |
| GET | `/tournaments/{id}/seasons` | Get seasons | - |
| GET | `/matches` | List matches | status, tournament_id, days_ahead, etc. |
| GET | `/matches/{id}` | Match details | - |
| GET | `/matches/{id}/h2h` | Head-to-head | - |
| GET | `/teams` | List teams | search, tournament_id |
| GET | `/teams/{id}/form` | Team form | n_matches, reference_date |
| GET | `/predictions` | List predictions | match_id, model_version_id, etc. |
| POST | `/predictions/generate` | Generate predictions | Body: match_ids, etc. |
| GET | `/value-bets` | Value bet opportunities | min_ev, days, max_matches |
| GET | `/models` | List models | active_only, algorithm |
| POST | `/models/{id}/activate` | Activate model | - |
| WS | `/ws` | WebSocket connection | - |

### WebSocket Event Types

| Event Type | Direction | Description |
|------------|-----------|-------------|
| `match_update` | Server → Client | Live match score/status update |
| `odds_update` | Server → Client | Real-time odds change |
| `prediction_ready` | Server → Client | New prediction generated |
| `subscribe` | Client → Server | Subscribe to specific match updates |
| `unsubscribe` | Client → Server | Unsubscribe from match updates |

### Component Naming Convention
- Pages: `page.tsx` (Next.js convention)
- Layouts: `layout.tsx`
- Loading states: `loading.tsx`
- Error boundaries: `error.tsx`
- Components: kebab-case (e.g., `match-card.tsx`)
- Hooks: camelCase with `use` prefix (e.g., `use-matches.ts`)
- Stores: kebab-case (e.g., `filter-store.ts`)
- Skeletons: `*-skeleton.tsx` (e.g., `match-list-skeleton.tsx`)
- Tests: `*.test.ts` or `*.spec.ts`

### NPM Scripts Reference

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Production build |
| `npm run lint` | Run ESLint |
| `npm run typecheck` | TypeScript type checking |
| `npm run test` | Run Vitest unit/integration tests |
| `npm run test:coverage` | Tests with coverage report |
| `npm run test:e2e` | Run Playwright E2E tests |
| `npm run generate:types` | Generate API types from OpenAPI |

---

*Document Version: 2.0*
*Last Updated: 2026-02-01*

### Changelog

#### v2.0 (2026-02-01)
- Added Zod runtime validation schemas (Section 2.4)
- Added MSW API mocking for development/testing (Section 2.5)
- Added WebSocket real-time updates (Section 2.9)
- Added Suspense boundaries and Error Handling patterns (Section 3.0)
- Added comprehensive skeleton components library
- Expanded testing section with MSW integration, Vitest, and Playwright examples
- Added CI/CD pipelines with GitHub Actions (Section 8.4)
- Added OpenAPI type generation workflow (Section 8.5)
- Updated Future Enhancements with completed items
