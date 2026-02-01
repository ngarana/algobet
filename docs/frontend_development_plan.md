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
| **Charts** | Recharts | Declarative, responsive, composable |
| **State** | Zustand | Lightweight, no boilerplate, devtools support |
| **Icons** | Lucide React | Consistent icon set |
| **Date Handling** | date-fns | Lightweight date manipulation |
| **HTTP Client** | Native fetch + tanstack-query | No axios needed with modern fetch |

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
│   │   └── index.ts
│   ├── utils/
│   │   ├── cn.ts                  # Tailwind class merger
│   │   ├── format.ts              # Date/formatting utilities
│   │   └── odds.ts                # Odds calculations
│   └── config.ts                  # Environment configuration
├── hooks/                         # Custom React hooks
│   ├── use-debounce.ts
│   ├── use-local-storage.ts
│   └── use-media-query.ts
├── stores/                        # Zustand stores
│   ├── filter-store.ts            # Global filter state
│   └── ui-store.ts                # UI state (sidebar, etc.)
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

### 2.4 API Client with Error Handling

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

### 2.6 Zustand Store Structure

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

### 7.2 Integration Tests
- API client error handling
- Query hook data fetching
- Form validation flows

### 7.3 E2E Tests (Playwright)
- Match listing and filtering
- Prediction generation workflow
- Model activation flow
- Value bets display

### 7.4 Test Structure
```
frontend/
├── tests/
│   ├── unit/
│   │   ├── utils/
│   │   └── hooks/
│   ├── integration/
│   │   └── api/
│   └── e2e/
│       ├── matches.spec.ts
│       ├── predictions.spec.ts
│       └── models.spec.ts
```

---

## Phase 8: Deployment Considerations

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

---

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
2. **WebSocket**: Real-time odds updates and live match status
3. **Mobile App**: React Native wrapper or PWA
4. **Export**: PDF reports, scheduled email digests
5. **Advanced Analytics**: A/B testing of models, ROI tracking per strategy
6. **Social Features**: Share predictions, community leaderboards
7. **Notifications**: Alert when value bets are detected
8. **Import**: User data import/export

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

### Component Naming Convention
- Pages: `page.tsx` (Next.js convention)
- Layouts: `layout.tsx`
- Loading states: `loading.tsx`
- Error boundaries: `error.tsx`
- Components: kebab-case (e.g., `match-card.tsx`)
- Hooks: camelCase with `use` prefix (e.g., `use-matches.ts`)
- Stores: kebab-case (e.g., `filter-store.ts`)

---

*Document Version: 1.1*
*Last Updated: 2026-02-01*
