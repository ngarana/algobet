/**
 * TypeScript types for AlgoBet API
 * Aligned with backend Pydantic schemas
 */

// Common types
export interface FormBreakdown {
  avg_points: number
  win_rate: number
  draw_rate: number
  loss_rate: number
  avg_goals_for: number
  avg_goals_against: number
}

// Tournament types
export interface Tournament {
  id: number
  name: string
  country: string
  url_slug: string
}

export interface Season {
  id: number
  tournament_id: number
  name: string
  start_year: number
  end_year: number
  url_suffix: string | null
}

// Team types
export interface Team {
  id: number
  name: string
}

export interface TeamWithStats extends Team {
  total_matches: number
  wins: number
  draws: number
  losses: number
  goals_for: number
  goals_against: number
  current_form: FormBreakdown
}

// Match types
export type MatchStatus = 'SCHEDULED' | 'FINISHED' | 'LIVE'
export type PredictedOutcome = 'H' | 'D' | 'A'

export interface Match {
  id: number
  tournament_id: number
  season_id: number
  home_team_id: number
  away_team_id: number
  match_date: string
  home_score: number | null
  away_score: number | null
  status: MatchStatus
  odds_home: number | null
  odds_draw: number | null
  odds_away: number | null
  num_bookmakers: number | null
  created_at: string
  updated_at: string
  result: PredictedOutcome | null
}

export interface MatchDetail extends Match {
  tournament: Tournament
  season: Season
  home_team: Team
  away_team: Team
  predictions: Prediction[]
  h2h_matches: Match[]
}

export interface MatchFilters {
  status?: MatchStatus
  tournament_id?: number
  season_id?: number
  team_id?: number
  from_date?: string
  to_date?: string
  days_ahead?: number
  has_odds?: boolean
  limit?: number
  offset?: number
}

// Prediction types
export interface Prediction {
  id: number
  match_id: number
  model_version_id: number
  prob_home: number
  prob_draw: number
  prob_away: number
  predicted_outcome: PredictedOutcome
  confidence: number
  predicted_at: string
  actual_roi: number | null
  max_probability: number
}

export interface PredictionWithMatch extends Prediction {
  match: MatchDetail
  model_version: ModelVersion
}

export interface PredictionFilters {
  match_id?: number
  model_version_id?: number
  has_result?: boolean
  from_date?: string
  to_date?: string
  min_confidence?: number
}

// Value Bet types
export interface ValueBet {
  match: Match
  prediction_id: number
  predicted_outcome: PredictedOutcome
  predicted_probability: number
  market_odds: number
  expected_value: number
  kelly_fraction: number
  confidence: number
}

// Model types
export interface ModelVersion {
  id: number
  name: string
  version: string
  algorithm: string
  accuracy: number | null
  file_path: string
  is_active: boolean
  created_at: string
  metrics: Record<string, unknown> | null
  hyperparameters: Record<string, unknown> | null
  feature_schema_version: string | null
  description: string | null
}

// API Response types
export interface ApiResponse<T> {
  data: T
  message?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  limit: number
  offset: number
}

// Scraping types (from existing implementation)
export interface ScrapingProgress {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  current_page: number
  total_pages: number
  matches_found: number
  message?: string
  error?: string
}

export interface ScrapingConfig {
  tournament_id: number
  season_id: number
  start_page?: number
  max_pages?: number
}
