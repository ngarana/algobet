/**
 * TypeScript types for AlgoBet API
 * Aligned with backend Pydantic schemas
 */

// Common types
export interface FormBreakdown {
  avg_points: number;
  win_rate: number;
  draw_rate: number;
  loss_rate: number;
  avg_goals_for: number;
  avg_goals_against: number;
}

// Tournament types
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

// Team types
export interface Team {
  id: number;
  name: string;
}

export interface TeamWithStats extends Team {
  total_matches: number;
  wins: number;
  draws: number;
  losses: number;
  goals_for: number;
  goals_against: number;
  current_form: FormBreakdown;
}

// Match types
export type MatchStatus = "SCHEDULED" | "FINISHED" | "LIVE";
export type PredictedOutcome = "H" | "D" | "A";

export interface Match {
  id: number;
  tournament_id: number | null;
  season_id: number | null;
  home_team_id: number;
  away_team_id: number;
  match_date: string;
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
  home_team_name?: string | null;
  away_team_name?: string | null;
  tournament_name?: string | null;
  season_name?: string | null;
}

export interface MatchDetail extends Match {
  tournament: Tournament;
  season: Season;
  home_team: TeamWithStats;
  away_team: TeamWithStats;
  predictions: Prediction[];
  h2h_matches: Match[];
}

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

// Prediction types
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
  match?: PredictionMatchSummary | null;
  model_version?: ModelVersion | null;
}

export interface PredictionMatchSummary {
  id: number;
  match_date: string;
  status: MatchStatus;
  home_team_name: string;
  away_team_name: string;
  tournament_name: string | null;
  season_name: string | null;
  home_score: number | null;
  away_score: number | null;
  odds_home: number | null;
  odds_draw: number | null;
  odds_away: number | null;
}

export interface PredictionWithMatch extends Omit<
  Prediction,
  "match" | "model_version"
> {
  match: MatchDetail;
  model_version: ModelVersion;
}

export interface PredictionFilters {
  match_id?: number;
  model_version_id?: number;
  has_result?: boolean;
  from_date?: string;
  to_date?: string;
  min_confidence?: number;
}

// Value Bet types
export interface ValueBet {
  match: Match;
  prediction_id: number;
  predicted_outcome: PredictedOutcome;
  predicted_probability: number;
  market_odds: number;
  expected_value: number;
  kelly_fraction: number;
  confidence: number;
}

// Daily workflow types
export type WatchlistEntryType = "team" | "tournament" | "match";
export type TotalGoalsPick = "OVER" | "UNDER";

export interface ProfilePreferences {
  profile_key: string;
  display_name: string;
  default_days_ahead: number;
  min_confidence: number;
  min_ev: number;
  favorite_bookie: string | null;
  followed_tournament_ids: number[];
}

export interface ProfilePreferencesUpdate {
  display_name?: string;
  default_days_ahead?: number;
  min_confidence?: number;
  min_ev?: number;
  favorite_bookie?: string | null;
  followed_tournament_ids?: number[];
}

export interface WatchlistEntry {
  id: number;
  entry_type: WatchlistEntryType;
  entry_id: number;
  label: string;
  meta: string | null;
  created_at: string;
}

export interface Watchlist {
  teams: WatchlistEntry[];
  tournaments: WatchlistEntry[];
  matches: WatchlistEntry[];
}

export interface WatchlistEntryRequest {
  entry_type: WatchlistEntryType;
  entry_id: number;
}

export interface UserPredictionRequest {
  match_id: number;
  pick_1x2?: PredictedOutcome | null;
  home_score?: number | null;
  away_score?: number | null;
  total_goals_line?: number | null;
  total_goals_pick?: TotalGoalsPick | null;
  notes?: string | null;
}

export interface UserPrediction {
  id: number;
  match_id: number;
  pick_1x2: PredictedOutcome | null;
  home_score: number | null;
  away_score: number | null;
  total_goals_line: number | null;
  total_goals_pick: TotalGoalsPick | null;
  notes: string | null;
  model_prediction: Prediction | null;
  is_correct_1x2: boolean | null;
  is_exact_score: boolean | null;
  points: number;
  created_at: string;
  updated_at: string;
}

export interface ResultsSummary {
  label: string;
  start_date: string;
  end_date: string;
  model_predictions: number;
  model_correct: number;
  model_accuracy: number | null;
  user_predictions: number;
  user_correct: number;
  user_accuracy: number | null;
}

export interface DailyWorkflow {
  date: string;
  today_matches: Prediction[];
  high_confidence: Prediction[];
  value_bets: ValueBet[];
  watched_fixtures: MatchDetail[];
  results_summary: ResultsSummary;
  watchlist: Watchlist;
}

export interface ResultsReviewItem {
  match: Match;
  model_prediction: Prediction | null;
  user_prediction: UserPrediction | null;
  actual_result: PredictedOutcome | null;
  model_correct: boolean | null;
  user_correct: boolean | null;
}

export interface ResultsReview {
  summaries: ResultsSummary[];
  items: ResultsReviewItem[];
}

export interface MatchOddsRow {
  bookmaker: string;
  odds_home: number;
  odds_draw: number;
  odds_away: number;
  scraped_at: string | null;
  source: string | null;
}

export interface RecentTeamMatch {
  match_id: number;
  match_date: string;
  opponent_name: string;
  venue: string;
  goals_for: number;
  goals_against: number;
  result: string;
}

export interface RecentForm {
  home: RecentTeamMatch[];
  away: RecentTeamMatch[];
}

export interface TeamStatsComparison {
  team_id: number;
  team_name: string;
  matches: number;
  avg_goals_for: number;
  avg_goals_against: number;
  avg_shots: number | null;
  avg_shots_on_target: number | null;
  avg_corners: number | null;
}

export interface StatsComparison {
  home: TeamStatsComparison;
  away: TeamStatsComparison;
}

export interface ModelFeatureExplanation {
  feature: string;
  label: string;
  value: number;
  direction: string;
  impact: number;
}

export interface SimilarAccuracy {
  sample_size: number;
  correct: number;
  accuracy: number | null;
  description: string;
}

export interface MatchWorkflowDetail {
  match: MatchDetail;
  odds_comparison: MatchOddsRow[];
  recent_form: RecentForm;
  stats_comparison: StatsComparison;
  model_explanation: ModelFeatureExplanation[];
  similar_accuracy: SimilarAccuracy;
  user_prediction: UserPrediction | null;
  watched: boolean;
}

// Model types
export interface ModelVersion {
  id: number;
  name: string;
  version: string;
  algorithm: string;
  accuracy: number | null;
  file_path: string;
  is_active: boolean;
  created_at: string;
  metrics: Record<string, unknown> | null;
  hyperparameters: Record<string, unknown> | null;
  feature_schema_version: string | null;
  description: string | null;
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

// Scraping types (from existing implementation)
export interface ScrapingProgress {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  current_page: number;
  total_pages: number;
  matches_found: number;
  message?: string;
  error?: string;
}

export interface ScrapingConfig {
  tournament_id: number;
  season_id: number;
  start_page?: number;
  max_pages?: number;
}
