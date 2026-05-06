/**
 * Zod schemas for runtime validation of API responses
 */

import { z } from "zod";

// Common schemas
export const FormBreakdownSchema = z.object({
  avg_points: z.number(),
  win_rate: z.number(),
  draw_rate: z.number(),
  loss_rate: z.number(),
  avg_goals_for: z.number(),
  avg_goals_against: z.number(),
});

// Tournament schemas
export const TournamentSchema = z.object({
  id: z.number(),
  name: z.string(),
  country: z.string(),
  url_slug: z.string(),
});

export const SeasonSchema = z.object({
  id: z.number(),
  tournament_id: z.number(),
  name: z.string(),
  start_year: z.number(),
  end_year: z.number(),
  url_suffix: z.string().nullable(),
});

// Team schemas
export const TeamSchema = z.object({
  id: z.number(),
  name: z.string(),
});

export const TeamWithStatsSchema = TeamSchema.extend({
  total_matches: z.number(),
  wins: z.number(),
  draws: z.number(),
  losses: z.number(),
  goals_for: z.number(),
  goals_against: z.number(),
  current_form: FormBreakdownSchema,
});

// Match schemas
export const MatchStatusSchema = z.enum(["SCHEDULED", "FINISHED", "LIVE"]);
export const PredictedOutcomeSchema = z.enum(["H", "D", "A"]);

export const MatchSchema = z.object({
  id: z.number(),
  tournament_id: z.number().nullable(),
  season_id: z.number().nullable(),
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
  home_team_name: z.string().nullable().optional(),
  away_team_name: z.string().nullable().optional(),
  tournament_name: z.string().nullable().optional(),
  season_name: z.string().nullable().optional(),
});

export const MatchDetailSchema = MatchSchema.extend({
  tournament: TournamentSchema,
  season: SeasonSchema,
  home_team: TeamWithStatsSchema,
  away_team: TeamWithStatsSchema,
  predictions: z.array(z.any()), // Will be defined after PredictionSchema
  h2h_matches: z.array(MatchSchema),
});

export const MatchFiltersSchema = z.object({
  status: MatchStatusSchema.optional(),
  tournament_id: z.number().optional(),
  season_id: z.number().optional(),
  team_id: z.number().optional(),
  from_date: z.string().optional(),
  to_date: z.string().optional(),
  days_ahead: z.number().optional(),
  has_odds: z.boolean().optional(),
  limit: z.number().min(1).max(100).default(50),
  offset: z.number().default(0),
});

// Prediction schemas
export const PredictionSchema = z.object({
  id: z.number(),
  match_id: z.number(),
  model_version_id: z.number(),
  prob_home: z.number().min(0).max(1),
  prob_draw: z.number().min(0).max(1),
  prob_away: z.number().min(0).max(1),
  predicted_outcome: PredictedOutcomeSchema,
  confidence: z.number().min(0).max(1),
  predicted_at: z.string(),
  actual_roi: z.number().nullable(),
  max_probability: z.number(),
});

export const PredictionMatchSummarySchema = z.object({
  id: z.number(),
  match_date: z.string(),
  status: MatchStatusSchema,
  home_team_name: z.string(),
  away_team_name: z.string(),
  tournament_name: z.string().nullable(),
  season_name: z.string().nullable(),
  home_score: z.number().nullable(),
  away_score: z.number().nullable(),
  odds_home: z.number().nullable(),
  odds_draw: z.number().nullable(),
  odds_away: z.number().nullable(),
});

// Update MatchDetailSchema predictions field now that PredictionSchema is defined
export const MatchDetailSchemaComplete = MatchSchema.extend({
  tournament: TournamentSchema,
  season: SeasonSchema,
  home_team: TeamWithStatsSchema,
  away_team: TeamWithStatsSchema,
  predictions: z.array(PredictionSchema),
  h2h_matches: z.array(MatchSchema),
});

export const PredictionWithMatchSchema = PredictionSchema.extend({
  match: MatchDetailSchemaComplete,
  model_version: z.any(), // Will be defined after ModelVersionSchema
});

export const PredictionFiltersSchema = z.object({
  match_id: z.number().optional(),
  model_version_id: z.number().optional(),
  has_result: z.boolean().optional(),
  from_date: z.string().optional(),
  to_date: z.string().optional(),
  min_confidence: z.number().min(0).max(1).optional(),
});

// Value Bet schemas
export const ValueBetSchema = z.object({
  match: MatchSchema,
  prediction_id: z.number(),
  predicted_outcome: PredictedOutcomeSchema,
  predicted_probability: z.number().min(0).max(1),
  market_odds: z.number(),
  expected_value: z.number(),
  kelly_fraction: z.number(),
  confidence: z.number().min(0).max(1),
});

export const WatchlistEntryTypeSchema = z.enum(["team", "tournament", "match"]);
export const TotalGoalsPickSchema = z.enum(["OVER", "UNDER"]);

export const ProfilePreferencesSchema = z.object({
  profile_key: z.string(),
  display_name: z.string(),
  default_days_ahead: z.number(),
  min_confidence: z.number(),
  min_ev: z.number(),
  favorite_bookie: z.string().nullable(),
  followed_tournament_ids: z.array(z.number()),
});

export const WatchlistEntrySchema = z.object({
  id: z.number(),
  entry_type: WatchlistEntryTypeSchema,
  entry_id: z.number(),
  label: z.string(),
  meta: z.string().nullable(),
  created_at: z.string(),
});

export const WatchlistSchema = z.object({
  teams: z.array(WatchlistEntrySchema),
  tournaments: z.array(WatchlistEntrySchema),
  matches: z.array(WatchlistEntrySchema),
});

export const UserPredictionSchema: z.ZodType<import("./api").UserPrediction> =
  z.object({
    id: z.number(),
    match_id: z.number(),
    pick_1x2: PredictedOutcomeSchema.nullable(),
    home_score: z.number().nullable(),
    away_score: z.number().nullable(),
    total_goals_line: z.number().nullable(),
    total_goals_pick: TotalGoalsPickSchema.nullable(),
    notes: z.string().nullable(),
    model_prediction: z.lazy(() => PredictionRecordSchema).nullable(),
    is_correct_1x2: z.boolean().nullable(),
    is_exact_score: z.boolean().nullable(),
    points: z.number(),
    created_at: z.string(),
    updated_at: z.string(),
  });

export const ResultsSummarySchema = z.object({
  label: z.string(),
  start_date: z.string(),
  end_date: z.string(),
  model_predictions: z.number(),
  model_correct: z.number(),
  model_accuracy: z.number().nullable(),
  user_predictions: z.number(),
  user_correct: z.number(),
  user_accuracy: z.number().nullable(),
});

export const DailyWorkflowSchema = z.object({
  date: z.string(),
  today_matches: z.array(z.lazy(() => PredictionRecordSchema)),
  high_confidence: z.array(z.lazy(() => PredictionRecordSchema)),
  value_bets: z.array(ValueBetSchema),
  watched_fixtures: z.array(MatchDetailSchema),
  results_summary: ResultsSummarySchema,
  watchlist: WatchlistSchema,
});

export const ResultsReviewItemSchema = z.object({
  match: MatchSchema,
  model_prediction: z.lazy(() => PredictionRecordSchema).nullable(),
  user_prediction: UserPredictionSchema.nullable(),
  actual_result: PredictedOutcomeSchema.nullable(),
  model_correct: z.boolean().nullable(),
  user_correct: z.boolean().nullable(),
});

export const ResultsReviewSchema = z.object({
  summaries: z.array(ResultsSummarySchema),
  items: z.array(ResultsReviewItemSchema),
});

export const MatchOddsRowSchema = z.object({
  bookmaker: z.string(),
  odds_home: z.number(),
  odds_draw: z.number(),
  odds_away: z.number(),
  scraped_at: z.string().nullable(),
  source: z.string().nullable(),
});

export const RecentTeamMatchSchema = z.object({
  match_id: z.number(),
  match_date: z.string(),
  opponent_name: z.string(),
  venue: z.string(),
  goals_for: z.number(),
  goals_against: z.number(),
  result: z.string(),
});

export const RecentFormSchema = z.object({
  home: z.array(RecentTeamMatchSchema),
  away: z.array(RecentTeamMatchSchema),
});

export const TeamStatsComparisonSchema = z.object({
  team_id: z.number(),
  team_name: z.string(),
  matches: z.number(),
  avg_goals_for: z.number(),
  avg_goals_against: z.number(),
  avg_shots: z.number().nullable(),
  avg_shots_on_target: z.number().nullable(),
  avg_corners: z.number().nullable(),
});

export const StatsComparisonSchema = z.object({
  home: TeamStatsComparisonSchema,
  away: TeamStatsComparisonSchema,
});

export const ModelFeatureExplanationSchema = z.object({
  feature: z.string(),
  label: z.string(),
  value: z.number(),
  direction: z.string(),
  impact: z.number(),
});

export const SimilarAccuracySchema = z.object({
  sample_size: z.number(),
  correct: z.number(),
  accuracy: z.number().nullable(),
  description: z.string(),
});

export const MatchWorkflowDetailSchema = z.object({
  match: MatchDetailSchema,
  odds_comparison: z.array(MatchOddsRowSchema),
  recent_form: RecentFormSchema,
  stats_comparison: StatsComparisonSchema,
  model_explanation: z.array(ModelFeatureExplanationSchema),
  similar_accuracy: SimilarAccuracySchema,
  user_prediction: UserPredictionSchema.nullable(),
  watched: z.boolean(),
});

// Model schemas
export const ModelVersionSchema = z.object({
  id: z.number(),
  name: z.string(),
  version: z.string(),
  algorithm: z.string(),
  accuracy: z.number().min(0).max(1).nullable(),
  file_path: z.string(),
  is_active: z.boolean(),
  created_at: z.string(),
  metrics: z.record(z.unknown()).nullable(),
  hyperparameters: z.record(z.unknown()).nullable(),
  feature_schema_version: z.string().nullable(),
  description: z.string().nullable(),
});

export const NullableModelVersionSchema = ModelVersionSchema.nullable();

export const PredictionRecordSchema = PredictionSchema.extend({
  match: PredictionMatchSummarySchema.nullable().optional(),
  model_version: NullableModelVersionSchema.optional(),
});

// Update PredictionWithMatchSchema with complete ModelVersionSchema
export const PredictionWithMatchSchemaComplete = PredictionSchema.extend({
  match: MatchDetailSchemaComplete,
  model_version: ModelVersionSchema,
});

// API Response schemas
export function createApiResponseSchema<T extends z.ZodTypeAny>(dataSchema: T) {
  return z.object({
    data: dataSchema,
    message: z.string().optional(),
  });
}

export function createPaginatedResponseSchema<T extends z.ZodTypeAny>(itemSchema: T) {
  return z.object({
    items: z.array(itemSchema),
    total: z.number(),
    limit: z.number(),
    offset: z.number(),
  });
}

// Scraping schemas (from existing implementation)
export const ScrapingProgressSchema = z.object({
  job_id: z.string(),
  status: z.enum(["pending", "running", "completed", "failed"]),
  progress: z.number(),
  current_page: z.number(),
  total_pages: z.number(),
  matches_found: z.number(),
  message: z.string().optional(),
  error: z.string().optional(),
});

export const ScrapingConfigSchema = z.object({
  tournament_id: z.number(),
  season_id: z.number(),
  start_page: z.number().optional(),
  max_pages: z.number().optional(),
});
