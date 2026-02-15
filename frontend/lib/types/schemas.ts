/**
 * Zod schemas for runtime validation of API responses
 */

import { z } from 'zod'

// Common schemas
export const FormBreakdownSchema = z.object({
  avg_points: z.number(),
  win_rate: z.number(),
  draw_rate: z.number(),
  loss_rate: z.number(),
  avg_goals_for: z.number(),
  avg_goals_against: z.number(),
})

// Tournament schemas
export const TournamentSchema = z.object({
  id: z.number(),
  name: z.string(),
  country: z.string(),
  url_slug: z.string(),
})

export const SeasonSchema = z.object({
  id: z.number(),
  tournament_id: z.number(),
  name: z.string(),
  start_year: z.number(),
  end_year: z.number(),
  url_suffix: z.string().nullable(),
})

// Team schemas
export const TeamSchema = z.object({
  id: z.number(),
  name: z.string(),
})

export const TeamWithStatsSchema = TeamSchema.extend({
  total_matches: z.number(),
  wins: z.number(),
  draws: z.number(),
  losses: z.number(),
  goals_for: z.number(),
  goals_against: z.number(),
  current_form: FormBreakdownSchema,
})

// Match schemas
export const MatchStatusSchema = z.enum(['SCHEDULED', 'FINISHED', 'LIVE'])
export const PredictedOutcomeSchema = z.enum(['H', 'D', 'A'])

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
})

export const MatchDetailSchema = MatchSchema.extend({
  tournament: TournamentSchema,
  season: SeasonSchema,
  home_team: TeamSchema,
  away_team: TeamSchema,
  predictions: z.array(z.any()), // Will be defined after PredictionSchema
  h2h_matches: z.array(MatchSchema),
})

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
})

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
})

// Update MatchDetailSchema predictions field now that PredictionSchema is defined
export const MatchDetailSchemaComplete = MatchSchema.extend({
  tournament: TournamentSchema,
  season: SeasonSchema,
  home_team: TeamSchema,
  away_team: TeamSchema,
  predictions: z.array(PredictionSchema),
  h2h_matches: z.array(MatchSchema),
})

export const PredictionWithMatchSchema = PredictionSchema.extend({
  match: MatchDetailSchemaComplete,
  model_version: z.any(), // Will be defined after ModelVersionSchema
})

export const PredictionFiltersSchema = z.object({
  match_id: z.number().optional(),
  model_version_id: z.number().optional(),
  has_result: z.boolean().optional(),
  from_date: z.string().optional(),
  to_date: z.string().optional(),
  min_confidence: z.number().min(0).max(1).optional(),
})

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
})

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
})

// Update PredictionWithMatchSchema with complete ModelVersionSchema
export const PredictionWithMatchSchemaComplete = PredictionSchema.extend({
  match: MatchDetailSchemaComplete,
  model_version: ModelVersionSchema,
})

// API Response schemas
export function createApiResponseSchema<T extends z.ZodTypeAny>(dataSchema: T) {
  return z.object({
    data: dataSchema,
    message: z.string().optional(),
  })
}

export function createPaginatedResponseSchema<T extends z.ZodTypeAny>(itemSchema: T) {
  return z.object({
    items: z.array(itemSchema),
    total: z.number(),
    limit: z.number(),
    offset: z.number(),
  })
}

// Scraping schemas (from existing implementation)
export const ScrapingProgressSchema = z.object({
  job_id: z.string(),
  status: z.enum(['pending', 'running', 'completed', 'failed']),
  progress: z.number(),
  current_page: z.number(),
  total_pages: z.number(),
  matches_found: z.number(),
  message: z.string().optional(),
  error: z.string().optional(),
})

export const ScrapingConfigSchema = z.object({
  tournament_id: z.number(),
  season_id: z.number(),
  start_page: z.number().optional(),
  max_pages: z.number().optional(),
})
