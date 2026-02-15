/**
 * TypeScript types and Zod schemas for ML operations
 */

import { z } from 'zod'

// =============================================================================
// Backtest Types
// =============================================================================

export const BacktestRequestSchema = z.object({
  model_version: z.string().optional(),
  start_date: z.string().optional(),
  end_date: z.string().optional(),
  min_matches: z.number().min(10).max(10000).default(100),
  min_edge: z.number().min(0).max(1).default(0).optional(),
})

export const ClassificationMetricsSchema = z.object({
  accuracy: z.number(),
  log_loss: z.number(),
  brier_score: z.number(),
  precision_macro: z.number(),
  recall_macro: z.number(),
  f1_macro: z.number(),
  precision_weighted: z.number(),
  recall_weighted: z.number(),
  f1_weighted: z.number(),
  per_class_precision: z.record(z.number()),
  per_class_recall: z.record(z.number()),
  per_class_f1: z.record(z.number()),
  confusion_matrix: z.array(z.array(z.number())),
  top_2_accuracy: z.number(),
  cohen_kappa: z.number(),
})

export const BettingMetricsSchema = z.object({
  total_bets: z.number(),
  winning_bets: z.number(),
  losing_bets: z.number(),
  total_stake: z.number(),
  total_return: z.number(),
  profit_loss: z.number(),
  roi_percent: z.number(),
  yield_percent: z.number(),
  sharpe_ratio: z.number(),
  max_drawdown: z.number(),
  win_rate: z.number(),
  average_winning_odds: z.number(),
  average_losing_odds: z.number(),
  average_kelly_fraction: z.number(),
  optimal_kelly_fraction: z.number(),
})

export const BacktestResultSchema = z.object({
  model_version: z.string(),
  evaluated_at: z.string(),
  num_samples: z.number(),
  date_range: z.tuple([z.string(), z.string()]).nullable(),
  classification: ClassificationMetricsSchema,
  betting: BettingMetricsSchema.nullable(),
  expected_calibration_error: z.number(),
  maximum_calibration_error: z.number(),
  outcome_accuracy: z.record(z.number()),
})

// =============================================================================
// Calibrate Types
// =============================================================================

export const CalibrateRequestSchema = z.object({
  model_version: z.string().optional(),
  method: z.enum(['isotonic', 'sigmoid']).default('isotonic'),
  validation_split: z.number().min(0.1).max(0.5).default(0.2),
  activate: z.boolean().default(true),
})

export const CalibrationMetricsSchema = z.object({
  expected_calibration_error: z.number(),
  maximum_calibration_error: z.number(),
  brier_score: z.number(),
  log_loss: z.number(),
})

export const CalibrateResultSchema = z.object({
  base_model_version: z.string(),
  calibrated_model_version: z.string(),
  method: z.string(),
  samples_used: z.number(),
  before_metrics: CalibrationMetricsSchema,
  after_metrics: CalibrationMetricsSchema,
  improvement: z.object({
    ece_improvement: z.number(),
    brier_improvement: z.number(),
    log_loss_improvement: z.number(),
  }),
  is_active: z.boolean(),
})

// =============================================================================
// TypeScript Interfaces
// =============================================================================

export type BacktestRequest = z.infer<typeof BacktestRequestSchema>
export type ClassificationMetrics = z.infer<typeof ClassificationMetricsSchema>
export type BettingMetrics = z.infer<typeof BettingMetricsSchema>
export type BacktestResult = z.infer<typeof BacktestResultSchema>

export type CalibrateRequest = z.infer<typeof CalibrateRequestSchema>
export type CalibrationMetrics = z.infer<typeof CalibrationMetricsSchema>
export type CalibrateResult = z.infer<typeof CalibrateResultSchema>
