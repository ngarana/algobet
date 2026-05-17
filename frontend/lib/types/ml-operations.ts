/**
 * TypeScript types and Zod schemas for ML operations
 */

import { z } from "zod";

const numberWithDefault = (
  defaultValue: number
): z.ZodEffects<z.ZodNumber, number, unknown> =>
  z.preprocess((value) => value ?? defaultValue, z.number());

// =============================================================================
// Train Model Types
// =============================================================================

export const TrainModelRequestSchema = z.object({
  model_type: z
    .enum(["xgboost", "lightgbm", "random_forest", "dixon_coles", "hybrid_poisson"])
    .default("xgboost"),
  tune_hyperparameters: z.boolean().default(false),
  description: z.string().trim().min(1).max(500).optional(),
  activate: z.boolean().default(true),
  // Data range settings
  start_date: z.string().optional(),
  end_date: z.string().optional(),
  min_matches: z.number().min(10).max(100000).default(100),
  // Filtering: tournament and team selection
  tournament_ids: z.array(z.number()).optional(),
  team_ids: z.array(z.number()).optional(),
  venue_filter: z.enum(["home", "away", "both"]).optional(),
  // Match quality filters
  min_total_goals: z.number().min(0).optional(),
  max_total_goals: z.number().min(0).optional(),
  // Train/val/test split ratios
  train_ratio: z.number().min(0.1).max(0.9).default(0.7),
  val_ratio: z.number().min(0.05).max(0.45).default(0.15),
  test_ratio: z.number().min(0.05).max(0.45).default(0.15),
  // Training settings
  random_seed: z.number().min(0).max(999999).default(42),
  early_stopping_rounds: z.number().min(10).max(500).default(50),
  tuning_trials: z.number().min(10).max(500).default(50),
  // Calibration settings
  calibrate_probabilities: z.boolean().default(true),
  calibration_method: z
    .enum(["isotonic", "sigmoid", "temperature", "venn_abers"])
    .default("temperature"),
  // Outcome balancing
  outcome_balance: z.boolean().optional(),
  outcome_balance_strength: z.number().min(0).max(1).default(0.5),
  // Feature groups selection
  feature_groups: z.array(z.string()).optional(),
  // Feature importance pruning
  feature_selection: z.boolean().default(false),
  feature_selection_threshold: z.number().min(0).max(1).default(0.01),
  min_samples_per_feature: z.number().min(1).optional(),
  // Ensemble training
  use_ensemble: z.boolean().default(false),
  ensemble_types: z.array(z.string()).optional(),
  // Split strategy
  split_strategy: z
    .enum(["temporal", "expanding_window", "season_aware", "walk_forward"])
    .default("temporal"),
  gap_days: z.number().min(0).max(30).default(0),
  // Expanding window params
  min_train_size: z.number().min(50).max(5000).default(100),
  ew_val_size: z.number().min(10).max(1000).default(50),
  ew_test_size: z.number().min(10).max(1000).default(50),
  step_size: z.number().min(10).max(500).default(50),
  // Season-aware params
  train_seasons: z.number().min(1).max(10).default(3),
  val_seasons: z.number().min(1).max(5).default(1),
  test_seasons: z.number().min(1).max(5).default(1),
  // Model tags
  tags: z.record(z.string()).default({}),
  // Custom hyperparameters
  hyperparameters: z.record(z.unknown()).default({}),
});

export const TrainModelResultSchema = z.object({
  model_id: z.number().nullable(),
  model_version: z.string(),
  model_type: z.string(),
  is_active: z.boolean(),
  feature_schema_version: z.string(),
  num_features: z.number(),
  trained_at: z.string(),
  training_duration_seconds: z.number(),
  train_metrics: z.record(z.number()),
  val_metrics: z.record(z.number()),
  test_metrics: z.record(z.number()),
  feature_importance: z.record(z.number()).nullable(),
  // Ensemble metadata
  ensemble_weights: z.record(z.number()).nullable(),
  ensemble_validation_metrics: z.record(z.number()).nullable(),
  ensemble_types: z.array(z.string()).nullable(),
});

// =============================================================================
// Backtest Types
// =============================================================================

export const BacktestRequestSchema = z.object({
  model_version: z.string().optional(),
  tournament_id: z.number().optional(),
  start_date: z.string().optional(),
  end_date: z.string().optional(),
  min_matches: z.number().min(10).max(10000).default(100),
  min_edge: z.number().min(0).max(1).default(0).optional(),
});

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
});

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
  mean_clv: numberWithDefault(0),
  clv_hit_rate: numberWithDefault(0),
  clv_weighted_roi: numberWithDefault(0),
});

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
});

// =============================================================================
// Calibrate Types
// =============================================================================

export const CalibrateRequestSchema = z.object({
  model_version: z.string().optional(),
  method: z
    .enum(["isotonic", "sigmoid", "temperature", "venn_abers"])
    .default("isotonic"),
  validation_split: z.number().min(0.1).max(0.5).default(0.2),
  activate: z.boolean().default(true),
});

export const CalibrationMetricsSchema = z.object({
  expected_calibration_error: z.number(),
  maximum_calibration_error: z.number(),
  brier_score: z.number(),
  log_loss: z.number(),
});

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
});

// =============================================================================
// TypeScript Interfaces
// =============================================================================

export type TrainModelRequest = z.infer<typeof TrainModelRequestSchema>;
export type TrainModelResult = z.infer<typeof TrainModelResultSchema>;

export type BacktestRequest = z.infer<typeof BacktestRequestSchema>;
export type ClassificationMetrics = z.infer<typeof ClassificationMetricsSchema>;
export type BettingMetrics = z.infer<typeof BettingMetricsSchema>;
export type BacktestResult = z.infer<typeof BacktestResultSchema>;

export type CalibrateRequest = z.infer<typeof CalibrateRequestSchema>;
export type CalibrationMetrics = z.infer<typeof CalibrationMetricsSchema>;
export type CalibrateResult = z.infer<typeof CalibrateResultSchema>;

// =============================================================================
// Backtest History Types
// =============================================================================

export const BacktestHistoryItemSchema = z.object({
  id: z.number(),
  model_version_id: z.number(),
  model_name: z.string().nullable(),
  model_version: z.string().nullable(),
  num_samples: z.number(),
  date_range_start: z.string().nullable(),
  date_range_end: z.string().nullable(),
  accuracy: z.number(),
  log_loss: z.number(),
  f1_macro: z.number(),
  roi_percent: z.number().nullable(),
  win_rate: z.number().nullable(),
  evaluated_at: z.string(),
});

export const BacktestHistoryListSchema = z.object({
  items: z.array(BacktestHistoryItemSchema),
  total: z.number(),
});

export type BacktestHistoryItem = z.infer<typeof BacktestHistoryItemSchema>;
export type BacktestHistoryList = z.infer<typeof BacktestHistoryListSchema>;
