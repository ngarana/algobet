/**
 * TypeScript types and Zod schemas for ablation / permutation importance
 */

import { z } from "zod";

// =============================================================================
// Permutation Importance Types
// =============================================================================

export const PermutationFamilyResultSchema = z.object({
  family: z.string(),
  features_in_family: z.array(z.string()),
  features_found: z.array(z.string()),
  baseline_log_loss: z.number(),
  permuted_log_loss: z.number(),
  log_loss_increase: z.number(),
  baseline_accuracy: z.number(),
  permuted_accuracy: z.number(),
  accuracy_decrease: z.number(),
  importance_score: z.number(),
  importance_rank: z.number(),
});

export const PermutationImportanceResponseSchema = z.object({
  method: z.literal("permutation"),
  model_version: z.string(),
  num_samples: z.number(),
  n_repeats: z.number(),
  baseline_log_loss: z.number(),
  baseline_accuracy: z.number(),
  families: z.array(PermutationFamilyResultSchema),
  raw_feature_importance: z.record(z.number()).nullable(),
});

// =============================================================================
// Ablation Types
// =============================================================================

export const AblationModelConfigSchema = z.object({
  model_type: z.enum(["xgboost", "lightgbm", "random_forest"]).default("xgboost"),
  tune_hyperparameters: z.boolean().default(false),
  early_stopping_rounds: z.number().min(10).max(500).default(50),
  calibrate_probabilities: z.boolean().default(true),
  calibration_method: z.enum(["isotonic", "sigmoid"]).default("sigmoid"),
  random_seed: z.number().min(0).max(999999).default(42),
});

export const AblationFamilyResultSchema = z.object({
  family: z.string(),
  features_excluded: z.array(z.string()),
  num_features_used: z.number(),
  model_version: z.string(),
  train_metrics: z.record(z.number()),
  val_metrics: z.record(z.number()),
  test_metrics: z.record(z.number()),
  log_loss_delta: z.number(),
  accuracy_delta: z.number(),
});

export const AblationStudyResponseSchema = z.object({
  method: z.literal("ablation"),
  baseline_model_version: z.string(),
  baseline_num_features: z.number(),
  baseline_train_metrics: z.record(z.number()),
  baseline_val_metrics: z.record(z.number()),
  baseline_test_metrics: z.record(z.number()),
  families: z.array(AblationFamilyResultSchema),
});

// =============================================================================
// Ablation Request
// =============================================================================

export const AblationRequestSchema = z.object({
  method: z.enum(["permutation", "ablation"]).default("permutation"),
  model_version: z.string().optional(),
  n_repeats: z.number().min(1).max(100).default(10),
  random_state: z.number().min(0).max(999999).default(42),
  feature_families: z.array(z.string()).optional(),
  group_by: z.enum(["family", "generator"]).default("family"),
  start_date: z.string().optional(),
  end_date: z.string().optional(),
  tournament_ids: z.array(z.number()).optional(),
  min_matches: z.number().min(10).max(10000).default(100),
  train_ratio: z.number().min(0.1).max(0.9).default(0.7),
  val_ratio: z.number().min(0.05).max(0.45).default(0.15),
  test_ratio: z.number().min(0.05).max(0.45).default(0.15),
  gap_days: z.number().min(0).max(30).default(0),
  ablation_model_config: AblationModelConfigSchema.optional(),
});

// =============================================================================
// TypeScript Interfaces
// =============================================================================

export type PermutationFamilyResult = z.infer<typeof PermutationFamilyResultSchema>;
export type PermutationImportanceResponse = z.infer<
  typeof PermutationImportanceResponseSchema
>;
export type AblationModelConfig = z.infer<typeof AblationModelConfigSchema>;
export type AblationFamilyResult = z.infer<typeof AblationFamilyResultSchema>;
export type AblationStudyResponse = z.infer<typeof AblationStudyResponseSchema>;
export type AblationRequest = z.infer<typeof AblationRequestSchema>;

export type AblationResponse = PermutationImportanceResponse | AblationStudyResponse;
