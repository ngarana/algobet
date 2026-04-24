/**
 * API functions for predictions
 */

import { apiGet, apiPost, buildQueryString } from "./client";
import type { Prediction, PredictionFilters, PaginatedResponse } from "@/lib/types/api";
import { z } from "zod";
import {
  PredictionRecordSchema,
  createPaginatedResponseSchema,
} from "@/lib/types/schemas";

const predictionArraySchema = createPaginatedResponseSchema(PredictionRecordSchema);
const generatePredictionsResultSchema = z.object({
  generated: z.number(),
  prediction_ids: z.array(z.number()),
  model_version: z.string(),
  matches_processed: z.number(),
  existing_predictions_skipped: z.number(),
});

export interface GeneratePredictionsRequest {
  match_ids?: number[];
  model_version?: string;
  tournament_id?: number;
  days_ahead?: number;
}

export interface GeneratePredictionsResult {
  generated: number;
  prediction_ids: number[];
  model_version: string;
  matches_processed: number;
  existing_predictions_skipped: number;
}

export async function getPredictions(
  filters?: PredictionFilters
): Promise<PaginatedResponse<Prediction>> {
  const queryString = filters ? buildQueryString(filters) : "";
  return apiGet(`/predictions${queryString}`, predictionArraySchema);
}

export async function generatePredictions(
  request: GeneratePredictionsRequest
): Promise<GeneratePredictionsResult> {
  return apiPost(
    "/predictions/generate",
    request,
    generatePredictionsResultSchema
  );
}

export async function getUpcomingPredictions(
  daysAhead?: number,
  modelVersionId?: number
): Promise<PaginatedResponse<Prediction>> {
  const queryString = buildQueryString({
    days: daysAhead,
    model_version_id: modelVersionId,
  });
  return apiGet(`/predictions/upcoming${queryString}`, predictionArraySchema);
}

export async function getPredictionHistory(
  params?: {
    from_date?: string;
    to_date?: string;
    model_version_id?: number;
    limit?: number;
  }
): Promise<PaginatedResponse<Prediction>> {
  const queryString = params ? buildQueryString(params) : "";
  return apiGet(`/predictions/history${queryString}`, predictionArraySchema);
}
