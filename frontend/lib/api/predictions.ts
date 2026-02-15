/**
 * API functions for predictions
 */

import { apiGet, apiPost, buildQueryString } from "./client";
import type { Prediction, PredictionFilters, PaginatedResponse } from "@/lib/types/api";
import { PredictionSchema, createPaginatedResponseSchema } from "@/lib/types/schemas";

const predictionArraySchema = createPaginatedResponseSchema(PredictionSchema);

export async function getPredictions(
  filters?: PredictionFilters
): Promise<PaginatedResponse<Prediction>> {
  const queryString = filters ? buildQueryString(filters) : "";
  return apiGet(`/predictions${queryString}`, predictionArraySchema);
}

export async function generatePredictions(matchIds: number[]): Promise<Prediction[]> {
  return apiPost("/predictions/generate", { match_ids: matchIds });
}

export async function getUpcomingPredictions(
  daysAhead?: number
): Promise<PaginatedResponse<Prediction>> {
  const queryString = daysAhead ? `?days_ahead=${daysAhead}` : "";
  return apiGet(`/predictions/upcoming${queryString}`, predictionArraySchema);
}

export async function getPredictionHistory(
  fromDate?: string,
  toDate?: string
): Promise<PaginatedResponse<Prediction>> {
  const params: Record<string, unknown> = {};
  if (fromDate) params.from_date = fromDate;
  if (toDate) params.to_date = toDate;
  const queryString = buildQueryString(params);
  return apiGet(`/predictions/history${queryString}`, predictionArraySchema);
}
