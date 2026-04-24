/**
 * API functions for model versions
 */

import { z } from "zod";
import { apiGet, apiPost, apiDelete } from "./client";
import type { ModelVersion } from "@/lib/types/api";
import {
  ModelVersionSchema,
  NullableModelVersionSchema,
  createPaginatedResponseSchema,
} from "@/lib/types/schemas";

const modelArraySchema = createPaginatedResponseSchema(ModelVersionSchema);
const modelMetricsSchema = z.object({
  model_id: z.number(),
  name: z.string(),
  version: z.string(),
  algorithm: z.string(),
  accuracy: z.number().nullable(),
  metrics: z.record(z.unknown()),
  hyperparameters: z.record(z.unknown()),
  feature_schema_version: z.string().nullable(),
  created_at: z.string(),
  is_active: z.boolean(),
});

export type ModelMetricsResponse = z.infer<typeof modelMetricsSchema>;

export async function getModels(): Promise<{ items: ModelVersion[] }> {
  return apiGet("/models", modelArraySchema);
}

export async function getActiveModel(): Promise<ModelVersion | null> {
  return apiGet("/models/active", NullableModelVersionSchema);
}

export async function getModel(id: number): Promise<ModelVersion> {
  return apiGet(`/models/${id}`, ModelVersionSchema);
}

export async function activateModel(id: number): Promise<{
  message: string;
  model_id: number;
  version: string;
}> {
  return apiPost(`/models/${id}/activate`, {});
}

export async function deleteModel(id: number): Promise<void> {
  return apiDelete(`/models/${id}`);
}

export async function getModelMetrics(id: number): Promise<ModelMetricsResponse> {
  return apiGet(`/models/${id}/metrics`, modelMetricsSchema);
}
