/**
 * API functions for model versions
 */

import { apiGet, apiPost, apiDelete } from "./client";
import type { ModelVersion } from "@/lib/types/api";
import { ModelVersionSchema, createPaginatedResponseSchema } from "@/lib/types/schemas";

const modelArraySchema = createPaginatedResponseSchema(ModelVersionSchema);

export async function getModels(): Promise<{ items: ModelVersion[] }> {
  return apiGet("/models", modelArraySchema);
}

export async function getActiveModel(): Promise<ModelVersion> {
  return apiGet("/models/active", ModelVersionSchema);
}

export async function getModel(id: number): Promise<ModelVersion> {
  return apiGet(`/models/${id}`, ModelVersionSchema);
}

export async function activateModel(id: number): Promise<ModelVersion> {
  return apiPost(`/models/${id}/activate`, {});
}

export async function deleteModel(id: number): Promise<void> {
  return apiDelete(`/models/${id}`);
}

export async function getModelMetrics(id: number): Promise<Record<string, unknown>> {
  return apiGet(`/models/${id}/metrics`);
}
