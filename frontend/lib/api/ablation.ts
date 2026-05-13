/**
 * API functions for ablation / permutation importance
 */

import { apiPost } from "./client";
import type { AblationRequest, AblationResponse } from "@/lib/types/ablation";
import {
  PermutationImportanceResponseSchema,
  AblationStudyResponseSchema,
} from "@/lib/types/ablation";

/**
 * Run ablation or permutation importance analysis
 */
export async function runAblation(request: AblationRequest): Promise<AblationResponse> {
  if (request.method === "permutation") {
    const result = await apiPost(
      "/ml/ablation",
      request,
      PermutationImportanceResponseSchema
    );
    return result as AblationResponse;
  }
  const result = await apiPost("/ml/ablation", request, AblationStudyResponseSchema);
  return result as AblationResponse;
}
