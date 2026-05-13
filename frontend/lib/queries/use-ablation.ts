/**
 * TanStack Query hooks for ablation / permutation importance
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { runAblation } from "@/lib/api/ablation";
import type { AblationRequest, AblationResponse } from "@/lib/types/ablation";

export const ablationKeys = {
  all: ["ablation"] as const,
  permutation: () => [...ablationKeys.all, "permutation"] as const,
  ablationStudy: () => [...ablationKeys.all, "study"] as const,
};

/**
 * Run ablation / permutation importance mutation
 */
export function useAblation() {
  const queryClient = useQueryClient();

  return useMutation<AblationResponse, Error, AblationRequest>({
    mutationFn: (request: AblationRequest) => runAblation(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ablationKeys.all });
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}
