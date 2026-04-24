/**
 * TanStack Query hooks for predictions
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getPredictions,
  generatePredictions,
  getUpcomingPredictions,
  getPredictionHistory,
} from "@/lib/api/predictions";
import type { PredictionFilters } from "@/lib/types/api";
import type { GeneratePredictionsRequest } from "@/lib/api/predictions";

export const predictionKeys = {
  all: ["predictions"] as const,
  lists: () => [...predictionKeys.all, "list"] as const,
  list: (filters: PredictionFilters | undefined) =>
    [...predictionKeys.lists(), filters] as const,
  upcoming: () => [...predictionKeys.all, "upcoming"] as const,
  history: () => [...predictionKeys.all, "history"] as const,
  detail: (id: number) => [...predictionKeys.all, "detail", id] as const,
};

export function usePredictions(filters?: PredictionFilters) {
  return useQuery({
    queryKey: predictionKeys.list(filters),
    queryFn: () => getPredictions(filters),
  });
}

export function useUpcomingPredictions(daysAhead?: number, modelVersionId?: number) {
  return useQuery({
    queryKey: [...predictionKeys.upcoming(), { daysAhead, modelVersionId }],
    queryFn: () => getUpcomingPredictions(daysAhead, modelVersionId),
  });
}

export function usePredictionHistory(params?: {
  from_date?: string;
  to_date?: string;
  model_version_id?: number;
  limit?: number;
}) {
  return useQuery({
    queryKey: [...predictionKeys.history(), params],
    queryFn: () => getPredictionHistory(params),
  });
}

export function useGeneratePredictions() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: GeneratePredictionsRequest) => generatePredictions(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.all });
    },
  });
}

export function useInvalidatePredictions() {
  const queryClient = useQueryClient();

  return {
    invalidateAll: () =>
      queryClient.invalidateQueries({ queryKey: predictionKeys.all }),
    invalidateList: () =>
      queryClient.invalidateQueries({ queryKey: predictionKeys.lists() }),
  };
}
