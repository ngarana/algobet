/**
 * TanStack Query hooks for predictions
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getPredictions,
  generatePredictions,
  getUpcomingPredictions,
  getPredictionHistory,
} from '@/lib/api/predictions'
import type { PredictionFilters } from '@/lib/types/api'

export const predictionKeys = {
  all: ['predictions'] as const,
  lists: () => [...predictionKeys.all, 'list'] as const,
  list: (filters: PredictionFilters) => [...predictionKeys.lists(), filters] as const,
  upcoming: () => [...predictionKeys.all, 'upcoming'] as const,
  history: () => [...predictionKeys.all, 'history'] as const,
  detail: (id: number) => [...predictionKeys.all, 'detail', id] as const,
}

export function usePredictions(filters?: PredictionFilters) {
  return useQuery({
    queryKey: predictionKeys.list(filters ?? {}),
    queryFn: () => getPredictions(filters),
  })
}

export function useUpcomingPredictions(daysAhead?: number) {
  return useQuery({
    queryKey: predictionKeys.upcoming(),
    queryFn: () => getUpcomingPredictions(daysAhead),
  })
}

export function usePredictionHistory(fromDate?: string, toDate?: string) {
  return useQuery({
    queryKey: [...predictionKeys.history(), { fromDate, toDate }],
    queryFn: () => getPredictionHistory(fromDate, toDate),
  })
}

export function useGeneratePredictions() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: generatePredictions,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.all })
    },
  })
}

export function useInvalidatePredictions() {
  const queryClient = useQueryClient()
  
  return {
    invalidateAll: () => queryClient.invalidateQueries({ queryKey: predictionKeys.all }),
    invalidateList: () => queryClient.invalidateQueries({ queryKey: predictionKeys.lists() }),
  }
}
