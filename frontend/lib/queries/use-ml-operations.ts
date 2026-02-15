/**
 * TanStack Query hooks for ML operations
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { runBacktest, runCalibrate } from '@/lib/api/ml-operations'
import type { BacktestRequest, CalibrateRequest } from '@/lib/types/ml-operations'

export const mlOperationsKeys = {
  all: ['ml-operations'] as const,
  backtest: () => [...mlOperationsKeys.all, 'backtest'] as const,
  calibrate: () => [...mlOperationsKeys.all, 'calibrate'] as const,
}

/**
 * Run backtest mutation
 */
export function useBacktest() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: BacktestRequest) => runBacktest(request),
    onSuccess: () => {
      // Invalidate models to refresh any cached data
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
}

/**
 * Run calibrate mutation
 */
export function useCalibrate() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: CalibrateRequest) => runCalibrate(request),
    onSuccess: () => {
      // Invalidate models to refresh the list with new calibrated model
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
}

/**
 * Hook to invalidate ML operations cache
 */
export function useInvalidateMLOperations() {
  const queryClient = useQueryClient()

  return {
    invalidateAll: () =>
      queryClient.invalidateQueries({ queryKey: mlOperationsKeys.all }),
    invalidateBacktest: () =>
      queryClient.invalidateQueries({ queryKey: mlOperationsKeys.backtest() }),
    invalidateCalibrate: () =>
      queryClient.invalidateQueries({ queryKey: mlOperationsKeys.calibrate() }),
  }
}
