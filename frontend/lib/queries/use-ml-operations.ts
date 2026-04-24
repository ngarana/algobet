/**
 * TanStack Query hooks for ML operations
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  runTrainModel,
  runBacktest,
  runCalibrate,
  getBacktestHistory,
  getBacktestDetail,
} from "@/lib/api/ml-operations";
import type {
  BacktestRequest,
  CalibrateRequest,
  TrainModelRequest,
} from "@/lib/types/ml-operations";

export const mlOperationsKeys = {
  all: ["ml-operations"] as const,
  train: () => [...mlOperationsKeys.all, "train"] as const,
  backtest: () => [...mlOperationsKeys.all, "backtest"] as const,
  backtestHistory: () => [...mlOperationsKeys.all, "backtest-history"] as const,
  backtestHistoryList: (filters?: {
    model_version_id?: number;
    limit?: number;
    offset?: number;
  }) => [...mlOperationsKeys.backtestHistory(), "list", filters] as const,
  backtestDetail: (id: number) =>
    [...mlOperationsKeys.backtestHistory(), "detail", id] as const,
  calibrate: () => [...mlOperationsKeys.all, "calibrate"] as const,
};

/**
 * Train model mutation
 */
export function useTrainModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: TrainModelRequest) => runTrainModel(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

/**
 * Run backtest mutation
 */
export function useBacktest() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: BacktestRequest) => runBacktest(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: mlOperationsKeys.backtestHistory() });
    },
  });
}

/**
 * Run calibrate mutation
 */
export function useCalibrate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CalibrateRequest) => runCalibrate(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

/**
 * Get backtest history
 */
export function useBacktestHistory(filters?: {
  model_version_id?: number;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: mlOperationsKeys.backtestHistoryList(filters),
    queryFn: () => getBacktestHistory(filters),
  });
}

/**
 * Get backtest detail
 */
export function useBacktestDetail(backtestId: number | null) {
  return useQuery({
    queryKey: mlOperationsKeys.backtestDetail(backtestId ?? 0),
    queryFn: () => getBacktestDetail(backtestId ?? 0),
    enabled: backtestId !== null,
  });
}

/**
 * Hook to invalidate ML operations cache
 */
export function useInvalidateMLOperations() {
  const queryClient = useQueryClient();

  return {
    invalidateAll: () =>
      queryClient.invalidateQueries({ queryKey: mlOperationsKeys.all }),
    invalidateBacktest: () =>
      queryClient.invalidateQueries({ queryKey: mlOperationsKeys.backtest() }),
    invalidateBacktestHistory: () =>
      queryClient.invalidateQueries({ queryKey: mlOperationsKeys.backtestHistory() }),
    invalidateCalibrate: () =>
      queryClient.invalidateQueries({ queryKey: mlOperationsKeys.calibrate() }),
  };
}
