/**
 * API functions for ML operations (backtest, calibrate)
 */

import { apiGet, apiPost } from "./client";
import type {
  BacktestResult,
  CalibrateResult,
  BacktestRequest,
  CalibrateRequest,
  BacktestHistoryItem,
  BacktestHistoryList,
} from "@/lib/types/ml-operations";
import {
  BacktestResultSchema,
  CalibrateResultSchema,
  BacktestHistoryItemSchema,
  BacktestHistoryListSchema,
} from "@/lib/types/ml-operations";

/**
 * Run a historical backtest on model predictions
 */
export async function runBacktest(request: BacktestRequest): Promise<BacktestResult> {
  return apiPost("/ml/backtest", request, BacktestResultSchema);
}

/**
 * Calibrate model probabilities
 */
export async function runCalibrate(
  request: CalibrateRequest
): Promise<CalibrateResult> {
  return apiPost("/ml/calibrate", request, CalibrateResultSchema);
}

/**
 * Get backtest history
 */
export async function getBacktestHistory(params?: {
  model_version_id?: number;
  limit?: number;
  offset?: number;
}): Promise<BacktestHistoryList> {
  const queryString = params
    ? "?" +
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${k}=${v}`)
        .join("&")
    : "";
  return apiGet(`/ml/backtest/history${queryString}`, BacktestHistoryListSchema);
}

/**
 * Get backtest detail by ID
 */
export async function getBacktestDetail(backtestId: number): Promise<BacktestResult> {
  return apiGet(`/ml/backtest/${backtestId}`, BacktestResultSchema);
}
