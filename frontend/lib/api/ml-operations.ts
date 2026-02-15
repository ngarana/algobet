/**
 * API functions for ML operations (backtest, calibrate)
 */

import { apiPost } from './client'
import type {
  BacktestResult,
  CalibrateResult,
  BacktestRequest,
  CalibrateRequest,
} from '@/lib/types/ml-operations'
import {
  BacktestResultSchema,
  CalibrateResultSchema,
} from '@/lib/types/ml-operations'

/**
 * Run a historical backtest on model predictions
 */
export async function runBacktest(
  request: BacktestRequest
): Promise<BacktestResult> {
  return apiPost('/ml/backtest', request, BacktestResultSchema)
}

/**
 * Calibrate model probabilities
 */
export async function runCalibrate(
  request: CalibrateRequest
): Promise<CalibrateResult> {
  return apiPost('/ml/calibrate', request, CalibrateResultSchema)
}
