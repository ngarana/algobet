/**
 * TanStack Query hooks for dashboard statistics
 */

import { useQuery } from '@tanstack/react-query'
import { usePredictions, useUpcomingPredictions } from './use-predictions'
import { useValueBets } from './use-value-bets'
import { useActiveModel } from './use-models'
import type { Prediction } from '@/lib/types/api'

export interface DashboardStats {
  totalProfit: number
  winRate: number
  totalPredictions: number
  successfulPredictions: number
  upcomingMatchesCount: number
  valueBetsCount: number
  avgConfidence: number
  activeModelAccuracy: number | null
}

export const dashboardKeys = {
  all: ['dashboard'] as const,
  stats: () => [...dashboardKeys.all, 'stats'] as const,
}

export function useDashboardStats() {
  // Get all predictions (both upcoming and historical)
  const { data: allPredictionsData, isLoading: predictionsLoading } = usePredictions()
  const { data: upcomingPredictionsData, isLoading: upcomingLoading } = useUpcomingPredictions()
  const { data: valueBetsData, isLoading: valueBetsLoading } = useValueBets({ max_matches: 100 })
  const { data: activeModelData, isLoading: modelLoading } = useActiveModel()

  const allPredictions = allPredictionsData?.items || []
  const upcomingPredictions = upcomingPredictionsData?.items || []
  const valueBets = valueBetsData || []

  // Calculate stats from available data
  const stats = calculateDashboardStats(
    allPredictions,
    upcomingPredictions,
    valueBets,
    activeModelData
  )

  return {
    data: stats,
    isLoading: predictionsLoading || upcomingLoading || valueBetsLoading || modelLoading,
  }
}

function calculateDashboardStats(
  allPredictions: Prediction[],
  upcomingPredictions: Prediction[],
  valueBets: any[], // Using any for now since we don't have the exact type
  activeModel: any // Using any for now since we don't have the exact type
): DashboardStats {
  // Calculate total profit (sum of actual_roi where available)
  const totalProfit = allPredictions
    .filter(p => p.actual_roi !== null)
    .reduce((sum, p) => sum + (p.actual_roi || 0), 0)

  // Calculate win rate (successful predictions / total predictions with results)
  const predictionsWithResults = allPredictions.filter(p => p.actual_roi !== null)
  const successfulPredictions = predictionsWithResults.filter(p => (p.actual_roi || 0) > 0).length
  const winRate = predictionsWithResults.length > 0 
    ? successfulPredictions / predictionsWithResults.length 
    : 0

  // Calculate average confidence
  const avgConfidence = allPredictions.length > 0
    ? allPredictions.reduce((sum, p) => sum + (p.confidence || 0), 0) / allPredictions.length
    : 0

  return {
    totalProfit,
    winRate,
    totalPredictions: allPredictions.length,
    successfulPredictions,
    upcomingMatchesCount: upcomingPredictions.length,
    valueBetsCount: valueBets.length,
    avgConfidence,
    activeModelAccuracy: activeModel?.accuracy || null,
  }
}