/**
 * TanStack Query hooks for value bets
 */

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { getValueBets } from '@/lib/api/value-bets'

export const valueBetKeys = {
  all: ['value-bets'] as const,
  lists: () => [...valueBetKeys.all, 'list'] as const,
  list: (filters: object) => [...valueBetKeys.lists(), filters] as const,
}

interface ValueBetFilters {
  min_ev?: number
  max_odds?: number
  days?: number
  model_version?: string
  min_confidence?: number
  max_matches?: number
}

export function useValueBets(filters?: ValueBetFilters) {
  return useQuery({
    queryKey: valueBetKeys.list(filters ?? {}),
    queryFn: () => getValueBets(filters),
  })
}

export function useInvalidateValueBets() {
  const queryClient = useQueryClient()
  
  return {
    invalidateAll: () => queryClient.invalidateQueries({ queryKey: valueBetKeys.all }),
    invalidateList: () => queryClient.invalidateQueries({ queryKey: valueBetKeys.lists() }),
  }
}
