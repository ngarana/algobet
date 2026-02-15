/**
 * TanStack Query hooks for teams
 */

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { getTeams, getTeam, getTeamForm, getTeamMatches } from '@/lib/api/teams'

export const teamKeys = {
  all: ['teams'] as const,
  lists: () => [...teamKeys.all, 'list'] as const,
  list: (filters: Record<string, unknown>) => [...teamKeys.lists(), filters] as const,
  detail: (id: number) => [...teamKeys.all, 'detail', id] as const,
  form: (id: number) => [...teamKeys.detail(id), 'form'] as const,
  matches: (id: number) => [...teamKeys.detail(id), 'matches'] as const,
}

interface TeamFilters {
  search?: string
  tournament_id?: number
  limit?: number
  offset?: number
}

export function useTeams(filters?: TeamFilters) {
  return useQuery({
    queryKey: teamKeys.list(filters ? { ...filters } : {}),
    queryFn: () => getTeams(filters),
  })
}

export function useTeam(id: number) {
  return useQuery({
    queryKey: teamKeys.detail(id),
    queryFn: () => getTeam(id),
    enabled: !!id,
  })
}

export function useTeamForm(id: number, nMatches?: number) {
  return useQuery({
    queryKey: [...teamKeys.form(id), { nMatches }],
    queryFn: () => getTeamForm(id, nMatches),
    enabled: !!id,
  })
}

export function useTeamMatches(id: number, venue?: 'home' | 'away' | 'all', limit?: number) {
  return useQuery({
    queryKey: [...teamKeys.matches(id), { venue, limit }],
    queryFn: () => getTeamMatches(id, venue, limit),
    enabled: !!id,
  })
}

export function useInvalidateTeams() {
  const queryClient = useQueryClient()
  
  return {
    invalidateAll: () => queryClient.invalidateQueries({ queryKey: teamKeys.all }),
    invalidateList: () => queryClient.invalidateQueries({ queryKey: teamKeys.lists() }),
    invalidateDetail: (id: number) => queryClient.invalidateQueries({ queryKey: teamKeys.detail(id) }),
  }
}
