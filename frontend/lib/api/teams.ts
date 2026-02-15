/**
 * API functions for teams
 */

import { apiGet, buildQueryString } from './client'
import type { Team, FormBreakdown } from '@/lib/types/api'
import { TeamSchema, FormBreakdownSchema } from '@/lib/types/schemas'
import { z } from 'zod'

const teamArraySchema = z.array(TeamSchema)

interface TeamFilters {
  search?: string
  tournament_id?: number
  limit?: number
  offset?: number
}

export async function getTeams(filters?: TeamFilters): Promise<Team[]> {
  const queryString = filters ? buildQueryString(filters) : ''
  return apiGet(`/teams${queryString}`, teamArraySchema)
}

export async function getTeam(id: number): Promise<Team> {
  return apiGet(`/teams/${id}`, TeamSchema)
}

export async function getTeamForm(id: number, nMatches?: number): Promise<FormBreakdown> {
  const queryString = nMatches ? `?n_matches=${nMatches}` : ''
  return apiGet(`/teams/${id}/form${queryString}`, FormBreakdownSchema)
}

export async function getTeamMatches(id: number, venue?: 'home' | 'away' | 'all', limit?: number): Promise<unknown[]> {
  const params: Record<string, unknown> = {}
  if (venue) params.venue = venue
  if (limit) params.limit = limit
  const queryString = buildQueryString(params)
  return apiGet(`/teams/${id}/matches${queryString}`)
}
