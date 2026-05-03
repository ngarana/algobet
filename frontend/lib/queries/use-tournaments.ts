/**
 * TanStack Query hooks for tournament operations
 */

import { useQuery } from "@tanstack/react-query";
import {
  getTournaments,
  getTournament,
  getTournamentSeasons,
} from "@/lib/api/tournaments";

export interface TournamentFilters {
  search?: string;
  limit?: number;
  offset?: number;
}

export const tournamentKeys = {
  all: ["tournaments"] as const,
  lists: () => [...tournamentKeys.all, "list"] as const,
  list: (filters?: TournamentFilters) => [...tournamentKeys.lists(), filters] as const,
  details: () => [...tournamentKeys.all, "detail"] as const,
  detail: (id: number) => [...tournamentKeys.details(), id] as const,
  seasons: (id: number) => [...tournamentKeys.detail(id), "seasons"] as const,
};

/**
 * Get all tournaments - cached for 5 minutes
 */
export function useTournaments(filters?: TournamentFilters) {
  return useQuery({
    queryKey: tournamentKeys.list(filters),
    queryFn: () => getTournaments(filters),
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Get a specific tournament
 */
export function useTournament(id: number | null) {
  return useQuery({
    queryKey: tournamentKeys.detail(id ?? 0),
    queryFn: () => getTournament(id ?? 0),
    enabled: id !== null,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Get all seasons for a tournament
 */
export function useTournamentSeasons(tournamentId: number | null) {
  return useQuery({
    queryKey: tournamentKeys.seasons(tournamentId ?? 0),
    queryFn: () => getTournamentSeasons(tournamentId ?? 0),
    enabled: tournamentId !== null,
    staleTime: 10 * 60 * 1000,
  });
}
