/**
 * TanStack Query hooks for tournament operations
 */

import { useQuery } from "@tanstack/react-query";
import {
  getTournaments,
  getTournament,
  getTournamentSeasons,
} from "@/lib/api/tournaments";

export const tournamentKeys = {
  all: ["tournaments"] as const,
  lists: () => [...tournamentKeys.all, "list"] as const,
  list: () => [...tournamentKeys.lists()] as const,
  details: () => [...tournamentKeys.all, "detail"] as const,
  detail: (id: number) => [...tournamentKeys.details(), id] as const,
  seasons: (id: number) => [...tournamentKeys.detail(id), "seasons"] as const,
};

/**
 * Get all tournaments - cached for 5 minutes
 */
export function useTournaments() {
  return useQuery({
    queryKey: tournamentKeys.list(),
    queryFn: getTournaments,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Get a specific tournament
 */
export function useTournament(id: number | null) {
  return useQuery({
    queryKey: tournamentKeys.detail(id!),
    queryFn: () => getTournament(id!),
    enabled: id !== null,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Get all seasons for a tournament
 */
export function useTournamentSeasons(tournamentId: number | null) {
  return useQuery({
    queryKey: tournamentKeys.seasons(tournamentId!),
    queryFn: () => getTournamentSeasons(tournamentId!),
    enabled: tournamentId !== null,
    staleTime: 10 * 60 * 1000,
  });
}
