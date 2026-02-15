/**
 * TanStack Query hooks for matches
 */

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getMatches, getMatch, getMatchPreview, getMatchH2H } from "@/lib/api/matches";
import type { MatchFilters } from "@/lib/types/api";

export const matchKeys = {
  all: ["matches"] as const,
  lists: () => [...matchKeys.all, "list"] as const,
  list: (filters: MatchFilters) => [...matchKeys.lists(), filters] as const,
  details: () => [...matchKeys.all, "detail"] as const,
  detail: (id: number) => [...matchKeys.details(), id] as const,
  preview: (id: number) => [...matchKeys.detail(id), "preview"] as const,
  h2h: (id: number) => [...matchKeys.detail(id), "h2h"] as const,
};

export function useMatches(filters?: MatchFilters) {
  return useQuery({
    queryKey: matchKeys.list(filters ?? {}),
    queryFn: () => getMatches(filters),
  });
}

export function useMatch(id: number) {
  return useQuery({
    queryKey: matchKeys.detail(id),
    queryFn: () => getMatch(id),
    enabled: !!id,
  });
}

export function useMatchPreview(id: number) {
  return useQuery({
    queryKey: matchKeys.preview(id),
    queryFn: () => getMatchPreview(id),
    enabled: !!id,
  });
}

export function useMatchH2H(id: number) {
  return useQuery({
    queryKey: matchKeys.h2h(id),
    queryFn: () => getMatchH2H(id),
    enabled: !!id,
  });
}

export function useInvalidateMatches() {
  const queryClient = useQueryClient();

  return {
    invalidateAll: () => queryClient.invalidateQueries({ queryKey: matchKeys.all }),
    invalidateList: () =>
      queryClient.invalidateQueries({ queryKey: matchKeys.lists() }),
    invalidateDetail: (id: number) =>
      queryClient.invalidateQueries({ queryKey: matchKeys.detail(id) }),
  };
}
