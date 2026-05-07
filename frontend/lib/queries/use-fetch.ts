/**
 * TanStack Query hooks for data fetching operations
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchUpcomingMatches,
  fetchResults,
  fetchByDate,
  importFootballData,
  getFetchJobs,
  getFetchJob,
  getFetchStats,
} from "@/lib/api/fetch";
import type {
  FetchUpcomingRequest,
  FetchResultsRequest,
  FetchByDateRequest,
  FootballDataImportRequest,
} from "@/lib/api/fetch";

export const fetchKeys = {
  all: ["fetch"] as const,
  jobs: () => [...fetchKeys.all, "jobs"] as const,
  jobsList: (filters?: { status?: string }) =>
    [...fetchKeys.jobs(), "list", filters] as const,
  job: (id: string) => [...fetchKeys.jobs(), "detail", id] as const,
  stats: () => [...fetchKeys.all, "stats"] as const,
};

/**
 * Hook to get all fetch jobs
 */
export function useFetchJobs(filters?: { status?: string }) {
  return useQuery({
    queryKey: fetchKeys.jobsList(filters),
    queryFn: () => getFetchJobs(filters?.status),
    refetchInterval: 5000,
  });
}

/**
 * Hook to get a specific fetch job by ID
 */
export function useFetchJob(jobId: string | null) {
  return useQuery({
    queryKey: fetchKeys.job(jobId ?? ""),
    queryFn: () => getFetchJob(jobId ?? ""),
    enabled: jobId !== null && jobId !== "",
    refetchInterval: 3000,
  });
}

/**
 * Hook to fetch upcoming matches
 */
export function useFetchUpcoming() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: FetchUpcomingRequest) => fetchUpcomingMatches(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fetchKeys.jobs() });
      queryClient.invalidateQueries({ queryKey: fetchKeys.stats() });
    },
  });
}

/**
 * Hook to fetch match results
 */
export function useFetchResults() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: FetchResultsRequest) => fetchResults(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fetchKeys.jobs() });
      queryClient.invalidateQueries({ queryKey: fetchKeys.stats() });
    },
  });
}

/**
 * Hook to fetch matches by date
 */
export function useFetchByDate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: FetchByDateRequest) => fetchByDate(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fetchKeys.jobs() });
      queryClient.invalidateQueries({ queryKey: fetchKeys.stats() });
    },
  });
}

/**
 * Hook to import from Football-Data.co.uk
 */
export function useImportFootballData() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: FootballDataImportRequest) => importFootballData(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fetchKeys.jobs() });
      queryClient.invalidateQueries({ queryKey: fetchKeys.stats() });
    },
  });
}

/**
 * Hook to get fetch statistics
 */
export function useFetchStats() {
  return useQuery({
    queryKey: fetchKeys.stats(),
    queryFn: () => getFetchStats(),
    refetchInterval: 10000,
  });
}

/**
 * Hook to invalidate fetch queries
 */
export function useInvalidateFetch() {
  const queryClient = useQueryClient();

  return {
    invalidateJobs: () => queryClient.invalidateQueries({ queryKey: fetchKeys.jobs() }),
    invalidateStats: () =>
      queryClient.invalidateQueries({ queryKey: fetchKeys.stats() }),
    invalidateAll: () => queryClient.invalidateQueries({ queryKey: fetchKeys.all }),
  };
}
