/**
 * TanStack Query hooks for scraping operations
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  scrapeUpcomingMatches,
  scrapeResults,
  scrapeByDate,
  getScrapingJobs,
  getScrapingJob,
  getScrapingStats,
} from "@/lib/api/scraping";
import type { ScrapeUpcomingRequest, ScrapeResultsRequest, ScrapeByDateRequest } from "@/lib/api/scraping";

export const scrapingKeys = {
  all: ["scraping"] as const,
  jobs: () => [...scrapingKeys.all, "jobs"] as const,
  jobsList: (filters?: { status?: string }) =>
    [...scrapingKeys.jobs(), "list", filters] as const,
  job: (id: string) => [...scrapingKeys.jobs(), "detail", id] as const,
  stats: () => [...scrapingKeys.all, "stats"] as const,
};

export function useScrapingJobs(filters?: { status?: string }) {
  return useQuery({
    queryKey: scrapingKeys.jobsList(filters),
    queryFn: () => getScrapingJobs(filters?.status),
    refetchInterval: 5000,
  });
}

export function useScrapingJob(jobId: string | null) {
  return useQuery({
    queryKey: scrapingKeys.job(jobId ?? ""),
    queryFn: () => getScrapingJob(jobId ?? ""),
    enabled: jobId !== null && jobId !== "",
    refetchInterval: 3000,
  });
}

export function useScrapeUpcoming() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ScrapeUpcomingRequest) => scrapeUpcomingMatches(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: scrapingKeys.jobs() });
      queryClient.invalidateQueries({ queryKey: scrapingKeys.stats() });
    },
  });
}

export function useScrapeResults() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ScrapeResultsRequest) => scrapeResults(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: scrapingKeys.jobs() });
      queryClient.invalidateQueries({ queryKey: scrapingKeys.stats() });
    },
  });
}

export function useScrapeByDate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ScrapeByDateRequest) => scrapeByDate(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: scrapingKeys.jobs() });
      queryClient.invalidateQueries({ queryKey: scrapingKeys.stats() });
    },
  });
}

export function useScrapingStats() {
  return useQuery({
    queryKey: scrapingKeys.stats(),
    queryFn: () => getScrapingStats(),
    refetchInterval: 10000,
  });
}

export function useInvalidateScraping() {
  const queryClient = useQueryClient();

  return {
    invalidateJobs: () =>
      queryClient.invalidateQueries({ queryKey: scrapingKeys.jobs() }),
    invalidateStats: () =>
      queryClient.invalidateQueries({ queryKey: scrapingKeys.stats() }),
    invalidateAll: () => queryClient.invalidateQueries({ queryKey: scrapingKeys.all }),
  };
}
