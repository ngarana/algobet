/**
 * Hook for managing fetch job operations with unified error handling
 */

import { useCallback } from "react";
import {
  useFetchByDate,
  useFetchJobs,
  useFetchResults,
  useFetchStats,
  useFetchUpcoming,
  useImportFootballData,
} from "@/lib/queries/use-fetch";
import type { LogLevelValue } from "@/lib/constants/fetch";
import { FetchDialogType, ERROR_MESSAGES } from "@/lib/constants/fetch";

interface UseJobOperationsOptions {
  addLog: (level: LogLevelValue, message: string) => void;
  onJobCreated?: (jobId: string) => void;
  onError?: (error: string) => void;
}

interface UseJobOperationsReturn {
  fetchUpcoming: (request?: {
    tournament_id?: number;
    tournament_url?: string;
    scope?: "all" | "league";
  }) => Promise<void>;
  fetchResults: (request: {
    tournament_id?: number;
    tournament_url?: string;
    period?: string;
    period_start?: string;
    period_end?: string;
    max_pages?: number;
  }) => Promise<void>;
  fetchByDate: (request?: {
    date?: string;
    tournament_id?: number;
    scope?: "all" | "league";
  }) => Promise<void>;
  importFootballData: (request: {
    division: string;
    season: string;
  }) => Promise<void>;
  refreshAll: () => Promise<void>;
  isPending: boolean;
  isRefreshing: boolean;
  jobsData: ReturnType<typeof useFetchJobs>["data"];
  stats: ReturnType<typeof useFetchStats>["data"];
}

/**
 * Hook to manage fetch job operations with unified error handling and logging
 */
export function useJobOperations(
  options: UseJobOperationsOptions
): UseJobOperationsReturn {
  const { addLog, onJobCreated, onError } = options;

  // Queries
  const {
    data: jobsData,
    isFetching: jobsFetching,
    refetch: refreshJobs,
  } = useFetchJobs();

  const {
    data: stats,
    isFetching: statsFetching,
    refetch: refreshStats,
  } = useFetchStats();

  // Mutations
  const fetchUpcomingMutation = useFetchUpcoming();
  const fetchResultsMutation = useFetchResults();
  const fetchByDateMutation = useFetchByDate();
  const importFootballDataMutation = useImportFootballData();

  const isPending =
    fetchUpcomingMutation.isPending ||
    fetchResultsMutation.isPending ||
    fetchByDateMutation.isPending ||
    importFootballDataMutation.isPending;

  const isRefreshing = jobsFetching || statsFetching;

  /**
   * Refresh all data queries
   */
  const refreshAll = useCallback(async () => {
    await Promise.all([refreshJobs(), refreshStats()]);
    addLog("SYS", "Data refreshed");
  }, [refreshJobs, refreshStats, addLog]);

  /**
   * Execute a mutation with unified error handling
   */
  const executeMutation = useCallback(
    async (
      mutation: () => Promise<{ id: string }>,
      logPrefix: string
    ): Promise<void> => {
      addLog("INF", `Starting ${logPrefix}...`);
      try {
        const job = await mutation();
        addLog("SUC", `Job created: ${job.id}`);
        onJobCreated?.(job.id);
        await refreshAll();
      } catch (error) {
        console.error(`Error starting ${logPrefix}:`, error);
        addLog("ERR", "Failed to start fetch operation");
        onError?.(ERROR_MESSAGES.FETCH_FAILED);
      }
    },
    [addLog, onJobCreated, onError, refreshAll]
  );

  /**
   * Fetch upcoming matches
   */
  const fetchUpcoming = useCallback(
    async (request?: {
      tournament_id?: number;
      tournament_url?: string;
      scope?: "all" | "league";
    }) => {
      await executeMutation(
        () =>
          fetchUpcomingMutation.mutateAsync({
            tournament_id: request?.tournament_id,
            tournament_url: request?.tournament_url,
            scope: request?.scope ?? "all",
          }),
        `${FetchDialogType.UPCOMING} fixtures fetch${
          request?.scope === "league" ? " for selected league" : ""
        }`
      );
    },
    [executeMutation, fetchUpcomingMutation]
  );

  /**
   * Fetch results for a tournament
   */
  const fetchResults = useCallback(
    async (request: {
      tournament_id?: number;
      tournament_url?: string;
      period?: string;
      period_start?: string;
      period_end?: string;
      max_pages?: number;
    }) => {
      if (!request.tournament_id && !request.tournament_url) {
        addLog("ERR", "League selection or URL is required");
        onError?.(ERROR_MESSAGES.TOURNAMENT_URL_REQUIRED);
        return;
      }
      await executeMutation(
        () =>
          fetchResultsMutation.mutateAsync({
            tournament_id: request.tournament_id,
            tournament_url: request.tournament_url,
            period: request.period,
            period_start: request.period_start,
            period_end: request.period_end,
            max_pages: request.max_pages,
          }),
        `${FetchDialogType.RESULTS} fetch for selected league`
      );
    },
    [executeMutation, fetchResultsMutation, addLog, onError]
  );

  /**
   * Fetch matches by date
   */
  const fetchByDate = useCallback(
    async (request?: {
      date?: string;
      tournament_id?: number;
      scope?: "all" | "league";
    }) => {
      await executeMutation(
        () =>
          fetchByDateMutation.mutateAsync({
            date: request?.date,
            tournament_id: request?.tournament_id,
            scope: request?.scope ?? "all",
          }),
        `date fetch for ${request?.date || "today"}`
      );
    },
    [executeMutation, fetchByDateMutation]
  );

  /**
   * Import data from Football-Data.co.uk
   */
  const importFootballData = useCallback(
    async (request: { division: string; season: string }) => {
      addLog("INF", `Starting Football-Data import...`);
      try {
        const result = await importFootballDataMutation.mutateAsync({
          division: request.division,
          season: request.season,
        });
        addLog("SUC", `Import job created: ${result.job_id}`);
        onJobCreated?.(result.job_id);
        await refreshAll();
      } catch (error) {
        console.error("Error starting Football-Data import:", error);
        addLog("ERR", "Failed to start import operation");
        onError?.(ERROR_MESSAGES.FETCH_FAILED);
      }
    },
    [addLog, importFootballDataMutation, onJobCreated, onError, refreshAll]
  );

  return {
    fetchUpcoming,
    fetchResults,
    fetchByDate,
    importFootballData,
    refreshAll,
    isPending,
    isRefreshing,
    jobsData,
    stats,
  };
}
