/**
 * TanStack Query hooks for the daily workflow.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  addWatchlistEntry,
  getDailyWorkflow,
  getMatchWorkflowDetail,
  getProfilePreferences,
  getResultsReview,
  getUserPredictions,
  getWatchlist,
  removeWatchlistEntry,
  updateProfilePreferences,
  upsertUserPrediction,
} from "@/lib/api/workflow";
import type {
  ProfilePreferencesUpdate,
  UserPredictionRequest,
  WatchlistEntryRequest,
} from "@/lib/types/api";

export const workflowKeys = {
  all: ["workflow"] as const,
  daily: (date?: string) => [...workflowKeys.all, "daily", date] as const,
  preferences: () => [...workflowKeys.all, "preferences"] as const,
  watchlist: () => [...workflowKeys.all, "watchlist"] as const,
  userPredictions: (matchId?: number) =>
    [...workflowKeys.all, "user-predictions", matchId] as const,
  results: () => [...workflowKeys.all, "results"] as const,
  matchDetail: (matchId: number) =>
    [...workflowKeys.all, "match-detail", matchId] as const,
};

export function useDailyWorkflow(date?: string) {
  return useQuery({
    queryKey: workflowKeys.daily(date),
    queryFn: () => getDailyWorkflow(date),
    staleTime: 60 * 1000,
  });
}

export function useProfilePreferences() {
  return useQuery({
    queryKey: workflowKeys.preferences(),
    queryFn: getProfilePreferences,
  });
}

export function useUpdateProfilePreferences() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ProfilePreferencesUpdate) =>
      updateProfilePreferences(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: workflowKeys.all });
    },
  });
}

export function useWatchlist() {
  return useQuery({
    queryKey: workflowKeys.watchlist(),
    queryFn: getWatchlist,
  });
}

export function useAddWatchlistEntry() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: WatchlistEntryRequest) => addWatchlistEntry(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: workflowKeys.all });
    },
  });
}

export function useRemoveWatchlistEntry() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      entryType,
      entryId,
    }: {
      entryType: string;
      entryId: number;
    }) => removeWatchlistEntry(entryType, entryId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: workflowKeys.all });
    },
  });
}

export function useUserPredictions(matchId?: number) {
  return useQuery({
    queryKey: workflowKeys.userPredictions(matchId),
    queryFn: () => getUserPredictions(matchId),
  });
}

export function useUpsertUserPrediction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: UserPredictionRequest) => upsertUserPrediction(request),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: workflowKeys.all });
      queryClient.invalidateQueries({
        queryKey: workflowKeys.matchDetail(variables.match_id),
      });
    },
  });
}

export function useResultsReview() {
  return useQuery({
    queryKey: workflowKeys.results(),
    queryFn: getResultsReview,
  });
}

export function useMatchWorkflowDetail(matchId: number) {
  return useQuery({
    queryKey: workflowKeys.matchDetail(matchId),
    queryFn: () => getMatchWorkflowDetail(matchId),
    enabled: !!matchId,
  });
}
