/**
 * API functions for the daily workflow.
 */

import { apiDelete, apiGet, apiPost, apiPut, buildQueryString } from "./client";
import {
  DailyWorkflowSchema,
  MatchWorkflowDetailSchema,
  ProfilePreferencesSchema,
  ResultsReviewSchema,
  UserPredictionSchema,
  WatchlistEntrySchema,
  WatchlistSchema,
} from "@/lib/types/schemas";
import { z } from "zod";
import type {
  DailyWorkflow,
  MatchWorkflowDetail,
  ProfilePreferences,
  ProfilePreferencesUpdate,
  ResultsReview,
  UserPrediction,
  UserPredictionRequest,
  Watchlist,
  WatchlistEntry,
  WatchlistEntryRequest,
} from "@/lib/types/api";

const userPredictionArraySchema = z.array(UserPredictionSchema);
const deleteResponseSchema = z.object({ deleted: z.boolean() });

export async function getDailyWorkflow(date?: string): Promise<DailyWorkflow> {
  const queryString = date ? buildQueryString({ date }) : "";
  return apiGet(`/workflow/dashboard/daily${queryString}`, DailyWorkflowSchema);
}

export async function getProfilePreferences(): Promise<ProfilePreferences> {
  return apiGet("/workflow/profile/preferences", ProfilePreferencesSchema);
}

export async function updateProfilePreferences(
  request: ProfilePreferencesUpdate
): Promise<ProfilePreferences> {
  return apiPut("/workflow/profile/preferences", request, ProfilePreferencesSchema);
}

export async function getWatchlist(): Promise<Watchlist> {
  return apiGet("/workflow/watchlist", WatchlistSchema);
}

export async function addWatchlistEntry(
  request: WatchlistEntryRequest
): Promise<WatchlistEntry> {
  return apiPost("/workflow/watchlist", request, WatchlistEntrySchema);
}

export async function removeWatchlistEntry(
  entryType: string,
  entryId: number
): Promise<{ deleted: boolean }> {
  return apiDelete(`/workflow/watchlist/${entryType}/${entryId}`, deleteResponseSchema);
}

export async function getUserPredictions(matchId?: number): Promise<UserPrediction[]> {
  const queryString = matchId ? buildQueryString({ match_id: matchId }) : "";
  return apiGet(`/workflow/user-predictions${queryString}`, userPredictionArraySchema);
}

export async function upsertUserPrediction(
  request: UserPredictionRequest
): Promise<UserPrediction> {
  return apiPost("/workflow/user-predictions", request, UserPredictionSchema);
}

export async function getResultsReview(): Promise<ResultsReview> {
  return apiGet("/workflow/results/review", ResultsReviewSchema);
}

export async function getMatchWorkflowDetail(
  matchId: number
): Promise<MatchWorkflowDetail> {
  return apiGet(`/workflow/matches/${matchId}/workflow`, MatchWorkflowDetailSchema);
}
