/**
 * API client functions for scraping operations.
 */

import { z } from "zod";
import { apiGet, apiPost } from "./client";
import type { PaginatedResponse } from "@/lib/types/api";

const JobStatusSchema = z.enum([
  "pending",
  "running",
  "completed",
  "failed",
  "cancelled",
]);

const FetchTypeSchema = z.enum(["upcoming", "results", "by-date"]);
const FetchScopeSchema = z.enum(["all", "league"]);

export const FetchProgressSchema = z.object({
  job_id: z.string(),
  progress: z.number(),
  message: z.string(),
  matches_fetched: z.number().default(0),
  matches_saved: z.number().default(0),
  current_page: z.number().optional(),
  total_pages: z.number().optional(),
  started_at: z.string().optional().nullable(),
  completed_at: z.string().optional().nullable(),
  error: z.string().optional().nullable(),
  status: JobStatusSchema.optional(),
  timestamp: z.string(),
});

export const FetchJobSchema = z.object({
  id: z.string(),
  fetch_type: FetchTypeSchema,
  scraping_type: FetchTypeSchema,
  tournament_url: z.string().nullable(),
  tournament_id: z.number().nullable().optional(),
  tournament_name: z.string().nullable(),
  season: z.string().nullable(),
  scope: FetchScopeSchema,
  country: z.string().nullable().optional(),
  league_name: z.string().nullable().optional(),
  period: z.string().nullable().optional(),
  status: JobStatusSchema,
  progress: z.number(),
  message: z.string().nullable(),
  created_at: z.string(),
  started_at: z.string().nullable(),
  completed_at: z.string().nullable(),
  matches_fetched: z.number().default(0),
  matches_saved: z.number().default(0),
  errors: z.array(z.string()),
});

export interface FetchUpcomingRequest {
  tournament_id?: number;
  tournament_url?: string;
  scope?: "all" | "league";
}

export interface FetchResultsRequest {
  tournament_id?: number;
  tournament_url?: string;
  period?: string;
  max_pages?: number;
}

export interface FetchByDateRequest {
  tournament_id?: number;
  tournament_url?: string;
  date?: string;
  scope?: "all" | "league";
}

export type FetchProgress = z.infer<typeof FetchProgressSchema>;
export type FetchJob = z.infer<typeof FetchJobSchema>;

function normalizeJob(data: Record<string, unknown>): FetchJob {
  const scrapingType = (data.scraping_type ||
    data.fetch_type) as FetchJob["fetch_type"];
  const normalized = {
    id: data.id as string,
    fetch_type: scrapingType,
    scraping_type: scrapingType,
    tournament_url: (data.tournament_url as string | null | undefined) ?? null,
    tournament_id: (data.tournament_id as number | null | undefined) ?? null,
    tournament_name: (data.tournament_name as string | null | undefined) ?? null,
    season: (data.season as string | null | undefined) ?? null,
    scope: ((data.scope as FetchJob["scope"] | undefined) ??
      "all") as FetchJob["scope"],
    country: (data.country as string | null | undefined) ?? null,
    league_name: (data.league_name as string | null | undefined) ?? null,
    period: (data.period as string | null | undefined) ?? null,
    status: data.status as FetchJob["status"],
    progress: (data.progress as number | undefined) ?? 0,
    message: (data.message as string | null | undefined) ?? null,
    created_at: data.created_at as string,
    started_at: (data.started_at as string | null | undefined) ?? null,
    completed_at: (data.completed_at as string | null | undefined) ?? null,
    matches_fetched:
      (data.matches_scraped as number | undefined) ??
      (data.matches_fetched as number | undefined) ??
      0,
    matches_saved: (data.matches_saved as number | undefined) ?? 0,
    errors: ((data.errors as string[] | undefined) ?? []).filter(Boolean),
  };

  return FetchJobSchema.parse(normalized);
}

export async function fetchUpcomingMatches(
  request: FetchUpcomingRequest = {}
): Promise<FetchJob> {
  const response = await apiPost(
    "/scraping/upcoming",
    {
      tournament_id: request.tournament_id,
      tournament_url: request.tournament_url,
      scope: request.scope ?? "all",
    },
    z.record(z.unknown())
  );
  return normalizeJob(response);
}

export async function fetchResults(
  request: FetchResultsRequest = {}
): Promise<FetchJob> {
  const response = await apiPost(
    "/scraping/results",
    {
      tournament_id: request.tournament_id,
      tournament_url: request.tournament_url,
      period: request.period,
      max_pages: request.max_pages,
    },
    z.record(z.unknown())
  );
  return normalizeJob(response);
}

export async function fetchByDate(request: FetchByDateRequest = {}): Promise<FetchJob> {
  const response = await apiPost(
    "/scraping/by-date",
    {
      date: request.date,
      tournament_id: request.tournament_id,
      tournament_url: request.tournament_url,
      scope: request.scope ?? "all",
    },
    z.record(z.unknown())
  );
  return normalizeJob(response);
}

export async function getFetchJobs(
  status?: string
): Promise<PaginatedResponse<FetchJob>> {
  const response = await apiGet("/scraping/jobs", z.record(z.unknown()));
  const rawItems = ((response.items || []) as Record<string, unknown>[]).filter(
    Boolean
  );
  const items = status
    ? rawItems.map(normalizeJob).filter((job) => job.status === status)
    : rawItems.map(normalizeJob);

  return {
    items,
    total: status
      ? items.length
      : ((response.total as number | undefined) ?? items.length),
    limit: (response.limit as number | undefined) ?? items.length,
    offset: (response.offset as number | undefined) ?? 0,
  };
}

export async function getFetchJob(jobId: string): Promise<FetchJob> {
  const response = await apiGet(`/scraping/jobs/${jobId}`, z.record(z.unknown()));
  return normalizeJob(response);
}

export async function getFetchStats(): Promise<FetchStats> {
  const response = await apiGet("/scraping/stats", z.record(z.unknown()));
  return FetchStatsSchema.parse({
    total_jobs: response.total_jobs,
    completed_jobs: response.completed_jobs,
    failed_jobs: response.failed_jobs,
    running_jobs: response.running_jobs,
    total_matches_fetched:
      (response.total_matches_scraped as number | undefined) ??
      (response.total_matches_fetched as number | undefined) ??
      0,
    average_duration_seconds:
      (response.average_duration_seconds as number | null | undefined) ?? null,
    success_rate: response.success_rate,
  });
}

export const FetchStatsSchema = z.object({
  total_jobs: z.number(),
  completed_jobs: z.number(),
  failed_jobs: z.number(),
  running_jobs: z.number(),
  total_matches_fetched: z.number(),
  average_duration_seconds: z.number().nullable(),
  success_rate: z.number(),
});

export type FetchStats = z.infer<typeof FetchStatsSchema>;
