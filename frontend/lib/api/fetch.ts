/**
 * API client functions for scraping operations (OddsPortal web scraping)
 */

import { apiGet, apiPost, buildQueryString } from "./client";
import { z } from "zod";
import { createPaginatedResponseSchema } from "@/lib/types/schemas";
import type { PaginatedResponse } from "@/lib/types/api";

// Zod schemas for runtime validation
export const FetchProgressSchema = z.object({
  job_id: z.string(),
  progress: z.number(),
  message: z.string(),
  matches_fetched: z.number().default(0),
  matches_saved: z.number().optional(),
  current_page: z.number().optional(),
  total_pages: z.number().optional(),
  started_at: z.string().optional().nullable(),
  completed_at: z.string().optional().nullable(),
  error: z.string().optional().nullable(),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]).optional(),
  timestamp: z.string(),
});

export const FetchJobSchema = z.object({
  id: z.string(),
  fetch_type: z.enum(["upcoming", "results", "by-date"]),
  tournament_url: z.string().nullable(),
  tournament_name: z.string().nullable(),
  season: z.string().nullable(),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]),
  progress: z.number(),
  message: z.string().nullable(),
  created_at: z.string(),
  started_at: z.string().nullable(),
  completed_at: z.string().nullable(),
  matches_fetched: z.number().default(0),
  errors: z.array(z.string()),
});

export const fetchJobArraySchema = createPaginatedResponseSchema(FetchJobSchema);

export const FetchUpcomingRequestSchema = z.object({
  tournament_url: z.string().optional(),
});

export const FetchResultsRequestSchema = z.object({
  tournament_url: z.string().optional(),
});

export const FetchByDateRequestSchema = z.object({
  date: z.string().optional(), // YYYY-MM-DD format
});

// Types derived from schemas
export type FetchProgress = z.infer<typeof FetchProgressSchema>;
export type FetchJob = z.infer<typeof FetchJobSchema>;
export type FetchUpcomingRequest = z.infer<typeof FetchUpcomingRequestSchema>;
export type FetchResultsRequest = z.infer<typeof FetchResultsRequestSchema>;
export type FetchByDateRequest = z.infer<typeof FetchByDateRequestSchema>;

/**
 * Transform backend response to frontend FetchJob type
 */
function transformJobResponse(data: Record<string, unknown>): FetchJob {
  return {
    id: data.id as string,
    fetch_type: (data.scraping_type || data.fetch_type) as FetchJob["fetch_type"],
    tournament_url: data.tournament_url as string | null,
    tournament_name: data.tournament_name as string | null,
    season: data.season as string | null,
    status: data.status as FetchJob["status"],
    progress: data.progress as number,
    message: data.message as string | null,
    created_at: data.created_at as string,
    started_at: data.started_at as string | null,
    completed_at: data.completed_at as string | null,
    matches_fetched: (data.matches_scraped || data.matches_fetched || 0) as number,
    errors: (data.errors || []) as string[],
  };
}

/**
 * Scrape upcoming matches from OddsPortal
 */
export async function fetchUpcomingMatches(
  request: FetchUpcomingRequest = {}
): Promise<FetchJob> {
  const params: Record<string, unknown> = {};
  if (request.tournament_url) {
    params.tournament_url = request.tournament_url;
  }
  const queryString = buildQueryString(params);
  const response = await apiPost(
    `/scraping/upcoming${queryString}`,
    {},
    z.record(z.unknown())
  );
  return transformJobResponse(response);
}

/**
 * Scrape historical results from OddsPortal
 */
export async function fetchResults(
  request: FetchResultsRequest = {}
): Promise<FetchJob> {
  const params: Record<string, unknown> = {};
  if (request.tournament_url) {
    params.tournament_url = request.tournament_url;
  }
  const queryString = buildQueryString(params);
  const response = await apiPost(
    `/scraping/results${queryString}`,
    {},
    z.record(z.unknown())
  );
  return transformJobResponse(response);
}

/**
 * Scrape all matches for a specific date from OddsPortal
 */
export async function fetchByDate(request: FetchByDateRequest = {}): Promise<FetchJob> {
  const params: Record<string, unknown> = {};
  if (request.date) {
    params.date = request.date;
  }
  const queryString = buildQueryString(params);
  const response = await apiPost(
    `/scraping/by-date${queryString}`,
    {},
    z.record(z.unknown())
  );
  return transformJobResponse(response);
}

/**
 * Get all fetch jobs
 */
export async function getFetchJobs(
  status?: string
): Promise<PaginatedResponse<FetchJob>> {
  const params: Record<string, unknown> = {};
  if (status) params.status_filter = status;

  const queryString = buildQueryString(params);
  const response = await apiGet(`/scraping/jobs${queryString}`, z.record(z.unknown()));

  const items = ((response.items || []) as Record<string, unknown>[]).map(
    transformJobResponse
  );

  return {
    items,
    total: response.total as number,
    limit: response.limit as number,
    offset: response.offset as number,
  };
}

/**
 * Get a specific fetch job by ID
 */
export async function getFetchJob(jobId: string): Promise<FetchJob> {
  const response = await apiGet(`/scraping/jobs/${jobId}`, z.record(z.unknown()));
  return transformJobResponse(response);
}

/**
 * Get fetch statistics
 */
export async function getFetchStats(): Promise<FetchStats> {
  const response = await apiGet("/scraping/stats", z.record(z.unknown()));
  return {
    total_jobs: response.total_jobs as number,
    completed_jobs: response.completed_jobs as number,
    failed_jobs: response.failed_jobs as number,
    running_jobs: response.running_jobs as number,
    total_matches_fetched: (response.total_matches_scraped ||
      response.total_matches_fetched ||
      0) as number,
    average_duration_seconds: response.average_duration_seconds as number | null,
    success_rate: response.success_rate as number,
  };
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
