/**
 * API client functions for scraping operations (API-Football integration)
 */

import { apiGet, apiPost, buildQueryString } from "./client";
import { z } from "zod";
import { createPaginatedResponseSchema } from "@/lib/types/schemas";
import type { PaginatedResponse } from "@/lib/types/api";

// Zod schemas for runtime validation
export const ScrapingProgressSchema = z.object({
  job_id: z.string(),
  progress: z.number(),
  message: z.string(),
  matches_scraped: z.number(),
  matches_saved: z.number().optional(),
  current_page: z.number().optional(),
  total_pages: z.number().optional(),
  started_at: z.string().optional().nullable(),
  completed_at: z.string().optional().nullable(),
  error: z.string().optional().nullable(),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]).optional(),
  timestamp: z.string(),
});

export const ScrapingJobSchema = z.object({
  id: z.string(),
  scraping_type: z.enum(["upcoming", "results", "by-date"]),
  tournament_url: z.string().nullable(),
  tournament_name: z.string().nullable(),
  season: z.string().nullable(),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]),
  progress: z.number(),
  message: z.string().nullable(),
  created_at: z.string(),
  started_at: z.string().nullable(),
  completed_at: z.string().nullable(),
  matches_scraped: z.number(),
  errors: z.array(z.string()),
  // API-Football fields
  league_ids: z.array(z.number()).nullable().optional(),
  league_id: z.number().nullable().optional(),
  max_results: z.number().nullable().optional(),
});

export const scrapingJobArraySchema = createPaginatedResponseSchema(ScrapingJobSchema);

export const ScrapeUpcomingRequestSchema = z.object({
  league_ids: z.array(z.number()).optional(),
});

export const ScrapeResultsRequestSchema = z.object({
  league_id: z.number().optional(),
  max_results: z.number().optional(),
});

export const ScrapeByDateRequestSchema = z.object({
  date: z.string().optional(), // YYYY-MM-DD format
  league_id: z.number().optional(),
});

// Types derived from schemas
export type ScrapingProgress = z.infer<typeof ScrapingProgressSchema>;
export type ScrapingJob = z.infer<typeof ScrapingJobSchema>;
export type ScrapeUpcomingRequest = z.infer<typeof ScrapeUpcomingRequestSchema>;
export type ScrapeResultsRequest = z.infer<typeof ScrapeResultsRequestSchema>;
export type ScrapeByDateRequest = z.infer<typeof ScrapeByDateRequestSchema>;

/**
 * Popular league IDs for API-Football
 */
export const POPULAR_LEAGUES = [
  { id: 39, name: "Premier League", country: "England", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿" },
  { id: 140, name: "La Liga", country: "Spain", flag: "🇪🇸" },
  { id: 135, name: "Serie A", country: "Italy", flag: "🇮🇹" },
  { id: 78, name: "Bundesliga", country: "Germany", flag: "🇩🇪" },
  { id: 61, name: "Ligue 1", country: "France", flag: "🇫🇷" },
  { id: 2, name: "Champions League", country: "Europe", flag: "🇪🇺" },
  { id: 3, name: "Europa League", country: "Europe", flag: "🇪🇺" },
  { id: 848, name: "Conference League", country: "Europe", flag: "🇪🇺" },
  { id: 886, name: "Eredivisie", country: "Netherlands", flag: "🇳🇱" },
  { id: 15, name: "FIFA World Cup", country: "World", flag: "🌍" },
  { id: 960, name: "UEFA Nations League", country: "Europe", flag: "🇪🇺" },
];

/**
 * Fetch upcoming matches using API-Football
 */
export async function scrapeUpcomingMatches(
  request: ScrapeUpcomingRequest = {}
): Promise<ScrapingJob> {
  const params: Record<string, unknown> = {};
  if (request.league_ids && request.league_ids.length > 0) {
    params.league_ids = request.league_ids.join(",");
  }
  const queryString = buildQueryString(params);
  return apiPost(`/scraping/upcoming${queryString}`, {}, ScrapingJobSchema);
}

/**
 * Fetch historical results using API-Football
 */
export async function scrapeResults(
  request: ScrapeResultsRequest = {}
): Promise<ScrapingJob> {
  const params: Record<string, unknown> = {};
  if (request.league_id) {
    params.league_id = request.league_id;
  }
  if (request.max_results) {
    params.max_results = request.max_results;
  }
  const queryString = buildQueryString(params);
  return apiPost(`/scraping/results${queryString}`, {}, ScrapingJobSchema);
}

/**
 * Fetch ALL matches for a specific date across all leagues.
 * This is the equivalent of scraping all matches from OddsPortal's main page.
 * Uses only 1 API request regardless of how many leagues have matches.
 */
export async function scrapeByDate(
  request: ScrapeByDateRequest = {}
): Promise<ScrapingJob> {
  const params: Record<string, unknown> = {};
  if (request.date) {
    params.date = request.date;
  }
  if (request.league_id) {
    params.league_id = request.league_id;
  }
  const queryString = buildQueryString(params);
  return apiPost(`/scraping/by-date${queryString}`, {}, ScrapingJobSchema);
}

/**
 * Get all scraping jobs
 */
export async function getScrapingJobs(
  status?: string
): Promise<PaginatedResponse<ScrapingJob>> {
  const params: Record<string, unknown> = {};
  if (status) params.status_filter = status;

  const queryString = buildQueryString(params);
  return apiGet(`/scraping/jobs${queryString}`, scrapingJobArraySchema);
}

/**
 * Get a specific scraping job by ID
 */
export async function getScrapingJob(jobId: string): Promise<ScrapingJob> {
  return apiGet(`/scraping/jobs/${jobId}`, ScrapingJobSchema);
}

/**
 * Get scraping statistics
 */
export async function getScrapingStats(): Promise<ScrapingStats> {
  return apiGet("/scraping/stats", ScrapingStatsSchema);
}

export const ScrapingStatsSchema = z.object({
  total_jobs: z.number(),
  completed_jobs: z.number(),
  failed_jobs: z.number(),
  running_jobs: z.number(),
  total_matches_scraped: z.number(),
  average_duration_seconds: z.number().nullable(),
  success_rate: z.number(),
});

export type ScrapingStats = z.infer<typeof ScrapingStatsSchema>;
