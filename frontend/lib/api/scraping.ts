/**
 * Scraping API client — semantic request interface aligned with backend contract.
 */

import { buildQueryString } from "./client";
import type { PaginatedResponse } from "@/lib/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export interface ScrapingJob {
  id: string;
  scraping_type: "upcoming" | "results" | "by-date";
  tournament_url: string | null;
  tournament_name: string | null;
  tournament_id?: number | null;
  season: string | null;
  scope: string;
  country?: string | null;
  league_name?: string | null;
  period?: string | null;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  matches_scraped: number;
  matches_saved: number;
  errors: string[];
}

export interface UpcomingScrapeRequest {
  url?: string;
  tournament_id?: number;
  scope?: "all" | "league";
}

export interface ResultsScrapeRequest {
  url?: string;
  tournament_id?: number;
  period?: string;
  max_pages?: number;
}

export interface DailyScrapeRequest {
  date?: string;
  url?: string;
  tournament_id?: number;
  scope?: "all" | "league";
}

async function post(path: string): Promise<ScrapingJob> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json() as Promise<ScrapingJob>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export async function scrapeUpcomingMatches(
  request: UpcomingScrapeRequest = {}
): Promise<ScrapingJob> {
  const params: Record<string, unknown> = {};
  if (request.url) params.tournament_url = request.url;
  if (request.tournament_id) params.tournament_id = request.tournament_id;
  return post(`/scraping/upcoming${buildQueryString(params)}`);
}

export async function scrapeResults(
  request: ResultsScrapeRequest = {}
): Promise<ScrapingJob> {
  const params: Record<string, unknown> = {};
  if (request.url) params.tournament_url = request.url;
  if (request.tournament_id) params.tournament_id = request.tournament_id;
  if (request.period) params.period = request.period;
  if (request.max_pages) params.max_pages = request.max_pages;
  return post(`/scraping/results${buildQueryString(params)}`);
}

export async function scrapeByDate(
  request: DailyScrapeRequest = {}
): Promise<ScrapingJob> {
  const params: Record<string, unknown> = {};
  if (request.date) params.date = request.date;
  if (request.url) params.tournament_url = request.url;
  if (request.tournament_id) params.tournament_id = request.tournament_id;
  return post(`/scraping/by-date${buildQueryString(params)}`);
}

export async function getScrapingJobs(
  status?: string
): Promise<PaginatedResponse<ScrapingJob>> {
  const params: Record<string, unknown> = {};
  if (status) params.status_filter = status;
  const data = await get<{
    items: ScrapingJob[];
    total: number;
    limit: number;
    offset: number;
  }>(`/scraping/jobs${buildQueryString(params)}`);
  return {
    items: data.items ?? [],
    total: data.total,
    limit: data.limit,
    offset: data.offset,
  };
}

export async function getScrapingJob(jobId: string): Promise<ScrapingJob> {
  return get<ScrapingJob>(`/scraping/jobs/${jobId}`);
}

export async function getScrapingStats() {
  return get<{
    total_jobs: number;
    completed_jobs: number;
    failed_jobs: number;
    running_jobs: number;
    total_matches_scraped: number;
    average_duration_seconds: number | null;
    success_rate: number;
  }>("/scraping/stats");
}
