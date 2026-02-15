/**
 * API client functions for scraping operations
 */

import { apiGet, apiPost, buildQueryString } from './client'
import { z } from 'zod'
import { createPaginatedResponseSchema } from '@/lib/types/schemas'
import type { PaginatedResponse } from '@/lib/types/api'

// Zod schemas for runtime validation
export const ScrapingProgressSchema = z.object({
  job_id: z.string(),
  progress: z.number(),
  message: z.string(),
  matches_scraped: z.number(),
  status: z.enum(['pending', 'running', 'completed', 'failed', 'cancelled']).optional(),
  timestamp: z.string(),
})

export const ScrapingJobSchema = z.object({
  id: z.string(),
  scraping_type: z.enum(['upcoming', 'results']),
  tournament_url: z.string().nullable(),
  tournament_name: z.string().nullable(),
  season: z.string().nullable(),
  status: z.enum(['pending', 'running', 'completed', 'failed', 'cancelled']),
  progress: z.number(),
  message: z.string().nullable(),
  created_at: z.string(),
  started_at: z.string().nullable(),
  completed_at: z.string().nullable(),
  matches_scraped: z.number(),
  errors: z.array(z.string()),
})

export const scrapingJobArraySchema = createPaginatedResponseSchema(ScrapingJobSchema)

export const ScrapeUpcomingRequestSchema = z.object({
  url: z.string().optional(),
})

export const ScrapeResultsRequestSchema = z.object({
  url: z.string(),
  max_pages: z.number().optional(),
})

// Types derived from schemas
export type ScrapingProgress = z.infer<typeof ScrapingProgressSchema>
export type ScrapingJob = z.infer<typeof ScrapingJobSchema>
export type ScrapeUpcomingRequest = z.infer<typeof ScrapeUpcomingRequestSchema>
export type ScrapeResultsRequest = z.infer<typeof ScrapeResultsRequestSchema>

/**
 * Scrape upcoming matches from OddsPortal
 */
export async function scrapeUpcomingMatches(request: ScrapeUpcomingRequest = {}): Promise<ScrapingJob> {
  return apiPost('/scraping/upcoming', request, ScrapingJobSchema)
}

/**
 * Scrape historical results from OddsPortal
 */
export async function scrapeResults(request: ScrapeResultsRequest): Promise<ScrapingJob> {
  return apiPost('/scraping/results', request, ScrapingJobSchema)
}

/**
 * Get all scraping jobs
 */
export async function getScrapingJobs(status?: string): Promise<PaginatedResponse<ScrapingJob>> {
  const params: Record<string, unknown> = {}
  if (status) params.status = status

  const queryString = buildQueryString(params)
  return apiGet(`/scraping/jobs${queryString}`, scrapingJobArraySchema)
}

/**
 * Get a specific scraping job by ID
 */
export async function getScrapingJob(jobId: string): Promise<ScrapingJob> {
  return apiGet(`/scraping/jobs/${jobId}`, ScrapingJobSchema)
}

/**
 * Cancel a scraping job
 */
export async function cancelScrapingJob(jobId: string): Promise<void> {
  return apiPost(`/scraping/jobs/${jobId}/cancel`, {})
}