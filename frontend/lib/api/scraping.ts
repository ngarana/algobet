// API client functions for scraping operations

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export interface ScrapingProgress {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  current_page: number;
  total_pages: number;
  matches_scraped: number;
  matches_saved: number;
  message: string;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface ScrapingJob {
  id: string;
  job_type: string;
  url: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  progress: ScrapingProgress | null;
}

export interface ScrapeUpcomingRequest {
  url?: string;
}

export interface ScrapeResultsRequest {
  url: string;
  max_pages?: number;
}

/**
 * Scrape upcoming matches from OddsPortal
 */
export async function scrapeUpcomingMatches(request: ScrapeUpcomingRequest = {}): Promise<ScrapingJob> {
  const response = await fetch(`${API_BASE_URL}/scraping/upcoming`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to scrape upcoming matches: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Scrape historical results from OddsPortal
 */
export async function scrapeResults(request: ScrapeResultsRequest): Promise<ScrapingJob> {
  const response = await fetch(`${API_BASE_URL}/scraping/results`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to scrape results: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get all scraping jobs
 */
export async function getScrapingJobs(status?: string): Promise<ScrapingJob[]> {
  const url = status
    ? `${API_BASE_URL}/scraping/jobs?status=${status}`
    : `${API_BASE_URL}/scraping/jobs`;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to get scraping jobs: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get a specific scraping job by ID
 */
export async function getScrapingJob(jobId: string): Promise<ScrapingJob> {
  const response = await fetch(`${API_BASE_URL}/scraping/jobs/${jobId}`);

  if (!response.ok) {
    throw new Error(`Failed to get scraping job: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Cancel a scraping job
 */
export async function cancelScrapingJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/scraping/jobs/${jobId}/cancel`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel scraping job: ${response.statusText}`);
  }
}
