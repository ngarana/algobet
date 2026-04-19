/**
 * Compatibility exports for scraping operations.
 */

export {
  fetchByDate as scrapeByDate,
  fetchResults as scrapeResults,
  fetchUpcomingMatches as scrapeUpcomingMatches,
  getFetchJob as getScrapingJob,
  getFetchJobs as getScrapingJobs,
  getFetchStats as getScrapingStats,
} from "./fetch";

export type {
  FetchByDateRequest as DailyScrapeRequest,
  FetchJob as ScrapingJob,
  FetchResultsRequest as ResultsScrapeRequest,
  FetchStats as ScrapingStats,
  FetchUpcomingRequest as UpcomingScrapeRequest,
} from "./fetch";
