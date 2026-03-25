import { beforeEach, describe, expect, it, vi } from "vitest";
import { getScrapingJobs, scrapeResults, scrapeUpcomingMatches } from "../scraping";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const baseJob = {
  id: "job-123",
  scraping_type: "upcoming" as const,
  tournament_url: null,
  tournament_name: null,
  season: null,
  status: "pending" as const,
  progress: 0,
  message: "Job created and queued",
  created_at: "2026-03-25T10:00:00Z",
  started_at: null,
  completed_at: null,
  matches_scraped: 0,
  errors: [],
};

describe("scraping api", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("encodes upcoming scrape URLs as query params", async () => {
    const tournamentUrl = "https://www.oddsportal.com/football/england/premier-league/";

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...baseJob,
        tournament_url: tournamentUrl,
      }),
    });

    await scrapeUpcomingMatches({ url: tournamentUrl });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [requestUrl, options] = mockFetch.mock.calls[0];
    expect(requestUrl).toContain("/scraping/upcoming?");
    expect(requestUrl).toContain(encodeURIComponent(tournamentUrl));
    expect(options.method).toBe("POST");
    expect(options.body).toBe(JSON.stringify({}));
  });

  it("encodes results scrape page limits as query params", async () => {
    const tournamentUrl = "https://www.oddsportal.com/football/spain/laliga/results/";

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...baseJob,
        scraping_type: "results",
        tournament_url: tournamentUrl,
      }),
    });

    await scrapeResults({ url: tournamentUrl, max_pages: 8 });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [requestUrl] = mockFetch.mock.calls[0];
    expect(requestUrl).toContain("/scraping/results?");
    expect(requestUrl).toContain(`tournament_url=${encodeURIComponent(tournamentUrl)}`);
    expect(requestUrl).toContain("max_pages=8");
  });

  it("uses status_filter for job history requests", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        items: [baseJob],
        total: 1,
        limit: 50,
        offset: 0,
      }),
    });

    await getScrapingJobs("running");

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [requestUrl, options] = mockFetch.mock.calls[0];
    expect(requestUrl).toContain("/scraping/jobs?status_filter=running");
    expect(options.headers).toEqual({ Accept: "application/json" });
  });
});
