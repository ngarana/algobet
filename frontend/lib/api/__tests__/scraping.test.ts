import { beforeEach, describe, expect, it, vi } from "vitest";
import { getScrapingJobs, scrapeResults, scrapeUpcomingMatches } from "../scraping";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const baseJob = {
  id: "job-123",
  scraping_type: "upcoming" as const,
  tournament_url: null,
  tournament_id: null,
  tournament_name: null,
  season: null,
  scope: "all" as const,
  country: null,
  league_name: null,
  period: null,
  status: "pending" as const,
  progress: 0,
  message: "Job created and queued",
  created_at: "2026-03-25T10:00:00Z",
  started_at: null,
  completed_at: null,
  matches_scraped: 0,
  matches_saved: 0,
  errors: [],
};

describe("scraping api", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("posts semantic upcoming scrape payloads", async () => {
    const tournamentId = 42;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({
        ...baseJob,
        scraping_type: "upcoming",
        tournament_id: tournamentId,
        scope: "league",
      }),
      text: async () =>
        JSON.stringify({
          ...baseJob,
          scraping_type: "upcoming",
          tournament_id: tournamentId,
          scope: "league",
        }),
    });

    await scrapeUpcomingMatches({ tournament_id: tournamentId, scope: "league" });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [requestUrl, options] = mockFetch.mock.calls[0];
    expect(requestUrl).toContain("/scraping/upcoming");
    expect(options.method).toBe("POST");
    expect(options.body).toBe(
      JSON.stringify({
        tournament_id: tournamentId,
        tournament_url: undefined,
        scope: "league",
      })
    );
  });

  it("posts results scrape payloads with period and page limit", async () => {
    const tournamentId = 15;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({
        ...baseJob,
        scraping_type: "results",
        tournament_id: tournamentId,
        scope: "league",
        period: "2023/2024",
      }),
      text: async () =>
        JSON.stringify({
          ...baseJob,
          scraping_type: "results",
          tournament_id: tournamentId,
          scope: "league",
          period: "2023/2024",
        }),
    });

    await scrapeResults({
      tournament_id: tournamentId,
      period: "2023/2024",
      max_pages: 8,
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [requestUrl, options] = mockFetch.mock.calls[0];
    expect(requestUrl).toContain("/scraping/results");
    expect(options.body).toBe(
      JSON.stringify({
        tournament_id: tournamentId,
        tournament_url: undefined,
        period: "2023/2024",
        max_pages: 8,
      })
    );
  });

  it("uses status_filter for job history requests", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({
        items: [baseJob],
        total: 1,
        limit: 50,
        offset: 0,
      }),
      text: async () =>
        JSON.stringify({
          items: [baseJob],
          total: 1,
          limit: 50,
          offset: 0,
        }),
    });

    await getScrapingJobs("running");

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [requestUrl, options] = mockFetch.mock.calls[0];
    expect(requestUrl).toContain("/scraping/jobs");
    expect(options.headers).toEqual({ Accept: "application/json" });
  });
});
