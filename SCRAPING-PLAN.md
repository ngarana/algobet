# Scraping Page Contract Alignment and Semantic Targeting

## Summary
- Rework scraping into three explicit user flows: `Daily Matches`, `Historical Matches`, and `Upcoming Matches`.
- Move the primary frontend UX from raw OddsPortal URLs to semantic selectors: `country`, `league`, and `period/date`, while keeping legacy `tournament_url` support in the API for backward compatibility.
- Make the backend contract authoritative and typed, then update the frontend page, hooks, and API client to use that contract end-to-end.
- Keep existing job history, stats, and WebSocket progress monitoring, but update them to display semantic target metadata instead of relying on parsing URLs.

## API and Interface Changes
- Keep the existing routes:
  - `POST /scraping/upcoming`
  - `POST /scraping/results`
  - `POST /scraping/by-date`
  - `GET /scraping/jobs`
  - `GET /scraping/jobs/{job_id}`
  - `GET /scraping/stats`
- Change the write routes to accept typed request bodies as the primary contract, while still accepting current query params as fallback for compatibility.
- Add request schemas:
  - `UpcomingScrapeRequest`
    - `scope: "all" | "league"` default `"all"`
    - `tournament_id?: number`
    - `tournament_url?: HttpUrl`
  - `HistoricalScrapeRequest`
    - `tournament_id?: number`
    - `tournament_url?: HttpUrl`
    - `period?: string` using season labels like `2023/2024` or `2023-2024`
    - `max_pages?: number` preserved for compatibility if already used elsewhere
  - `DailyScrapeRequest`
    - `scope: "all" | "league"` default `"all"`
    - `date?: date` default today
    - `tournament_id?: number`
    - `tournament_url?: HttpUrl`
- Extend job metadata returned by `ScrapingJobResponse` so the UI can render targets without URL parsing:
  - `scope`
  - `country`
  - `league_name`
  - `period`
  - keep existing `scraping_type`, `tournament_url`, `tournament_name`, timestamps, status, progress, counts
- Preserve job list pagination shape as `items`, `total`, `limit`, `offset`; update stale tests/docs still expecting `jobs/page/page_size`.
- Backend request-resolution rules:
  - If `tournament_id` is present, resolve tournament from `/tournaments` data in the DB and build the OddsPortal target URL server-side.
  - If `period` is present for historical scraping, resolve season via `Season.name` for that tournament and use `Season.url_suffix` when available to build the season-specific results URL.
  - If only `tournament_url` is provided, use it exactly as today.
  - For `daily` and `upcoming`, allow `scope="all"` with no tournament.
  - For `historical`, require a league target via `tournament_id` or `tournament_url`.

## Implementation Changes
- Backend
  - Refactor scraping router to parse body-first requests, resolve semantic inputs to URLs, and store semantic metadata on jobs.
  - Add a small target-resolution layer in the scraping service/router to convert `tournament_id + period` into the correct results URL.
  - Fix progress wiring so background jobs pass `progress_callback`, forward incremental progress to WebSockets, set `started_at`, and preserve `matches_scraped/matches_saved`.
  - Make `by-date` perform true date-driven scraping rather than overloading `season` with the selected date.
- Frontend
  - Replace the current generic dialog with a structured form that has:
    - mode selector: `Daily`, `Historical`, `Upcoming`
    - country selector populated from `/tournaments`
    - league selector filtered by country
    - period selector populated from `/tournaments/{id}/seasons` for historical mode
    - date picker for daily mode
    - scope toggle for daily: `All leagues` or `Specific league`
  - Submit typed request bodies through `frontend/lib/api/fetch.ts`; stop sending empty POST bodies plus URL-only query strings as the primary path.
  - Update page state, hooks, and cards to use semantic fields from job responses and show clearer labels in history/live monitor.
  - Keep legacy URL entry only as an advanced fallback if needed, not the main UI.
- Docs and tests
  - Update frontend tests that still import deleted `lib/api/scraping.ts`.
  - Update backend tests to the current pagination model and the new semantic request contract.
  - Update API docs/examples to show semantic body payloads first, with legacy query examples marked as compatibility mode.

## Test Plan
- Backend
  - Create jobs for upcoming all-leagues, daily all-leagues, daily specific-league, and historical season-based league scrape.
  - Verify `tournament_id` resolves to correct target metadata and URL.
  - Verify `period` maps to the selected season and historical route rejects missing league target.
  - Verify job responses include semantic metadata plus existing status/progress fields.
  - Verify progress callbacks update `started_at`, broadcast WebSocket progress, and stats aggregate `matches_scraped` correctly.
- Frontend
  - Form tests for mode switching, dependent country/league selectors, season loading, and daily scope toggle behavior.
  - API client tests for semantic request bodies, compatibility fallback, and response normalization.
  - Page tests ensuring created jobs appear in history and live monitor with country/league/period labels.
- End-to-end
  - Start a daily all-leagues scrape, a daily single-league scrape, and a historical season scrape; confirm the correct request payload, job creation, polling/WebSocket updates, and final job rendering.

## Assumptions and Defaults
- `/tournaments` and `/tournaments/{id}/seasons` are the source of truth for selector options.
- Historical scraping is season-first; daily scraping uses a date picker and can target either all leagues or a selected league.
- Existing route paths and job polling/WebSocket endpoints remain unchanged.
- Legacy `tournament_url` support remains in place during migration, but the primary UX and examples move to semantic inputs.
