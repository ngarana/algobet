Wired the last missing pieces in algobet/api/routers/scraping.py, algobet/services/scraping_service.py, and frontend/hooks/useFetchProgress.ts. BY DATE now uses the real OddsPortal dated path like /matches/football/YYYYMMDD/, the backend actually scrapes/filter-saves for the selected day, job timestamps are UTC-consistent, and the frontend websocket hook no longer reconnects on every render.
I also fixed the stats crash that was breaking the page’s live monitor indirectly, and added coverage in tests/test_scraping_router_progress.py.

Verified:

Real browser run on http://localhost:3001/scraping submitted {"date":"2026-04-23","scope":"all"} from the page.
The page created the job, showed Daily · All leagues · 2026-04-23, and the live monitor stayed on websocket with Socket connected and live progress instead of falling back.
Real API run for daily scraping used https://www.oddsportal.com/matches/football/20260421/ and streamed 5% -> 25% -> 75% -> completed.
Historical semantic contract still resolves correctly: posting {"tournament_id":1,"period":"2023/2024","max_pages":1} returned https://www.oddsportal.com/football/albania/abissnet-superiore-2023-2024/results/.
Results dialog still shows Country, League, Period, and Max Pages.
pytest -q tests/test_scraping_schemas.py tests/test_scraping_router_progress.py passed.
pnpm --dir frontend typecheck passed.