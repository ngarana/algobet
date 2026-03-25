# API-Football Integration Development Report

## Executive Summary

This document details the migration from web scraping (OddsPortal/Playwright) to API-Football for fetching football match data. The migration was necessary due to OddsPortal changing their page structure, which broke the existing scraper.

**Key Achievement**: Replaced unreliable web scraping with a stable JSON API that fetches all matches for a day in a single request.

---

## Problem Statement

### Original Issue
When clicking "Start" on `http://localhost:3000/scraping`, the error appeared:
```
Could not start the upcoming scrape. Double-check the URL and try again.
```

### Root Cause Analysis

1. **Frontend Port Mismatch**: Frontend was configured for port 8000, but API ran on port 8001
2. **OddsPortal Page Structure Changed**: CSS selectors like `div[data-testid="game-row"]` no longer exist
3. **JavaScript Bug**: Undeclared `currentSlug` variable caused ReferenceError
4. **Docker Network Issues**: Containers couldn't reach external APIs

### Investigation Results

| Test | Result |
|------|--------|
| Page loads | ✅ 361KB content |
| `data-testid` selectors | ❌ 0 matches found |
| Match-related classes | ❌ None exist |
| API-Football test | ✅ 260 matches returned |

---

## Solution: API-Football Integration

### Why API-Football?

| Criteria | OddsPortal Scraping | API-Football |
|----------|---------------------|--------------|
| Reliability | ❌ Breaks when page changes | ✅ Stable JSON API |
| Speed | ❌ 30+ seconds per page | ✅ ~1 second |
| Data Format | ❌ HTML parsing required | ✅ Clean JSON |
| Rate Limits | ❌ Can get blocked | ✅ 100 req/day free |
| Anti-Bot | ❌ CAPTCHA risk | ✅ No issues |
| Coverage | ✅ All leagues | ✅ 900+ leagues |

### API-Football Free Tier
- **100 requests/day**
- **900+ leagues covered**
- **Fixtures, results, odds included**
- **No credit card required**

Register at: https://dashboard.api-football.com/register

---

## Implementation

### Files Changed

#### Backend (Python)

| File | Change |
|------|--------|
| `algobet/infrastructure/api_football_client.py` | **New** - API client (668 lines) |
| `algobet/infrastructure/config.py` | Added `APIFootballConfig` class |
| `algobet/services/scraping_service.py` | Replaced Playwright with API-Football |
| `algobet/api/routers/scraping.py` | Added `/by-date` endpoint |
| `algobet/api/schemas/scraping.py` | Added `BY_DATE` scraping type |
| `algobet/matches/models.py` | Added `api_football_id`, `predictions` relationship |
| `algobet/teams/models.py` | Added `api_football_id` to Team, Tournament |

#### Frontend (TypeScript/React)

| File | Change |
|------|--------|
| `frontend/lib/api/scraping.ts` | Added `POPULAR_LEAGUES`, `scrapeByDate()` |
| `frontend/components/scraping/ScrapeFormCard.tsx` | League selection UI |
| `frontend/app/scraping/page.tsx` | "Fetch All Today" button |
| `frontend/lib/queries/use-scraping.ts` | Added `useScrapeByDate` hook |

#### Infrastructure

| File | Change |
|------|--------|
| `docker-compose.yml` | `network_mode: host` for API access |
| `.env.example` | Added API-Football configuration |

---

## API Endpoints

### New Endpoints

#### Fetch All Matches for a Date
```bash
POST /api/v1/scraping/by-date
```

**Parameters:**
- `date` (optional): YYYY-MM-DD format, defaults to today
- `league_id` (optional): Filter to specific league

**Example:**
```bash
# All matches today
curl -X POST "http://localhost:8001/api/v1/scraping/by-date"

# Specific date
curl -X POST "http://localhost:8001/api/v1/scraping/by-date?date=2026-03-25"

# Premier League only
curl -X POST "http://localhost:8001/api/v1/scraping/by-date?date=2026-03-25&league_id=39"
```

**Response:**
```json
{
  "scraping_type": "by-date",
  "status": "completed",
  "message": "Completed! Fetched 260 matches from 1 API request(s), saved 260.",
  "matches_scraped": 260
}
```

### Updated Endpoints

#### Fetch Upcoming Matches by League
```bash
POST /api/v1/scraping/upcoming?league_ids=39,140,135
```

#### Fetch Historical Results
```bash
POST /api/v1/scraping/results?league_id=39&max_results=20
```

---

## Popular League IDs

| League | ID | Country |
|--------|-----|---------|
| Premier League | 39 | England |
| La Liga | 140 | Spain |
| Serie A | 135 | Italy |
| Bundesliga | 78 | Germany |
| Ligue 1 | 61 | France |
| Champions League | 2 | Europe |
| Europa League | 3 | Europe |
| Eredivisie | 886 | Netherlands |
| Conference League | 848 | Europe |
| FIFA World Cup | 15 | World |

---

## Docker Configuration

### Network Mode Change

**Before (Broken):**
```yaml
api:
  ports:
    - "8001:8000"
  environment:
    - POSTGRES_HOST=db
```

**After (Working):**
```yaml
api:
  network_mode: host
  environment:
    - POSTGRES_HOST=localhost
    - API_PORT=8001
```

### Why Host Network Mode?

Docker's default bridge network couldn't reach external APIs. Using `network_mode: host` allows the container to use the host's network stack directly.

---

## Database Schema Changes

### New Columns

```sql
-- Added to teams table
ALTER TABLE teams ADD COLUMN api_football_id INTEGER UNIQUE;

-- Added to tournaments table
ALTER TABLE tournaments ADD COLUMN api_football_id INTEGER UNIQUE;

-- Added to matches table
ALTER TABLE matches ADD COLUMN api_football_id INTEGER UNIQUE;
```

These columns enable deduplication when re-fetching the same matches.

---

## Frontend Changes

### League Selection UI

The scraping page now shows checkboxes for popular leagues instead of a URL input:

```
┌─────────────────────────────────────────────────────────────┐
│  Select Leagues                                    [Select All] │
├─────────────────────────────────────────────────────────────┤
│  ☑ 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League        ☑ 🇪🇸 La Liga             │
│  England                        Spain                        │
│                                                               │
│  ☑ 🇮🇹 Serie A                ☑ 🇩🇪 Bundesliga            │
│  Italy                          Germany                       │
│                                                               │
│  ☑ 🇫🇷 Ligue 1                ☐ 🇪🇺 Champions League       │
│  France                         Europe                        │
└─────────────────────────────────────────────────────────────┘
```

### "Fetch All Today" Button

A prominent button at the top of the page fetches all matches for today with a single click:

```
┌─────────────────────────────────────────────────────────────┐
│  [📅 Fetch All Today] [1 req]                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Results

### API-Football Integration Test

```
Date: 2026-03-25
Endpoint: POST /api/v1/scraping/by-date?date=2026-03-25
Result: ✅ SUCCESS

{
  "status": "completed",
  "message": "Completed! Fetched 260 matches from 1 API request(s), saved 260.",
  "matches_scraped": 260,
  "requests_made": 1
}
```

### Deduplication Test

```
Run 1: Fetched 260 matches, Saved 260
Run 2: Fetched 260 matches, Saved 0 (duplicates detected)
```

---

## Rate Limit Management

### Daily Budget (Free Tier: 100 requests)

| Operation | Requests | Frequency |
|-----------|----------|-----------|
| Fetch by date | 1 | Daily |
| Fetch upcoming (5 leagues) | 5 | Daily |
| Fetch results (5 leagues) | 5 | Daily |
| **Total** | **11** | **Daily** |

**Remaining**: 89 requests for ad-hoc queries

---

## Migration Checklist

- [x] Create API-Football client
- [x] Update scraping service
- [x] Add database columns
- [x] Update API endpoints
- [x] Update frontend UI
- [x] Fix Docker networking
- [x] Update documentation
- [x] Test integration
- [x] Merge branches

---

## Git History

```bash
* 793fc4c (HEAD -> main) Merge branch 'feature/architecture-refactoring'
* 1d4d66f fix: Docker network connectivity for external API access
* 1de009 feat: add get_all_fixtures_by_date endpoint
* 73d8f97 docs: update README for API-Football integration
* 338d01f feat: integrate API-Football as replacement for web scraping
```

---

## Environment Variables

```bash
# Required
ALGOBET_API_FOOTBALL__API_KEY=your_api_key_here

# Optional
ALGOBET_API_FOOTBALL__BASE_URL=https://v3.football.api-sports.io
ALGOBET_API_FOOTBALL__RATE_LIMIT_PER_DAY=100
ALGOBET_API_FOOTBALL__TIMEOUT=30
ALGOBET_SCRAPING__DEFAULT_LEAGUE_IDS=[39,140,135,78,61]
```

---

## Troubleshooting

### "Network is unreachable"
**Solution**: Ensure `network_mode: host` in docker-compose.yml

### "0 matches saved"
**Solution**: Matches already exist in database (deduplication working)

### "403 Forbidden"
**Solution**: Check API key is set correctly in .env file

### "Not Found" on endpoint
**Solution**: Verify API is running on correct port (8001)

---

## Future Enhancements

1. **Caching**: Cache API responses to reduce request count
2. **Historical Data**: Bulk fetch past seasons for ML training
3. **Live Odds**: Real-time odds updates during matches
4. **Statistics**: Add team/player statistics from API-Football
5. **Predictions**: Use API-Football's prediction data

---

## References

- [API-Football Documentation](https://www.api-football.com/documentation-v3)
- [API-Football Registration](https://dashboard.api-football.com/register)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)

---

## Conclusion

The migration from OddsPortal web scraping to API-Football was successful. The new implementation:

- ✅ Is reliable and won't break with page changes
- ✅ Fetches all matches in 1 API request (~1 second)
- ✅ Includes odds data
- ✅ Has a clean JSON response format
- ✅ Never gets blocked or CAPTCHA'd
- ✅ Supports 900+ leagues

The free tier (100 requests/day) is sufficient for daily use.
