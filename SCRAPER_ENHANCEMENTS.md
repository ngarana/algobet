# Scraper Enhancements - League-Specific Upcoming Matches

## Summary

Enhanced the OddsPortal scraper and frontend to support scraping upcoming matches from league-specific URLs (e.g., `https://www.oddsportal.com/football/england/premier-league/`).

## Changes Made

### 1. Backend - Scraper (`algobet/scraper.py`)

**Updated `navigate_to_upcoming()` method:**
- Enhanced docstring to clarify support for both global and league-specific URLs
- Examples:
  - Global: `https://www.oddsportal.com/matches/`
  - League: `https://www.oddsportal.com/football/england/premier-league/`

No code changes were needed - the scraper already supports any URL pattern.

### 2. Frontend - FetchDialog Component (`frontend/components/scraping/FetchDialog.tsx`)

**Added league URL input mode for upcoming matches:**

1. **New State Variables:**
   - `upcomingInputMode`: Toggle between "select" (dropdown) and "link" (URL input)
   - `upcomingLeagueLink`: Store the pasted league URL

2. **Updated Type Definition:**
   ```typescript
   type FetchDialogSubmitData = {
     type: "upcoming";
     scope: "all" | "league";
     tournament_id?: number;
     tournament_url?: string;  // ← Added
     team_id?: number;
   } | ...
   ```

3. **New UI Elements:**
   - Radio buttons to switch between "Select from dropdown" and "Paste league link"
   - Text input field for league URL with placeholder
   - Help text explaining the feature

4. **Updated Logic:**
   - `requiresLeague`: Now considers upcoming link mode
   - `handleConfirm`: Passes `tournament_url` when in link mode
   - `isConfirmDisabled`: Validates URL input when in link mode

### 3. Frontend - Scraping Page (`frontend/app/scraping/page.tsx`)

**Updated `handleDialogConfirm`:**
- Added `tournament_url` parameter to upcoming match type
- Passes URL to `fetchUpcoming()` function

### 4. Frontend - Job Operations Hook (`frontend/hooks/useJobOperations.ts`)

**Updated `fetchUpcoming` function:**
- Added `tournament_url` parameter to interface
- Passes URL to mutation function

## Usage

### Via Frontend UI

1. Navigate to the Scraping page
2. Click "UPCOMING" button
3. Select "Specific league" scope
4. Choose "Paste league link" mode
5. Enter league URL (e.g., `https://www.oddsportal.com/football/germany/bundesliga/`)
6. Click "Start Fetch"

### Via API

```bash
curl -X POST "http://localhost:8000/api/v1/scraping/upcoming" \
  -H "Content-Type: application/json" \
  -d '{
    "tournament_url": "https://www.oddsportal.com/football/england/premier-league/",
    "scope": "league"
  }'
```

## Benefits

1. **Flexibility**: Users can scrape any league without it being in the database
2. **Speed**: Direct URL input is faster than navigating dropdowns
3. **Discovery**: Users can explore leagues not yet tracked in the system
4. **Consistency**: Matches the existing pattern used for results scraping

## Testing

Test with various league URLs:
- Premier League: `https://www.oddsportal.com/football/england/premier-league/`
- Bundesliga: `https://www.oddsportal.com/football/germany/bundesliga/`
- La Liga: `https://www.oddsportal.com/football/spain/laliga/`
- Serie A: `https://www.oddsportal.com/football/italy/serie-a/`

## Notes

- The scraper's `scrape_upcoming_matches()` method already handles league-specific pages correctly
- The "Show more" button clicking and infinite scroll logic work for both global and league pages
- Tournament metadata extraction from the page works automatically
