# League-Specific Page Analysis

## Summary

After inspecting the HTML structure of league-specific upcoming matches pages (e.g., `https://www.oddsportal.com/football/england/premier-league/`), **the scraper already works correctly without modifications**.

## Findings

### Page Structure

League-specific pages use the **same HTML structure** as the global `/matches/` page:

- `div[data-testid="game-row"]` - Match rows
- `div[data-testid="secondary-header"]` - Date headers
- `div[data-testid="sport-country-league-item"]` - Tournament headers
- `div[data-testid="odd-container-default"]` - Odds containers

### Duplicate Elements

The page contains duplicate `game-row` elements for each match:
1. **With odds** - In a bordered container (extracted)
2. **Without odds** - In a different container (filtered out)

Example from Premier League page:
- Total `game-row` elements: 30
- Unique matches: 15
- Each match appears twice with different parent classes

The scraper correctly filters duplicates by requiring odds to be present.

### Test Results

#### Premier League
```
URL: https://www.oddsportal.com/football/england/premier-league/
- Found 32 game-row elements
- Extracted 15 unique matches with odds
- Tournament metadata correctly extracted: "Premier League", "England"
```

#### Bundesliga
```
URL: https://www.oddsportal.com/football/germany/bundesliga/
- Extracted 12 matches
- Sample: St. Pauli vs Mainz (2.9 / 3.38 / 2.68)
- Tournament: "Bundesliga", Country: "Germany"
```

### Filtering Behavior

The scraper has built-in filtering:

1. **Matches without odds** - Skipped (correct behavior)
2. **Past matches** - Filtered by `only_future_matches=True` (default)
3. **Buffer time** - 30-minute buffer before match start (configurable)

## Conclusion

✅ **No changes needed** - The scraper works correctly for league-specific URLs

✅ **Same extraction logic** - Both global and league pages use identical selectors

✅ **Proper filtering** - Duplicates and invalid matches are correctly filtered

## Usage

The scraper can be used with league URLs directly:

```python
from algobet.scraper import OddsPortalScraper

with OddsPortalScraper(headless=True) as scraper:
    scraper.navigate_to_upcoming("https://www.oddsportal.com/football/england/premier-league/")
    matches = scraper.scrape_upcoming_matches()
```

Or via the API:

```bash
curl -X POST "http://localhost:8000/api/v1/scraping/upcoming" \
  -H "Content-Type: application/json" \
  -d '{
    "tournament_url": "https://www.oddsportal.com/football/england/premier-league/",
    "scope": "league"
  }'
```

## Notes

- The "Show more" button clicking works on league pages
- Infinite scroll is handled correctly
- Tournament metadata is extracted from page headers
- Date parsing handles "Today", "Tomorrow", and explicit dates
