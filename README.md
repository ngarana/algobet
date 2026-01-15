# AlgoBet - Football Match Database & OddsPortal Scraper

A Python tool to scrape historical football match data (teams, results, betting odds) from [OddsPortal](https://www.oddsportal.com) and store it in a SQLite database.

## Features

- 📊 **Database**: SQLite with SQLAlchemy ORM models for tournaments, seasons, teams, and matches
- 🌐 **Web Scraper**: Playwright-based scraper that handles JavaScript rendering
- 📅 **Multi-Season**: Support for scraping ranges of seasons (`--from-season`, `--to-season`)
- 📤 **Export**: CSV export with filters by tournament/season

## Installation

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .
uv run playwright install chromium

# Or using pip
pip install -e .
playwright install chromium
```

## Usage

### Scrape Current Season
```bash
python -m algobet.cli scrape --url "https://www.oddsportal.com/football/england/premier-league/results/"
```

### Scrape Past Season
```bash
python -m algobet.cli scrape --url "https://www.oddsportal.com/football/england/premier-league-2023-2024/results/"
```

### Scrape Multiple Seasons
```bash
# All available seasons
python -m algobet.cli scrape-all --url "https://www.oddsportal.com/football/england/premier-league/results/"

# Specific range
python -m algobet.cli scrape-all --url "..." --from-season "2020/2021" --to-season "2023/2024"
```

### List Available Seasons
```bash
python -m algobet.cli seasons --url "https://www.oddsportal.com/football/england/premier-league/results/"
```

### Export to CSV
```bash
python -m algobet.cli export -o matches.csv --tournament "Premier" --season "2023/2024"
```

## Database Schema

| Table | Columns |
|-------|---------|
| tournaments | id, name, country, url_slug |
| seasons | id, tournament_id, name, start_year, end_year, url_suffix |
| teams | id, name |
| matches | id, tournament_id, season_id, home/away_team_id, match_date, home/away_score, odds_home/draw/away, num_bookmakers |

## URL Patterns

- **Current Season**: `https://www.oddsportal.com/football/england/premier-league/results/`
- **Past Seasons**: `https://www.oddsportal.com/football/england/premier-league-2023-2024/results/`

## Notes

- The scraper uses headless Chromium by default
- Use `--no-headless` flag to see the browser (may have timeout issues on slow connections)
- Matches are deduplicated based on tournament, season, teams, and date
