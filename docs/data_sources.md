# Free Football Data Sources

This document explores free alternatives to OddsPortal scraping for historical football match data and betting odds.

## Overview

The AlgoBet project currently uses Playwright-based web scraping of OddsPortal, which is:
- **Unreliable**: Subject to rate limiting, CAPTCHAs, and UI changes
- **Slow**: Requires browser automation and JavaScript rendering
- **Fragile**: Breaks when website structure changes

Football-Data.co.uk provides a reliable, free alternative with ready-to-download CSV files containing historical match data and betting odds.

---

## Football-Data.co.uk

### About

**Website**: https://www.football-data.co.uk/

Football-Data.co.uk is a free resource providing historical football results and betting odds data in CSV format. The data is updated weekly and covers major European leagues since the 1993/1994 season.

### Key Advantages

| Aspect | Football-Data.co.uk | OddsPortal Scraping |
|--------|---------------------|---------------------|
| **Reliability** | High - static CSV files | Low - subject to blocking |
| **Speed** | Fast - direct download | Slow - browser automation |
| **Maintenance** | Minimal - stable format | High - UI changes break scraper |
| **Historical Data** | 30+ seasons available | Limited by scraping capacity |
| **Betting Odds** | Multiple bookmakers | Single aggregated view |
| **Cost** | Free | Free (but higher dev cost) |

### Available Leagues

The site covers major European leagues with division codes:

| Code | League | Country |
|------|--------|---------|
| E0 | Premier League | England |
| E1 | Championship | England |
| E2 | League One | England |
| E3 | League Two | England |
| SC0 | Scottish Premiership | Scotland |
| D1 | Bundesliga | Germany |
| D2 | 2. Bundesliga | Germany |
| I1 | Serie A | Italy |
| I2 | Serie B | Italy |
| SP1 | La Liga | Spain |
| SP2 | La Liga 2 | Spain |
| F1 | Ligue 1 | France |
| F2 | Ligue 2 | France |
| N1 | Eredivisie | Netherlands |
| B1 | First Division A | Belgium |
| P1 | Primeira Liga | Portugal |
| T1 | Super Lig | Turkey |
| G1 | Super League | Greece |

### URL Pattern

CSV files are available at:
```
https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
```

Where:
- `{season}`: 4-digit season code (e.g., `2324` for 2023/2024)
- `{league}`: Division code (e.g., `E0` for Premier League)

**Examples**:
- Premier League 2023/2024: `https://www.football-data.co.uk/mmz4281/2324/E0.csv`
- La Liga 2023/2024: `https://www.football-data.co.uk/mmz4281/2324/SP1.csv`
- Serie A 2022/2023: `https://www.football-data.co.uk/mmz4281/2223/I1.csv`

---

## CSV Structure Analysis

### Column Overview

The CSV files contain **106 columns** organized into categories:

#### 1. Match Information (Columns 1-11)

| Column | Name | Description | Sample Value |
|--------|------|-------------|--------------|
| 1 | `Div` | League division code | `E0` |
| 2 | `Date` | Match date | `11/08/2023` |
| 3 | `Time` | Kick-off time | `20:00` |
| 4 | `HomeTeam` | Home team name | `Burnley` |
| 5 | `AwayTeam` | Away team name | `Man City` |
| 6 | `FTHG` | Full Time Home Goals | `0` |
| 7 | `FTAG` | Full Time Away Goals | `3` |
| 8 | `FTR` | Full Time Result (H/D/A) | `A` |
| 9 | `HTHG` | Half Time Home Goals | `0` |
| 10 | `HTAG` | Half Time Away Goals | `2` |
| 11 | `HTR` | Half Time Result (H/D/A) | `A` |

#### 2. Match Statistics (Columns 12-24)

| Column | Name | Description | Sample Value |
|--------|------|-------------|--------------|
| 12 | `Referee` | Match referee name | `C Pawson` |
| 13 | `HS` | Home Team Shots | `6` |
| 14 | `AS` | Away Team Shots | `17` |
| 15 | `HST` | Home Team Shots on Target | `1` |
| 16 | `AST` | Away Team Shots on Target | `8` |
| 17 | `HF` | Home Team Fouls | `11` |
| 18 | `AF` | Away Team Fouls | `8` |
| 19 | `HC` | Home Team Corners | `6` |
| 20 | `AC` | Away Team Corners | `5` |
| 21 | `HY` | Home Team Yellow Cards | `0` |
| 22 | `AY` | Away Team Yellow Cards | `0` |
| 23 | `HR` | Home Team Red Cards | `1` |
| 24 | `AR` | Away Team Red Cards | `0` |

#### 3. Match Odds (Columns 25-48)

Opening odds from multiple bookmakers:

| Column | Name | Description | Sample Value |
|--------|------|-------------|--------------|
| 25-27 | `B365H/D/A` | Bet365 odds | `8.0`, `5.5`, `1.33` |
| 28-30 | `BWH/D/A` | Bet&Win odds | `8.75`, `5.25`, `1.34` |
| 31-33 | `IWH/D/A` | Interwetten odds | `8.0`, `5.5`, `1.35` |
| 34-36 | `PSH/D/A` | Pinnacle odds | `8.58`, `5.51`, `1.37` |
| 37-39 | `WHH/D/A` | William Hill odds | `8.0`, `5.0`, `1.25` |
| 40-42 | `VCH/D/A` | VC Bet odds | `9.5`, `5.25`, `1.33` |
| 43-45 | `MaxH/D/A` | Market maximum odds | `9.5`, `5.68`, `1.39` |
| 46-48 | `AvgH/D/A` | Market average odds | `9.02`, `5.35`, `1.35` |

#### 4. Goals Over/Under Odds (Columns 49-56)

| Column | Name | Description |
|--------|------|-------------|
| 49-50 | `B365>2.5/<2.5` | Bet365 over/under 2.5 goals |
| 51-52 | `P>2.5/<2.5` | Pinnacle over/under 2.5 goals |
| 53-54 | `Max>2.5/<2.5` | Maximum over/under 2.5 goals |
| 55-56 | `Avg>2.5/<2.5` | Average over/under 2.5 goals |

#### 5. Asian Handicap Odds (Columns 57-65)

| Column | Name | Description |
|--------|------|-------------|
| 57 | `AHh` | Asian handicap line |
| 58-59 | `B365AHH/AHA` | Bet365 Asian handicap odds |
| 60-61 | `PAHH/PAHA` | Pinnacle Asian handicap odds |
| 62-63 | `MaxAHH/MaxAHA` | Maximum Asian handicap odds |
| 64-65 | `AvgAHH/AvgAHA` | Average Asian handicap odds |

#### 6. Closing Odds (Columns 66-106)

Same structure as opening odds but with `C` suffix (e.g., `B365CH`, `B365CD`, `B365CA`) representing odds at market close.

### Sample Data

```csv
Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,...
E0,11/08/2023,20:00,Burnley,Man City,0,3,A,0,2,A,C Pawson,6,17,1,8,11,8,6,5,0,0,1,0,8,5.5,1.33,...
E0,12/08/2023,12:30,Arsenal,Nott'm Forest,2,1,H,2,0,H,M Oliver,15,6,7,2,12,12,8,3,2,2,0,0,1.18,7,15,...
```

### Data Types

| Field Type | Format | Example |
|------------|--------|---------|
| Date | `DD/MM/YYYY` | `11/08/2023` |
| Time | `HH:MM` (24-hour) | `20:00` |
| Result | `H` / `D` / `A` | `A` |
| Goals | Integer | `0`, `1`, `2`, `3` |
| Odds | Decimal (float) | `1.33`, `5.5`, `8.0` |
| Team Name | String | `Man City`, `Arsenal` |

---

## Column Mapping to AlgoBet Models

### Tournament Mapping

| Football-Data Field | AlgoBet Model | Mapping Logic |
|---------------------|---------------|---------------|
| `Div` | `Tournament.name` | Map division code to league name |
| `Div` | `Tournament.country` | Map division code to country |
| `Div` | `Tournament.url_slug` | Generate from division code |

**Division Code Mapping Table**:

```python
DIVISION_MAPPING = {
    "E0": {"name": "Premier League", "country": "England", "url_slug": "premier-league"},
    "E1": {"name": "Championship", "country": "England", "url_slug": "championship"},
    "D1": {"name": "Bundesliga", "country": "Germany", "url_slug": "bundesliga"},
    "I1": {"name": "Serie A", "country": "Italy", "url_slug": "serie-a"},
    "SP1": {"name": "La Liga", "country": "Spain", "url_slug": "la-liga"},
    "F1": {"name": "Ligue 1", "country": "France", "url_slug": "ligue-1"},
    # ... additional mappings
}
```

### Season Mapping

| Source | AlgoBet Model | Mapping Logic |
|--------|---------------|---------------|
| CSV filename | `Season.name` | Convert `2324` → `2023/2024` |
| CSV filename | `Season.start_year` | Extract `2023` from `2324` |
| CSV filename | `Season.end_year` | Extract `2024` from `2324` |

**Season Code Conversion**:

```python
def parse_season_code(code: str) -> dict:
    """Convert season code like '2324' to season info."""
    start_year = 2000 + int(code[:2])
    end_year = 2000 + int(code[2:])
    return {
        "name": f"{start_year}/{end_year}",
        "start_year": start_year,
        "end_year": end_year,
    }
```

### Team Mapping

| Football-Data Field | AlgoBet Model | Notes |
|---------------------|---------------|-------|
| `HomeTeam` | `Team.name` | Home team name |
| `AwayTeam` | `Team.name` | Away team name |

**Team Name Normalization**:

Some team names in Football-Data differ from OddsPortal. A normalization mapping may be needed:

```python
TEAM_NAME_MAPPING = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "Wolves": "Wolverhampton Wanderers",
    # ... additional mappings
}
```

### Match Mapping

| Football-Data Field | AlgoBet Model | Notes |
|---------------------|---------------|-------|
| `Date` + `Time` | `Match.match_date` | Combine date and time |
| `HomeTeam` | `Match.home_team_id` | FK to Team |
| `AwayTeam` | `Match.away_team_id` | FK to Team |
| `FTHG` | `Match.home_score` | Full-time home goals |
| `FTAG` | `Match.away_score` | Full-time away goals |
| - | `Match.status` | Always `FINISHED` for historical data |
| `AvgH` | `Match.odds_home` | Average market odds |
| `AvgD` | `Match.odds_draw` | Average market odds |
| `AvgA` | `Match.odds_away` | Average market odds |
| - | `Match.num_bookmakers` | Count available odds columns |

**Match Data Transformation**:

```python
from datetime import datetime

def parse_match_date(date_str: str, time_str: str) -> datetime:
    """Parse Football-Data date and time into datetime."""
    # Date format: DD/MM/YYYY
    # Time format: HH:MM
    return datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
```

### Complete Mapping Example

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class FootballDataMatch:
    """Parsed match data from Football-Data.co.uk."""

    # Tournament/Season
    division: str
    season_code: str

    # Match info
    match_date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: str = "FINISHED"

    # Betting odds (using average market odds)
    odds_home: Optional[float]
    odds_draw: Optional[float]
    odds_away: Optional[float]

    # Additional data available
    half_time_home_score: int
    half_time_away_score: int
    referee: str
    home_shots: int
    away_shots: int
    home_shots_on_target: int
    away_shots_on_target: int
    home_corners: int
    away_corners: int
    home_fouls: int
    away_fouls: int
    home_yellow_cards: int
    away_yellow_cards: int
    home_red_cards: int
    away_red_cards: int


def parse_csv_row(row: dict, season_code: str) -> FootballDataMatch:
    """Parse a CSV row into a FootballDataMatch."""
    return FootballDataMatch(
        division=row["Div"],
        season_code=season_code,
        match_date=parse_match_date(row["Date"], row["Time"]),
        home_team=row["HomeTeam"],
        away_team=row["AwayTeam"],
        home_score=int(row["FTHG"]),
        away_score=int(row["FTAG"]),
        odds_home=float(row["AvgH"]) if row.get("AvgH") else None,
        odds_draw=float(row["AvgD"]) if row.get("AvgD") else None,
        odds_away=float(row["AvgA"]) if row.get("AvgA") else None,
        half_time_home_score=int(row["HTHG"]) if row.get("HTHG") else 0,
        half_time_away_score=int(row["HTAG"]) if row.get("HTAG") else 0,
        referee=row.get("Referee", ""),
        home_shots=int(row["HS"]) if row.get("HS") else 0,
        away_shots=int(row["AS"]) if row.get("AS") else 0,
        home_shots_on_target=int(row["HST"]) if row.get("HST") else 0,
        away_shots_on_target=int(row["AST"]) if row.get("AST") else 0,
        home_corners=int(row["HC"]) if row.get("HC") else 0,
        away_corners=int(row["AC"]) if row.get("AC") else 0,
        home_fouls=int(row["HF"]) if row.get("HF") else 0,
        away_fouls=int(row["AF"]) if row.get("AF") else 0,
        home_yellow_cards=int(row["HY"]) if row.get("HY") else 0,
        away_yellow_cards=int(row["AY"]) if row.get("AY") else 0,
        home_red_cards=int(row["HR"]) if row.get("HR") else 0,
        away_red_cards=int(row["AR"]) if row.get("AR") else 0,
    )
```

---

## Data Comparison: Football-Data vs OddsPortal

### Data Available in Both Sources

| Data Point | Football-Data | OddsPortal |
|------------|---------------|------------|
| Match date/time | ✅ | ✅ |
| Home/Away teams | ✅ | ✅ |
| Full-time score | ✅ | ✅ |
| Home win odds | ✅ | ✅ |
| Draw odds | ✅ | ✅ |
| Away win odds | ✅ | ✅ |

### Data Only in Football-Data

| Data Point | Description |
|------------|-------------|
| Half-time scores | `HTHG`, `HTAG` |
| Match statistics | Shots, corners, fouls, cards |
| Multiple bookmaker odds | Bet365, Pinnacle, William Hill, etc. |
| Opening vs Closing odds | Market movement tracking |
| Asian handicap odds | Alternative betting market |
| Over/Under odds | Goals market |
| Referee information | Match official name |

### Data Only in OddsPortal

| Data Point | Description |
|------------|-------------|
| Live/in-play odds | Real-time odds during matches |
| Upcoming matches | Future fixtures |
| Number of bookmakers | Aggregated count |
| Current season live | Ongoing season data |

---

## Implementation Recommendations

### 1. Create a Data Importer Service

```python
# algobet/services/football_data_importer.py

import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator
from urllib.request import urlopen

@dataclass
class FootballDataConfig:
    """Configuration for Football-Data.co.uk."""
    base_url: str = "https://www.football-data.co.uk/mmz4281"

    def get_csv_url(self, season_code: str, division: str) -> str:
        """Build CSV download URL."""
        return f"{self.base_url}/{season_code}/{division}.csv"


class FootballDataImporter:
    """Import match data from Football-Data.co.uk."""

    def __init__(self, config: FootballDataConfig | None = None):
        self.config = config or FootballDataConfig()

    def download_csv(self, season_code: str, division: str) -> str:
        """Download CSV file content."""
        url = self.config.get_csv_url(season_code, division)
        with urlopen(url) as response:
            return response.read().decode("utf-8")

    def parse_csv(self, content: str) -> list[dict]:
        """Parse CSV content into list of dictionaries."""
        reader = csv.DictReader(content.splitlines())
        return list(reader)

    def import_season(
        self,
        season_code: str,
        divisions: list[str]
    ) -> Iterator[dict]:
        """Import all matches for a season across divisions."""
        for division in divisions:
            try:
                content = self.download_csv(season_code, division)
                for row in self.parse_csv(content):
                    yield {"division": division, "season": season_code, **row}
            except Exception as e:
                # Log error and continue with other divisions
                print(f"Error importing {division} {season_code}: {e}")
                continue
```

### 2. Add CLI Command

```python
# algobet/cli/commands/import_data.py

import click

@click.command()
@click.option("--season", required=True, help="Season code (e.g., 2324)")
@click.option("--division", multiple=True, help="Division codes (e.g., E0, SP1)")
@click.option("--all-leagues", is_flag=True, help="Import all major leagues")
def import_data(season: str, division: tuple[str, ...], all_leagues: bool) -> None:
    """Import match data from Football-Data.co.uk."""
    from algobet.services.football_data_importer import FootballDataImporter

    importer = FootballDataImporter()

    if all_leagues:
        divisions = ["E0", "D1", "I1", "SP1", "F1", "N1"]
    else:
        divisions = list(division)

    matches = list(importer.import_season(season, divisions))
    click.echo(f"Imported {len(matches)} matches")
```

### 3. Database Migration

No schema changes required. The existing `Match` model supports all core fields:
- `match_date`, `home_team_id`, `away_team_id`
- `home_score`, `away_score`, `status`
- `odds_home`, `odds_draw`, `odds_away`

### 4. Hybrid Approach

For a transition period, use both sources:

```python
class MatchDataProvider:
    """Provides match data from multiple sources."""

    def __init__(self):
        self.football_data = FootballDataImporter()
        self.oddsportal = OddsPortalScraper()

    def get_historical_matches(
        self,
        tournament: str,
        season: str
    ) -> list[Match]:
        """Get historical matches from Football-Data.co.uk."""
        # Use Football-Data for completed seasons
        pass

    def get_upcoming_matches(self, tournament: str) -> list[Match]:
        """Get upcoming matches from OddsPortal."""
        # Use OddsPortal for live/upcoming data
        pass
```

---

## Limitations and Considerations

### Football-Data.co.uk Limitations

1. **Update Frequency**: Data is updated weekly, not real-time
2. **No Live Data**: Cannot get in-play odds or live match updates
3. **No Upcoming Fixtures**: Only completed matches are available
4. **Team Name Variations**: May differ from other sources
5. **Missing Data**: Some fields may be empty for older seasons

### Recommended Use Cases

| Use Case | Recommended Source |
|----------|-------------------|
| Historical analysis (30+ seasons) | Football-Data.co.uk |
| Model training data | Football-Data.co.uk |
| Backtesting strategies | Football-Data.co.uk |
| Live/upcoming matches | OddsPortal (or API) |
| Real-time odds | OddsPortal (or API) |

---

## Alternative Data Sources

### Free APIs

1. **API-Football** (https://www.api-football.com/)
   - Free tier: 100 requests/day
   - Covers 100+ leagues
   - Real-time data available

2. **Football-Data.org** (https://www.football-data.org/)
   - Free tier: 10 requests/minute
   - European leagues focus
   - Good API structure

3. **TheSportsDB** (https://www.thesportsdb.com/)
   - Free, community-maintained
   - Multiple sports
   - Less reliable for odds

### Commercial APIs

1. **Sportmonks** - Comprehensive football data
2. **RapidAPI** - Multiple football data providers
3. **Betfair API** - Betting exchange data

---

## Conclusion

Football-Data.co.uk provides an excellent free alternative to OddsPortal scraping for historical match data and betting odds. The key benefits are:

1. **Reliability**: Static CSV files, no scraping required
2. **Rich Data**: Match statistics, multiple bookmaker odds
3. **Historical Depth**: 30+ seasons of data
4. **Cost**: Completely free

**Recommendation**: Use Football-Data.co.uk as the primary source for historical data and model training. Keep OddsPortal scraping only for live/upcoming matches until a suitable API integration is implemented.

---

## Next Steps

1. [ ] Implement `FootballDataImporter` service
2. [ ] Add CLI commands for data import
3. [ ] Create team name normalization mapping
4. [ ] Add database seeding from Football-Data
5. [ ] Update prediction pipeline to use imported data
6. [ ] Consider API integration for live data
