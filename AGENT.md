# AlgoBet

Football match database and OddsPortal scraper for historical data, betting odds, and team statistics.

## Code Style

- Use Python 3.10+ type hints with `from __future__ import annotations` for forward references
- Follow SQLAlchemy 2.0+ patterns: use `Mapped[T]` with `mapped_column()` for model fields
- Use `DeclarativeBase` as base class for all ORM models
- Use Click decorators for CLI: `@click.group()`, `@cli.command()`, `@click.option()`
- Use `session_scope()` context manager for all database transactions
- Keep functions under 50 lines; extract helpers if longer
- Use f-strings for string formatting

## Architecture

```
algobet/
├── __init__.py       # Package version info
├── cli.py            # Click CLI commands (scrape, export, init, seasons)
├── database.py       # DB connection, session management
├── models.py         # SQLAlchemy ORM models
├── scraper.py        # Playwright-based scraper
└── predictions/      # Prediction engine package
    ├── __init__.py
    ├── data/         # Data access layer
    │   ├── __init__.py
    │   └── queries.py  # MatchRepository for querying match data
    ├── features/     # Feature engineering
    │   ├── __init__.py
    │   └── form_features.py  # FormCalculator for team form metrics
    └── models/       # ML model management
        ├── __init__.py
        └── registry.py  # ModelRegistry for model lifecycle
```

- CLI commands defined in `cli.py`; each command is a separate function
- Models in `models.py` follow declarative pattern with relationships
- Scraper uses context manager: `with OddsPortalScraper(headless=True) as scraper:`
- Database operations wrapped in `session_scope()` for automatic commit/rollback
- `MatchRepository` provides data access for historical matches, team matches, H2H
- `FormCalculator` computes team form metrics (points, goals, venue-specific)
- `ModelRegistry` manages model versioning, storage, and activation

## Database Schema

### Models

| Table | Key Fields | Unique Constraint |
|-------|------------|-------------------|
| tournaments | id, name, country, url_slug | url_slug |
| seasons | id, tournament_id, name, start_year, end_year | (tournament_id, name) |
| teams | id, name | name |
| matches | id, tournament_id, season_id, home_team_id, away_team_id, match_date | (tournament_id, season_id, home_team_id, away_team_id, match_date) |
| model_versions | id, name, version, algorithm, file_path, is_active | version |

### Relationships
- Tournament → Seasons (cascade delete-orphan)
- Tournament → Matches (cascade delete-orphan)
- Season → Matches (cascade delete-orphan)
- Team → home_matches, away_matches (cascade delete-orphan)

### Match Status Values
- `SCHEDULED` - Upcoming match, scores are None
- `FINISHED` - Completed match with scores
- `LIVE` - In progress

## Common Patterns

### Get-or-Create Pattern
```python
def get_or_create_tournament(session, country: str, name: str, slug: str) -> Tournament:
    tournament = session.execute(
        select(Tournament).where(Tournament.url_slug == slug)
    ).scalar_one_or_none()
    if not tournament:
        tournament = Tournament(name=name, country=country, url_slug=slug)
        session.add(tournament)
        session.flush()
    return tournament
```

### Season Naming
- Format: `"YYYY/YYYY+1"` (e.g., `"2023/2024"`)
- Parse with: `re.findall(r"\d{4}", season_name)` to extract years

### URL Patterns
- Current: `https://www.oddsportal.com/football/{country}/{league}/results/`
- Past: `https://www.oddsportal.com/football/{country}/{league}-{start}-{end}/results/`
- Parse with: `r"/football/([^/]+)/([^/]+?)(?:-\d{4}-\d{4})?/results/"`

### Match Deduplication
Always check for existing match before insert:
```python
existing = session.execute(
    select(Match).where(
        Match.tournament_id == tournament.id,
        Match.season_id == season.id,
        Match.home_team_id == home_team.id,
        Match.away_team_id == away_team.id,
        Match.match_date == scraped.match_date,
    )
).scalar_one_or_none()
```

### MatchRepository Pattern
Use `MatchRepository` for querying match data within `session_scope()`:
```python
with session_scope() as session:
    repo = MatchRepository(session)
    matches = repo.get_historical_matches(
        min_date=datetime(2023, 1, 1),
        tournament_id=tournament.id
    )
```

### FormCalculator Pattern
Compute team form metrics using `FormCalculator`:
```python
with session_scope() as session:
    repo = MatchRepository(session)
    calc = FormCalculator(repo)
    form = calc.calculate_recent_form(
        team_id=team.id,
        reference_date=match.match_date,
        n_matches=5
    )
```

### ModelRegistry Pattern
Save and load trained models using `ModelRegistry`:
```python
with session_scope() as session:
    registry = ModelRegistry(storage_path=Path("data/models"), session=session)
    version = registry.save_model(
        model=trained_model,
        name="xgboost_predictor",
        metrics={"accuracy": 0.65, "log_loss": 0.85}
    )
    registry.activate_model(version)

## Environment Variables

Database connection (loaded via `python-dotenv`):
- `POSTGRES_USER` (default: "algobet")
- `POSTGRES_PASSWORD` (default: "password")
- `POSTGRES_DB` (default: "football")
- `POSTGRES_HOST` (default: "localhost")
- `POSTGRES_PORT` (default: "5432")

## Dependencies

- `playwright>=1.40.0` - Headless browser scraping
- `sqlalchemy>=2.0.0` - ORM
- `psycopg2-binary>=2.9.9` - PostgreSQL adapter
- `click>=8.0.0` - CLI framework
- `python-dotenv>=1.0.0` - Environment variables
- `pandas` - Data manipulation for ML features
- `numpy` - Numerical operations

## Testing

- Use pytest for all tests
- Mock Playwright browser interactions with `unittest.mock`
- Test database with in-memory SQLite or test PostgreSQL instance
- Aim for >80% code coverage on business logic

## Security

- Never commit API keys, secrets, or credentials to repository
- Use environment variables for all sensitive data
- Validate all user inputs before database operations
- Use parameterized queries (SQLAlchemy handles this automatically)
- Never log or print sensitive information (passwords, tokens)

