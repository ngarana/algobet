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
api/              # FastAPI REST API (routers, schemas, websockets)
cli/              # CLI command modules (commands, presenters)
services/         # Business logic layer (scraping, prediction, scheduler)
predictions/      # Prediction engine (data, features, models, training, evaluation)
scheduler/        # Background task scheduling
```

### Layer Responsibilities

- **API Layer**: HTTP request/response handling only. No business logic. Routers delegate to services.
- **CLI Layer**: Command-line interface. No business logic. Commands call services.
- **Service Layer**: Business logic orchestration. Coordinates infrastructure and domain. Uses repositories for data access.
- **Data Layer**: Data access abstraction. Repositories encapsulate query logic.

## Anti-Patterns to Avoid

### GOD Modules
- Split large files (>500 lines) into focused modules
- Use composition over inheritance for code reuse
- Each class should have a single reason to change

### DRY Violations
- Get-or-create patterns must be centralized in repository classes
- Form calculation logic must not be repeated across generators
- URL parsing logic must be centralized in a single module

### Magic Numbers
- All constants must be named and documented
- Create `algobet/constants.py` for shared constants
- Use `algobet/config.py` with pydantic-settings for configuration

### Hard-coded Values
- Use environment variables for deployment-specific values
- Use configuration files for application settings
- Never hard-code URLs, paths, or timeouts

## Service Layer Guidelines

1. **Single Responsibility**: Each service handles one domain area
2. **No Direct Queries**: Services use repositories for data access
3. **Transaction Management**: Services manage transactions, repositories don't
4. **Stateless by Default**: No class-level mutable state in services
5. **Domain Exceptions**: Use custom exceptions, never generic ones

## Database Schema

| Table | Unique Constraint |
|-------|-------------------|
| tournaments | url_slug |
| seasons | (tournament_id, name) |
| teams | name |
| matches | (tournament_id, season_id, home_team_id, away_team_id, match_date) |
| model_versions | version |
| scheduled_tasks | name |

### Match Status Values
- `SCHEDULED` - Upcoming match, scores are None
- `FINISHED` - Completed match with scores
- `LIVE` - In progress

## Common Patterns

- **Season Naming**: Format `"YYYY/YYYY+1"` (e.g., `"2023/2024"`)
- **URL Patterns**: Parse with `r"/football/([^/]+)/([^/]+?)(?:-\d{4}-\d{4})?/results/"`
- **Match Deduplication**: Always check for existing match before insert using all key fields

## Environment Variables

- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`
- `ODDSPORTAL_BASE_URL`, `SCRAPER_HEADLESS`, `SCRAPER_TIMEOUT_MS`
- `MODELS_STORAGE_PATH`

## Code Review Checklist

- [ ] No magic numbers - all constants are named and documented
- [ ] No hard-coded URLs or paths - use configuration
- [ ] No duplicated logic - extract to shared functions/classes
- [ ] Functions under 50 lines - extract helpers if needed
- [ ] Services use repositories for data access
- [ ] No class-level mutable state in services
- [ ] All exceptions are domain-specific, not generic
- [ ] Type hints on all public functions
- [ ] Docstrings on all public classes and functions
- [ ] Tests for new functionality
