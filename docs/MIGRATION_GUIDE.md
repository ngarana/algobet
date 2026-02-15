# AlgoBet CLI Migration Guide

## Overview

This guide helps developers migrate from the old monolithic CLI architecture to the new modular, service-oriented architecture with async support. The refactored system improves maintainability, testability, and extensibility.

## Key Changes

### 1. From Monolithic to Modular Architecture

**Before (Old Approach):**
- Single 922-line `dev_tools.py` file
- Mixed concerns: CLI parsing, business logic, display formatting
- Direct database operations in CLI commands
- No separation of concerns

**After (New Approach):**
- Modular structure with clear separation of concerns
- Service layer for business logic
- Dependency injection for managing dependencies
- Async support for I/O-bound operations
- Structured logging and error handling

### 2. Command Structure Changes

| Old Command | New Command | Reason |
|-------------|-------------|---------|
| `algobet init-db` | `algobet db init` | Better grouping of database commands |
| `algobet reset-db` | `algobet db reset` | Consistent naming convention |
| `algobet db-stats` | `algobet db stats` | Grouped under `db` namespace |
| `algobet list-tournaments` | `algobet list tournaments` | Grouped under `list` namespace |
| `algobet list-teams` | `algobet list teams` | Consistent with other list commands |
| `algobet upcoming-matches` | `algobet list upcoming` | Logical grouping |
| `algobet delete-model` | `algobet model delete` | Grouped under `model` namespace |
| `algobet backtest` | `algobet analyze backtest` | Grouped under `analyze` namespace |
| `algobet value-bets` | `algobet analyze value-bets` | Consistent analysis commands |
| `algobet calibrate` | `algobet analyze calibrate` | Part of analysis workflow |

## Migration Steps

### Step 1: Update Imports and Dependencies

**Old Way:**
```python
# Direct imports from various modules
from algobet.database import get_session
from algobet.models import Tournament, Team
from algobet.predictions import backtest_model
```

**New Way:**
```python
# Use dependency injection container
from algobet.cli.container import Container
from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import get_logger
```

### Step 2: Migrate Business Logic to Services

**Old Way:**
```python
@click.command()
def get_db_stats():
    session = get_session()
    tournament_count = session.query(Tournament).count()
    team_count = session.query(Team).count()
    match_count = session.query(Match).count()
    session.close()

    click.echo(f"Tournaments: {tournament_count}")
    click.echo(f"Teams: {team_count}")
    click.echo(f"Matches: {match_count}")
```

**New Way:**
```python
from algobet.cli.container import ServiceLocator

@click.command()
@handle_errors
def get_db_stats():
    logger = get_logger(__name__)
    db_service = ServiceLocator.get_database_service()

    stats = db_service.get_stats()

    logger.info("Database statistics retrieved", extra=stats)
    click.echo(f"Tournaments: {stats['tournaments']}")
    click.echo(f"Teams: {stats['teams']}")
    click.echo(f"Matches: {stats['matches']}")
```

### Step 3: Add Error Handling

**Old Way:**
```python
def some_operation():
    # No error handling
    result = risky_operation()
    return result
```

**New Way:**
```python
@handle_errors
def some_operation():
    # Automatic error handling
    result = risky_operation()
    return result
```

### Step 4: Add Logging

**Old Way:**
```python
def process_data():
    print("Starting data processing...")
    # Process data
    print("Data processing completed")
```

**New Way:**
```python
@handle_errors
def process_data():
    logger = get_logger(__name__)
    logger.info("Starting data processing")

    # Process data
    logger.info("Data processing completed")
```

### Step 5: Implement Async Operations (When Appropriate)

**Old Way:**
```python
@click.command()
def fetch_data():
    data = fetch_from_api()  # Synchronous operation
    process_data(data)
```

**New Way:**
```python
from algobet.cli.async_runner import click_async

@click.command()
@click_async
async def fetch_data():
    data = await async_fetch_from_api()  # Asynchronous operation
    await async_process_data(data)
```

## Working with Services

### Sync Services
```python
from algobet.cli.container import ServiceLocator

# Get sync services
db_service = ServiceLocator.get_database_service()
query_service = ServiceLocator.get_query_service()
model_service = ServiceLocator.get_model_management_service()
```

### Async Services
```python
from algobet.cli.container import ServiceLocator

# Get async services
async_db_service = ServiceLocator.get_async_database_service()
async_query_service = ServiceLocator.get_async_query_service()
async_model_service = ServiceLocator.get_async_model_management_service()
```

## Configuration Updates

### Accessing Configuration
**Old Way:**
```python
import os
db_url = os.getenv('DATABASE_URL')
debug = os.getenv('DEBUG', 'false').lower() == 'true'
```

**New Way:**
```python
from algobet.config import get_config

config = get_config()
db_url = config.database.url
debug = config.app.debug
```

## Testing Updates

### Unit Tests for Services
```python
# tests/unit/services/test_my_service.py
import pytest
from unittest.mock import MagicMock
from algobet.services.my_service import MyService

class TestMyService:
    def test_my_method(self):
        mock_session = MagicMock()
        service = MyService(mock_session)

        result = service.my_method()

        assert result is not None
```

### CLI Command Tests
```python
# tests/unit/cli/test_my_command.py
from click.testing import CliRunner
from algobet.cli.dev_tools import my_command

def test_my_command():
    runner = CliRunner()
    result = runner.invoke(my_command)

    assert result.exit_code == 0
```

## Common Migration Patterns

### Pattern 1: Simple Database Query
**Before:**
```python
@click.command()
def list_tournaments():
    session = get_session()
    tournaments = session.query(Tournament).all()
    session.close()

    for t in tournaments:
        click.echo(t.name)
```

**After:**
```python
@click.command()
@handle_errors
def list_tournaments():
    query_service = ServiceLocator.get_query_service()
    tournaments = query_service.list_tournaments()

    for t in tournaments:
        click.echo(t.name)
```

### Pattern 2: Async Operation
**Before:**
```python
@click.command()
def scrape_data():
    data = scraper.scrape_tournament(url)
    # Save to database
```

**After:**
```python
@click.command()
@click_async
async def scrape_data():
    async_query_service = ServiceLocator.get_async_query_service()
    async_db_service = ServiceLocator.get_async_database_service()

    data = await scraper.async_scrape_tournament(url)
    # Save using async service
    await async_db_service.save_tournament(data)
```

## Environment Variables

Update your `.env` file with the new configuration options:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/algobet

# Application
DEBUG=true
LOG_LEVEL=INFO

# API
API_BASE_URL=http://localhost:8000
API_TIMEOUT=30

# Scraping
SCRAPER_DELAY=1
SCRAPER_MAX_RETRIES=3
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to use the DI container instead of direct imports
2. **Session Management**: Let the DI container handle session lifecycle
3. **Missing Error Handling**: Always wrap commands with `@handle_errors`
4. **Logging Issues**: Use structured logging instead of print statements
5. **Async/Sync Mixups**: Be consistent with async/sync patterns in a single function

### Debugging Tips

1. Enable debug mode: `algobet --debug command`
2. Check logs for detailed error information
3. Use the container's wire method to verify dependencies are available
4. Look at existing command implementations for reference

## Performance Considerations

- Use async services for I/O-bound operations
- Leverage connection pooling for database operations
- Consider caching for expensive computations
- Monitor memory usage in long-running operations
- Profile performance-critical paths

## Next Steps

1. Update your existing CLI commands following this guide
2. Write unit tests for your services
3. Add integration tests for your command flows
4. Review and optimize performance of critical paths
5. Document any custom services you create
