# AlgoBet CLI Architecture Documentation

## Overview

The AlgoBet CLI has been refactored into a production-ready system with proper architecture, following SOLID principles and modern Python practices. The system now includes configuration management, structured logging, comprehensive error handling, service layers, dependency injection, and async support.

## Architecture Components

### 1. Configuration System
Located in `algobet/config.py`, the configuration system uses Pydantic Settings for type-safe, validated configuration management.

**Key Features:**
- Nested configuration classes for different domains
- Environment variable support
- Validation and type checking
- Multiple configuration profiles (development, production)

**Example Usage:**
```python
from algobet.config import get_config

config = get_config()
db_url = config.database.url
debug_mode = config.app.debug
```

### 2. Logging System
The logging system in `algobet/logging_config.py` and `algobet/cli/logger.py` provides structured logging with multiple output formats.

**Features:**
- JSON and text output formats
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Integration with Click CLI commands
- Support for structured logging with context

**Example Usage:**
```python
from algobet.cli.logger import get_logger

logger = get_logger(__name__)
logger.info("Database initialized", extra={"table_count": 5})
```

### 3. Error Handling Framework
Located in `algobet/exceptions.py` and `algobet/cli/error_handler.py`, the error handling system provides a comprehensive exception hierarchy and centralized error handling.

**Exception Hierarchy:**
- `AlgoBetException` (base)
  - `ConfigError`
  - `DatabaseError`
    - `DatabaseConnectionError`
    - `DatabaseIntegrityError`
  - `ValidationError`
  - `ScrapingError`
  - `PredictionError`
  - `ServiceError`

**Error Handler:**
- Global error decorator `@handle_errors`
- Maps exceptions to appropriate exit codes
- Provides user-friendly error messages
- Supports debug mode with full stack traces

### 4. Service Layer
The service layer in `algobet/services/` extracts business logic from CLI commands into testable, reusable services.

**Service Types:**
- `DatabaseService` / `AsyncDatabaseService` - Database operations
- `QueryService` / `AsyncQueryService` - Data querying and filtering
- `ModelManagementService` / `AsyncModelManagementService` - Model management
- `AnalysisService` - Analytics and predictions
- `BaseService` / `AsyncBaseService` - Base service classes

**Example Service Usage:**
```python
from algobet.services.database_service import DatabaseService

service = DatabaseService(session)
stats = service.get_stats()
```

### 5. Dependency Injection Container
The DI container in `algobet/cli/container.py` manages service dependencies and lifecycles.

**Features:**
- Automatic service instantiation
- Session management
- Provider pattern for lazy loading
- Wiring for module integration

**Container Usage:**
```python
from algobet.cli.container import Container

container = Container()
db_service = container.database_service()
```

### 6. Async Support
Async support is implemented through:
- `algobet/cli/async_runner.py` - Async CLI execution
- Async versions of all services
- `@click_async` decorator for async commands
- Async database operations

**Async Example:**
```python
from algobet.cli.async_runner import click_async
import click

@click.command()
@click_async
async def my_async_command():
    # Async operations here
    pass
```

## CLI Command Structure

The CLI follows a hierarchical command structure:

```
algobet
├── db                 # Database management
│   ├── init           # Initialize database
│   ├── reset          # Reset database
│   └── stats          # Show database statistics
├── list               # Data listing
│   ├── tournaments    # List tournaments
│   ├── teams          # List teams
│   └── upcoming       # List upcoming matches
├── model              # Model management
│   ├── list           # List models
│   └── delete         # Delete model
├── analyze            # Analysis commands
│   ├── backtest       # Backtesting
│   ├── value-bets     # Value bet identification
│   └── calibrate      # Model calibration
├── async-db           # Async database commands
├── async-list         # Async listing commands
└── --debug, --verbose # Global options
```

## Development Best Practices

### Creating New Commands
1. Add to appropriate command file in `algobet/cli/commands/`
2. Use dependency injection for services
3. Handle errors with `@handle_errors` decorator
4. Use logging instead of print statements
5. Follow async patterns if I/O bound

### Testing
- Unit tests for services in `tests/unit/services/`
- Integration tests for DI container in `tests/integration/`
- CLI command tests in `tests/unit/cli/`
- Performance benchmarks in `tests/performance/`

### Error Handling
Always use the error handling framework:
```python
from algobet.cli.error_handler import handle_errors

@handle_errors
def my_command():
    # Command logic here
    pass
```

### Logging
Use structured logging:
```python
from algobet.cli.logger import get_logger

logger = get_logger(__name__)
logger.info("Operation completed", extra={"duration": 1.2})
```

## Performance Considerations

- Use async services for I/O-bound operations
- Leverage connection pooling for database operations
- Cache expensive computations when appropriate
- Monitor memory usage in long-running operations

## Migration Guide

For developers migrating to the new architecture:

1. **Configuration**: Access config through `get_config()` instead of environment variables directly
2. **Database**: Use services instead of direct SQLAlchemy queries in CLI commands
3. **Logging**: Replace `print()` with structured logging
4. **Errors**: Use custom exceptions and the error handler decorator
5. **Services**: Depend on services via DI container instead of importing modules directly
6. **Async**: Use async services and the `@click_async` decorator for async operations