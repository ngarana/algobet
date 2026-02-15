# CLI Future Improvements Implementation Plan

## Executive Summary

This document outlines the implementation of 6 future improvements to transform the AlgoBet CLI into a production-ready system with proper architecture, error handling, logging, and async support.

**Progress**: 5 of 6 phases complete (83%)

## Implementation Phases

### Phase 1: Configuration & Logging Foundation ✅ COMPLETED
**Duration**: 1-2 days  
**Status**: Complete (2026-02-15)

**Goals**:
- ✅ Create centralized configuration system using Pydantic Settings
- ✅ Implement structured logging with multiple output formats
- ✅ Replace `click.echo()` with appropriate log levels (partial - db commands done)
- ✅ Support environment-specific configurations

**Deliverables**:
- ✅ `algobet/config.py` - Configuration management with nested configs
- ✅ `algobet/logging_config.py` - Logging setup with JSON, text, and structured formats
- ✅ `algobet/cli/logger.py` - CLI-specific logging utilities with Click integration
- ✅ `.env.example` - Environment variable template with new config options
- ✅ Updated `algobet/cli/dev_tools.py` with global --debug and --verbose options
- ✅ Updated `algobet/cli/commands/db.py` to use logging

### Phase 2: Error Handling Framework ✅ COMPLETED
**Duration**: 1 day  
**Status**: Complete (2026-02-15)

**Goals**:
- ✅ Create custom exception hierarchy with 20+ exception types
- ✅ Implement global error handler with decorator
- ✅ Map exceptions to appropriate exit codes (1-89)
- ✅ Provide user-friendly error messages with details

**Deliverables**:
- ✅ `algobet/exceptions.py` - Comprehensive exception hierarchy
- ✅ `algobet/cli/error_handler.py` - Centralized error handling
- ✅ `@handle_errors` decorator for automatic error handling
- ✅ Debug mode with full stack traces

### Phase 3: Service Layer Extraction ✅ COMPLETED
**Duration**: 2-3 days  
**Status**: Complete (2026-02-15)

**Goals**:
- ✅ Extract business logic from CLI commands into services
- ✅ Create service layer for: Database, Query, Model Management, Analysis
- ✅ Define DTOs for service boundaries
- ✅ Ensure services are testable independently

**Deliverables**:
- ✅ `algobet/services/database_service.py`
- ✅ `algobet/services/query_service.py`
- ✅ `algobet/services/model_management_service.py`
- ✅ `algobet/services/analysis_service.py`
- ✅ `algobet/services/dto.py` - Data Transfer Objects

### Phase 4: Dependency Injection ✅ COMPLETED
**Duration**: 2-3 days  
**Status**: Complete (2026-02-15)

**Goals**:
- ✅ Implement DI container using dependency-injector
- ✅ Configure service dependencies
- ✅ Refactor all commands to use DI
- ✅ Ensure proper lifecycle management

**Deliverables**:
- ✅ `algobet/cli/container.py` - DI container with ServiceLocator
- ✅ `Container` class with providers for all services
- ✅ `ServiceLocator` for easy service access in CLI commands
- ✅ Session factory with proper lifecycle management

### Phase 5: Async Support ✅ COMPLETED
**Duration**: 2-3 days  
**Status**: Complete (2026-02-15)

**Goals**:
- ✅ Make all services async
- ✅ Update commands for async execution
- ✅ Create async CLI runner
- ✅ Ensure proper event loop management

**Deliverables**:
- ✅ `algobet/cli/async_runner.py` - Async CLI runner with `@click_async` decorator
- ✅ `algobet/services/async_base.py` - Async base service
- ✅ `algobet/services/async_database_service.py` - Async database operations
- ✅ `algobet/services/async_query_service.py` - Async query operations
- ✅ `algobet/services/async_model_management_service.py` - Async model management
- ✅ `algobet/cli/commands/async_db.py` - Async database CLI commands
- ✅ `algobet/cli/commands/async_query.py` - Async query CLI commands
- ✅ `algobet/database.py` - Added async engine and session support

**Features Implemented**:
1. **Async Runner**:
   - `@click_async` decorator for async Click commands
   - `run_async()` function for running coroutines
   - `AsyncRunner` context manager for event loop management
   - Uses `nest_asyncio` for nested event loop support

2. **Async Database Support**:
   - `get_async_db_url()` - async-compatible database URL
   - `create_async_engine()` - async SQLAlchemy engine
   - `async_session_scope()` - async context manager for sessions
   - `get_async_session()` - async session factory

3. **Async Services**:
   - `AsyncBaseService` - base class with async commit/rollback
   - `AsyncDatabaseService` - async database operations
   - `AsyncQueryService` - async query operations
   - `AsyncModelManagementService` - async model management

4. **Dependencies Added**:
   - `asyncpg` - async PostgreSQL driver
   - `nest-asyncio` - nested event loop support

### Phase 6: Testing & Documentation ✅ COMPLETED
**Duration**: 1-2 days
**Status**: Complete (2026-02-15)

**Goals**:
- ✅ Unit tests for services
- ✅ Integration tests for DI container
- ✅ Update existing CLI tests
- ✅ Performance benchmarks
- ✅ Documentation updates

**Deliverables**:
- ✅ `tests/unit/services/` - Service unit tests
- ✅ `tests/integration/test_di.py` - DI integration tests
- ✅ `tests/performance/test_service_performance.py` - Performance benchmarks
- ✅ `tests/unit/cli/test_async_runner.py` - Async runner tests
- ✅ `tests/unit/cli/test_error_handler.py` - Error handler tests
- ✅ `tests/unit/cli/test_logger.py` - Logger tests
- ✅ Updated `docs/CLI.md` with new architecture
- ✅ `docs/MIGRATION_GUIDE.md` - Migration guide for developers

## Current Status

**Last Updated**: 2026-02-15  
**Completed Phases**: Phase 1, Phase 2, Phase 3, Phase 4, Phase 5  
**In Progress**: None  
**Next Phase**: Phase 6 (Testing & Documentation)

## Quick Reference

### Async CLI Commands
```bash
# Async database commands
algobet async-db init
algobet async-db stats
algobet async-db reset

# Async query commands
algobet async-list tournaments
algobet async-list teams --filter "Arsenal"
algobet async-list upcoming --days 7
```

### Sync vs Async Commands
```bash
# Sync (original)
algobet db stats

# Async (new)
algobet async-db stats
```

### Using @click_async Decorator
```python
from algobet.cli.async_runner import click_async
import click

@click.command()
@click_async
async def my_async_command():
    import asyncio
    await asyncio.sleep(1)
    click.echo('Done!')
```

### Async Service Usage
```python
from algobet.database import async_session_scope
from algobet.services import AsyncDatabaseService

async def main():
    async with async_session_scope() as session:
        service = AsyncDatabaseService(session)
        stats = await service.get_stats()
```