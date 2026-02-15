"""Async database management CLI commands.

This module demonstrates the async pattern for CLI commands using
the @click_async decorator and async services.
"""

from __future__ import annotations

import click

from algobet.cli.async_runner import click_async
from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import info, success
from algobet.database import async_session_scope
from algobet.services import AsyncDatabaseService
from algobet.services.dto import DatabaseStatsRequest


@click.group(name="async-db")
def async_db_cli() -> None:
    """Async database management commands."""
    pass


@async_db_cli.command(name="init")
@click.option("--drop-existing", is_flag=True, help="Drop existing tables first")
@handle_errors
@click_async
async def init_db(drop_existing: bool) -> None:
    """Initialize the database with all tables (async version)."""
    info("Creating database tables...")

    from algobet.database import create_async_db_engine
    from algobet.models import Base

    engine = create_async_db_engine()
    async with engine.begin() as conn:
        if drop_existing:
            info("Dropping existing tables...")
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    success("Database initialized successfully")


@async_db_cli.command(name="reset")
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
@handle_errors
@click_async
async def reset_db() -> None:
    """Reset the database by dropping and recreating all tables (async version)."""
    info("Dropping all tables...")

    from algobet.database import create_async_db_engine
    from algobet.models import Base

    engine = create_async_db_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        info("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    success("Database reset successfully")


@async_db_cli.command(name="stats")
@handle_errors
@click_async
async def db_stats() -> None:
    """Display database statistics (async version)."""
    async with async_session_scope() as session:
        service = AsyncDatabaseService(session)
        response = await service.get_stats(DatabaseStatsRequest())

        click.echo("\n" + "=" * 40)
        click.echo("AlgoBet Database Statistics (Async)")
        click.echo("=" * 40)
        click.echo(f"{'Tournaments':20s}: {response.tournaments_count:,}")
        click.echo(f"{'Seasons':20s}: {response.seasons_count:,}")
        click.echo(f"{'Teams':20s}: {response.teams_count:,}")
        click.echo(f"{'Matches':20s}: {response.matches_count:,}")
        click.echo(f"{'  - Finished':20s}: {response.finished_matches_count:,}")
        click.echo(f"{'  - Scheduled':20s}: {response.scheduled_matches_count:,}")
        click.echo(f"{'Model Versions':20s}: {response.model_versions_count:,}")
        click.echo(f"{'Scheduled Tasks':20s}: {response.scheduled_tasks_count:,}")
        click.echo("=" * 40 + "\n")
