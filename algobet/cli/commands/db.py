"""Database management CLI commands."""

from __future__ import annotations

import click

from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import info, success
from algobet.database import session_scope
from algobet.services import DatabaseService
from algobet.services.dto import (
    DatabaseStatsRequest,
)


@click.group(name="db")
def db_cli() -> None:
    """Database management commands."""
    pass


@db_cli.command(name="init")
@handle_errors
def init_db() -> None:
    """Initialize the database with all tables."""
    info("Creating database tables...")

    # DatabaseService.initialize doesn't need a session, it creates its own engine
    from algobet.database import create_db_engine
    from algobet.models import Base

    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)
    success("Database initialized successfully")


@db_cli.command(name="reset")
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
@handle_errors
def reset_db() -> None:
    """Reset the database by dropping and recreating all tables."""
    info("Dropping all tables...")

    from algobet.database import create_db_engine
    from algobet.models import Base

    engine = create_db_engine()
    Base.metadata.drop_all(bind=engine)
    info("Creating tables...")
    Base.metadata.create_all(bind=engine)
    success("Database reset successfully")


@db_cli.command(name="stats")
@handle_errors
def db_stats() -> None:
    """Display database statistics."""
    with session_scope() as session:
        service = DatabaseService(session)
        response = service.get_stats(DatabaseStatsRequest())

        click.echo("\n" + "=" * 40)
        click.echo("AlgoBet Database Statistics")
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
