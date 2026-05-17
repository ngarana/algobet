"""Database management CLI commands."""

from __future__ import annotations

import click

from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import info, success
from algobet.infrastructure.database import session_scope
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
    from algobet.infrastructure.database import create_db_engine
    from algobet.infrastructure.models import Base

    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)
    success("Database initialized successfully")


@db_cli.command(name="reset")
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
@handle_errors
def reset_db() -> None:
    """Reset the database by dropping and recreating all tables."""
    info("Dropping all tables...")

    from algobet.infrastructure.database import create_db_engine
    from algobet.infrastructure.models import Base

    engine = create_db_engine()
    Base.metadata.drop_all(bind=engine)
    info("Creating tables...")
    Base.metadata.create_all(bind=engine)
    success("Database reset successfully")


@db_cli.command(name="migrate")
@click.option("--message", "-m", help="Migration message")
@handle_errors
def migrate(message: str | None) -> None:
    """Create a new Alembic migration from model changes."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "alembic", "revision", "--autogenerate"]
    if message:
        cmd.extend(["-m", message])
    info("Generating migration...")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        success("Migration created successfully")


@db_cli.command(name="upgrade")
@click.argument("revision", default="head")
@handle_errors
def upgrade(revision: str) -> None:
    """Run Alembic migrations (default: head)."""
    import subprocess
    import sys

    info(f"Running migrations up to: {revision}")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", revision],
        check=False,
    )
    if result.returncode == 0:
        success("Migrations applied successfully")


@db_cli.command(name="downgrade")
@click.argument("revision", default="-1")
@handle_errors
def downgrade(revision: str) -> None:
    """Downgrade Alembic migrations."""
    import subprocess
    import sys

    info(f"Downgrading to: {revision}")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", revision],
        check=False,
    )
    if result.returncode == 0:
        success("Downgrade completed")


@db_cli.command(name="current")
@handle_errors
def current() -> None:
    """Show current Alembic migration version."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "alembic", "current"],
        check=False,
    )


@db_cli.command(name="history")
@handle_errors
def history() -> None:
    """Show Alembic migration history."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "alembic", "history"],
        check=False,
    )


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


@db_cli.command(name="repair-bundesliga-country")
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    help="Apply the repair. Without this flag the command only reports findings.",
)
@handle_errors
def repair_bundesliga_country(apply_changes: bool) -> None:
    """Repair German Bundesliga rows imported into a non-Germany tournament."""
    from sqlalchemy import func, or_, select
    from sqlalchemy.orm import aliased

    from algobet.models import Match, Team, Tournament

    german_markers = [
        "Bayern",
        "Dortmund",
        "Leverkusen",
        "Schalke",
        "Hoffenheim",
        "RB Leipzig",
        "Werder Bremen",
        "Wolfsburg",
        "Freiburg",
        "Mainz",
        "Augsburg",
        "Stuttgart",
    ]

    with session_scope() as session:
        tournament = session.execute(
            select(Tournament).where(Tournament.url_slug == "bundesliga")
        ).scalar_one_or_none()
        if tournament is None:
            info("No tournament with url_slug='bundesliga' was found.")
            return

        home_team = aliased(Team)
        away_team = aliased(Team)
        marker_conditions = []
        for marker in german_markers:
            pattern = f"%{marker}%"
            marker_conditions.append(home_team.name.ilike(pattern))
            marker_conditions.append(away_team.name.ilike(pattern))

        total_matches = session.execute(
            select(func.count(Match.id)).where(Match.tournament_id == tournament.id)
        ).scalar_one()
        german_marker_matches = session.execute(
            select(func.count(Match.id))
            .join(home_team, Match.home_team_id == home_team.id)
            .join(away_team, Match.away_team_id == away_team.id)
            .where(
                Match.tournament_id == tournament.id,
                or_(*marker_conditions),
            )
        ).scalar_one()

        click.echo(
            f"bundesliga tournament id={tournament.id}, "
            f"name={tournament.name}, country={tournament.country}, "
            f"matches={total_matches}, german_marker_matches={german_marker_matches}"
        )

        if tournament.country == "Germany":
            success("bundesliga tournament is already marked as Germany.")
            return

        if german_marker_matches == 0:
            info(
                "No German team markers were found, so the command will not "
                "change the tournament country."
            )
            return

        if not apply_changes:
            info("Dry run only. Re-run with --apply to set country='Germany'.")
            return

        tournament.name = "Bundesliga"
        tournament.country = "Germany"
        session.add(tournament)
        success(
            f"Updated tournament id={tournament.id} country to Germany "
            f"based on {german_marker_matches} German-marker match(es)."
        )
