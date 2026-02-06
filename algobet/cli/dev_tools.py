"""Development tools for AlgoBet."""

from __future__ import annotations

import click
from sqlalchemy import func

from algobet.database import session_scope
from algobet.models import (
    Base,
    Match,
    ModelVersion,
    Prediction,
    Season,
    Team,
    Tournament,
)


@click.group()
def cli() -> None:
    """AlgoBet Development Tools."""
    pass


@cli.command()
def init_db() -> None:
    """Initialize the database with all tables."""
    from algobet.database import create_db_engine

    click.echo("Creating database tables...")
    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)
    click.echo("✓ Database initialized successfully")


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
def reset_db() -> None:
    """Reset the database by dropping and recreating all tables."""
    from algobet.database import create_db_engine

    click.echo("Dropping all tables...")
    engine = create_db_engine()
    Base.metadata.drop_all(bind=engine)
    click.echo("Creating tables...")
    Base.metadata.create_all(bind=engine)
    click.echo("✓ Database reset successfully")


@cli.command()
def db_stats() -> None:
    """Display database statistics."""
    with session_scope() as session:
        stats = {
            "Tournaments": session.query(func.count(Tournament.id)).scalar(),
            "Seasons": session.query(func.count(Season.id)).scalar(),
            "Teams": session.query(func.count(Team.id)).scalar(),
            "Matches": session.query(func.count(Match.id)).scalar(),
            "Predictions": session.query(func.count(Prediction.id)).scalar(),
            "Model Versions": session.query(func.count(ModelVersion.id)).scalar(),
        }

        click.echo("\n" + "=" * 40)
        click.echo("AlgoBet Database Statistics")
        click.echo("=" * 40)
        for name, count in stats.items():
            click.echo(f"{name:20s}: {count:,}")
        click.echo("=" * 40 + "\n")


@cli.command()
@click.option("--tournament", help="Filter by tournament name")
def list_tournaments(tournament: str | None = None) -> None:
    """List all tournaments in the database."""
    with session_scope() as session:
        query = session.query(Tournament)
        if tournament:
            query = query.filter(Tournament.name.ilike(f"%{tournament}%"))

        tournaments = query.order_by(Tournament.country, Tournament.name).all()

        if not tournaments:
            click.echo("No tournaments found.")
            return

        click.echo(f"\nFound {len(tournaments)} tournament(s):\n")
        for t in tournaments:
            click.echo(f"  {t.country}: {t.name} (slug: {t.url_slug})")
            if t.seasons:
                click.echo(f"    Seasons: {', '.join(s.name for s in t.seasons)}")
        click.echo()


@cli.command()
@click.option("--team", help="Filter by team name")
def list_teams(team: str | None = None) -> None:
    """List all teams in the database."""
    with session_scope() as session:
        query = session.query(Team)
        if team:
            query = query.filter(Team.name.ilike(f"%{team}%"))

        teams = query.order_by(Team.name).all()

        if not teams:
            click.echo("No teams found.")
            return

        click.echo(f"\nFound {len(teams)} team(s):\n")
        for t in teams:
            click.echo(f"  {t.name}")
            home_matches = (
                session.query(func.count(Match.id))
                .filter(Match.home_team_id == t.id)
                .scalar()
            )
            away_matches = (
                session.query(func.count(Match.id))
                .filter(Match.away_team_id == t.id)
                .scalar()
            )
            click.echo(
                f"    Home matches: {home_matches}, Away matches: {away_matches}"
            )
        click.echo()


@cli.command()
@click.option("--days", type=int, default=7, help="Number of days ahead")
def upcoming_matches(days: int) -> None:
    """List upcoming matches within the specified number of days."""
    from datetime import datetime, timedelta

    with session_scope() as session:
        max_date = datetime.now() + timedelta(days=days)
        matches = (
            session.query(Match)
            .filter(Match.status == "SCHEDULED")
            .filter(Match.match_date <= max_date)
            .order_by(Match.match_date)
            .all()
        )

        if not matches:
            click.echo(f"No upcoming matches in the next {days} days.")
            return

        click.echo(f"\nUpcoming matches (next {days} days):\n")
        for m in matches:
            click.echo(f"  {m.match_date.strftime('%Y-%m-%d %H:%M')}")
            click.echo(f"    {m.home_team.name} vs {m.away_team.name}")
            if m.tournament:
                click.echo(f"    {m.tournament.name}")
            if m.odds_home:
                click.echo(
                    f"    Odds: {m.odds_home:.2f} - "
                    f"{m.odds_draw:.2f} - {m.odds_away:.2f}"
                )
            click.echo()


@cli.command()
@click.argument("model_id", type=int)
def delete_model(model_id: int) -> None:
    """Delete a model version from the database."""
    from pathlib import Path

    with session_scope() as session:
        model = session.query(ModelVersion).filter(ModelVersion.id == model_id).first()
        if not model:
            click.echo(f"Model version with ID {model_id} not found.")
            return

        # Delete model file
        models_path = Path("data/models")
        model_file = models_path / f"{model.version}.pkl"
        if model_file.exists():
            model_file.unlink()
            click.echo(f"✓ Deleted model file: {model_file}")

        # Delete database record
        session.delete(model)
        click.echo(f"✓ Deleted model version: {model.version}")


if __name__ == "__main__":
    cli()
