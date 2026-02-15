"""Query/listing CLI commands."""

from __future__ import annotations

from datetime import datetime, timedelta

import click

from algobet.cli.error_handler import handle_errors
from algobet.database import session_scope
from algobet.exceptions import DataNotFoundError
from algobet.services import QueryService
from algobet.services.dto import MatchFilter, TeamFilter, TournamentFilter


@click.group(name="list")
def list_cli() -> None:
    """List/query commands for viewing data."""
    pass


@list_cli.command(name="tournaments")
@click.option("--filter", "filter_name", help="Filter by tournament name")
@handle_errors
def list_tournaments(filter_name: str | None = None) -> None:
    """List all tournaments in the database."""
    with session_scope() as session:
        service = QueryService(session)
        request = TournamentFilter(name=filter_name)
        response = service.list_tournaments(request)

        if not response.tournaments:
            raise DataNotFoundError(
                "No tournaments found.",
                details={"filter": filter_name} if filter_name else None,
            )

        click.echo(f"\nFound {len(response.tournaments)} tournament(s):\n")
        for t in response.tournaments:
            click.echo(f"  {t.name} (slug: {t.url_slug})")
            click.echo(f"    Seasons: {t.seasons_count}")
        click.echo()


@list_cli.command(name="teams")
@click.option("--filter", "filter_name", help="Filter by team name")
@handle_errors
def list_teams(filter_name: str | None = None) -> None:
    """List all teams in the database."""
    with session_scope() as session:
        service = QueryService(session)
        request = TeamFilter(name=filter_name)
        response = service.list_teams(request)

        if not response.teams:
            raise DataNotFoundError(
                "No teams found.",
                details={"filter": filter_name} if filter_name else None,
            )

        click.echo(f"\nFound {len(response.teams)} team(s):\n")
        for t in response.teams:
            click.echo(f"  {t.name}")
            click.echo(f"    Matches played: {t.matches_played}")
        click.echo()


@list_cli.command(name="upcoming")
@click.option("--days", type=int, default=7, help="Number of days ahead")
@handle_errors
def list_upcoming_matches(days: int) -> None:
    """List upcoming matches within the specified number of days."""
    with session_scope() as session:
        service = QueryService(session)

        # Use match filter with SCHEDULED status
        # Note: Date filtering is done in CLI layer since
        # MatchFilter doesn't support date ranges
        request = MatchFilter(status="SCHEDULED", limit=100)
        response = service.list_matches(request)

        # Filter by date range in CLI layer
        now = datetime.now()
        max_date = now + timedelta(days=days)

        upcoming_matches = [
            m
            for m in response.matches
            if m.match_date and now <= m.match_date <= max_date
        ]

        if not upcoming_matches:
            raise DataNotFoundError(
                f"No upcoming matches in the next {days} days.",
                details={"days": days},
            )

        click.echo(f"\nUpcoming matches (next {days} days):\n")
        for m in upcoming_matches:
            date_str = (
                m.match_date.strftime("%Y-%m-%d %H:%M")
                if m.match_date
                else "TBD"
            )
            click.echo(f"  {date_str}")
            click.echo(f"    {m.home_team} vs {m.away_team}")
            click.echo(f"    {m.tournament_name}")
            click.echo()
