"""Async query/listing CLI commands.

This module demonstrates the async pattern for CLI commands using
the @click_async decorator and async services.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import click

from algobet.cli.async_runner import click_async
from algobet.cli.error_handler import handle_errors
from algobet.database import async_session_scope
from algobet.exceptions import DataNotFoundError
from algobet.services import AsyncQueryService
from algobet.services.dto import MatchFilter, TeamFilter, TournamentFilter


@click.group(name="async-list")
def async_list_cli() -> None:
    """Async list/query commands for viewing data."""
    pass


@async_list_cli.command(name="tournaments")
@click.option("--filter", "filter_name", help="Filter by tournament name")
@handle_errors
@click_async
async def list_tournaments(filter_name: str | None = None) -> None:
    """List all tournaments in the database (async version)."""
    async with async_session_scope() as session:
        service = AsyncQueryService(session)
        request = TournamentFilter(name=filter_name)
        response = await service.list_tournaments(request)

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


@async_list_cli.command(name="teams")
@click.option("--filter", "filter_name", help="Filter by team name")
@handle_errors
@click_async
async def list_teams(filter_name: str | None = None) -> None:
    """List all teams in the database (async version)."""
    async with async_session_scope() as session:
        service = AsyncQueryService(session)
        request = TeamFilter(name=filter_name)
        response = await service.list_teams(request)

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


@async_list_cli.command(name="upcoming")
@click.option("--days", type=int, default=7, help="Number of days ahead")
@handle_errors
@click_async
async def list_upcoming_matches(days: int) -> None:
    """List upcoming matches within the specified number of days (async version)."""
    async with async_session_scope() as session:
        service = AsyncQueryService(session)

        # Use match filter with SCHEDULED status
        request = MatchFilter(status="SCHEDULED", limit=100)
        response = await service.list_matches(request)

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
                m.match_date.strftime("%Y-%m-%d %H:%M") if m.match_date else "TBD"
            )
            click.echo(f"  {date_str}")
            click.echo(f"    {m.home_team} vs {m.away_team}")
            click.echo(f"    {m.tournament_name}")
            click.echo()


@async_list_cli.command(name="matches")
@click.option(
    "--status",
    type=click.Choice(["SCHEDULED", "FINISHED", "LIVE"]),
    help="Filter by match status",
)
@click.option("--team", "team_name", help="Filter by team name")
@click.option("--limit", type=int, default=50, help="Maximum number of matches")
@handle_errors
@click_async
async def list_matches(status: str | None, team_name: str | None, limit: int) -> None:
    """List matches with optional filters (async version)."""
    async with async_session_scope() as session:
        service = AsyncQueryService(session)

        request = MatchFilter(
            status=status,
            team_name=team_name,
            limit=limit,
        )
        response = await service.list_matches(request)

        if not response.matches:
            raise DataNotFoundError(
                "No matches found with the specified filters.",
                details={"status": status, "team": team_name},
            )

        click.echo(
            f"\nFound {len(response.matches)} match(es) "
            f"(total: {response.total_count}):\n"
        )
        for m in response.matches:
            date_str = (
                m.match_date.strftime("%Y-%m-%d %H:%M") if m.match_date else "TBD"
            )
            score = (
                f" ({m.home_score} - {m.away_score})"
                if m.home_score is not None and m.away_score is not None
                else ""
            )
            click.echo(f"  {date_str}{score}")
            click.echo(f"    {m.home_team} vs {m.away_team}")
            click.echo(f"    {m.tournament_name} - {m.season_name}")
            click.echo(f"    Status: {m.status}")
            click.echo()
