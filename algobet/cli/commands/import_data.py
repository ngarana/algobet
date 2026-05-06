"""CLI commands for importing football data and enriching with stats.

Usage:
    # Enrich existing matches with Understat/ESPN stats
    algobet import-data enrich "ENG-Premier League" --season 2024
    algobet import-data enrich-understat "ENG-Premier League" --season 2024
    algobet import-data enrich-players "ENG-Premier League" --season 2024
"""

from __future__ import annotations

import click

from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import info, success
from algobet.importers.soccerdata_importer import SoccerDataImporter
from algobet.infrastructure.database import session_scope


@click.group(name="import-data")
def import_cli() -> None:
    """Import football data and enrich with soccerdata stats.

    Commands for enriching existing matches with xG and player stats.

    \\b
    Examples:
        algobet import-data enrich "ENG-Premier League" --season 2024
        algobet import-data enrich-understat "ENG-Premier League" --season 2024
        algobet import-data enrich-players "ENG-Premier League" --season 2024
    """
    pass


# ---------------------------------------------------------------------------
# Soccerdata enrichment commands (Understat + ESPN)
# ---------------------------------------------------------------------------


@import_cli.command(name="enrich")
@click.argument("league")
@click.option("--season", required=True, help="Season (e.g., '2024')")
@handle_errors
def enrich(league: str, season: str) -> None:
    """Enrich matches with xG and player stats via soccerdata.

    LEAGUE is the soccerdata league ID (e.g., 'ENG-Premier League').

    \\b
    Example:
        algobet import-data enrich "ENG-Premier League" --season 2024
    """
    info(f"Enriching {league} season {season}...")

    with session_scope() as session:
        importer = SoccerDataImporter(session)
        result = importer.enrich_all(league=league, season=season)

    success(
        f"Enriched {result['understat_enriched']} matches and "
        f"{result['players_added']} players in {result['matches_processed']} matches"
    )


@import_cli.command(name="enrich-understat")
@click.argument("league")
@click.option("--season", required=True, help="Season (e.g., '2024')")
@handle_errors
def enrich_understat(league: str, season: str) -> None:
    """Enrich with Understat xG, npxG, PPDA, deep completions.

    \\b
    Example:
        algobet import-data enrich-understat "ENG-Premier League" --season 2024
    """
    info(f"Enriching {league} {season} with Understat metrics...")

    with session_scope() as session:
        importer = SoccerDataImporter(session)
        count = importer.enrich_understat_stats(league=league, season=season)

    success(f"Enriched {count} matches with Understat advanced metrics")


@import_cli.command(name="enrich-players")
@click.argument("league")
@click.option("--season", required=True, help="Season (e.g., '2024')")
@handle_errors
def enrich_players(league: str, season: str) -> None:
    """Enrich with ESPN per-player match stats.

    \\b
    Example:
        algobet import-data enrich-players "ENG-Premier League" --season 2024
    """
    info(f"Enriching {league} {season} with player stats...")

    with session_scope() as session:
        importer = SoccerDataImporter(session)
        result = importer.enrich_player_stats(league=league, season=season)

    success(
        f"Added {result['players_added']} player records "
        f"in {result['matches_processed']} matches"
    )
