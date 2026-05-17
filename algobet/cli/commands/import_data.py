"""CLI commands for importing football data and enriching with stats.

Usage:
    # Import from Football-Data.co.uk (25+ years of historical data)
    algobet import-data fd-import "ENG-Premier League" --season 2024
    algobet import-data fd-top5-2025

    # Enrich existing matches with Understat/ESPN stats
    algobet import-data enrich "ENG-Premier League" --season 2024
    algobet import-data enrich-understat "ENG-Premier League" --season 2024
    algobet import-data enrich-players "ENG-Premier League" --season 2024
    # Enrich with FBref player stats (bypasses Cloudflare-blocked soccerdata)
    algobet import-data fbref-players "ENG-Premier League" --season 2020-2021
"""

from __future__ import annotations

from typing import Any

import click

from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import info, success
from algobet.importers.fd_importer import FDImporter
from algobet.importers.soccerdata_importer import SoccerDataImporter
from algobet.infrastructure.database import session_scope


@click.group(name="import-data")
def import_cli() -> None:
    """Import football data and enrich with stats.

    Commands for importing historical data and enriching matches.

    \b
    Examples:
        algobet import-data fd-import "ENG-Premier League" --season 2024
        algobet import-data fd-top5-2025
    """
    pass


# ---------------------------------------------------------------------------
# Football-Data.co.uk import commands
# ---------------------------------------------------------------------------


@import_cli.command(name="fd-import")
@click.argument("league")
@click.option("--season", required=True, help="Season (e.g., '2024' for 2024/25)")
@click.option("--no-stats", is_flag=True, help="Skip match statistics")
@click.option("--no-odds", is_flag=True, help="Skip betting odds")
@handle_errors
def fd_import(league: str, season: str, no_stats: bool, no_odds: bool) -> None:
    """Import historical data from Football-Data.co.uk.

    LEAGUE is the soccerdata league ID (e.g., 'ENG-Premier League').
    Provides 25+ years of results, stats, and odds without CAPTCHA issues.

    \b
    Example:
        algobet import-data fd-import "ENG-Premier League" --season 2024
    """
    info(f"Importing {league} season {season} from Football-Data.co.uk...")

    with session_scope() as session:
        importer = FDImporter(session)
        result = importer.import_season(
            league,
            season,
            include_stats=not no_stats,
            include_odds=not no_odds,
        )

    success(result.message)


@import_cli.command(name="fd-top5-2025")
@click.option("--no-stats", is_flag=True, help="Skip match statistics")
@click.option("--no-odds", is_flag=True, help="Skip betting odds")
@handle_errors
def fd_top5_2025(no_stats: bool, no_odds: bool) -> None:
    """Import 2024/25 season for top 5 European leagues.

    Imports Premier League, La Liga, Ligue 1, Bundesliga, and Serie A.

    \b
    Example:
        algobet import-data fd-top5-2025
    """
    info("Importing 2024/25 season for top 5 European leagues...")

    with session_scope() as session:
        importer = FDImporter(session)
        results = importer.import_top_5_leagues_2024_25()

    total_matches = sum(r.progress.matches_created for r in results if r.success)
    success(f"Imported {total_matches} matches across {len(results)} leagues")


@import_cli.command(name="fd-top5-range")
@click.argument("start_season")
@click.argument("end_season")
@click.option("--no-stats", is_flag=True, help="Skip match statistics")
@click.option("--no-odds", is_flag=True, help="Skip betting odds")
@handle_errors
def fd_top5_range(
    start_season: str, end_season: str, no_stats: bool, no_odds: bool
) -> None:
    """Import top 5 leagues for a season range.

    START_SEASON and END_SEASON are the starting years (e.g., 2012 2023
    imports 2012/13 through 2023/24).

    \b
    Example:
        algobet import-data fd-top5-range 2012 2023
        algobet import-data fd-top5-range 2015 2020 --no-stats
    """
    info(
        f"Importing top 5 leagues from {start_season}/{int(start_season) + 1} "
        f"to {end_season}/{int(end_season) + 1}..."
    )

    with session_scope() as session:
        importer = FDImporter(session)
        results = importer.import_top_5_leagues_range(
            start_season,
            end_season,
            include_stats=not no_stats,
            include_odds=not no_odds,
        )

    total_matches = sum(r.progress.matches_created for r in results if r.success)
    total_updated = sum(r.progress.matches_updated for r in results if r.success)
    success(
        f"Imported {total_matches} matches, updated {total_updated} "
        f"across {len(results)} league-seasons"
    )


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


# ---------------------------------------------------------------------------
# FBref Playwright scraper commands (bypasses soccerdata Cloudflare blocks)
# ---------------------------------------------------------------------------


@import_cli.command(name="fbref-players")
@click.argument("league")
@click.option("--season", required=True, help="Season (e.g., '2020-2021')")
@click.option(
    "--max-matches",
    default=None,
    type=int,
    help="Max matches to scrape (default: all)",
)
@click.option(
    "--start-from", default=0, type=int, help="Start from Nth match (0-based)"
)
@click.option(
    "--headless/--no-headless",
    default=False,
    help="Run browser headless (default: visible for CAPTCHA)",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    help="Skip matches with existing FBref stats",
)
@click.option(
    "--storage-state",
    default=None,
    type=click.Path(),
    help="Path to persist browser state (solves CAPTCHA once)",
)
@handle_errors
def fbref_players(
    league: str,
    season: str,
    max_matches: int | None,
    start_from: int,
    headless: bool,
    skip_existing: bool,
    storage_state: str | None,
) -> None:
    """Enrich with per-player match stats scraped from FBref.

    Uses a Playwright browser with stealth patches to scrape FBref
    directly, bypassing soccerdata's Cloudflare-blocked API.

    When headless=False (default) the browser window is visible so you
    can solve any CAPTCHA manually.  Browser state (cookies, local
    storage) is persisted so you only need to solve the CAPTCHA once.

    LEAGUE is the soccerdata league ID (e.g., 'ENG-Premier League').

    \\b
    Example:
        algobet import-data fbref-players "ENG-Premier League" --season 2020-2021
        algobet import-data fbref-players "ENG-Premier League" \\
            --season 2020-2021 --max-matches 10
    """
    from algobet.fbref_importer import FBrefImporter
    from algobet.fbref_scraper import FBrefScraper

    info(f"Scraping FBref player stats for {league} {season}...")

    scraper_kwargs: dict[str, Any] = {"headless": headless}
    if storage_state:
        scraper_kwargs["storage_state_file"] = storage_state

    with FBrefScraper(**scraper_kwargs) as scraper, session_scope() as session:
        importer = FBrefImporter(session, scraper)
        result = importer.import_player_stats(
            league=league,
            season=season,
            max_matches=max_matches,
            skip_existing=skip_existing,
            start_from_match=start_from,
        )

    success(
        f"FBref import: {result['players_added']} players added in "
        f"{result['matches_processed']} matches "
        f"({result['matches_skipped']} skipped)"
    )
