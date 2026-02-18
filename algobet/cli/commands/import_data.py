"""CLI commands for importing football data from Football-Data.co.uk.

This module provides commands for importing historical match data and
betting odds from Football-Data.co.uk CSV files.

Usage:
    # Import from local file
    algobet import-data file data.csv --season 2023/2024

    # Import from URL
    algobet import-data url https://football-data.co.uk/mmz4281/2324/E0.csv \
        --season 2023/2024

    # Import a season (auto-URL construction)
    algobet import-data season E0 --season 2023/2024  # Premier League 23/24

    # Import multiple historical seasons
    algobet import-data historical E0 --from 2020 --to 2023  # 20/21-23/24
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import error, info, success
from algobet.database import session_scope
from algobet.importers.football_data import (
    DIVISION_MAPPING,
    FootballDataImporter,
    ImportProgress,
)

# Division codes for help text
DIVISION_HELP = """Division code (league identifier). Common codes:
  E0  - Premier League (England)
  E1  - Championship (England)
  SP1 - La Liga (Spain)
  SP2 - La Liga 2 (Spain)
  D1  - Bundesliga (Germany)
  D2  - Bundesliga 2 (Germany)
  I1  - Serie A (Italy)
  I2  - Serie B (Italy)
  F1  - Ligue 1 (France)
  F2  - Ligue 2 (France)
"""


def format_progress(progress: ImportProgress) -> str:
    """Format import progress for display.

    Args:
        progress: Import progress data

    Returns:
        Formatted string for display
    """
    lines = [
        f"  Rows processed: {progress.processed_rows}/{progress.total_rows}",
        f"  Matches created: {progress.matches_created}",
        f"  Matches skipped (duplicates): {progress.matches_skipped}",
        f"  Teams created: {progress.teams_created}",
    ]
    if progress.errors:
        lines.append(f"  Errors: {len(progress.errors)}")
    return "\n".join(lines)


def create_progress_callback() -> Any:
    """Create a progress callback for the importer.

    Returns:
        Callback function that prints progress updates
    """

    def callback(progress: ImportProgress) -> None:
        if progress.total_rows > 0:
            pct = (progress.processed_rows / progress.total_rows) * 100
            processed = progress.processed_rows
            total = progress.total_rows
            msg = f"\r  Progress: {pct:.1f}% ({processed}/{total})"
            click.echo(msg, nl=False)

    return callback


def season_name_to_code(season_name: str) -> str:
    """Convert season name like '2023/2024' to code like '2324'.

    Args:
        season_name: Season name in YYYY/YYYY format

    Returns:
        4-digit season code
    """
    # Handle both "2023/2024" and "2023/24" formats
    parts = season_name.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid season format: {season_name}. Use format like '2023/2024'"
        )

    start_year = parts[0]
    # Take last 2 digits of each year
    return f"{start_year[-2:]}{parts[1][-2:]}"


def validate_division(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Validate division code.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Division code value

    Returns:
        Validated division code

    Raises:
        click.BadParameter: If division code is unknown
    """
    if value not in DIVISION_MAPPING:
        known_codes = ", ".join(sorted(DIVISION_MAPPING.keys()))
        raise click.BadParameter(
            f"Unknown division code: {value}.\nKnown codes: {known_codes}"
        )
    return value


@click.group(name="import-data")
def import_cli() -> None:
    """Import football data from Football-Data.co.uk.

    Commands for importing historical match data and betting odds from
    Football-Data.co.uk CSV files.

    \b
    Examples:
        algobet import-data file data.csv --season 2023/2024
        algobet import-data url https://www.football-data.co.uk/mmz4281/2324/E0.csv
        algobet import-data season E0 --season 2023/2024
        algobet import-data historical E0 --from 2020 --to 2023
    """
    pass


@import_cli.command(name="file")
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--season",
    "season_name",
    required=True,
    help="Season name in YYYY/YYYY format (e.g., '2023/2024')",
)
@click.option(
    "--division",
    callback=validate_division,
    help="Division code override (e.g., 'E0'). Inferred from CSV if not provided.",
)
@handle_errors
def import_file(file_path: Path, season_name: str, division: str | None) -> None:
    """Import match data from a local CSV file.

    FILE_PATH is the path to the CSV file to import.

    \b
    Example:
        algobet import-data file data.csv --season 2023/2024
    """
    info(f"Importing from file: {file_path}")

    season_code = season_name_to_code(season_name)
    info(f"Season: {season_name} (code: {season_code})")

    if division:
        info(f"Division: {division} ({DIVISION_MAPPING[division]['name']})")

    progress_callback = create_progress_callback()

    with session_scope() as session:
        importer = FootballDataImporter(session, progress_callback=progress_callback)
        result = importer.import_from_file(
            file_path=file_path,
            season_code=season_code,
            division=division,
        )

    click.echo()  # New line after progress

    if result.success:
        success(result.message)
        click.echo("\nImport Statistics:")
        click.echo(format_progress(result.progress))
    else:
        error(f"Import failed: {result.message}")
        if result.progress.errors:
            click.echo("\nErrors:")
            for err in result.progress.errors[:10]:  # Show first 10 errors
                click.echo(f"  - {err}")


@import_cli.command(name="url")
@click.argument("url")
@click.option(
    "--season",
    "season_name",
    required=True,
    help="Season name in YYYY/YYYY format (e.g., '2023/2024')",
)
@click.option(
    "--division",
    callback=validate_division,
    help="Division code override (e.g., 'E0'). Inferred from URL if not provided.",
)
@handle_errors
def import_url(url: str, season_name: str, division: str | None) -> None:
    """Import match data from a Football-Data.co.uk URL.

    URL is the full URL to the CSV file on Football-Data.co.uk.

    \b
    Example:
        algobet import-data url https://football-data.co.uk/mmz4281/2324/E0.csv \
            --season 2023/2024
    """
    info(f"Importing from URL: {url}")

    season_code = season_name_to_code(season_name)
    info(f"Season: {season_name} (code: {season_code})")

    if division:
        info(f"Division: {division} ({DIVISION_MAPPING[division]['name']})")

    progress_callback = create_progress_callback()

    with session_scope() as session:
        importer = FootballDataImporter(session, progress_callback=progress_callback)

        # Extract division from URL if not provided
        if not division:
            # URL format: .../mmz4281/2324/E0.csv
            parts = url.split("/")
            if len(parts) >= 2:
                potential_div = parts[-1].replace(".csv", "")
                if potential_div in DIVISION_MAPPING:
                    division = potential_div
                    info(f"Inferred division: {division}")

        if not division:
            error("Could not infer division from URL. Please specify --division.")
            return

        result = importer.import_from_url(
            season_code=season_code,
            divisions=[division],
        )

    click.echo()  # New line after progress

    if result.success:
        success(result.message)
        click.echo("\nImport Statistics:")
        click.echo(format_progress(result.progress))
    else:
        error(f"Import failed: {result.message}")
        if result.progress.errors:
            click.echo("\nErrors:")
            for err in result.progress.errors[:10]:
                click.echo(f"  - {err}")


@import_cli.command(name="season")
@click.argument("division", callback=validate_division)
@click.option(
    "--season",
    "season_name",
    required=True,
    help="Season name in YYYY/YYYY format (e.g., '2023/2024')",
)
@handle_errors
def import_season(division: str, season_name: str) -> None:
    """Import a full season for a specific league.

    DIVISION is the Football-Data.co.uk division code (e.g., E0 for Premier League).

    This command automatically constructs the URL and downloads the data
    from Football-Data.co.uk.

    \b
    Examples:
        algobet import-data season E0 --season 2023/2024   # Premier League 23/24
        algobet import-data season SP1 --season 2023/2024  # La Liga 23/24
        algobet import-data season D1 --season 2023/2024   # Bundesliga 23/24
    """
    tournament_name = DIVISION_MAPPING[division]["name"]
    season_code = season_name_to_code(season_name)

    info(f"Importing {tournament_name} season {season_name}")
    info(f"Division code: {division}, Season code: {season_code}")

    progress_callback = create_progress_callback()

    with session_scope() as session:
        importer = FootballDataImporter(session, progress_callback=progress_callback)
        result = importer.import_from_url(
            season_code=season_code,
            divisions=[division],
        )

    click.echo()  # New line after progress

    if result.success:
        success(result.message)
        click.echo("\nImport Statistics:")
        click.echo(format_progress(result.progress))
    else:
        error(f"Import failed: {result.message}")
        if result.progress.errors:
            click.echo("\nErrors:")
            for err in result.progress.errors[:10]:
                click.echo(f"  - {err}")


@import_cli.command(name="historical")
@click.argument("division", callback=validate_division)
@click.option(
    "--from",
    "from_year",
    type=int,
    required=True,
    help="Start year (e.g., 2020 for 2020/2021 season)",
)
@click.option(
    "--to",
    "to_year",
    type=int,
    required=True,
    help="End year (e.g., 2023 for 2023/2024 season)",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    default=False,
    help="Continue importing even if a season fails",
)
@handle_errors
def import_historical(
    division: str,
    from_year: int,
    to_year: int,
    continue_on_error: bool,
) -> None:
    """Import multiple historical seasons for a league.

    DIVISION is the Football-Data.co.uk division code (e.g., E0 for Premier League).

    This command imports data for multiple seasons, from the --from year
    to the --to year (inclusive).

    \b
    Examples:
        algobet import-data historical E0 --from 2020 --to 2023  # 20/21 through 23/24
        algobet import-data historical SP1 --from 2018 --to 2022  # 18/19 through 22/23
    """
    if from_year > to_year:
        error("--from year must be less than or equal to --to year")
        return

    tournament_name = DIVISION_MAPPING[division]["name"]

    # Generate season codes
    season_codes = []
    for year in range(from_year, to_year + 1):
        next_year = year + 1
        season_codes.append(f"{str(year)[-2:]}{str(next_year)[-2:]}")

    info(f"Importing {tournament_name} historical data")
    info(f"Seasons: {from_year}/{from_year + 1} through {to_year}/{to_year + 1}")
    info(f"Total seasons to import: {len(season_codes)}")

    results_summary: list[dict[str, str | int]] = []

    with session_scope() as session:
        importer = FootballDataImporter(session)

        for i, season_code in enumerate(season_codes, 1):
            season_name = f"20{season_code[:2]}/20{season_code[2:]}"
            click.echo(f"\n[{i}/{len(season_codes)}] Importing season {season_name}...")

            try:
                result = importer.import_from_url(
                    season_code=season_code,
                    divisions=[division],
                )

                if result.success:
                    success(f"  {result.message}")
                    results_summary.append(
                        {
                            "season": season_name,
                            "status": "success",
                            "matches": result.progress.matches_created,
                            "skipped": result.progress.matches_skipped,
                        }
                    )
                else:
                    error(f"  Failed: {result.message}")
                    results_summary.append(
                        {
                            "season": season_name,
                            "status": "failed",
                            "error": result.message,
                        }
                    )
                    if not continue_on_error:
                        break

            except Exception as e:
                error(f"  Error: {e}")
                results_summary.append(
                    {
                        "season": season_name,
                        "status": "error",
                        "error": str(e),
                    }
                )
                if not continue_on_error:
                    break

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Import Summary")
    click.echo("=" * 50)

    total_matches = 0
    total_skipped = 0
    success_count = 0

    for r in results_summary:
        if r["status"] == "success":
            matches_val = r["matches"]
            skipped_val = r["skipped"]
            click.echo(f"  {r['season']}: {matches_val} matches, {skipped_val} skipped")
            total_matches += int(matches_val)
            total_skipped += int(skipped_val)
            success_count += 1
        else:
            click.echo(f"  {r['season']}: FAILED - {r.get('error', 'Unknown error')}")

    click.echo("-" * 50)
    click.echo(f"Total: {success_count}/{len(season_codes)} seasons imported")
    click.echo(f"       {total_matches} matches created, {total_skipped} skipped")
    click.echo("=" * 50)


@import_cli.command(name="list-divisions")
def list_divisions() -> None:
    """List all available division codes.

    Shows all division codes supported by Football-Data.co.uk
    along with their corresponding tournament names and countries.
    """
    click.echo("\nAvailable Division Codes:")
    click.echo("=" * 50)

    # Group by country
    by_country: dict[str, list[tuple[str, str]]] = {}
    for code, division_info in sorted(DIVISION_MAPPING.items()):
        country = division_info["country"]
        if country not in by_country:
            by_country[country] = []
        by_country[country].append((code, division_info["name"]))

    for country in sorted(by_country.keys()):
        click.echo(f"\n{country}:")
        for code, name in by_country[country]:
            click.echo(f"  {code:4s} - {name}")

    click.echo("\n" + "=" * 50)
