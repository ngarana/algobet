"""Command-line interface for AlgoBet scraper."""

import csv
import re
from typing import Any

import click
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import init_db, session_scope
from .models import Match, Season, Team, Tournament
from .predictions_cli import predictions
from .scraper import OddsPortalScraper, ScrapedMatch


def parse_league_info(url: str) -> tuple[str, str, str]:
    """Extract country, league name, and slug from URL.

    Args:
        url: OddsPortal results URL.

    Returns:
        Tuple of (country, league_name, url_slug).
    """
    # Pattern: /football/{country}/{league-slug}/results/
    match = re.search(r"/football/([^/]+)/([^/]+?)(?:-\d{4}-\d{4})?/results/", url)
    if not match:
        raise ValueError(f"Could not parse league info from URL: {url}")

    country = match.group(1).replace("-", " ").title()
    slug = match.group(2)
    league_name = slug.replace("-", " ").title()

    return country, league_name, slug


def extract_season_from_url(url: str) -> str | None:
    """Extract season suffix from URL if present.

    Args:
        url: OddsPortal results URL.

    Returns:
        Season suffix (e.g., "2023-2024") or None for current season.
    """
    match = re.search(r"-(\d{4}-\d{4})/results/", url)
    return match.group(1) if match else None


def get_or_create_tournament(
    session: Session, country: str, name: str, slug: str
) -> Tournament:
    """Get or create a tournament."""
    tournament = session.execute(
        select(Tournament).where(Tournament.url_slug == slug)
    ).scalar_one_or_none()

    if not tournament:
        tournament = Tournament(name=name, country=country, url_slug=slug)
        session.add(tournament)
        session.flush()

    return tournament


def get_or_create_season(
    session: Session, tournament: Tournament, name: str, url_suffix: str | None
) -> Season:
    """Get or create a season."""
    season = session.execute(
        select(Season).where(Season.tournament_id == tournament.id, Season.name == name)
    ).scalar_one_or_none()

    if not season:
        # Parse years from name (e.g., "2023/2024")
        years = re.findall(r"\d{4}", name)
        start_year = int(years[0]) if years else 2024
        end_year = int(years[1]) if len(years) > 1 else start_year + 1

        season = Season(
            tournament_id=tournament.id,
            name=name,
            start_year=start_year,
            end_year=end_year,
            url_suffix=url_suffix,
        )
        session.add(season)
        session.flush()

    return season


def get_or_create_team(session: Session, name: str) -> Team:
    """Get or create a team."""
    team = session.execute(select(Team).where(Team.name == name)).scalar_one_or_none()

    if not team:
        team = Team(name=name)
        session.add(team)
        session.flush()

    return team


def save_matches_to_db(
    session: Session,
    matches: list[ScrapedMatch],
    tournament: Tournament,
    season: Season,
) -> int:
    """Save scraped matches to database.

    Returns:
        Number of new matches saved.
    """
    saved_count = 0

    for scraped in matches:
        home_team = get_or_create_team(session, scraped.home_team)
        away_team = get_or_create_team(session, scraped.away_team)

        # Check if match already exists
        existing = session.execute(
            select(Match).where(
                Match.tournament_id == tournament.id,
                Match.season_id == season.id,
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
                Match.match_date == scraped.match_date,
            )
        ).scalar_one_or_none()

        if existing:
            continue

        match = Match(
            tournament_id=tournament.id,
            season_id=season.id,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=scraped.match_date,
            home_score=scraped.home_score,
            away_score=scraped.away_score,
            odds_home=scraped.odds_home,
            odds_draw=scraped.odds_draw,
            odds_away=scraped.odds_away,
            num_bookmakers=scraped.num_bookmakers,
        )
        session.add(match)
        saved_count += 1

    return saved_count


def save_upcoming_matches(session: Session, matches_data: list[dict[str, Any]]) -> int:
    """Save upcoming matches to database.

    Args:
        session: Database session.
        matches_data: List of match dictionaries.

    Returns:
        Number of matches saved/updated.
    """
    saved_count = 0

    # Cache for tournaments/seasons to avoid repeated DB lookups
    # Key: (country, name) -> Tournament object
    tournaments_cache = {}
    # Key: (tournament_id, name) -> Season object
    seasons_cache = {}

    for data in matches_data:
        # 1. Get/Create Tournament
        country = data["country_name"]
        tourn_name = data["tournament_name"]
        slug = tourn_name.lower().replace(" ", "-")  # Simple slug generation

        tourn_key = (country, tourn_name)
        if tourn_key not in tournaments_cache:
            tournaments_cache[tourn_key] = get_or_create_tournament(
                session, country, tourn_name, slug
            )
        tournament = tournaments_cache[tourn_key]

        # 2. Get/Create Season
        # For upcoming, we assume current season based on date
        match_date = data["match_date"]
        # Simple heuristic: "YYYY/YYYY+1"
        start_year = match_date.year
        if match_date.month < 7:  # Second half of season
            start_year -= 1
        season_name = f"{start_year}/{start_year + 1}"

        season_key = (tournament.id, season_name)
        if season_key not in seasons_cache:
            seasons_cache[season_key] = get_or_create_season(
                session, tournament, season_name, None
            )
        season = seasons_cache[season_key]

        # 3. Get/Create Teams
        home_team = get_or_create_team(session, data["home_team"])
        away_team = get_or_create_team(session, data["away_team"])

        # 4. Save/Update Match
        # Check if exists
        existing = session.execute(
            select(Match).where(
                Match.tournament_id == tournament.id,
                Match.season_id == season.id,
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
                Match.match_date == match_date,
            )
        ).scalar_one_or_none()

        if existing:
            # Update odds/status if needed
            existing.odds_home = data["odds_home"]
            existing.odds_draw = data["odds_draw"]
            existing.odds_away = data["odds_away"]
            existing.num_bookmakers = data["num_bookmakers"]
            existing.status = "SCHEDULED"  # Or update if live
            continue  # Don't count as new save, but we updated it

        match = Match(
            tournament_id=tournament.id,
            season_id=season.id,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=match_date,
            home_score=None,
            away_score=None,
            odds_home=data["odds_home"],
            odds_draw=data["odds_draw"],
            odds_away=data["odds_away"],
            num_bookmakers=data["num_bookmakers"],
            status="SCHEDULED",
        )
        session.add(match)
        saved_count += 1

    return saved_count


@click.group()
def cli() -> None:
    """AlgoBet - Football match database and OddsPortal scraper."""
    pass


# Add predictions as a subcommand group
cli.add_command(predictions)


@cli.command()
def init() -> None:
    """Initialize the database."""
    import time

    from sqlalchemy.exc import OperationalError

    max_retries = 30
    click.echo("Waiting for database connection...")

    for i in range(max_retries):
        try:
            init_db()
            click.echo("Database initialized successfully!")
            return
        except OperationalError:
            if i < max_retries - 1:
                time.sleep(1)
                continue
            raise
        except Exception as e:
            click.echo(f"Error initializing database: {e}")
            raise


@cli.command()
@click.option(
    "--url",
    default="https://www.oddsportal.com/matches/football/",
    help="OddsPortal upcoming matches URL",
)
@click.option("--headless/--no-headless", default=True, help="Run browser headlessly")
def scrape_upcoming(url: str, headless: bool) -> None:
    """Scrape upcoming matches from OddsPortal."""
    click.echo(f"Scraping upcoming matches from: {url}")

    with OddsPortalScraper(headless=headless) as scraper:
        click.echo("Navigating to upcoming matches page...")
        scraper.navigate_to_upcoming(url)

        click.echo("Scraping matches...")
        matches_data = scraper.scrape_upcoming_matches()

        if not matches_data:
            click.echo("No upcoming matches found.")
            return

        click.echo(f"Found {len(matches_data)} matches")

        # Save to DB
        with session_scope() as session:
            saved = save_upcoming_matches(session, matches_data)
            click.echo(f"Saved {saved} new matches to database")


@cli.command()
@click.option("--url", required=True, help="OddsPortal results page URL")
@click.option(
    "--pages", type=int, default=None, help="Max pages to scrape (all if not set)"
)
@click.option("--headless/--no-headless", default=True, help="Run browser headlessly")
def scrape(url: str, pages: int | None, headless: bool) -> None:
    """Scrape matches from an OddsPortal results page."""

    # Ensure database exists/connected
    init_db()

    # Parse league info
    country, league_name, slug = parse_league_info(url)
    season_suffix = extract_season_from_url(url)

    # Determine season name
    if season_suffix:
        years = season_suffix.split("-")
        season_name = f"{years[0]}/{years[1]}"
    else:
        # Current season - we'll detect from page
        season_name = "2025/2026"  # Default, might be updated

    click.echo(f"Scraping: {league_name} ({country}) - Season {season_name}")
    click.echo(f"URL: {url}")

    with OddsPortalScraper(headless=headless) as scraper:
        # Navigate to results page
        click.echo("Navigating to results page...")
        scraper.navigate_to_results(url)

        # Try to get actual season from page
        seasons = scraper.get_available_seasons()
        current_season = next((s for s in seasons if s.is_current), None)
        if current_season and not season_suffix:
            season_name = current_season.name
            click.echo(f"Detected current season: {season_name}")

        # Scrape matches
        click.echo(f"Scraping matches (max pages: {pages or 'all'})...")
        matches = scraper.scrape_all_pages(max_pages=pages)

        click.echo(f"Scraped {len(matches)} matches")

        # Save to database
        if matches:
            with session_scope() as session:
                tournament = get_or_create_tournament(
                    session, country, league_name, slug
                )
                season = get_or_create_season(
                    session, tournament, season_name, season_suffix
                )
                saved = save_matches_to_db(session, matches, tournament, season)
                click.echo(f"Saved {saved} new matches to database")


@cli.command()
@click.option(
    "--url", required=True, help="OddsPortal results page URL (current season)"
)
@click.option("--headless/--no-headless", default=True, help="Run browser headlessly")
def seasons(url: str, headless: bool) -> None:
    """List available seasons for a league."""
    with OddsPortalScraper(headless=headless) as scraper:
        scraper.navigate_to_results(url)
        available_seasons = scraper.get_available_seasons()

        click.echo(f"Available seasons ({len(available_seasons)}):")
        for season in available_seasons:
            marker = " (current)" if season.is_current else ""
            click.echo(f"  - {season.name}{marker}")


@cli.command("scrape-all")
@click.option(
    "--url", required=True, help="OddsPortal results page URL (current season)"
)
@click.option("--from-season", "from_season", help="Start season (e.g., 2020/2021)")
@click.option("--to-season", "to_season", help="End season (e.g., 2023/2024)")
@click.option(
    "--pages", type=int, default=None, help="Max pages per season (all if not set)"
)
@click.option("--headless/--no-headless", default=True, help="Run browser headlessly")
def scrape_all(
    url: str,
    from_season: str | None,
    to_season: str | None,
    pages: int | None,
    headless: bool,
) -> None:
    """Scrape matches from multiple seasons.

    Examples:
        # Scrape all available seasons
        algobet scrape-all --url "https://www.oddsportal.com/football/england/premier-league/results/"

        # Scrape from 2020/2021 to 2023/2024
        algobet scrape-all --url "..." --from-season "2020/2021" --to-season "2023/2024"
    """
    init_db()

    country, league_name, slug = parse_league_info(url)
    click.echo(f"Scraping: {league_name} ({country})")

    with OddsPortalScraper(headless=headless) as scraper:
        # Navigate to get available seasons
        click.echo("Fetching available seasons...")
        scraper.navigate_to_results(url)
        available_seasons = scraper.get_available_seasons()

        if not available_seasons:
            click.echo("No seasons found!")
            return

        click.echo(f"Found {len(available_seasons)} seasons")

        # Filter seasons by range if specified
        seasons_to_scrape = []
        for season in available_seasons:
            # Parse season years
            years = re.findall(r"\d{4}", season.name)
            if not years:
                continue
            start_year = int(years[0])

            # Apply from_season filter
            if from_season:
                from_years = re.findall(r"\d{4}", from_season)
                if from_years and start_year < int(from_years[0]):
                    continue

            # Apply to_season filter
            if to_season:
                to_years = re.findall(r"\d{4}", to_season)
                if to_years and start_year > int(to_years[0]):
                    continue

            seasons_to_scrape.append(season)

        click.echo(f"Scraping {len(seasons_to_scrape)} seasons...")

        total_matches = 0
        for i, season in enumerate(seasons_to_scrape, 1):
            click.echo(f"\n[{i}/{len(seasons_to_scrape)}] Season: {season.name}")

            # Build season URL
            season_url = scraper.get_season_url(url, season)
            click.echo(f"  URL: {season_url}")

            # Navigate and scrape
            scraper.navigate_to_results(season_url)
            matches = scraper.scrape_all_pages(max_pages=pages)

            click.echo(f"  Scraped {len(matches)} matches")

            # Save to database
            if matches:
                with session_scope() as session:
                    tournament = get_or_create_tournament(
                        session, country, league_name, slug
                    )
                    db_season = get_or_create_season(
                        session, tournament, season.name, season.url_suffix
                    )
                    saved = save_matches_to_db(session, matches, tournament, db_season)
                    click.echo(f"  Saved {saved} new matches")
                    total_matches += saved

        click.echo(f"\nTotal: {total_matches} new matches saved to database")


@cli.command()
@click.option(
    "--output", "-o", required=True, type=click.Path(), help="Output CSV file"
)
@click.option("--tournament", help="Filter by tournament name")
@click.option("--season", help="Filter by season (e.g., 2023/2024)")
def export(
    output: str,
    tournament: str | None,
    season: str | None,
) -> None:
    """Export matches to CSV."""
    with session_scope() as session:
        query = select(Match)

        # Apply filters
        if tournament:
            query = query.join(Tournament).where(
                Tournament.name.ilike(f"%{tournament}%")
            )
        if season:
            query = query.join(Season).where(Season.name == season)

        matches = session.execute(query).scalars().all()

        if not matches:
            click.echo("No matches found matching the criteria")
            return

        # Write CSV
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "date",
                    "tournament",
                    "season",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                    "result",
                    "odds_home",
                    "odds_draw",
                    "odds_away",
                    "num_bookmakers",
                ]
            )

            for match in matches:
                writer.writerow(
                    [
                        match.match_date.isoformat(),
                        match.tournament.name,
                        match.season.name,
                        match.home_team.name,
                        match.away_team.name,
                        match.home_score,
                        match.away_score,
                        match.result,
                        match.odds_home,
                        match.odds_draw,
                        match.odds_away,
                        match.num_bookmakers,
                    ]
                )

        click.echo(f"Exported {len(matches)} matches to {output}")


if __name__ == "__main__":
    cli()
