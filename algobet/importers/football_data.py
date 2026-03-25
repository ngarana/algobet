"""Football-Data.co.uk CSV importer for historical match data and betting odds.

This module provides a service for importing football match data from
Football-Data.co.uk, a free resource providing historical results and
betting odds in CSV format.

Usage:
    from algobet.importers import FootballDataImporter
    from algobet.infrastructure.database import session_scope

    with session_scope() as session:
        importer = FootballDataImporter(session)
        result = importer.import_from_file(
            "data/sample/premier-league-2023-24.csv",
            season_code="2324"
        )
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import Match, Season, Team, Tournament

logger = logging.getLogger(__name__)


# Division code to tournament mapping
DIVISION_MAPPING: dict[str, dict[str, str]] = {
    "E0": {
        "name": "Premier League",
        "country": "England",
        "url_slug": "premier-league",
    },
    "E1": {"name": "Championship", "country": "England", "url_slug": "championship"},
    "E2": {"name": "League One", "country": "England", "url_slug": "league-one"},
    "E3": {"name": "League Two", "country": "England", "url_slug": "league-two"},
    "SC0": {
        "name": "Scottish Premiership",
        "country": "Scotland",
        "url_slug": "scottish-premiership",
    },
    "D1": {"name": "Bundesliga", "country": "Germany", "url_slug": "bundesliga"},
    "D2": {"name": "2. Bundesliga", "country": "Germany", "url_slug": "2-bundesliga"},
    "I1": {"name": "Serie A", "country": "Italy", "url_slug": "serie-a"},
    "I2": {"name": "Serie B", "country": "Italy", "url_slug": "serie-b"},
    "SP1": {"name": "La Liga", "country": "Spain", "url_slug": "la-liga"},
    "SP2": {"name": "La Liga 2", "country": "Spain", "url_slug": "la-liga-2"},
    "F1": {"name": "Ligue 1", "country": "France", "url_slug": "ligue-1"},
    "F2": {"name": "Ligue 2", "country": "France", "url_slug": "ligue-2"},
    "N1": {"name": "Eredivisie", "country": "Netherlands", "url_slug": "eredivisie"},
    "B1": {
        "name": "First Division A",
        "country": "Belgium",
        "url_slug": "first-division-a",
    },
    "P1": {"name": "Primeira Liga", "country": "Portugal", "url_slug": "primeira-liga"},
    "T1": {"name": "Super Lig", "country": "Turkey", "url_slug": "super-lig"},
    "G1": {"name": "Super League", "country": "Greece", "url_slug": "super-league"},
}

# Team name normalization mapping (Football-Data -> AlgoBet standard)
TEAM_NAME_MAPPING: dict[str, str] = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "Wolves": "Wolverhampton Wanderers",
    "West Ham": "West Ham United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Leeds": "Leeds United",
    "Sheffield United": "Sheffield United",
    "QPR": "Queens Park Rangers",
    "Middlesbrough": "Middlesbrough",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "Huddersfield": "Huddersfield Town",
    "Cardiff": "Cardiff City",
    "Blackburn": "Blackburn Rovers",
    "Bolton": "Bolton Wanderers",
    "Preston": "Preston North End",
    "Rotherham": "Rotherham United",
    "Peterboro": "Peterborough United",
    "Nottm County": "Notts County",
    "Milton K Dons": "Milton Keynes Dons",
    "Morecambe": "Morecambe",
    "Cheltenham": "Cheltenham Town",
    "Cambridge Utd": "Cambridge United",
    "Burton A": "Burton Albion",
    "Shrewsbury": "Shrewsbury Town",
    "Wycombe": "Wycombe Wanderers",
    "Portsmouth": "Portsmouth",
    "Plymouth": "Plymouth Argyle",
    "Ipswich": "Ipswich Town",
    "Derby": "Derby County",
    "Barnsley": "Barnsley",
    "Charlton": "Charlton Athletic",
    "Wigan": "Wigan Athletic",
    "Reading": "Reading",
    "Luton": "Luton Town",
    "Hull": "Hull City",
    "Bristol City": "Bristol City",
    "Millwall": "Millwall",
    "Coventry": "Coventry City",
    "Sunderland": "Sunderland",
    "Watford": "Watford",
    "Birmingham": "Birmingham City",
    "Bristol Rovers": "Bristol Rovers",
    "Exeter": "Exeter City",
    "Fleetwood": "Fleetwood Town",
    "Lincoln": "Lincoln City",
    "Oxford Utd": "Oxford United",
    "Accrington": "Accrington Stanley",
    "Forest Green": "Forest Green Rovers",
    "Grimsby": "Grimsby Town",
    "Harrogate": "Harrogate Town",
    "Barrow": "Barrow",
    "Bradford City": "Bradford City",
    "Newport": "Newport County",
    "Sutton Utd": "Sutton United",
    "Tranmere": "Tranmere Rovers",
    "Crawley Town": "Crawley Town",
    "Colchester": "Colchester United",
    "Swindon": "Swindon Town",
    "Northampton": "Northampton Town",
    "Leyton Orient": "Leyton Orient",
    "Salford": "Salford City",
    "Mansfield": "Mansfield Town",
    "Crewe": "Crewe Alexandra",
    "Doncaster": "Doncaster Rovers",
    "Gillingham": "Gillingham",
    "Hartlepool": "Hartlepool United",
    "Rochdale": "Rochdale",
    "AFC Wimbledon": "AFC Wimbledon",
    "Carlisle": "Carlisle United",
    "Stevenage": "Stevenage",
    "Walsall": "Walsall",
    "Stockport": "Stockport County",
    "Wrexham": "Wrexham",
    "Notts County": "Notts County",
    "Woking": "Woking",
    "York": "York City",
    "Aldershot": "Aldershot Town",
    "Dagenham & R": "Dagenham & Redbridge",
    "Boreham Wood": "Boreham Wood",
    "Wealdstone": "Wealdstone",
    "Eastleigh": "Eastleigh",
    "Bromley": "Bromley",
    "Chesterfield": "Chesterfield",
    "Gateshead": "Gateshead",
    "Dover": "Dover Athletic",
    "Torquay": "Torquay United",
    "Notts Co": "Notts County",
}


@dataclass
class ImportProgress:
    """Progress update for an import operation."""

    total_rows: int = 0
    processed_rows: int = 0
    matches_created: int = 0
    matches_skipped: int = 0
    teams_created: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the import."""
        if self.processed_rows == 0:
            return 0.0
        return (self.matches_created / self.processed_rows) * 100


@dataclass
class ImportResult:
    """Result of an import operation."""

    success: bool
    progress: ImportProgress
    message: str = ""
    season_id: int | None = None
    tournament_id: int | None = None


class FootballDataImporter:
    """Import match data from Football-Data.co.uk CSV files.

    This service handles downloading, parsing, and importing football
    match data from Football-Data.co.uk into the AlgoBet database.

    Attributes:
        session: SQLAlchemy database session
        progress_callback: Optional callback for progress updates
    """

    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    def __init__(
        self,
        session: Session,
        progress_callback: Callable[[ImportProgress], None] | None = None,
    ) -> None:
        """Initialize the importer.

        Args:
            session: SQLAlchemy database session
            progress_callback: Optional callback for progress updates
        """
        self.session = session
        self.progress_callback = progress_callback

    def _emit_progress(self, progress: ImportProgress) -> None:
        """Emit progress update to callback if registered.

        Args:
            progress: Progress update to emit
        """
        if self.progress_callback:
            self.progress_callback(progress)

    def normalize_team_name(self, name: str) -> str:
        """Normalize a team name to AlgoBet standard.

        Args:
            name: Raw team name from CSV

        Returns:
            Normalized team name
        """
        return TEAM_NAME_MAPPING.get(name, name)

    def parse_season_code(self, code: str) -> dict[str, int | str]:
        """Convert season code like '2324' to season info.

        Args:
            code: 4-digit season code (e.g., "2324" for 2023/2024)

        Returns:
            Dictionary with name, start_year, end_year
        """
        start_year = 2000 + int(code[:2])
        end_year = 2000 + int(code[2:])
        return {
            "name": f"{start_year}/{end_year}",
            "start_year": start_year,
            "end_year": end_year,
        }

    def parse_match_date(self, date_str: str, time_str: str | None = None) -> datetime:
        """Parse Football-Data date and time into datetime.

        Args:
            date_str: Date string in DD/MM/YYYY or DD/MM/YY format
            time_str: Optional time string in HH:MM format

        Returns:
            Parsed datetime object
        """
        if time_str and time_str.strip():
            try:
                return datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            except ValueError:
                return datetime.strptime(f"{date_str} {time_str}", "%d/%m/%y %H:%M")
        try:
            return datetime.strptime(f"{date_str} 15:00", "%d/%m/%Y %H:%M")
        except ValueError:
            return datetime.strptime(f"{date_str} 15:00", "%d/%m/%y %H:%M")

    def get_or_create_tournament(self, division_code: str) -> Tournament | None:
        """Get or create a tournament from division code.

        Args:
            division_code: Football-Data division code (e.g., "E0")

        Returns:
            Tournament instance or None if division is unknown
        """
        if division_code not in DIVISION_MAPPING:
            logger.warning(f"Unknown division code: {division_code}")
            return None

        info = DIVISION_MAPPING[division_code]
        tournament = self.session.execute(
            select(Tournament).where(Tournament.url_slug == info["url_slug"])
        ).scalar_one_or_none()

        if not tournament:
            tournament = Tournament(
                name=info["name"],
                country=info["country"],
                url_slug=info["url_slug"],
            )
            self.session.add(tournament)
            self.session.flush()

        return tournament

    def get_or_create_team(self, name: str) -> Team:
        """Get or create a team by name.

        Args:
            name: Team name (will be normalized)

        Returns:
            Team instance
        """
        normalized_name = self.normalize_team_name(name)
        team = self.session.execute(
            select(Team).where(Team.name == normalized_name)
        ).scalar_one_or_none()

        if not team:
            team = Team(name=normalized_name)
            self.session.add(team)
            self.session.flush()
            return team

        return team

    def get_or_create_season(self, tournament: Tournament, season_code: str) -> Season:
        """Get or create a season for a tournament.

        Args:
            tournament: Tournament instance
            season_code: 4-digit season code

        Returns:
            Season instance
        """
        season_info = self.parse_season_code(season_code)
        season = self.session.execute(
            select(Season).where(
                Season.tournament_id == tournament.id,
                Season.name == season_info["name"],
            )
        ).scalar_one_or_none()

        if not season:
            season = Season(
                tournament_id=tournament.id,
                name=season_info["name"],
                start_year=season_info["start_year"],
                end_year=season_info["end_year"],
            )
            self.session.add(season)
            self.session.flush()

        return season

    def parse_csv_row(self, row: dict[str, str]) -> dict[str, Any] | None:
        """Parse a CSV row into match data.

        Args:
            row: CSV row as dictionary

        Returns:
            Parsed match data or None if row is invalid
        """
        # Skip rows without required fields
        required_fields = ["Div", "Date", "HomeTeam", "AwayTeam"]
        if not all(field in row and row[field] for field in required_fields):
            return None

        try:
            # Parse date and time
            match_date = self.parse_match_date(row["Date"], row.get("Time", ""))

            # Parse scores (may be missing for scheduled matches)
            home_score = int(row["FTHG"]) if row.get("FTHG") else None
            away_score = int(row["FTAG"]) if row.get("FTAG") else None

            # Parse odds (use average odds)
            odds_home = float(row["AvgH"]) if row.get("AvgH") else None
            odds_draw = float(row["AvgD"]) if row.get("AvgD") else None
            odds_away = float(row["AvgA"]) if row.get("AvgA") else None

            # Determine match status
            status = "FINISHED" if home_score is not None else "SCHEDULED"

            return {
                "division": row["Div"],
                "match_date": match_date,
                "home_team": row["HomeTeam"],
                "away_team": row["AwayTeam"],
                "home_score": home_score,
                "away_score": away_score,
                "status": status,
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
            }
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse row: {e}")
            return None

    def create_match(
        self,
        match_data: dict[str, Any],
        tournament: Tournament,
        season: Season,
    ) -> Match | None:
        """Create a match record if it doesn't exist.

        Args:
            match_data: Parsed match data
            tournament: Tournament instance
            season: Season instance

        Returns:
            Created Match instance or None if already exists
        """
        home_team = self.get_or_create_team(match_data["home_team"])
        away_team = self.get_or_create_team(match_data["away_team"])

        # Check for existing match
        existing = self.session.execute(
            select(Match).where(
                Match.tournament_id == tournament.id,
                Match.season_id == season.id,
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
                Match.match_date == match_data["match_date"],
            )
        ).scalar_one_or_none()

        if existing:
            return None

        match = Match(
            tournament_id=tournament.id,
            season_id=season.id,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=match_data["match_date"],
            home_score=match_data["home_score"],
            away_score=match_data["away_score"],
            status=match_data["status"],
            odds_home=match_data["odds_home"],
            odds_draw=match_data["odds_draw"],
            odds_away=match_data["odds_away"],
        )
        self.session.add(match)
        return match

    def download_csv(self, season_code: str, division: str) -> str:
        """Download CSV file content from Football-Data.co.uk.

        Args:
            season_code: 4-digit season code (e.g., "2324")
            division: Division code (e.g., "E0")

        Returns:
            CSV content as string

        Raises:
            urllib.error.URLError: If download fails
        """
        url = f"{self.BASE_URL}/{season_code}/{division}.csv"
        logger.info(f"Downloading CSV from {url}")
        with urlopen(url) as response:
            content: str = response.read().decode("utf-8")
            return content

    def parse_csv(self, content: str) -> list[dict[str, str]]:
        """Parse CSV content into list of dictionaries.

        Args:
            content: CSV content as string

        Returns:
            List of row dictionaries
        """
        if content.startswith("\ufeff"):
            content = content[1:]
        reader = csv.DictReader(content.splitlines())
        return list(reader)

    def import_from_url(
        self,
        season_code: str,
        divisions: list[str],
    ) -> ImportResult:
        """Import match data from Football-Data.co.uk URLs.

        Args:
            season_code: 4-digit season code (e.g., "2324")
            divisions: List of division codes to import

        Returns:
            ImportResult with import statistics
        """
        progress = ImportProgress()
        all_matches_created = 0
        all_matches_skipped = 0
        all_teams_created = 0
        tournament_id = None
        season_id = None

        for division in divisions:
            try:
                content = self.download_csv(season_code, division)
                rows = self.parse_csv(content)
                progress.total_rows += len(rows)

                for row in rows:
                    progress.processed_rows += 1
                    match_data = self.parse_csv_row(row)

                    if not match_data:
                        progress.errors.append(f"Invalid row: {row}")
                        continue

                    tournament = self.get_or_create_tournament(match_data["division"])
                    if not tournament:
                        progress.errors.append(
                            f"Unknown division: {match_data['division']}"
                        )
                        continue

                    if tournament_id is None:
                        tournament_id = tournament.id

                    season = self.get_or_create_season(tournament, season_code)
                    if season_id is None:
                        season_id = season.id

                    initial_team_count = self.session.query(Team).count()
                    match = self.create_match(match_data, tournament, season)
                    final_team_count = self.session.query(Team).count()

                    if match:
                        all_matches_created += 1
                        all_teams_created += final_team_count - initial_team_count
                    else:
                        all_matches_skipped += 1

                    self._emit_progress(progress)

            except Exception as e:
                error_msg = f"Error importing {division} {season_code}: {e}"
                logger.error(error_msg)
                progress.errors.append(error_msg)

        progress.matches_created = all_matches_created
        progress.matches_skipped = all_matches_skipped
        progress.teams_created = all_teams_created

        return ImportResult(
            success=len(progress.errors) == 0,
            progress=progress,
            message=f"Imported {all_matches_created} matches, "
            f"skipped {all_matches_skipped} duplicates",
            season_id=season_id,
            tournament_id=tournament_id,
        )

    def import_from_file(
        self,
        file_path: str | Path,
        season_code: str | None = None,
        division: str | None = None,
    ) -> ImportResult:
        """Import match data from a local CSV file.

        Args:
            file_path: Path to the CSV file
            season_code: Optional season code (inferred from filename if not provided)
            division: Optional division code (inferred from CSV if not provided)

        Returns:
            ImportResult with import statistics
        """
        file_path = Path(file_path)
        progress = ImportProgress()

        # Infer season code from filename if not provided
        if not season_code:
            # Try to extract from filename like "premier-league-2023-24.csv"
            import re

            match = re.search(r"(\d{2})(\d{2})", file_path.stem)
            if match:
                season_code = f"{match.group(1)}{match.group(2)}"
            else:
                season_code = "2324"  # Default to current season
                logger.warning(
                    f"Could not infer season code from filename, using {season_code}"
                )

        # Read CSV file
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return ImportResult(
                success=False,
                progress=progress,
                message=f"File not found: {file_path}",
            )

        rows = self.parse_csv(content)
        progress.total_rows = len(rows)

        tournament_id = None
        season_id = None

        for row in rows:
            progress.processed_rows += 1
            match_data = self.parse_csv_row(row)

            if not match_data:
                continue

            # Use provided division or from CSV
            div_code = division or match_data["division"]

            tournament = self.get_or_create_tournament(div_code)
            if not tournament:
                progress.errors.append(f"Unknown division: {div_code}")
                continue

            if tournament_id is None:
                tournament_id = tournament.id

            season = self.get_or_create_season(tournament, season_code)
            if season_id is None:
                season_id = season.id

            initial_team_count = self.session.query(Team).count()
            new_match = self.create_match(match_data, tournament, season)
            final_team_count = self.session.query(Team).count()

            if new_match:
                progress.matches_created += 1
                progress.teams_created += final_team_count - initial_team_count
            else:
                progress.matches_skipped += 1

            # Emit progress every 10 rows
            if progress.processed_rows % 10 == 0:
                self._emit_progress(progress)

        self._emit_progress(progress)

        return ImportResult(
            success=len(progress.errors) == 0,
            progress=progress,
            message=f"Imported {progress.matches_created} matches, "
            f"skipped {progress.matches_skipped} duplicates",
            season_id=season_id,
            tournament_id=tournament_id,
        )

    def import_multiple_seasons(
        self,
        season_codes: list[str],
        divisions: list[str],
    ) -> list[ImportResult]:
        """Import multiple seasons of data.

        Args:
            season_codes: List of season codes (e.g., ["2223", "2324"])
            divisions: List of division codes to import

        Returns:
            List of ImportResult objects, one per season
        """
        results = []
        for season_code in season_codes:
            result = self.import_from_url(season_code, divisions)
            results.append(result)
        return results
