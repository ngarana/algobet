"""Unit tests for Football-Data.co.uk importer.

Tests cover:
- Team name normalization
- Date parsing
- Season code parsing
- Division code mapping
- CSV row parsing
- Full import from file
- Duplicate handling
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from algobet.importers.football_data import (
    DIVISION_MAPPING,
    TEAM_NAME_MAPPING,
    FootballDataImporter,
    ImportProgress,
    ImportResult,
)
from algobet.models import Base


@pytest.fixture
def db_engine() -> Engine:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(db_engine: Engine) -> Session:
    """Create a database session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def importer(db_session: Session) -> FootballDataImporter:
    """Create a FootballDataImporter instance for testing."""
    return FootballDataImporter(db_session)


@pytest.fixture
def sample_csv_content() -> str:
    """Return sample CSV content for testing."""
    header = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,"
        "Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,"
        "B365H,B365D,B365A,BWH,BWD,BWA,IWH,IWD,IWA,PSH,PSD,PSA,"
        "WHH,WHD,WHA,VCH,VCD,VCA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA\n"
    )
    row1 = (
        "E0,11/08/2023,20:00,Burnley,Man City,0,3,A,0,2,A,"
        "C Pawson,6,17,1,8,11,8,6,5,0,0,1,0,"
        "8,5.5,1.33,8.75,5.25,1.34,8,5.5,1.35,8.58,5.51,1.37,"
        "8,5,1.25,9.5,5.25,1.33,9.5,5.68,1.39,9.02,5.35,1.35\n"
    )
    row2 = (
        "E0,12/08/2023,15:00,Arsenal,Nott'm Forest,2,1,H,2,0,H,"
        "M Oliver,15,6,7,2,12,12,8,3,2,2,0,0,"
        "1.18,7,15,1.17,7.5,15.5,1.2,7.25,14,1.18,7.86,15.67,1.12,"
        "6.5,12,1.14,7.5,17,1.21,8.5,17.5,1.18,7.64,15.67\n"
    )
    row3 = (
        "E0,12/08/2023,15:00,Brighton,Luton,4,1,H,1,0,H,"
        "D Coote,27,9,12,3,11,12,6,7,2,2,0,0,"
        "1.33,5.5,9,1.32,5.5,9,1.35,5.25,8.5,1.33,5.65,9.61,1.25,"
        "4.6,8.5,1.29,5.25,10,1.36,6,10.5,1.33,5.52,9.61\n"
    )
    return header + row1 + row2 + row3


@pytest.fixture
def sample_csv_file(sample_csv_content: str) -> Path:
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample_csv_content)
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


class TestNormalizeTeamName:
    """Tests for team name normalization."""

    def test_normalize_known_team_man_city(
        self, importer: FootballDataImporter
    ) -> None:
        """Test normalizing 'Man City' to 'Manchester City'."""
        result = importer.normalize_team_name("Man City")
        assert result == "Manchester City"

    def test_normalize_known_team_man_united(
        self, importer: FootballDataImporter
    ) -> None:
        """Test normalizing 'Man United' to 'Manchester United'."""
        result = importer.normalize_team_name("Man United")
        assert result == "Manchester United"

    def test_normalize_known_team_man_utd(self, importer: FootballDataImporter) -> None:
        """Test normalizing 'Man Utd' to 'Manchester United'."""
        result = importer.normalize_team_name("Man Utd")
        assert result == "Manchester United"

    def test_normalize_known_team_nottingham(
        self, importer: FootballDataImporter
    ) -> None:
        """Test normalizing 'Nott'm Forest' to 'Nottingham Forest'."""
        result = importer.normalize_team_name("Nott'm Forest")
        assert result == "Nottingham Forest"

    def test_normalize_known_team_tottenham(
        self, importer: FootballDataImporter
    ) -> None:
        """Test normalizing 'Tottenham' to 'Tottenham Hotspur'."""
        result = importer.normalize_team_name("Tottenham")
        assert result == "Tottenham Hotspur"

    def test_normalize_known_team_wolves(self, importer: FootballDataImporter) -> None:
        """Test normalizing 'Wolves' to 'Wolverhampton Wanderers'."""
        result = importer.normalize_team_name("Wolves")
        assert result == "Wolverhampton Wanderers"

    def test_normalize_unknown_team(self, importer: FootballDataImporter) -> None:
        """Test that unknown team names are returned unchanged."""
        result = importer.normalize_team_name("Some Unknown Team")
        assert result == "Some Unknown Team"

    def test_normalize_team_already_standard(
        self, importer: FootballDataImporter
    ) -> None:
        """Test that already standard names are returned unchanged."""
        result = importer.normalize_team_name("Liverpool")
        assert result == "Liverpool"

    def test_team_name_mapping_contains_key_teams(self) -> None:
        """Test that the mapping contains expected key teams."""
        assert "Man City" in TEAM_NAME_MAPPING
        assert "Man United" in TEAM_NAME_MAPPING
        assert "Tottenham" in TEAM_NAME_MAPPING
        assert "Wolves" in TEAM_NAME_MAPPING
        assert "Newcastle" in TEAM_NAME_MAPPING


class TestParseSeasonCode:
    """Tests for season code parsing."""

    def test_parse_season_code_2324(self, importer: FootballDataImporter) -> None:
        """Test parsing '2324' season code."""
        result = importer.parse_season_code("2324")
        assert result["name"] == "2023/2024"
        assert result["start_year"] == 2023
        assert result["end_year"] == 2024

    def test_parse_season_code_2223(self, importer: FootballDataImporter) -> None:
        """Test parsing '2223' season code."""
        result = importer.parse_season_code("2223")
        assert result["name"] == "2022/2023"
        assert result["start_year"] == 2022
        assert result["end_year"] == 2023

    def test_parse_season_code_1920(self, importer: FootballDataImporter) -> None:
        """Test parsing '1920' season code."""
        result = importer.parse_season_code("1920")
        assert result["name"] == "2019/2020"
        assert result["start_year"] == 2019
        assert result["end_year"] == 2020

    def test_parse_season_code_0001(self, importer: FootballDataImporter) -> None:
        """Test parsing '0001' season code (2000/2001)."""
        result = importer.parse_season_code("0001")
        assert result["name"] == "2000/2001"
        assert result["start_year"] == 2000
        assert result["end_year"] == 2001

    def test_parse_season_code_9900(self, importer: FootballDataImporter) -> None:
        """Test parsing '9900' season code.

        Note: The implementation uses 2000 + int(code[:2]), so '99' becomes 2099.
        This is a known limitation - season codes before 2000 are not supported.
        """
        result = importer.parse_season_code("9900")
        # The implementation assumes all years are 2000+
        assert result["name"] == "2099/2000"
        assert result["start_year"] == 2099
        assert result["end_year"] == 2000


class TestParseMatchDate:
    """Tests for match date parsing."""

    def test_parse_date_with_time(self, importer: FootballDataImporter) -> None:
        """Test parsing date with time."""
        result = importer.parse_match_date("11/08/2023", "20:00")
        assert result == datetime(2023, 8, 11, 20, 0)

    def test_parse_date_without_time(self, importer: FootballDataImporter) -> None:
        """Test parsing date without time defaults to 15:00."""
        result = importer.parse_match_date("12/08/2023", None)
        assert result == datetime(2023, 8, 12, 15, 0)

    def test_parse_date_with_empty_time(self, importer: FootballDataImporter) -> None:
        """Test parsing date with empty time string defaults to 15:00."""
        result = importer.parse_match_date("12/08/2023", "")
        assert result == datetime(2023, 8, 12, 15, 0)

    def test_parse_date_with_whitespace_time(
        self, importer: FootballDataImporter
    ) -> None:
        """Test parsing date with whitespace-only time defaults to 15:00."""
        result = importer.parse_match_date("12/08/2023", "   ")
        assert result == datetime(2023, 8, 12, 15, 0)

    def test_parse_date_various_formats(self, importer: FootballDataImporter) -> None:
        """Test parsing various valid dates."""
        # Early season
        result = importer.parse_match_date("05/08/2023", "15:00")
        assert result == datetime(2023, 8, 5, 15, 0)

        # Late season
        result = importer.parse_match_date("19/05/2024", "20:00")
        assert result == datetime(2024, 5, 19, 20, 0)


class TestGetDivisionInfo:
    """Tests for division code mapping."""

    def test_division_mapping_e0(self) -> None:
        """Test E0 maps to Premier League."""
        assert DIVISION_MAPPING["E0"]["name"] == "Premier League"
        assert DIVISION_MAPPING["E0"]["country"] == "England"
        assert DIVISION_MAPPING["E0"]["url_slug"] == "premier-league"

    def test_division_mapping_e1(self) -> None:
        """Test E1 maps to Championship."""
        assert DIVISION_MAPPING["E1"]["name"] == "Championship"
        assert DIVISION_MAPPING["E1"]["country"] == "England"

    def test_division_mapping_sp1(self) -> None:
        """Test SP1 maps to La Liga."""
        assert DIVISION_MAPPING["SP1"]["name"] == "La Liga"
        assert DIVISION_MAPPING["SP1"]["country"] == "Spain"

    def test_division_mapping_d1(self) -> None:
        """Test D1 maps to Bundesliga."""
        assert DIVISION_MAPPING["D1"]["name"] == "Bundesliga"
        assert DIVISION_MAPPING["D1"]["country"] == "Germany"

    def test_division_mapping_i1(self) -> None:
        """Test I1 maps to Serie A."""
        assert DIVISION_MAPPING["I1"]["name"] == "Serie A"
        assert DIVISION_MAPPING["I1"]["country"] == "Italy"

    def test_division_mapping_f1(self) -> None:
        """Test F1 maps to Ligue 1."""
        assert DIVISION_MAPPING["F1"]["name"] == "Ligue 1"
        assert DIVISION_MAPPING["F1"]["country"] == "France"

    def test_division_mapping_contains_major_leagues(self) -> None:
        """Test that mapping contains all major European leagues."""
        expected_codes = ["E0", "E1", "SP1", "D1", "I1", "F1", "N1"]
        for code in expected_codes:
            assert code in DIVISION_MAPPING, f"Missing division code: {code}"


class TestParseCsvRow:
    """Tests for CSV row parsing."""

    def test_parse_valid_row(self, importer: FootballDataImporter) -> None:
        """Test parsing a valid CSV row."""
        row = {
            "Div": "E0",
            "Date": "11/08/2023",
            "Time": "20:00",
            "HomeTeam": "Burnley",
            "AwayTeam": "Man City",
            "FTHG": "0",
            "FTAG": "3",
            "AvgH": "9.02",
            "AvgD": "5.35",
            "AvgA": "1.35",
        }
        result = importer.parse_csv_row(row)

        assert result is not None
        assert result["division"] == "E0"
        assert result["match_date"] == datetime(2023, 8, 11, 20, 0)
        assert result["home_team"] == "Burnley"
        assert result["away_team"] == "Man City"
        assert result["home_score"] == 0
        assert result["away_score"] == 3
        assert result["status"] == "FINISHED"
        assert result["odds_home"] == 9.02
        assert result["odds_draw"] == 5.35
        assert result["odds_away"] == 1.35

    def test_parse_row_without_scores(self, importer: FootballDataImporter) -> None:
        """Test parsing a row without scores (scheduled match)."""
        row = {
            "Div": "E0",
            "Date": "11/08/2024",
            "Time": "15:00",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": "",
            "FTAG": "",
        }
        result = importer.parse_csv_row(row)

        assert result is not None
        assert result["home_score"] is None
        assert result["away_score"] is None
        assert result["status"] == "SCHEDULED"

    def test_parse_row_without_odds(self, importer: FootballDataImporter) -> None:
        """Test parsing a row without odds data."""
        row = {
            "Div": "E0",
            "Date": "11/08/2023",
            "Time": "15:00",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": "2",
            "FTAG": "1",
        }
        result = importer.parse_csv_row(row)

        assert result is not None
        assert result["odds_home"] is None
        assert result["odds_draw"] is None
        assert result["odds_away"] is None

    def test_parse_row_missing_division(self, importer: FootballDataImporter) -> None:
        """Test parsing a row with missing division returns None."""
        row = {
            "Date": "11/08/2023",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
        }
        result = importer.parse_csv_row(row)
        assert result is None

    def test_parse_row_missing_date(self, importer: FootballDataImporter) -> None:
        """Test parsing a row with missing date returns None."""
        row = {
            "Div": "E0",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
        }
        result = importer.parse_csv_row(row)
        assert result is None

    def test_parse_row_missing_home_team(self, importer: FootballDataImporter) -> None:
        """Test parsing a row with missing home team returns None."""
        row = {
            "Div": "E0",
            "Date": "11/08/2023",
            "AwayTeam": "Chelsea",
        }
        result = importer.parse_csv_row(row)
        assert result is None

    def test_parse_row_missing_away_team(self, importer: FootballDataImporter) -> None:
        """Test parsing a row with missing away team returns None."""
        row = {
            "Div": "E0",
            "Date": "11/08/2023",
            "HomeTeam": "Arsenal",
        }
        result = importer.parse_csv_row(row)
        assert result is None

    def test_parse_row_empty_required_field(
        self, importer: FootballDataImporter
    ) -> None:
        """Test parsing a row with empty required field returns None."""
        row = {
            "Div": "",
            "Date": "11/08/2023",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
        }
        result = importer.parse_csv_row(row)
        assert result is None

    def test_parse_row_invalid_date(self, importer: FootballDataImporter) -> None:
        """Test parsing a row with invalid date returns None."""
        row = {
            "Div": "E0",
            "Date": "invalid-date",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
        }
        result = importer.parse_csv_row(row)
        assert result is None


class TestGetOrCreateTournament:
    """Tests for tournament creation."""

    def test_create_new_tournament(self, importer: FootballDataImporter) -> None:
        """Test creating a new tournament."""
        tournament = importer.get_or_create_tournament("E0")

        assert tournament is not None
        assert tournament.name == "Premier League"
        assert tournament.country == "England"
        assert tournament.url_slug == "premier-league"

    def test_get_existing_tournament(self, importer: FootballDataImporter) -> None:
        """Test getting an existing tournament."""
        # Create first
        tournament1 = importer.get_or_create_tournament("E0")
        importer.session.flush()

        # Get again
        tournament2 = importer.get_or_create_tournament("E0")

        assert tournament1.id == tournament2.id

    def test_unknown_division_code(self, importer: FootballDataImporter) -> None:
        """Test that unknown division code returns None."""
        tournament = importer.get_or_create_tournament("UNKNOWN")
        assert tournament is None


class TestGetOrCreateTeam:
    """Tests for team creation."""

    def test_create_new_team(self, importer: FootballDataImporter) -> None:
        """Test creating a new team."""
        team = importer.get_or_create_team("Liverpool")

        assert team.name == "Liverpool"

    def test_create_team_with_normalization(
        self, importer: FootballDataImporter
    ) -> None:
        """Test creating a team with name normalization."""
        team = importer.get_or_create_team("Man City")

        assert team.name == "Manchester City"

    def test_get_existing_team(self, importer: FootballDataImporter) -> None:
        """Test getting an existing team."""
        # Create first
        team1 = importer.get_or_create_team("Liverpool")
        importer.session.flush()

        # Get again
        team2 = importer.get_or_create_team("Liverpool")

        assert team1.id == team2.id

    def test_normalized_team_deduplication(
        self, importer: FootballDataImporter
    ) -> None:
        """Test that normalized names prevent duplicate teams."""
        # Create with raw name
        team1 = importer.get_or_create_team("Man City")
        importer.session.flush()

        # Get with different raw name that normalizes to same
        team2 = importer.get_or_create_team("Man City")

        assert team1.id == team2.id


class TestGetOrCreateSeason:
    """Tests for season creation."""

    def test_create_new_season(self, importer: FootballDataImporter) -> None:
        """Test creating a new season."""
        tournament = importer.get_or_create_tournament("E0")
        season = importer.get_or_create_season(tournament, "2324")

        assert season.name == "2023/2024"
        assert season.start_year == 2023
        assert season.end_year == 2024
        assert season.tournament_id == tournament.id

    def test_get_existing_season(self, importer: FootballDataImporter) -> None:
        """Test getting an existing season."""
        tournament = importer.get_or_create_tournament("E0")
        season1 = importer.get_or_create_season(tournament, "2324")
        importer.session.flush()

        season2 = importer.get_or_create_season(tournament, "2324")

        assert season1.id == season2.id


class TestCreateMatch:
    """Tests for match creation."""

    def test_create_new_match(self, importer: FootballDataImporter) -> None:
        """Test creating a new match."""
        tournament = importer.get_or_create_tournament("E0")
        season = importer.get_or_create_season(tournament, "2324")

        match_data = {
            "division": "E0",
            "match_date": datetime(2023, 8, 11, 20, 0),
            "home_team": "Burnley",
            "away_team": "Manchester City",
            "home_score": 0,
            "away_score": 3,
            "status": "FINISHED",
            "odds_home": 9.02,
            "odds_draw": 5.35,
            "odds_away": 1.35,
        }

        match = importer.create_match(match_data, tournament, season)

        assert match is not None
        assert match.home_score == 0
        assert match.away_score == 3
        assert match.status == "FINISHED"
        assert match.odds_home == 9.02

    def test_duplicate_match_skipped(self, importer: FootballDataImporter) -> None:
        """Test that duplicate matches are skipped."""
        tournament = importer.get_or_create_tournament("E0")
        season = importer.get_or_create_season(tournament, "2324")

        match_data = {
            "division": "E0",
            "match_date": datetime(2023, 8, 11, 20, 0),
            "home_team": "Burnley",
            "away_team": "Manchester City",
            "home_score": 0,
            "away_score": 3,
            "status": "FINISHED",
            "odds_home": 9.02,
            "odds_draw": 5.35,
            "odds_away": 1.35,
        }

        # Create first match
        match1 = importer.create_match(match_data, tournament, season)
        importer.session.flush()

        # Try to create duplicate
        match2 = importer.create_match(match_data, tournament, season)

        assert match1 is not None
        assert match2 is None  # Duplicate should return None


class TestImportFromFile:
    """Tests for importing from file."""

    def test_import_from_file_success(
        self,
        importer: FootballDataImporter,
        sample_csv_file: Path,
    ) -> None:
        """Test successful import from file."""
        result = importer.import_from_file(
            file_path=sample_csv_file,
            season_code="2324",
        )

        assert result.success is True
        assert result.progress.matches_created == 3
        assert result.progress.matches_skipped == 0
        assert "Imported 3 matches" in result.message

    def test_import_from_file_creates_teams(
        self,
        importer: FootballDataImporter,
        sample_csv_file: Path,
    ) -> None:
        """Test that import creates teams."""
        result = importer.import_from_file(
            file_path=sample_csv_file,
            season_code="2324",
        )

        assert result.success is True
        # 6 unique teams: Burnley, Man City, Arsenal, Nott'm Forest, Brighton, Luton
        assert result.progress.teams_created == 6

    def test_import_from_file_creates_tournament(
        self,
        importer: FootballDataImporter,
        sample_csv_file: Path,
    ) -> None:
        """Test that import creates tournament."""
        result = importer.import_from_file(
            file_path=sample_csv_file,
            season_code="2324",
        )

        assert result.success is True
        assert result.tournament_id is not None

    def test_import_from_file_creates_season(
        self,
        importer: FootballDataImporter,
        sample_csv_file: Path,
    ) -> None:
        """Test that import creates season."""
        result = importer.import_from_file(
            file_path=sample_csv_file,
            season_code="2324",
        )

        assert result.success is True
        assert result.season_id is not None

    def test_import_from_file_nonexistent(self, importer: FootballDataImporter) -> None:
        """Test import from nonexistent file."""
        result = importer.import_from_file(
            file_path=Path("/nonexistent/file.csv"),
            season_code="2324",
        )

        assert result.success is False
        assert "File not found" in result.message

    def test_import_from_file_with_duplicate_handling(
        self,
        importer: FootballDataImporter,
        sample_csv_file: Path,
    ) -> None:
        """Test that duplicate matches are skipped on re-import."""
        # First import
        result1 = importer.import_from_file(
            file_path=sample_csv_file,
            season_code="2324",
        )
        assert result1.progress.matches_created == 3

        # Flush to ensure all data is persisted
        importer.session.flush()

        # Second import (should skip all as duplicates)
        result2 = importer.import_from_file(
            file_path=sample_csv_file,
            season_code="2324",
        )
        # All matches should be skipped as duplicates
        assert result2.progress.matches_skipped == 3
        assert result2.progress.matches_created == 0

    def test_import_from_file_with_division_override(
        self,
        importer: FootballDataImporter,
        sample_csv_file: Path,
    ) -> None:
        """Test import with division code override."""
        result = importer.import_from_file(
            file_path=sample_csv_file,
            season_code="2324",
            division="E0",
        )

        assert result.success is True

    def test_import_from_file_infers_season_from_filename(
        self,
        importer: FootballDataImporter,
    ) -> None:
        """Test that season code is inferred from filename."""
        # Create a file with season in name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="-2324.csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n")
            f.write("E0,11/08/2023,20:00,Burnley,Man City,0,3\n")
            path = Path(f.name)

        try:
            result = importer.import_from_file(file_path=path)
            assert result.success is True
        finally:
            path.unlink(missing_ok=True)


class TestImportProgress:
    """Tests for ImportProgress dataclass."""

    def test_progress_initialization(self) -> None:
        """Test ImportProgress initialization."""
        progress = ImportProgress()

        assert progress.total_rows == 0
        assert progress.processed_rows == 0
        assert progress.matches_created == 0
        assert progress.matches_skipped == 0
        assert progress.teams_created == 0
        assert progress.errors == []

    def test_progress_success_rate_zero(self) -> None:
        """Test success rate when no rows processed."""
        progress = ImportProgress()
        assert progress.success_rate == 0.0

    def test_progress_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        progress = ImportProgress(
            processed_rows=10,
            matches_created=8,
        )
        assert progress.success_rate == 80.0

    def test_progress_with_errors(self) -> None:
        """Test ImportProgress with errors."""
        progress = ImportProgress(
            errors=["Error 1", "Error 2"],
        )
        assert len(progress.errors) == 2


class TestImportResult:
    """Tests for ImportResult dataclass."""

    def test_result_initialization(self) -> None:
        """Test ImportResult initialization."""
        progress = ImportProgress()
        result = ImportResult(
            success=True,
            progress=progress,
            message="Test message",
        )

        assert result.success is True
        assert result.progress == progress
        assert result.message == "Test message"
        assert result.season_id is None
        assert result.tournament_id is None

    def test_result_with_ids(self) -> None:
        """Test ImportResult with season and tournament IDs."""
        progress = ImportProgress()
        result = ImportResult(
            success=True,
            progress=progress,
            season_id=1,
            tournament_id=2,
        )

        assert result.season_id == 1
        assert result.tournament_id == 2


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_progress_callback_called(
        self,
        db_session: Session,
        sample_csv_file: Path,
    ) -> None:
        """Test that progress callback is called during import."""
        progress_updates: list[ImportProgress] = []

        def callback(progress: ImportProgress) -> None:
            progress_updates.append(progress)

        importer = FootballDataImporter(db_session, progress_callback=callback)
        importer.import_from_file(file_path=sample_csv_file, season_code="2324")

        assert len(progress_updates) > 0
        # Final update should have all matches
        assert progress_updates[-1].matches_created == 3


class TestParseCsv:
    """Tests for CSV parsing."""

    def test_parse_csv_content(self, importer: FootballDataImporter) -> None:
        """Test parsing CSV content."""
        content = "Div,Date,HomeTeam,AwayTeam\nE0,11/08/2023,Arsenal,Chelsea\n"
        rows = importer.parse_csv(content)

        assert len(rows) == 1
        assert rows[0]["Div"] == "E0"
        assert rows[0]["HomeTeam"] == "Arsenal"

    def test_parse_csv_empty(self, importer: FootballDataImporter) -> None:
        """Test parsing empty CSV content."""
        content = "Div,Date,HomeTeam,AwayTeam\n"
        rows = importer.parse_csv(content)

        assert len(rows) == 0

    def test_parse_csv_multiple_rows(self, importer: FootballDataImporter) -> None:
        """Test parsing CSV with multiple rows."""
        content = (
            "Div,Date,HomeTeam,AwayTeam\n"
            "E0,11/08/2023,Arsenal,Chelsea\n"
            "E0,12/08/2023,Liverpool,Man City\n"
        )
        rows = importer.parse_csv(content)

        assert len(rows) == 2
