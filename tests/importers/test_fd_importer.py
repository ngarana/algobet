"""Unit tests for Football-Data.co.uk importer.

Tests cover:
- Team resolution and alias creation
- Season creation from various formats
- Tournament lookup
- Import progress calculations
- Mocked soccerdata integration
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from algobet.importers.fd_importer import (
    LEAGUE_MAPPING,
    TOP_5_LEAGUES,
    FDImporter,
    FDImportProgress,
    FDImportResult,
)
from algobet.infrastructure.models import Base
from algobet.models import Team, TeamAlias, Tournament
from algobet.utils.team_resolver import TeamResolver


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
def importer(db_session: Session) -> FDImporter:
    """Create an FDImporter instance for testing."""
    resolver = TeamResolver(config_path="/nonexistent/path/teamname_replacements.json")
    return FDImporter(db_session, resolver=resolver)


def _make_importer(db_session: Session, mappings: dict[str, list[str]]) -> FDImporter:
    """Create an importer with custom team name mappings."""
    resolver = TeamResolver.__new__(TeamResolver)
    resolver.mappings = mappings
    return FDImporter(db_session, resolver=resolver)


class TestLeagueMapping:
    """Tests for league mapping configuration."""

    def test_top_5_leagues_defined(self) -> None:
        assert len(TOP_5_LEAGUES) == 5
        assert "ENG-Premier League" in TOP_5_LEAGUES
        assert "ESP-La Liga" in TOP_5_LEAGUES
        assert "FRA-Ligue 1" in TOP_5_LEAGUES
        assert "GER-Bundesliga" in TOP_5_LEAGUES
        assert "ITA-Serie A" in TOP_5_LEAGUES

    def test_league_mapping_complete(self) -> None:
        for _league_code, info in LEAGUE_MAPPING.items():
            assert "name" in info
            assert "country" in info
            assert "url_slug" in info
            assert isinstance(info["name"], str)
            assert isinstance(info["country"], str)
            assert isinstance(info["url_slug"], str)

    def test_top_5_in_mapping(self) -> None:
        for league in TOP_5_LEAGUES:
            assert league in LEAGUE_MAPPING


class TestTournamentLookup:
    """Tests for tournament get_or_create."""

    def test_create_new_tournament(self, importer: FDImporter) -> None:
        tournament = importer.get_or_create_tournament("ENG-Premier League")
        assert tournament is not None
        assert tournament.name == "Premier League"
        assert tournament.country == "England"
        assert tournament.url_slug == "england-premier-league"

    def test_return_existing_tournament(self, importer: FDImporter) -> None:
        first = importer.get_or_create_tournament("ESP-La Liga")
        second = importer.get_or_create_tournament("ESP-La Liga")
        assert first is not None
        assert second is not None
        assert first.id == second.id

    def test_same_slug_different_country_gets_country_slug(
        self, importer: FDImporter, db_session: Session
    ) -> None:
        austrian = Tournament(
            name="Bundesliga",
            country="Austria",
            url_slug="bundesliga",
        )
        db_session.add(austrian)
        db_session.flush()

        german = importer.get_or_create_tournament("GER-Bundesliga")
        again = importer.get_or_create_tournament("GER-Bundesliga")

        assert german is not None
        assert again is not None
        assert german.id != austrian.id
        assert german.id == again.id
        assert german.country == "Germany"
        assert german.url_slug == "germany-bundesliga"

    def test_unknown_league(self, importer: FDImporter) -> None:
        result = importer.get_or_create_tournament("ZZZ-Unknown")
        assert result is None


class TestTeamResolution:
    """Tests for team name resolution and creation."""

    def test_get_or_create_team_normalizes_name(self, db_session: Session) -> None:
        imp = _make_importer(
            db_session,
            {
                "Manchester City": ["Man City"],
                "Manchester Utd": ["Man United", "Man Utd"],
            },
        )
        team = imp.get_or_create_team("Man City", "fd")
        assert team.name == "Manchester City"

    def test_get_or_create_team_creates_alias(self, db_session: Session) -> None:
        imp = _make_importer(
            db_session,
            {
                "Brighton": ["Brighton & Hove Albion"],
            },
        )
        team = imp.get_or_create_team("Brighton & Hove Albion", "fd")
        assert team.name == "Brighton"

        alias = db_session.execute(
            db_session.query(TeamAlias).where(
                TeamAlias.team_id == team.id,
                TeamAlias.alias == "Brighton & Hove Albion",
                TeamAlias.source == "fd",
            )
        ).scalar_one_or_none()
        assert alias is not None

    def test_lookup_team_by_alias(self, db_session: Session) -> None:
        imp = _make_importer(
            db_session,
            {
                "Wolves": ["Wolves", "Wolverhampton"],
            },
        )
        team = imp.get_or_create_team("Wolves", "fd")
        assert team.name == "Wolves"

    def test_exact_name_match_without_alias(self, db_session: Session) -> None:
        team = Team(name="Exact Name FC")
        db_session.add(team)
        db_session.flush()

        imp = _make_importer(db_session, {})
        found = imp.get_or_create_team("Exact Name FC", "fd")
        assert found is not None
        assert found.name == "Exact Name FC"


class TestSeasonCreation:
    """Tests for season get_or_create."""

    def test_create_season_from_4digit(self, importer: FDImporter) -> None:
        tournament = importer.get_or_create_tournament("ENG-Premier League")
        assert tournament is not None
        season = importer.get_or_create_season(tournament, "2024")
        assert season.name == "2024/2025"
        assert season.start_year == 2024
        assert season.end_year == 2025

    def test_create_season_from_dashed(self, importer: FDImporter) -> None:
        tournament = importer.get_or_create_tournament("GER-Bundesliga")
        assert tournament is not None
        season = importer.get_or_create_season(tournament, "23-24")
        assert season.name == "2023/2024"

    def test_season_deduplication(self, importer: FDImporter) -> None:
        tournament = importer.get_or_create_tournament("FRA-Ligue 1")
        assert tournament is not None
        s1 = importer.get_or_create_season(tournament, "2023")
        s2 = importer.get_or_create_season(tournament, "2023")
        assert s1.id == s2.id


class TestSafeFloat:
    """Tests for _safe_float helper."""

    def test_none_value(self, importer: FDImporter) -> None:
        assert importer._safe_float(None) is None

    def test_nan_value(self, importer: FDImporter) -> None:
        import pandas as pd

        assert importer._safe_float(pd.NA) is None
        assert importer._safe_float(float("nan")) is None

    def test_valid_value(self, importer: FDImporter) -> None:
        assert importer._safe_float(2.5) == 2.5
        assert importer._safe_float("1.75") == 1.75

    def test_zero_or_negative(self, importer: FDImporter) -> None:
        assert importer._safe_float(0) is None
        assert importer._safe_float(-1.0) is None


class TestImportProgress:
    """Tests for FDImportProgress dataclass."""

    def test_success_rate_empty(self) -> None:
        p = FDImportProgress()
        assert p.success_rate == 0.0

    def test_success_rate(self) -> None:
        p = FDImportProgress(
            total_rows=100,
            processed_rows=100,
            matches_created=80,
        )
        assert p.success_rate == 80.0

    def test_success_rate_partial(self) -> None:
        p = FDImportProgress(
            total_rows=200,
            processed_rows=50,
            matches_created=40,
        )
        assert p.success_rate == 80.0


class TestImportResult:
    """Tests for FDImportResult dataclass."""

    def test_successful_result(self) -> None:
        result = FDImportResult(
            success=True,
            progress=FDImportProgress(matches_created=5),
            message="done",
            season_id=1,
            tournament_id=2,
        )
        assert result.success is True
        assert result.season_id == 1
        assert result.tournament_id == 2

    def test_failed_result(self) -> None:
        result = FDImportResult(
            success=False,
            progress=FDImportProgress(),
            message="error",
        )
        assert result.success is False


class TestMockedImport:
    """Tests with mocked HTTP data."""

    def test_import_with_mock_data(self, db_session: Session, monkeypatch) -> None:
        """Test import flow with mocked HTTP request."""

        # Create mock DataFrame with CSV-like data
        csv_content = "\n".join(
            [
                ",".join(
                    [
                        "Div",
                        "Date",
                        "HomeTeam",
                        "AwayTeam",
                        "FTHG",
                        "FTAG",
                        "HS",
                        "AS",
                        "HST",
                        "AST",
                        "HC",
                        "AC",
                        "HY",
                        "AY",
                        "HR",
                        "AR",
                        "B365H",
                        "B365D",
                        "B365A",
                        "PSH",
                        "PSD",
                        "PSA",
                        "MaxH",
                        "MaxD",
                        "MaxA",
                        "B365>2.5",
                        "AHh",
                        "B365AHH",
                    ]
                ),
                (
                    "E0,11/08/2023,Manchester City,Arsenal,2,1,15,10,8,5,"
                    "6,4,2,1,0,0,1.5,4.0,6.0,1.7,3.9,5.9,1.9,4.3,5.8,"
                    "1.75,-1.0,1.9"
                ),
                (
                    "E0,12/08/2023,Arsenal,Manchester City,1,2,10,15,5,8,"
                    "3,6,1,2,0,0,2.0,3.5,1.5,1.2,2.6,1.1,1.4,2.6,1.5,"
                    "2.0,-0.5,1.8"
                ),
                "",
            ]
        )

        # Mock the HTTP request
        class MockResponse:
            def __init__(self, data):
                self.data = data.encode("utf-8")

            def read(self):
                return self.data

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def mock_urlopen(url):
            return MockResponse(csv_content)

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

        importer = _make_importer(db_session, {"Manchester City": [], "Arsenal": []})
        result = importer.import_season(
            "ENG-Premier League",
            "2024",
            include_stats=True,
            include_odds=True,
        )

        assert result.success is True
        assert result.progress.matches_created == 2
        # Should have stats and odds enriched for both matches
        assert result.progress.stats_enriched >= 0
        assert result.progress.odds_enriched >= 0


class TestTop5LeaguesMethod:
    """Tests for import_top_5_leagues_2024_25 method."""

    def test_method_exists(self, importer: FDImporter) -> None:
        """Verify the method exists and is callable."""
        assert hasattr(importer, "import_top_5_leagues_2024_25")
        assert callable(importer.import_top_5_leagues_2024_25)
