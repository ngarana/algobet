"""Unit tests for soccerdata-based importer.

Tests cover:
- Team name resolution (teamname_replacements.json + TeamAlias)
- Score parsing
- Date parsing
- Tournament/Season lookup
- Team alias creation
- Import flow (mocked soccerdata)
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from algobet.importers.soccerdata_importer import (
    LEAGUE_MAPPING,
    ImportProgress,
    ImportResult,
    SoccerDataImporter,
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
def importer(db_session: Session) -> SoccerDataImporter:
    """Create a SoccerDataImporter instance for testing."""
    resolver = TeamResolver(config_path="/nonexistent/path/teamname_replacements.json")
    return SoccerDataImporter(db_session, resolver=resolver)


def _make_importer(
    db_session: Session, mappings: dict[str, list[str]]
) -> SoccerDataImporter:
    """Create an importer with custom team name mappings."""
    resolver = TeamResolver.__new__(TeamResolver)
    resolver.mappings = mappings
    return SoccerDataImporter(db_session, resolver=resolver)


class TestScoreParsing:
    """Tests for score string parsing."""

    def test_en_dash_separator(self, importer: SoccerDataImporter) -> None:
        home, away = importer._parse_score("3\u20131")
        assert home == 3
        assert away == 1

    def test_hyphen_separator(self, importer: SoccerDataImporter) -> None:
        home, away = importer._parse_score("0-0")
        assert home == 0
        assert away == 0

    def test_no_score(self, importer: SoccerDataImporter) -> None:
        home, away = importer._parse_score("")
        assert home is None
        assert away is None

    def test_none_score(self, importer: SoccerDataImporter) -> None:
        home, away = importer._parse_score(None)
        assert home is None
        assert away is None

    def test_large_score(self, importer: SoccerDataImporter) -> None:
        home, away = importer._parse_score("10\u20131")
        assert home == 10
        assert away == 1


class TestDateParsing:
    """Tests for date parsing."""

    def test_iso_date(self, importer: SoccerDataImporter) -> None:
        result = importer._parse_date("2024-05-06", "15:00")
        assert result == datetime(2024, 5, 6, 15, 0)

    def test_date_with_time_in_parens(self, importer: SoccerDataImporter) -> None:
        result = importer._parse_date("2024-05-06", "15:00 (16:00)")
        assert result == datetime(2024, 5, 6, 15, 0)

    def test_date_no_time(self, importer: SoccerDataImporter) -> None:
        result = importer._parse_date("2024-12-25")
        assert result == datetime(2024, 12, 25, 15, 0)


class TestTournamentLookup:
    """Tests for tournament get_or_create."""

    def test_create_new_tournament(self, importer: SoccerDataImporter) -> None:
        tournament = importer.get_or_create_tournament("ENG-Premier League")
        assert tournament is not None
        assert tournament.name == "Premier League"
        assert tournament.country == "England"
        assert tournament.url_slug == "england-premier-league"

    def test_return_existing_tournament(self, importer: SoccerDataImporter) -> None:
        first = importer.get_or_create_tournament("ESP-La Liga")
        second = importer.get_or_create_tournament("ESP-La Liga")
        assert first is not None
        assert second is not None
        assert first.id == second.id

    def test_same_slug_different_country_gets_country_slug(
        self, importer: SoccerDataImporter, db_session: Session
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

    def test_unknown_league(self, importer: SoccerDataImporter) -> None:
        result = importer.get_or_create_tournament("ZZZ-Unknown")
        assert result is None

    def test_all_league_codes_valid(self) -> None:
        for _code, info in LEAGUE_MAPPING.items():
            assert "name" in info
            assert "country" in info
            assert "url_slug" in info
            assert isinstance(info["name"], str)
            assert isinstance(info["country"], str)
            assert isinstance(info["url_slug"], str)


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
        team = imp.get_or_create_team("Man City", "fbref")
        assert team.name == "Manchester City"

    def test_get_or_create_team_creates_alias(self, db_session: Session) -> None:
        imp = _make_importer(
            db_session,
            {
                "Brighton": ["Brighton & Hove Albion"],
            },
        )
        team = imp.get_or_create_team("Brighton & Hove Albion", "fbref")
        assert team.name == "Brighton"

        alias = db_session.execute(
            select(TeamAlias).where(
                TeamAlias.team_id == team.id,
                TeamAlias.alias == "Brighton & Hove Albion",
                TeamAlias.source == "fbref",
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
        team = imp.get_or_create_team("Wolves", "fbref")
        assert team.name == "Wolves"

        found = imp.lookup_team("Wolverhampton", "fbref")
        assert found is not None
        assert found.id == team.id

    def test_lookup_team_not_found(self, db_session: Session) -> None:
        imp = _make_importer(db_session, {})
        found = imp.lookup_team("Nonexistent FC", "fbref")
        assert found is None

    def test_exact_name_match_without_alias(self, db_session: Session) -> None:
        team = Team(name="Exact Name FC")
        db_session.add(team)
        db_session.flush()

        imp = _make_importer(db_session, {})
        found = imp.lookup_team("Exact Name FC", "fbref")
        assert found is not None
        assert found.name == "Exact Name FC"


class TestSeasonCreation:
    """Tests for season get_or_create."""

    def test_create_season_from_4digit(self, importer: SoccerDataImporter) -> None:
        tournament = importer.get_or_create_tournament("ENG-Premier League")
        assert tournament is not None
        season = importer.get_or_create_season(tournament, "2024")
        assert season.name == "2024/2025"
        assert season.start_year == 2024
        assert season.end_year == 2025

    def test_create_season_from_dashed(self, importer: SoccerDataImporter) -> None:
        tournament = importer.get_or_create_tournament("GER-Bundesliga")
        assert tournament is not None
        season = importer.get_or_create_season(tournament, "23-24")
        assert season.name == "2023/2024"

    def test_season_deduplication(self, importer: SoccerDataImporter) -> None:
        tournament = importer.get_or_create_tournament("FRA-Ligue 1")
        assert tournament is not None
        s1 = importer.get_or_create_season(tournament, "2023")
        s2 = importer.get_or_create_season(tournament, "2023")
        assert s1.id == s2.id


class TestTeamAliasManagement:
    """Tests for _ensure_alias helper."""

    def test_creates_alias_when_different(
        self, importer: SoccerDataImporter, db_session: Session
    ) -> None:
        team = Team(name="Standard Name")
        db_session.add(team)
        db_session.flush()

        importer._ensure_alias(team, "SourceName", "fbref")

        alias = db_session.execute(
            select(TeamAlias).where(
                TeamAlias.team_id == team.id,
                TeamAlias.alias == "SourceName",
            )
        ).scalar_one_or_none()
        assert alias is not None
        assert alias.source == "fbref"

    def test_skips_when_same_as_team_name(
        self, importer: SoccerDataImporter, db_session: Session
    ) -> None:
        team = Team(name="Same Name")
        db_session.add(team)
        db_session.flush()

        importer._ensure_alias(team, "Same Name", "fbref")

        alias = db_session.execute(
            select(TeamAlias).where(
                TeamAlias.team_id == team.id,
            )
        ).scalar_one_or_none()
        assert alias is None

    def test_skips_duplicate_alias(
        self, importer: SoccerDataImporter, db_session: Session
    ) -> None:
        team = Team(name="Unique FC")
        db_session.add(team)
        db_session.flush()

        importer._ensure_alias(team, "Alias1", "fbref")
        importer._ensure_alias(team, "Alias1", "fbref")

        count = db_session.execute(
            select(TeamAlias).where(TeamAlias.team_id == team.id)
        ).all()
        assert len(count) == 1


class TestImportProgress:
    """Tests for ImportProgress dataclass."""

    def test_success_rate_empty(self) -> None:
        p = ImportProgress()
        assert p.success_rate == 0.0

    def test_success_rate(self) -> None:
        p = ImportProgress(
            total_rows=100,
            processed_rows=100,
            matches_created=80,
            matches_skipped=20,
        )
        assert p.success_rate == 80.0


class TestImportResult:
    """Tests for ImportResult dataclass."""

    def test_successful_result(self) -> None:
        result = ImportResult(
            success=True,
            progress=ImportProgress(matches_created=5),
            message="done",
            season_id=1,
            tournament_id=2,
        )
        assert result.success is True
        assert result.season_id == 1
        assert result.tournament_id == 2
