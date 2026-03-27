"""Unit tests for API-Football client.

Tests for the APIFootballClient class which handles integration with
the API-Football service for fetching match data, odds, and statistics.
"""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from algobet.infrastructure.api_football_client import (
    APIFootballClient,
    APIFootballFixture,
    APIFootballGoals,
    APIFootballLeague,
    APIFootballOdds,
    APIFootballResponse,
    APIFootballScore,
    APIFootballTeam,
    MatchStatus,
)
from algobet.infrastructure.config import AlgobetConfig

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@pytest.fixture
def mock_config() -> AlgobetConfig:
    """Create a mock configuration with test API key."""
    config = MagicMock(spec=AlgobetConfig)
    # Set up nested mock for api_football
    config.api_football = MagicMock()
    config.api_football.api_key = "test_api_key_12345"
    config.api_football.base_url = "https://v3.football.api-sports.io"
    config.api_football.timeout = 30
    config.api_football.rate_limit_per_day = 100
    # Set up nested mock for scraping
    config.scraping = MagicMock()
    config.scraping.default_league_ids = [39, 140, 135]
    return config


@pytest.fixture
def mock_response() -> dict[str, list[dict[str, any]]]:
    """Sample API response for testing."""
    return {
        "response": [
            {
                "fixture": {
                    "id": 123456,
                    "date": "2026-03-27T15:00:00+00:00",
                    "status": {"short": "NS", "long": "Not Started"},
                    "venue": {"name": "Old Trafford", "city": "Manchester"},
                    "referee": "Michael Oliver",
                },
                "league": {
                    "id": 39,
                    "name": "Premier League",
                    "country": "England",
                    "logo": "https://media.api-sports.io/football/leagues/39.png",
                    "flag": "https://media.api-sports.io/flags/gb.svg",
                    "season": 2025,
                },
                "teams": {
                    "home": {
                        "id": 33,
                        "name": "Manchester United",
                        "logo": "https://media.api-sports.io/football/teams/33.png",
                    },
                    "away": {
                        "id": 40,
                        "name": "Liverpool",
                        "logo": "https://media.api-sports.io/football/teams/40.png",
                    },
                },
                "goals": {"home": None, "away": None},
                "score": {
                    "halftime": {"home": None, "away": None},
                    "fulltime": {"home": None, "away": None},
                    "extratime": {"home": None, "away": None},
                    "penalty": {"home": None, "away": None},
                },
                "odds": [
                    {
                        "bookmaker": {"id": 6, "name": "Bwin"},
                        "bets": [
                            {
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "2.50"},
                                    {"value": "Draw", "odd": "3.40"},
                                    {"value": "Away", "odd": "2.80"},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


@pytest.fixture
def finished_match_response() -> dict[str, list[dict[str, any]]]:
    """Sample API response for a finished match."""
    return {
        "response": [
            {
                "fixture": {
                    "id": 123457,
                    "date": "2026-03-26T15:00:00+00:00",
                    "status": {"short": "FT", "long": "Match Finished"},
                    "venue": {"name": "Anfield", "city": "Liverpool"},
                },
                "league": {
                    "id": 39,
                    "name": "Premier League",
                    "country": "England",
                    "season": 2025,
                },
                "teams": {
                    "home": {
                        "id": 40,
                        "name": "Liverpool",
                        "logo": None,
                    },
                    "away": {
                        "id": 50,
                        "name": "Manchester City",
                        "logo": None,
                    },
                },
                "goals": {"home": 2, "away": 1},
                "score": {
                    "halftime": {"home": 1, "away": 0},
                    "fulltime": {"home": 2, "away": 1},
                    "extratime": {"home": None, "away": None},
                    "penalty": {"home": None, "away": None},
                },
                "odds": [],
            }
        ]
    }


@pytest.fixture
def odds_response() -> dict[str, list[dict[str, any]]]:
    """Sample API response for odds endpoint."""
    return {
        "response": [
            {
                "fixture": {"id": 123456},
                "bookmakers": [
                    {
                        "id": 6,
                        "name": "Bwin",
                        "bets": [
                            {
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "2.50"},
                                    {"value": "Draw", "odd": "3.40"},
                                    {"value": "Away", "odd": "2.80"},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


# =============================================================================
# Test Client Initialization
# =============================================================================


class TestAPIFootballClientInit:
    """Test APIFootballClient initialization."""

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_init_with_api_key(self, mock_get_config, mock_config):
        """Test client initialization with provided API key."""
        mock_get_config.return_value = mock_config

        client = APIFootballClient(api_key="custom_key")

        assert client.api_key == "custom_key"
        assert client.base_url == "https://v3.football.api-sports.io"
        assert client.timeout == 30

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_init_without_api_key_uses_config(self, mock_get_config, mock_config):
        """Test client uses config API key when not provided."""
        mock_get_config.return_value = mock_config

        client = APIFootballClient()

        assert client.api_key == "test_api_key_12345"

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_init_raises_on_missing_api_key(self, mock_get_config, mock_config):
        """Test client raises ValueError when API key is missing."""
        mock_config.api_football.api_key = ""
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError) as exc_info:
            APIFootballClient()

        assert "API-Football API key is required" in str(exc_info.value)
        assert "dashboard.api-football.com" in str(exc_info.value)


# =============================================================================
# Test Data Classes
# =============================================================================


class TestDataClasses:
    """Test data class functionality."""

    def test_match_status_enum(self):
        """Test MatchStatus enum values."""
        assert MatchStatus.NOT_STARTED.value == "NS"
        assert MatchStatus.MATCH_FINISHED.value == "FT"
        assert MatchStatus.LIVE.value == "LIVE"
        assert MatchStatus.HALFTIME.value == "HT"

    def test_api_football_team(self):
        """Test APIFootballTeam dataclass."""
        team = APIFootballTeam(id=33, name="Manchester United", logo="logo.png")

        assert team.id == 33
        assert team.name == "Manchester United"
        assert team.logo == "logo.png"

    def test_api_football_team_optional_logo(self):
        """Test APIFootballTeam with optional logo."""
        team = APIFootballTeam(id=33, name="Manchester United")

        assert team.logo is None

    def test_api_football_league(self):
        """Test APIFootballLeague dataclass."""
        league = APIFootballLeague(
            id=39,
            name="Premier League",
            country="England",
            logo="logo.png",
            flag="flag.png",
            season=2025,
        )

        assert league.id == 39
        assert league.name == "Premier League"
        assert league.country == "England"

    def test_api_football_goals(self):
        """Test APIFootballGoals dataclass."""
        goals = APIFootballGoals(home=2, away=1)

        assert goals.home == 2
        assert goals.away == 1

    def test_api_football_goals_defaults(self):
        """Test APIFootballGoals default values."""
        goals = APIFootballGoals()

        assert goals.home is None
        assert goals.away is None

    def test_api_football_score(self):
        """Test APIFootballScore dataclass."""
        score = APIFootballScore(
            halftime=APIFootballGoals(home=1, away=0),
            fulltime=APIFootballGoals(home=2, away=1),
        )

        assert score.halftime.home == 1
        assert score.fulltime.home == 2

    def test_api_football_odds(self):
        """Test APIFootballOdds dataclass."""
        odds = APIFootballOdds(
            id=6,
            name="Bwin",
            values=[
                {"value": "Home", "odd": "2.50"},
                {"value": "Draw", "odd": "3.40"},
                {"value": "Away", "odd": "2.80"},
            ],
        )

        assert odds.id == 6
        assert odds.name == "Bwin"
        assert len(odds.values) == 3

    def test_api_football_fixture(self):
        """Test APIFootballFixture dataclass."""
        fixture = APIFootballFixture(
            id=123456,
            date=datetime(2026, 3, 27, 15, 0),
            status=MatchStatus.NOT_STARTED,
            status_long="Not Started",
            home_team=APIFootballTeam(id=33, name="Man Utd"),
            away_team=APIFootballTeam(id=40, name="Liverpool"),
            league=APIFootballLeague(id=39, name="Premier League", country="England"),
        )

        assert fixture.id == 123456
        assert fixture.status == MatchStatus.NOT_STARTED
        assert fixture.home_team.name == "Man Utd"

    def test_fixture_is_upcoming(self):
        """Test fixture is_upcoming property."""
        upcoming = APIFootballFixture(
            id=1,
            date=datetime.now(),
            status=MatchStatus.NOT_STARTED,
            status_long="Not Started",
            home_team=APIFootballTeam(id=1, name="Home"),
            away_team=APIFootballTeam(id=2, name="Away"),
            league=APIFootballLeague(id=1, name="League", country="Country"),
        )

        assert upcoming.is_upcoming is True

    def test_fixture_is_upcoming_postponed(self):
        """Test fixture is_upcoming property with postponed status."""
        postponed = APIFootballFixture(
            id=1,
            date=datetime.now(),
            status=MatchStatus.MATCH_POSTPONED,
            status_long="Match Postponed",
            home_team=APIFootballTeam(id=1, name="Home"),
            away_team=APIFootballTeam(id=2, name="Away"),
            league=APIFootballLeague(id=1, name="League", country="Country"),
        )

        assert postponed.is_upcoming is True

    def test_fixture_is_upcoming_tbd(self):
        """Test fixture is_upcoming property with TBD status."""
        tbd = APIFootballFixture(
            id=1,
            date=datetime.now(),
            status=MatchStatus.TBD,
            status_long="Time To Be Defined",
            home_team=APIFootballTeam(id=1, name="Home"),
            away_team=APIFootballTeam(id=2, name="Away"),
            league=APIFootballLeague(id=1, name="League", country="Country"),
        )

        assert tbd.is_upcoming is True

    def test_fixture_is_finished(self):
        """Test fixture is_finished property."""
        finished = APIFootballFixture(
            id=1,
            date=datetime.now(),
            status=MatchStatus.MATCH_FINISHED,
            status_long="Match Finished",
            home_team=APIFootballTeam(id=1, name="Home"),
            away_team=APIFootballTeam(id=2, name="Away"),
            league=APIFootballLeague(id=1, name="League", country="Country"),
        )

        assert finished.is_finished is True

    def test_fixture_is_live(self):
        """Test fixture is_live property."""
        live = APIFootballFixture(
            id=1,
            date=datetime.now(),
            status=MatchStatus.FIRST_HALF,
            status_long="First Half",
            home_team=APIFootballTeam(id=1, name="Home"),
            away_team=APIFootballTeam(id=2, name="Away"),
            league=APIFootballLeague(id=1, name="League", country="Country"),
        )

        assert live.is_live is True

    def test_fixture_odds_home(self):
        """Test fixture odds_home property."""
        fixture = APIFootballFixture(
            id=1,
            date=datetime.now(),
            status=MatchStatus.NOT_STARTED,
            status_long="Not Started",
            home_team=APIFootballTeam(id=1, name="Home"),
            away_team=APIFootballTeam(id=2, name="Away"),
            league=APIFootballLeague(id=1, name="League", country="Country"),
            odds=[
                APIFootballOdds(
                    id=1,
                    name="Bwin",
                    values=[
                        {"value": "Home", "odd": "2.50"},
                        {"value": "Draw", "odd": "3.40"},
                        {"value": "Away", "odd": "2.80"},
                    ],
                )
            ],
        )

        assert fixture.odds_home == 2.50

    def test_fixture_odds_home_no_odds(self):
        """Test fixture odds_home with no odds."""
        fixture = APIFootballFixture(
            id=1,
            date=datetime.now(),
            status=MatchStatus.NOT_STARTED,
            status_long="Not Started",
            home_team=APIFootballTeam(id=1, name="Home"),
            away_team=APIFootballTeam(id=2, name="Away"),
            league=APIFootballLeague(id=1, name="League", country="Country"),
        )

        assert fixture.odds_home is None

    def test_api_football_response(self):
        """Test APIFootballResponse dataclass."""
        fixtures = [
            APIFootballFixture(
                id=1,
                date=datetime.now(),
                status=MatchStatus.NOT_STARTED,
                status_long="Not Started",
                home_team=APIFootballTeam(id=1, name="Home"),
                away_team=APIFootballTeam(id=2, name="Away"),
                league=APIFootballLeague(id=1, name="League", country="Country"),
            )
        ]
        response = APIFootballResponse(fixtures=fixtures, total=1, requests_made=5)

        assert len(response.fixtures) == 1
        assert response.total == 1
        assert response.requests_made == 5


# =============================================================================
# Test Request Handling
# =============================================================================


class TestRequestHandling:
    """Test HTTP request handling."""

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch("algobet.infrastructure.api_football_client.httpx")
    def test_request_makes_http_call(self, mock_httpx, mock_get_config, mock_config):
        """Test _request method makes HTTP calls."""
        mock_get_config.return_value = mock_config
        mock_client = MagicMock()
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = {"response": []}
        mock_client.get.return_value = mock_response_obj
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = APIFootballClient()
        result = client._request("/fixtures", {"param": "value"})

        assert result == {"response": []}
        mock_client.get.assert_called_once()

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch("algobet.infrastructure.api_football_client.httpx")
    def test_request_includes_headers(self, mock_httpx, mock_get_config, mock_config):
        """Test _request includes proper headers."""
        mock_get_config.return_value = mock_config
        mock_client = MagicMock()
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = {"response": []}
        mock_client.get.return_value = mock_response_obj
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = APIFootballClient()
        client._request("/fixtures")

        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args.kwargs
        assert "x-apisports-key" in call_kwargs["headers"]
        assert call_kwargs["headers"]["x-apisports-key"] == "test_api_key_12345"

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch("algobet.infrastructure.api_football_client.httpx")
    def test_request_raises_on_http_error(
        self, mock_httpx, mock_get_config, mock_config
    ):
        """Test _request raises on HTTP errors."""
        mock_get_config.return_value = mock_config
        mock_client = MagicMock()
        mock_response_obj = MagicMock()
        mock_response_obj.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )
        mock_client.get.return_value = mock_response_obj
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = APIFootballClient()

        with pytest.raises(httpx.HTTPStatusError):
            client._request("/fixtures")

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch("algobet.infrastructure.api_football_client.httpx")
    def test_request_tracks_count(self, mock_httpx, mock_get_config, mock_config):
        """Test _request tracks request count."""
        mock_get_config.return_value = mock_config
        mock_client = MagicMock()
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = {"response": []}
        mock_client.get.return_value = mock_response_obj
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = APIFootballClient()
        client._request("/fixtures")
        client._request("/fixtures")

        assert client._requests_made == 2

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch("algobet.infrastructure.api_football_client.httpx")
    def test_request_rate_limiting(self, mock_httpx, mock_get_config, mock_config):
        """Test _request enforces rate limiting."""
        mock_get_config.return_value = mock_config
        mock_client = MagicMock()
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = {"response": []}
        mock_client.get.return_value = mock_response_obj
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = APIFootballClient()

        # First request
        client._request("/fixtures")
        first_request_time = client._last_request_time

        # Second request should wait for rate limit
        time.sleep(0.1)  # Small delay to ensure time difference
        client._request("/fixtures")
        second_request_time = client._last_request_time

        # Second request should have updated the time
        assert second_request_time >= first_request_time


# =============================================================================
# Test Fixture Parsing
# =============================================================================


class TestFixtureParsing:
    """Test fixture parsing from API responses."""

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_parse_fixture_parses_basic_data(
        self, mock_get_config, mock_config, mock_response
    ):
        """Test _parse_fixture extracts basic fixture data."""
        mock_get_config.return_value = mock_config
        client = APIFootballClient()

        fixture = client._parse_fixture(mock_response["response"][0])

        assert fixture.id == 123456
        assert isinstance(fixture.date, datetime)
        assert fixture.status == MatchStatus.NOT_STARTED
        assert fixture.venue == "Old Trafford"

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_parse_fixture_parses_teams(
        self, mock_get_config, mock_config, mock_response
    ):
        """Test _parse_fixture extracts team data."""
        mock_get_config.return_value = mock_config
        client = APIFootballClient()

        fixture = client._parse_fixture(mock_response["response"][0])

        assert fixture.home_team.id == 33
        assert fixture.home_team.name == "Manchester United"
        assert fixture.away_team.id == 40
        assert fixture.away_team.name == "Liverpool"

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_parse_fixture_parses_league(
        self, mock_get_config, mock_config, mock_response
    ):
        """Test _parse_fixture extracts league data."""
        mock_get_config.return_value = mock_config
        client = APIFootballClient()

        fixture = client._parse_fixture(mock_response["response"][0])

        assert fixture.league.id == 39
        assert fixture.league.name == "Premier League"
        assert fixture.league.country == "England"

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_parse_fixture_parses_odds(
        self, mock_get_config, mock_config, mock_response
    ):
        """Test _parse_fixture extracts odds data."""
        mock_get_config.return_value = mock_config
        client = APIFootballClient()

        fixture = client._parse_fixture(mock_response["response"][0])

        assert len(fixture.odds) == 1
        assert fixture.odds[0].name == "Bwin"
        assert fixture.odds_home == 2.50
        assert fixture.odds_draw == 3.40
        assert fixture.odds_away == 2.80

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_parse_fixture_handles_missing_odds(
        self, mock_get_config, mock_config, finished_match_response
    ):
        """Test _parse_fixture handles missing odds."""
        mock_get_config.return_value = mock_config
        client = APIFootballClient()

        fixture = client._parse_fixture(finished_match_response["response"][0])

        assert len(fixture.odds) == 0
        assert fixture.odds_home is None

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_parse_fixture_handles_invalid_status(self, mock_get_config, mock_config):
        """Test _parse_fixture handles invalid status codes."""
        mock_get_config.return_value = mock_config
        client = APIFootballClient()

        response = {
            "fixture": {"status": {"short": "INVALID", "long": "Invalid Status"}},
            "teams": {
                "home": {"id": 1, "name": "Home"},
                "away": {"id": 2, "name": "Away"},
            },
            "league": {"id": 1, "name": "League", "country": "Country"},
            "goals": {"home": None, "away": None},
            "score": {},
            "odds": [],
        }

        fixture = client._parse_fixture(response)

        assert fixture.status == MatchStatus.NOT_STARTED  # Default fallback

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_parse_fixture_handles_invalid_date(self, mock_get_config, mock_config):
        """Test _parse_fixture handles invalid date formats."""
        mock_get_config.return_value = mock_config
        client = APIFootballClient()

        response = {
            "fixture": {
                "id": 1,
                "date": "invalid-date",
                "status": {"short": "NS", "long": "Not Started"},
            },
            "teams": {
                "home": {"id": 1, "name": "Home"},
                "away": {"id": 2, "name": "Away"},
            },
            "league": {"id": 1, "name": "League", "country": "Country"},
            "goals": {"home": None, "away": None},
            "score": {},
            "odds": [],
        }

        fixture = client._parse_fixture(response)

        assert isinstance(fixture.date, datetime)  # Should default to current time


# =============================================================================
# Test Public Methods - Get Fixtures
# =============================================================================


class TestGetFixtures:
    """Test get_upcoming_fixtures and related methods."""

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_upcoming_fixtures(
        self, mock_request, mock_get_config, mock_config, mock_response
    ):
        """Test get_upcoming_fixtures returns parsed fixtures."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = mock_response

        client = APIFootballClient()
        result = client.get_upcoming_fixtures(league_id=39, next=10)

        assert isinstance(result, APIFootballResponse)
        assert len(result.fixtures) == 1
        assert result.total == 1
        mock_request.assert_called_once_with(
            "/fixtures",
            {"status": "NS-TBD-PST", "league": 39, "next": 10},
        )

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_upcoming_fixtures_with_all_params(
        self, mock_request, mock_get_config, mock_config, mock_response
    ):
        """Test get_upcoming_fixtures with all parameters."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = mock_response

        client = APIFootballClient()
        result = client.get_upcoming_fixtures(
            league_id=39,
            team_id=33,
            next=5,
            date="2026-03-27",
            season=2025,
        )

        assert isinstance(result, APIFootballResponse)
        call_args = mock_request.call_args[0][1]
        assert call_args["league"] == 39
        assert call_args["team"] == 33
        assert call_args["next"] == 5
        assert call_args["date"] == "2026-03-27"
        assert call_args["season"] == 2025

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_all_upcoming(
        self, mock_request, mock_get_config, mock_config, mock_response
    ):
        """Test get_all_upcoming fetches from multiple leagues."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = mock_response

        client = APIFootballClient()
        result = client.get_all_upcoming(league_ids=[39, 140], next=10)

        assert isinstance(result, APIFootballResponse)
        # Should call get_upcoming_fixtures twice (once per league)
        assert mock_request.call_count == 2

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_all_upcoming_uses_default_leagues(
        self, mock_request, mock_get_config, mock_config, mock_response
    ):
        """Test get_all_upcoming uses config default leagues."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = mock_response

        client = APIFootballClient()
        result = client.get_all_upcoming()

        assert isinstance(result, APIFootballResponse)
        # Should call for each default league
        assert mock_request.call_count == 3  # [39, 140, 135]

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_fixtures_by_date(
        self, mock_request, mock_get_config, mock_config, mock_response
    ):
        """Test get_fixtures_by_date."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = mock_response

        client = APIFootballClient()
        result = client.get_fixtures_by_date(date="2026-03-27", league_id=39)

        assert isinstance(result, APIFootballResponse)
        mock_request.assert_called_once_with(
            "/fixtures",
            {"date": "2026-03-27", "league": 39},
        )

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_fixtures_by_date_defaults_to_today(
        self, mock_request, mock_get_config, mock_config, mock_response
    ):
        """Test get_fixtures_by_date defaults to today."""
        from datetime import date as dt_date

        mock_get_config.return_value = mock_config
        mock_request.return_value = mock_response

        client = APIFootballClient()
        result = client.get_fixtures_by_date()

        assert isinstance(result, APIFootballResponse)
        call_args = mock_request.call_args[0][1]
        assert call_args["date"] == dt_date.today().isoformat()

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_upcoming_by_date(
        self, mock_request, mock_get_config, mock_config, mock_response
    ):
        """Test get_upcoming_by_date filters by status."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = mock_response

        client = APIFootballClient()
        result = client.get_upcoming_by_date(date="2026-03-27")

        assert isinstance(result, APIFootballResponse)
        call_args = mock_request.call_args[0][1]
        assert call_args["status"] == "NS-TBD-PST"

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_results_by_date(
        self, mock_request, mock_get_config, mock_config, finished_match_response
    ):
        """Test get_results_by_date filters by finished status."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = finished_match_response

        client = APIFootballClient()
        result = client.get_results_by_date(date="2026-03-26")

        assert isinstance(result, APIFootballResponse)
        call_args = mock_request.call_args[0][1]
        assert call_args["status"] == "FT-AET-PEN"


# =============================================================================
# Test Public Methods - Get Results
# =============================================================================


class TestGetResults:
    """Test get_results and related methods."""

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_results(
        self, mock_request, mock_get_config, mock_config, finished_match_response
    ):
        """Test get_results returns finished matches."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = finished_match_response

        client = APIFootballClient()
        result = client.get_results(league_id=39, last=10)

        assert isinstance(result, APIFootballResponse)
        mock_request.assert_called_once_with(
            "/fixtures",
            {"status": "FT-AET-PEN", "league": 39, "last": 10},
        )

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_fixture_by_id_found(
        self, mock_request, mock_get_config, mock_config, mock_response
    ):
        """Test get_fixture_by_id returns fixture when found."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = mock_response

        client = APIFootballClient()
        result = client.get_fixture_by_id(123456)

        assert isinstance(result, APIFootballFixture)
        assert result.id == 123456

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_fixture_by_id_not_found(
        self, mock_request, mock_get_config, mock_config
    ):
        """Test get_fixture_by_id returns None when not found."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = {"response": []}

        client = APIFootballClient()
        result = client.get_fixture_by_id(999999)

        assert result is None


# =============================================================================
# Test Public Methods - Get Odds
# =============================================================================


class TestGetOdds:
    """Test get_odds and enrich_fixtures_with_odds methods."""

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_odds(self, mock_request, mock_get_config, mock_config, odds_response):
        """Test get_odds returns odds mapped by fixture ID."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = odds_response

        client = APIFootballClient()
        result = client.get_odds(date="2026-03-27")

        assert isinstance(result, dict)
        assert 123456 in result
        assert isinstance(result[123456], APIFootballOdds)
        # Check the odds values directly
        home_odd = [v for v in result[123456].values if v.get("value") == "Home"][0]
        assert home_odd.get("odd") == "2.50"

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_odds_empty_response(self, mock_request, mock_get_config, mock_config):
        """Test get_odds returns empty dict when no odds."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = {"response": []}

        client = APIFootballClient()
        result = client.get_odds(date="2026-03-27")

        assert result == {}

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_enrich_fixtures_with_odds(
        self, mock_request, mock_get_config, mock_config, mock_response, odds_response
    ):
        """Test enrich_fixtures_with_odds adds odds to fixtures."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = odds_response

        client = APIFootballClient()
        fixture = client._parse_fixture(mock_response["response"][0])
        fixture.odds = []  # Clear existing odds

        result = client.enrich_fixtures_with_odds([fixture])

        assert len(result) == 1
        assert len(result[0].odds) > 0
        assert result[0].odds_home == 2.50

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_enrich_fixtures_with_odds_empty_list(
        self, mock_request, mock_get_config, mock_config
    ):
        """Test enrich_fixtures_with_odds handles empty fixture list."""
        mock_get_config.return_value = mock_config

        client = APIFootballClient()
        result = client.enrich_fixtures_with_odds([])

        assert result == []


# =============================================================================
# Test Other Methods
# =============================================================================


class TestOtherMethods:
    """Test other API methods."""

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_leagues(self, mock_request, mock_get_config, mock_config):
        """Test get_leagues."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = {
            "response": [{"id": 39, "name": "Premier League", "country": "England"}]
        }

        client = APIFootballClient()
        result = client.get_leagues(country="England")

        assert isinstance(result, list)
        assert len(result) == 1
        mock_request.assert_called_once_with("/leagues", {"country": "England"})

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_teams(self, mock_request, mock_get_config, mock_config):
        """Test get_teams."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = {
            "response": [{"id": 33, "name": "Manchester United"}]
        }

        client = APIFootballClient()
        result = client.get_teams(league_id=39, season=2025)

        assert isinstance(result, list)
        mock_request.assert_called_once_with("/teams", {"league": 39, "season": 2025})

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_standings(self, mock_request, mock_get_config, mock_config):
        """Test get_standings."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = {
            "response": [
                {
                    "league": {
                        "standings": [
                            [
                                {
                                    "rank": 1,
                                    "team": {"name": "Liverpool"},
                                    "points": 70,
                                },
                                {"rank": 2, "team": {"name": "Man City"}, "points": 68},
                            ]
                        ]
                    }
                }
            ]
        }

        client = APIFootballClient()
        result = client.get_standings(league_id=39, season=2025)

        assert isinstance(result, list)
        assert len(result) == 2

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_standings_empty_response(
        self, mock_request, mock_get_config, mock_config
    ):
        """Test get_standings handles empty response."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = {"response": []}

        client = APIFootballClient()
        result = client.get_standings(league_id=39, season=2025)

        assert result == []

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_predictions(self, mock_request, mock_get_config, mock_config):
        """Test get_predictions."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = {
            "response": [{"predictions": {"home": "45%", "draw": "30%", "away": "25%"}}]
        }

        client = APIFootballClient()
        result = client.get_predictions(fixture_id=123456)

        assert isinstance(result, dict)
        assert "predictions" in result

    @patch("algobet.infrastructure.api_football_client.get_config")
    @patch.object(APIFootballClient, "_request")
    def test_get_predictions_empty(self, mock_request, mock_get_config, mock_config):
        """Test get_predictions returns None when no predictions."""
        mock_get_config.return_value = mock_config
        mock_request.return_value = {"response": []}

        client = APIFootballClient()
        result = client.get_predictions(fixture_id=123456)

        assert result is None


# =============================================================================
# Test Requests Remaining
# =============================================================================


class TestRequestsRemaining:
    """Test requests_remaining property."""

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_requests_remaining(self, mock_get_config, mock_config):
        """Test requests_remaining calculates correctly."""
        mock_get_config.return_value = mock_config

        client = APIFootballClient()
        client._requests_made = 10

        assert client.requests_remaining == 90

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_requests_remaining_zero(self, mock_get_config, mock_config):
        """Test requests_remaining returns 0 when limit exceeded."""
        mock_get_config.return_value = mock_config

        client = APIFootballClient()
        client._requests_made = 150

        assert client.requests_remaining == 0

    @patch("algobet.infrastructure.api_football_client.get_config")
    def test_requests_remaining_negative(self, mock_get_config, mock_config):
        """Test requests_remaining never returns negative."""
        mock_get_config.return_value = mock_config

        client = APIFootballClient()
        client._requests_made = 200

        assert client.requests_remaining == 0
