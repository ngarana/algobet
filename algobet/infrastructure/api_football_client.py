"""API-Football client for fetching match data.

This module provides a client for the API-Football service (api-football.com),
which provides reliable JSON API access to football fixtures, results, odds,
and statistics without web scraping.

API Documentation: https://www.api-football.com/documentation-v3
Free Tier: 100 requests/day
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import httpx

from algobet.infrastructure.config import get_config


class MatchStatus(str, Enum):
    """Match status from API-Football."""

    NOT_STARTED = "NS"  # Not Started
    FIRST_HALF = "1H"  # First Half, Kick Off
    HALFTIME = "HT"  # Halftime
    SECOND_HALF = "2H"  # Second Half, Second Half Started
    EXTRA_TIME = "ET"  # Extra Time
    PENALTY = "P"  # Penalty In Progress
    MATCH_SUSPENDED = "SUSP"  # Match Suspended
    MATCH_INTERRUPTED = "INT"  # Match Interrupted
    MATCH_FINISHED = "FT"  # Match Finished
    MATCH_FINISHED_AFTER_ET = "AET"  # Match Finished After Extra Time
    MATCH_FINISHED_AFTER_PEN = "PEN"  # Match Finished After Penalty
    BREAK_TIME = "BT"  # Break Time (in Extra Time)
    MATCH_POSTPONED = "PST"  # Match Postponed
    MATCH_CANCELLED = "CANC"  # Match Cancelled
    MATCH_ABANDONED = "ABD"  # Match Abandoned
    TECHNICAL_LOSS = "AWD"  # Technical Loss
    WALKOVER = "WO"  # Walkover
    LIVE = "LIVE"  # In Progress
    TBD = "TBD"  # Time To Be Defined


@dataclass
class APIFootballTeam:
    """Team data from API-Football."""

    id: int
    name: str
    logo: str | None = None


@dataclass
class APIFootballLeague:
    """League data from API-Football."""

    id: int
    name: str
    country: str
    logo: str | None = None
    flag: str | None = None
    season: int | None = None


@dataclass
class APIFootballGoals:
    """Goals data from API-Football."""

    home: int | None = None
    away: int | None = None


@dataclass
class APIFootballScore:
    """Score data from API-Football."""

    halftime: APIFootballGoals = field(default_factory=APIFootballGoals)
    fulltime: APIFootballGoals = field(default_factory=APIFootballGoals)
    extratime: APIFootballGoals = field(default_factory=APIFootballGoals)
    penalty: APIFootballGoals = field(default_factory=APIFootballGoals)


@dataclass
class APIFootballOdds:
    """Odds data from API-Football."""

    id: int
    name: str  # Bookmaker name
    values: list[dict[str, Any]]  # [{value: "Home", odd: "1.50"}, ...]


@dataclass
class APIFootballFixture:
    """Fixture/match data from API-Football."""

    id: int
    date: datetime
    status: MatchStatus
    status_long: str
    home_team: APIFootballTeam
    away_team: APIFootballTeam
    league: APIFootballLeague
    goals: APIFootballGoals = field(default_factory=APIFootballGoals)
    score: APIFootballScore = field(default_factory=APIFootballScore)
    odds: list[APIFootballOdds] = field(default_factory=list)
    venue: str | None = None
    referee: str | None = None

    @property
    def is_upcoming(self) -> bool:
        """Check if match is upcoming."""
        return self.status in (
            MatchStatus.NOT_STARTED,
            MatchStatus.TBD,
            MatchStatus.POSTPONED,
        )

    @property
    def is_finished(self) -> bool:
        """Check if match is finished."""
        return self.status in (
            MatchStatus.MATCH_FINISHED,
            MatchStatus.MATCH_FINISHED_AFTER_ET,
            MatchStatus.MATCH_FINISHED_AFTER_PEN,
        )

    @property
    def is_live(self) -> bool:
        """Check if match is live."""
        return self.status in (
            MatchStatus.FIRST_HALF,
            MatchStatus.SECOND_HALF,
            MatchStatus.HALFTIME,
            MatchStatus.EXTRA_TIME,
            MatchStatus.PENALTY,
            MatchStatus.LIVE,
        )

    @property
    def odds_home(self) -> float | None:
        """Get home team odds (from first bookmaker)."""
        if self.odds and self.odds[0].values:
            for v in self.odds[0].values:
                if v.get("value") == "Home":
                    return float(v.get("odd", 0))
        return None

    @property
    def odds_draw(self) -> float | None:
        """Get draw odds (from first bookmaker)."""
        if self.odds and self.odds[0].values:
            for v in self.odds[0].values:
                if v.get("value") == "Draw":
                    return float(v.get("odd", 0))
        return None

    @property
    def odds_away(self) -> float | None:
        """Get away team odds (from first bookmaker)."""
        if self.odds and self.odds[0].values:
            for v in self.odds[0].values:
                if v.get("value") == "Away":
                    return float(v.get("odd", 0))
        return None


@dataclass
class APIFootballResponse:
    """Response wrapper from API-Football."""

    fixtures: list[APIFootballFixture]
    total: int
    requests_made: int = 1


class APIFootballClient:
    """Client for API-Football API.

    Provides methods to fetch fixtures, results, and odds data
    from the API-Football service.

    Example:
        client = APIFootballClient()
        upcoming = client.get_upcoming_fixtures(league_id=39, next=10)
        results = client.get_results(league_id=39, last=10)
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the API-Football client.

        Args:
            api_key: API key (defaults to config value)
        """
        config = get_config()
        self.api_key = api_key or config.api_football.api_key
        self.base_url = config.api_football.base_url
        self.timeout = config.api_football.timeout
        self._requests_made = 0
        self._last_request_time: float = 0

        if not self.api_key:
            raise ValueError(
                "API-Football API key is required. "
                "Set ALGOBET_API_FOOTBALL__API_KEY or pass api_key parameter. "
                "Get your free key at https://dashboard.api-football.com/register"
            )

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "x-apisports-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a request to the API-Football API.

        Args:
            endpoint: API endpoint (e.g., "/fixtures")
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            httpx.HTTPError: If request fails
        """
        # Rate limiting: ensure at least 1 second between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        url = f"{self.base_url}{endpoint}"
        self._requests_made += 1
        self._last_request_time = time.time()

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._get_headers(), params=params or {})
            response.raise_for_status()
            return response.json()

    def _parse_fixture(self, data: dict[str, Any]) -> APIFootballFixture:
        """Parse fixture data from API response.

        Args:
            data: Raw fixture data from API

        Returns:
            Parsed APIFootballFixture
        """
        fixture = data.get("fixture", {})
        teams = data.get("teams", {})
        goals = data.get("goals", {})
        score = data.get("score", {})
        league = data.get("league", {})
        odds_data = data.get("odds", [])

        # Parse status
        status_short = fixture.get("status", {}).get("short", "NS")
        try:
            status = MatchStatus(status_short)
        except ValueError:
            status = MatchStatus.NOT_STARTED

        # Parse odds
        odds = []
        for odd_group in odds_data:
            bookmaker = odd_group.get("bookmaker", {})
            for bet in odd_group.get("bets", []):
                if bet.get("name") == "Match Winner":
                    odds.append(
                        APIFootballOdds(
                            id=bookmaker.get("id", 0),
                            name=bookmaker.get("name", "Unknown"),
                            values=bet.get("values", []),
                        )
                    )

        # Parse date
        date_str = fixture.get("date", "")
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            date = datetime.now()

        return APIFootballFixture(
            id=fixture.get("id", 0),
            date=date,
            status=status,
            status_long=fixture.get("status", {}).get("long", "Not Started"),
            home_team=APIFootballTeam(
                id=teams.get("home", {}).get("id", 0),
                name=teams.get("home", {}).get("name", "Unknown"),
                logo=teams.get("home", {}).get("logo"),
            ),
            away_team=APIFootballTeam(
                id=teams.get("away", {}).get("id", 0),
                name=teams.get("away", {}).get("name", "Unknown"),
                logo=teams.get("away", {}).get("logo"),
            ),
            league=APIFootballLeague(
                id=league.get("id", 0),
                name=league.get("name", "Unknown"),
                country=league.get("country", "Unknown"),
                logo=league.get("logo"),
                flag=league.get("flag"),
                season=league.get("season"),
            ),
            goals=APIFootballGoals(
                home=goals.get("home"),
                away=goals.get("away"),
            ),
            score=APIFootballScore(
                halftime=APIFootballGoals(
                    home=score.get("halftime", {}).get("home"),
                    away=score.get("halftime", {}).get("away"),
                ),
                fulltime=APIFootballGoals(
                    home=score.get("fulltime", {}).get("home"),
                    away=score.get("fulltime", {}).get("away"),
                ),
            ),
            odds=odds,
            venue=fixture.get("venue", {}).get("name"),
            referee=fixture.get("referee"),
        )

    def get_upcoming_fixtures(
        self,
        league_id: int | None = None,
        team_id: int | None = None,
        next: int = 10,
        date: str | None = None,
        season: int | None = None,
    ) -> APIFootballResponse:
        """Get upcoming fixtures.

        Args:
            league_id: Filter by league ID
            team_id: Filter by team ID
            next: Number of upcoming fixtures to return
            date: Filter by date (YYYY-MM-DD)
            season: Filter by season year

        Returns:
            APIFootballResponse with upcoming fixtures
        """
        params: dict[str, Any] = {"status": "NS-TBD-PST"}
        if league_id:
            params["league"] = league_id
        if team_id:
            params["team"] = team_id
        if next:
            params["next"] = next
        if date:
            params["date"] = date
        if season:
            params["season"] = season

        data = self._request("/fixtures", params)
        fixtures = [self._parse_fixture(f) for f in data.get("response", [])]

        return APIFootballResponse(
            fixtures=fixtures,
            total=len(fixtures),
            requests_made=self._requests_made,
        )

    def get_all_upcoming(
        self,
        league_ids: list[int] | None = None,
        next: int = 10,
    ) -> APIFootballResponse:
        """Get upcoming fixtures for multiple leagues.

        Args:
            league_ids: List of league IDs (defaults to config defaults)
            next: Number of fixtures per league

        Returns:
            APIFootballResponse with all upcoming fixtures
        """
        config = get_config()
        if league_ids is None:
            league_ids = config.scraping.default_league_ids

        all_fixtures: list[APIFootballFixture] = []
        for league_id in league_ids:
            try:
                response = self.get_upcoming_fixtures(league_id=league_id, next=next)
                all_fixtures.extend(response.fixtures)
            except Exception as e:
                print(f"Warning: Failed to fetch fixtures for league {league_id}: {e}")
                continue

        # Sort by date
        all_fixtures.sort(key=lambda f: f.date)

        return APIFootballResponse(
            fixtures=all_fixtures,
            total=len(all_fixtures),
            requests_made=self._requests_made,
        )

    def get_fixtures_by_date(
        self,
        date: str | None = None,
        league_id: int | None = None,
        season: int | None = None,
    ) -> APIFootballResponse:
        """Get ALL fixtures for a specific date across all leagues.

        This is the equivalent of scraping all matches from OddsPortal's
        main page - it returns every football match scheduled for a given date.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
            league_id: Optional filter by specific league
            season: Optional filter by season year

        Returns:
            APIFootballResponse with all fixtures for that date

        Example:
            client = APIFootballClient()
            # Get all matches today
            response = client.get_fixtures_by_date()
            # Get all matches on a specific date
            response = client.get_fixtures_by_date(date="2026-03-25")
            # Get Premier League matches only
            response = client.get_fixtures_by_date(date="2026-03-25", league_id=39)
        """
        if date is None:
            from datetime import date as dt_date

            date = dt_date.today().isoformat()

        params: dict[str, Any] = {"date": date}
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season

        data = self._request("/fixtures", params)
        fixtures = [self._parse_fixture(f) for f in data.get("response", [])]

        # Sort by league then by time
        fixtures.sort(key=lambda f: (f.league.name, f.date))

        return APIFootballResponse(
            fixtures=fixtures,
            total=len(fixtures),
            requests_made=self._requests_made,
        )

    def get_upcoming_by_date(
        self,
        date: str | None = None,
        league_id: int | None = None,
    ) -> APIFootballResponse:
        """Get upcoming (not started) fixtures for a specific date.

        Similar to get_fixtures_by_date but only returns matches that
        haven't started yet.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
            league_id: Optional filter by specific league

        Returns:
            APIFootballResponse with upcoming fixtures for that date
        """
        if date is None:
            from datetime import date as dt_date

            date = dt_date.today().isoformat()

        params: dict[str, Any] = {
            "date": date,
            "status": "NS-TBD-PST",
        }
        if league_id:
            params["league"] = league_id

        data = self._request("/fixtures", params)
        fixtures = [self._parse_fixture(f) for f in data.get("response", [])]

        # Sort by league then by time
        fixtures.sort(key=lambda f: (f.league.name, f.date))

        return APIFootballResponse(
            fixtures=fixtures,
            total=len(fixtures),
            requests_made=self._requests_made,
        )

    def get_results_by_date(
        self,
        date: str | None = None,
        league_id: int | None = None,
    ) -> APIFootballResponse:
        """Get completed results for a specific date.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
            league_id: Optional filter by specific league

        Returns:
            APIFootballResponse with results for that date
        """
        if date is None:
            from datetime import date as dt_date

            date = dt_date.today().isoformat()

        params: dict[str, Any] = {
            "date": date,
            "status": "FT-AET-PEN",
        }
        if league_id:
            params["league"] = league_id

        data = self._request("/fixtures", params)
        fixtures = [self._parse_fixture(f) for f in data.get("response", [])]

        # Sort by league then by time
        fixtures.sort(key=lambda f: (f.league.name, f.date))

        return APIFootballResponse(
            fixtures=fixtures,
            total=len(fixtures),
            requests_made=self._requests_made,
        )

    def get_results(
        self,
        league_id: int | None = None,
        team_id: int | None = None,
        last: int = 10,
        date: str | None = None,
        season: int | None = None,
    ) -> APIFootballResponse:
        """Get match results.

        Args:
            league_id: Filter by league ID
            team_id: Filter by team ID
            last: Number of past results to return
            date: Filter by date (YYYY-MM-DD)
            season: Filter by season year

        Returns:
            APIFootballResponse with match results
        """
        params: dict[str, Any] = {"status": "FT-AET-PEN"}
        if league_id:
            params["league"] = league_id
        if team_id:
            params["team"] = team_id
        if last:
            params["last"] = last
        if date:
            params["date"] = date
        if season:
            params["season"] = season

        data = self._request("/fixtures", params)
        fixtures = [self._parse_fixture(f) for f in data.get("response", [])]

        return APIFootballResponse(
            fixtures=fixtures,
            total=len(fixtures),
            requests_made=self._requests_made,
        )

    def get_fixture_by_id(self, fixture_id: int) -> APIFootballFixture | None:
        """Get a specific fixture by ID.

        Args:
            fixture_id: Fixture ID

        Returns:
            APIFootballFixture or None if not found
        """
        data = self._request(f"/fixtures?id={fixture_id}")
        responses = data.get("response", [])
        if responses:
            return self._parse_fixture(responses[0])
        return None

    def get_leagues(
        self, country: str | None = None, season: int | None = None
    ) -> list[dict[str, Any]]:
        """Get available leagues.

        Args:
            country: Filter by country name
            season: Filter by season year

        Returns:
            List of league data
        """
        params: dict[str, Any] = {}
        if country:
            params["country"] = country
        if season:
            params["season"] = season

        data = self._request("/leagues", params)
        return data.get("response", [])

    def get_teams(self, league_id: int, season: int) -> list[dict[str, Any]]:
        """Get teams in a league.

        Args:
            league_id: League ID
            season: Season year

        Returns:
            List of team data
        """
        params = {"league": league_id, "season": season}
        data = self._request("/teams", params)
        return data.get("response", [])

    def get_standings(self, league_id: int, season: int) -> list[dict[str, Any]]:
        """Get league standings.

        Args:
            league_id: League ID
            season: Season year

        Returns:
            List of standings data
        """
        params = {"league": league_id, "season": season}
        data = self._request("/standings", params)
        response = data.get("response", [])
        if response:
            return response[0].get("league", {}).get("standings", [[]])[0]
        return []

    def get_predictions(self, fixture_id: int) -> dict[str, Any] | None:
        """Get match predictions.

        Args:
            fixture_id: Fixture ID

        Returns:
            Prediction data or None
        """
        data = self._request(f"/predictions?fixture={fixture_id}")
        responses = data.get("response", [])
        if responses:
            return responses[0]
        return None

    def get_odds(
        self,
        fixture_id: int | None = None,
        league_id: int | None = None,
        date: str | None = None,
        season: int | None = None,
    ) -> dict[int, APIFootballOdds]:
        """Get odds data for fixtures.

        Odds are NOT included in the fixtures endpoint response. This method
        fetches odds from the dedicated /odds endpoint and returns them mapped
        by fixture ID.

        Args:
            fixture_id: Filter by specific fixture ID
            league_id: Filter by league ID
            date: Filter by date (YYYY-MM-DD)
            season: Filter by season year

        Returns:
            Dict mapping fixture IDs to APIFootballOdds (first bookmaker's Match Winner)

        Example:
            odds_map = client.get_odds(date="2026-03-26")
            # odds_map[fixture_id] contains odds for that fixture
        """
        params: dict[str, Any] = {}
        if fixture_id:
            params["fixture"] = fixture_id
        if league_id:
            params["league"] = league_id
        if date:
            params["date"] = date
        if season:
            params["season"] = season

        data = self._request("/odds", params)
        response_list = data.get("response", [])

        odds_by_fixture: dict[int, APIFootballOdds] = {}
        for item in response_list:
            fixture_id = item.get("fixture", {}).get("id")
            if not fixture_id:
                continue

            bookmakers = item.get("bookmakers", [])
            if not bookmakers:
                continue

            first_bookmaker = bookmakers[0]
            for bet in first_bookmaker.get("bets", []):
                if bet.get("name") == "Match Winner":
                    odds_by_fixture[fixture_id] = APIFootballOdds(
                        id=first_bookmaker.get("id", 0),
                        name=first_bookmaker.get("name", "Unknown"),
                        values=bet.get("values", []),
                    )
                    break

        return odds_by_fixture

    def enrich_fixtures_with_odds(
        self,
        fixtures: list[APIFootballFixture],
        league_id: int | None = None,
    ) -> list[APIFootballFixture]:
        """Enrich fixture list with odds data from the dedicated odds endpoint.

        Since odds are NOT returned by the fixtures endpoint, this makes an
        additional API call to fetch odds and attaches them to fixtures.

        Args:
            fixtures: List of fixtures to enrich
            league_id: Optional league filter for odds request

        Returns:
            Same fixtures list with odds populated

        Note:
            This uses 1 additional API request. Consider batching by date
            rather than calling for each fixture individually.
        """
        if not fixtures:
            return fixtures

        dates = {f.date.strftime("%Y-%m-%d") for f in fixtures}
        all_odds: dict[int, APIFootballOdds] = {}

        for date_str in dates:
            try:
                odds_map = self.get_odds(date=date_str, league_id=league_id)
                all_odds.update(odds_map)
            except Exception:
                continue

        for fixture in fixtures:
            if fixture.id in all_odds:
                fixture.odds = [all_odds[fixture.id]]

        return fixtures

    @property
    def requests_remaining(self) -> int:
        """Get estimated remaining requests for today."""
        config = get_config()
        return max(0, config.api_football.rate_limit_per_day - self._requests_made)
