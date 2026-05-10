"""Unit tests for query service classes."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from algobet.services.async_query_service import AsyncQueryService
from algobet.services.query_service import QueryService


class TestQueryService:
    """Test cases for the QueryService class."""

    def test_query_service_initialization(self):
        """Test QueryService initialization."""
        mock_session = MagicMock()
        service = QueryService(mock_session)

        assert service.session == mock_session

    def test_list_tournaments(self):
        """Test QueryService list_tournaments method."""
        mock_session = MagicMock()
        mock_tournament1 = MagicMock()
        mock_tournament1.id = 1
        mock_tournament1.name = "Premier League"
        mock_tournament1.url_slug = "premier-league"
        mock_tournament2 = MagicMock()
        mock_tournament2.id = 2
        mock_tournament2.name = "La Liga"
        mock_tournament2.url_slug = "la-liga"

        mock_limited_query = MagicMock()
        mock_limited_query.all.return_value = [mock_tournament1, mock_tournament2]

        mock_query = MagicMock()
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_limited_query
        mock_count_query = MagicMock()
        mock_count_query.scalar.return_value = 0
        mock_query.filter.return_value = mock_count_query

        mock_session.query.return_value = mock_query

        service = QueryService(mock_session)

        from algobet.services.dto import TournamentFilter

        filter_request = TournamentFilter(name=None, limit=10)
        response = service.list_tournaments(filter_request)

        assert hasattr(response, "tournaments")
        assert len(response.tournaments) == 2
        assert response.tournaments[0].name == "Premier League"
        assert response.tournaments[1].name == "La Liga"

    def test_list_teams(self):
        """Test QueryService list_teams method."""
        mock_session = MagicMock()
        mock_team1 = MagicMock()
        mock_team1.id = 1
        mock_team1.name = "Arsenal"
        mock_team2 = MagicMock()
        mock_team2.id = 2
        mock_team2.name = "Barcelona"

        mock_limited_query = MagicMock()
        mock_limited_query.all.return_value = [mock_team1, mock_team2]

        mock_query = MagicMock()
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_limited_query
        mock_count_query = MagicMock()
        mock_count_query.scalar.return_value = 10
        mock_query.filter.return_value = mock_count_query

        mock_session.query.return_value = mock_query

        service = QueryService(mock_session)

        from algobet.services.dto import TeamFilter

        filter_request = TeamFilter(name=None, limit=10)
        response = service.list_teams(filter_request)

        assert hasattr(response, "teams")
        assert len(response.teams) == 2
        assert response.teams[0].name == "Arsenal"
        assert response.teams[1].name == "Barcelona"

    def test_list_matches(self):
        """Test QueryService list_matches method."""
        mock_session = MagicMock()
        mock_match1 = MagicMock()
        mock_match1.id = 1
        mock_match1.home_team = MagicMock()
        mock_match1.home_team.name = "Arsenal"
        mock_match1.away_team = MagicMock()
        mock_match1.away_team.name = "Chelsea"
        mock_match1.match_date = "2023-01-01"
        mock_match1.status = "SCHEDULED"
        mock_match1.home_score = None
        mock_match1.away_score = None
        mock_match1.tournament = MagicMock()
        mock_match1.tournament.name = "Premier League"
        mock_match1.season = MagicMock()
        mock_match1.season.name = "2023-2024"

        mock_match2 = MagicMock()
        mock_match2.id = 2
        mock_match2.home_team = MagicMock()
        mock_match2.home_team.name = "Barcelona"
        mock_match2.away_team = MagicMock()
        mock_match2.away_team.name = "Real Madrid"
        mock_match2.match_date = "2023-01-02"
        mock_match2.status = "FINISHED"
        mock_match2.home_score = 2
        mock_match2.away_score = 1
        mock_match2.tournament = MagicMock()
        mock_match2.tournament.name = "La Liga"
        mock_match2.season = MagicMock()
        mock_match2.season.name = "2023-2024"

        mock_joined_query = MagicMock()
        mock_joined_query.join.return_value = mock_joined_query
        mock_joined_query.filter.return_value = mock_joined_query
        mock_joined_query.order_by.return_value = mock_joined_query
        mock_joined_query.count.return_value = 2
        mock_ordered_query = MagicMock()
        mock_ordered_query.limit.return_value = mock_ordered_query
        mock_ordered_query.all.return_value = [mock_match1, mock_match2]
        mock_joined_query.order_by.return_value = mock_ordered_query

        mock_session.query.return_value = mock_joined_query

        service = QueryService(mock_session)

        from algobet.services.dto import MatchFilter

        filter_request = MatchFilter(limit=10)
        response = service.list_matches(filter_request)

        assert hasattr(response, "matches")
        assert len(response.matches) == 2
        assert response.matches[0].home_team == "Arsenal"
        assert response.matches[1].home_team == "Barcelona"


class TestAsyncQueryService:
    """Test cases for the AsyncQueryService class."""

    @pytest.mark.asyncio
    async def test_async_query_service_initialization(self):
        """Test AsyncQueryService initialization."""
        mock_session = AsyncMock()
        service = AsyncQueryService(mock_session)

        assert service.session == mock_session

    @pytest.mark.asyncio
    async def test_async_list_tournaments(self):
        """Test AsyncQueryService list_tournaments method."""
        mock_session = AsyncMock()

        mock_tournament1 = MagicMock()
        mock_tournament1.id = 1
        mock_tournament1.name = "Premier League"
        mock_tournament1.url_slug = "premier-league"
        mock_tournament2 = MagicMock()
        mock_tournament2.id = 2
        mock_tournament2.name = "La Liga"
        mock_tournament2.url_slug = "la-liga"

        mock_scalars_result = MagicMock()
        mock_scalars_result.all.return_value = [mock_tournament1, mock_tournament2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars_result
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        async def mock_execute(query):
            if "func.count" in str(query):
                return mock_count_result
            return mock_result

        mock_session.execute = mock_execute

        service = AsyncQueryService(mock_session)

        from algobet.services.dto import TournamentFilter

        filter_request = TournamentFilter(name=None, limit=10)
        response = await service.list_tournaments(filter_request)

        assert hasattr(response, "tournaments")
        assert len(response.tournaments) == 2
        assert response.tournaments[0].name == "Premier League"
        assert response.tournaments[1].name == "La Liga"

    @pytest.mark.asyncio
    async def test_async_list_teams(self):
        """Test AsyncQueryService list_teams method."""
        mock_session = AsyncMock()

        mock_team1 = MagicMock()
        mock_team1.id = 1
        mock_team1.name = "Arsenal"
        mock_team2 = MagicMock()
        mock_team2.id = 2
        mock_team2.name = "Barcelona"

        mock_scalars_result = MagicMock()
        mock_scalars_result.all.return_value = [mock_team1, mock_team2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars_result
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 10

        async def mock_execute(query):
            if "func.count" in str(query):
                return mock_count_result
            return mock_result

        mock_session.execute = mock_execute

        service = AsyncQueryService(mock_session)

        from algobet.services.dto import TeamFilter

        filter_request = TeamFilter(name=None, limit=10)
        response = await service.list_teams(filter_request)

        assert hasattr(response, "teams")
        assert len(response.teams) == 2
        assert response.teams[0].name == "Arsenal"
        assert response.teams[1].name == "Barcelona"

    @pytest.mark.asyncio
    async def test_async_list_matches(self):
        """Test AsyncQueryService list_matches method."""
        mock_session = AsyncMock()

        mock_match1 = MagicMock()
        mock_match1.id = 1
        mock_match1.home_team = MagicMock()
        mock_match1.home_team.name = "Arsenal"
        mock_match1.away_team = MagicMock()
        mock_match1.away_team.name = "Chelsea"
        mock_match1.match_date = "2023-01-01"
        mock_match1.status = "SCHEDULED"
        mock_match1.home_score = None
        mock_match1.away_score = None
        mock_match1.tournament = MagicMock()
        mock_match1.tournament.name = "Premier League"
        mock_match1.season = MagicMock()
        mock_match1.season.name = "2023-2024"

        mock_match2 = MagicMock()
        mock_match2.id = 2
        mock_match2.home_team = MagicMock()
        mock_match2.home_team.name = "Barcelona"
        mock_match2.away_team = MagicMock()
        mock_match2.away_team.name = "Real Madrid"
        mock_match2.match_date = "2023-01-02"
        mock_match2.status = "FINISHED"
        mock_match2.home_score = 2
        mock_match2.away_score = 1
        mock_match2.tournament = MagicMock()
        mock_match2.tournament.name = "La Liga"
        mock_match2.season = MagicMock()
        mock_match2.season.name = "2023-2024"

        mock_unique_result = MagicMock()
        mock_unique_result.all.return_value = [mock_match1, mock_match2]
        mock_scalars_result = MagicMock()
        mock_scalars_result.unique.return_value = mock_unique_result
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars_result
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 2

        async def mock_execute(query):
            if "func.count" in str(query):
                return mock_count_result
            return mock_result

        mock_session.execute = mock_execute

        service = AsyncQueryService(mock_session)

        from algobet.services.dto import MatchFilter

        filter_request = MatchFilter(limit=10)
        response = await service.list_matches(filter_request)

        assert hasattr(response, "matches")
        assert len(response.matches) == 2
        assert response.matches[0].home_team == "Arsenal"
        assert response.matches[1].home_team == "Barcelona"
