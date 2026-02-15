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
        # Mock the query chain
        mock_query = MagicMock()
        mock_filtered_query = MagicMock()
        mock_limited_query = MagicMock()
        mock_tournament1 = MagicMock()
        mock_tournament1.id = 1
        mock_tournament1.name = "Premier League"
        mock_tournament1.url_slug = "premier-league"
        mock_tournament2 = MagicMock()
        mock_tournament2.id = 2
        mock_tournament2.name = "La Liga"
        mock_tournament2.url_slug = "la-liga"
        mock_limited_query.all.return_value = [mock_tournament1, mock_tournament2]

        # Mock the count query separately
        mock_count_query = MagicMock()
        mock_count_query.count.return_value = 0  # Seasons count for each tournament
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_filtered_query
        mock_filtered_query.order_by.return_value = mock_filtered_query
        mock_filtered_query.limit.return_value = mock_limited_query

        # Mock the seasons count query
        seasons_count_query = MagicMock()
        seasons_count_query.scalar.return_value = 0
        mock_session.query.return_value.filter.return_value = seasons_count_query

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
        # Mock the query chain
        mock_query = MagicMock()
        mock_filtered_query = MagicMock()
        mock_limited_query = MagicMock()
        mock_team1 = MagicMock()
        mock_team1.id = 1
        mock_team1.name = "Arsenal"
        mock_team2 = MagicMock()
        mock_team2.id = 2
        mock_team2.name = "Barcelona"
        mock_limited_query.all.return_value = [mock_team1, mock_team2]

        # Mock the matches count queries
        home_matches_query = MagicMock()
        home_matches_query.scalar.return_value = 10
        away_matches_query = MagicMock()
        away_matches_query.scalar.return_value = 10

        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_filtered_query
        mock_filtered_query.order_by.return_value = mock_filtered_query
        mock_filtered_query.limit.return_value = mock_limited_query

        # Mock the matches count queries for each team
        mock_session.query.return_value.filter.return_value = home_matches_query
        mock_session.query.return_value.filter.return_value = (
            away_matches_query  # This gets overridden
        )

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
        # Mock the query chain
        mock_query = MagicMock()
        mock_joined_query = MagicMock()
        mock_filtered_query = MagicMock()
        mock_ordered_query = MagicMock()
        mock_limited_query = MagicMock()
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

        mock_limited_query.all.return_value = [mock_match1, mock_match2]
        mock_joined_query.count.return_value = 2  # Total count

        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_joined_query
        mock_joined_query.join.return_value = mock_joined_query
        mock_joined_query.filter.return_value = mock_filtered_query
        mock_filtered_query.order_by.return_value = mock_ordered_query
        mock_ordered_query.limit.return_value = mock_limited_query

        service = QueryService(mock_session)

        from algobet.services.dto import MatchFilter

        filter_request = MatchFilter(limit=10)
        response = service.list_matches(filter_request)

        assert hasattr(response, "matches")
        assert len(response.matches) == 2
        assert response.total_count == 2
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

        # Mock tournament objects
        mock_tournament1 = MagicMock()
        mock_tournament1.id = 1
        mock_tournament1.name = "Premier League"
        mock_tournament1.url_slug = "premier-league"
        mock_tournament2 = MagicMock()
        mock_tournament2.id = 2
        mock_tournament2.name = "La Liga"
        mock_tournament2.url_slug = "la-liga"

        # Mock the execute result
        mock_result = AsyncMock()
        mock_scalars_result = AsyncMock()
        mock_scalars_result.all.return_value = [mock_tournament1, mock_tournament2]
        mock_result.scalars.return_value = mock_scalars_result

        # Mock the count query result
        mock_count_result = AsyncMock()
        mock_count_result.scalar.return_value = 0  # seasons count

        # Mock execute to return different results based on the query
        async def mock_execute(query):
            # If it's a count query, return count result
            if "func.count" in str(query):
                return mock_count_result
            else:
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

        # Mock team objects
        mock_team1 = MagicMock()
        mock_team1.id = 1
        mock_team1.name = "Arsenal"
        mock_team2 = MagicMock()
        mock_team2.id = 2
        mock_team2.name = "Barcelona"

        # Mock the execute result
        mock_result = AsyncMock()
        mock_scalars_result = AsyncMock()
        mock_scalars_result.all.return_value = [mock_team1, mock_team2]
        mock_result.scalars.return_value = mock_scalars_result

        # Mock the count query results
        mock_home_count_result = AsyncMock()
        mock_home_count_result.scalar.return_value = 10
        mock_away_count_result = AsyncMock()
        mock_away_count_result.scalar.return_value = 10

        # Mock execute to return different results based on the query
        async def mock_execute(query):
            # If it's a count query, return count result
            if "func.count" in str(query):
                if "home_team_id" in str(query):
                    return mock_home_count_result
                elif "away_team_id" in str(query):
                    return mock_away_count_result
                else:
                    return mock_home_count_result  # Default
            else:
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

        # Mock match objects
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

        # Mock the execute result
        mock_result = AsyncMock()
        mock_scalars_result = AsyncMock()
        mock_unique_result = AsyncMock()
        mock_unique_result.all.return_value = [mock_match1, mock_match2]
        mock_scalars_result.unique.return_value = mock_unique_result
        mock_result.scalars.return_value = mock_scalars_result

        # Mock the count query result
        mock_count_result = AsyncMock()
        mock_count_result.scalar.return_value = 2  # total count

        # Mock execute to return different results based on the query
        async def mock_execute(query):
            # If it's a count query, return count result
            if "func.count" in str(query):
                return mock_count_result
            else:
                return mock_result

        mock_session.execute = mock_execute

        service = AsyncQueryService(mock_session)

        from algobet.services.dto import MatchFilter

        filter_request = MatchFilter(limit=10)
        response = await service.list_matches(filter_request)

        assert hasattr(response, "matches")
        assert len(response.matches) == 2
        assert response.total_count == 2
        assert response.matches[0].home_team == "Arsenal"
        assert response.matches[1].home_team == "Barcelona"
