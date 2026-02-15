"""Performance benchmarks for AlgoBet services."""

import time
import pytest
import asyncio
from unittest.mock import MagicMock
from algobet.services.database_service import DatabaseService
from algobet.services.async_database_service import AsyncDatabaseService
from algobet.services.query_service import QueryService
from algobet.services.async_query_service import AsyncQueryService
from algobet.services.model_management_service import ModelManagementService
from algobet.services.async_model_management_service import AsyncModelManagementService


class TestServicePerformance:
    """Performance benchmarks for service operations."""
    
    def test_database_service_get_stats_performance(self):
        """Benchmark DatabaseService.get_stats() performance."""
        mock_session = MagicMock()
        service = DatabaseService(mock_session)
        
        # Mock the query results to simulate realistic DB operations
        mock_count_obj = MagicMock()
        mock_count_obj.scalar.return_value = 1000  # Simulate 1000 records
        mock_session.query.return_value.filter.return_value.filter.return_value = mock_count_obj
        
        start_time = time.time()
        from algobet.services.dto import DatabaseStatsRequest
        request = DatabaseStatsRequest()
        stats = service.get_stats(request)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Assert that the operation completes within an acceptable time (e.g., 1 second)
        assert execution_time < 1.0
        assert hasattr(stats, 'tournaments_count')
        assert hasattr(stats, 'teams_count')
        assert hasattr(stats, 'matches_count')
    
    @pytest.mark.asyncio
    async def test_async_database_service_get_stats_performance(self):
        """Benchmark AsyncDatabaseService.get_stats() performance."""
        mock_session = MagicMock()
        # Mock scalar to return different counts for different calls
        side_effects = [100, 1000, 5000, 0, 0, 0, 0]  # various counts for different stats
        mock_session.execute.return_value.scalar.return_value = 1000  # Return 1000 for all scalars
        
        service = AsyncDatabaseService(mock_session)
        
        start_time = time.time()
        from algobet.services.dto import DatabaseStatsRequest
        request = DatabaseStatsRequest()
        stats = await service.get_stats(request)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Assert that the operation completes within an acceptable time (e.g., 1 second)
        assert execution_time < 1.0
        assert hasattr(stats, 'tournaments_count')
        assert hasattr(stats, 'teams_count')
        assert hasattr(stats, 'matches_count')
    
    def test_query_service_list_operations_performance(self):
        """Benchmark QueryService list operations performance."""
        mock_session = MagicMock()
        # Mock query results with 1000 items
        mock_items = [MagicMock() for _ in range(1000)]
        mock_session.query.return_value.all.return_value = mock_items
        mock_session.query.return_value.count.return_value = 1000  # Total count
        
        service = QueryService(mock_session)
        
        start_time = time.time()
        from algobet.services.dto import TournamentFilter, TeamFilter
        tournament_filter = TournamentFilter(name=None, limit=1000)
        team_filter = TeamFilter(name=None, limit=1000)
        tournaments = service.list_tournaments(tournament_filter)
        teams = service.list_teams(team_filter)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert len(tournaments.tournaments) == 1000
        assert len(teams.teams) == 1000
        # Assert that the operation completes within an acceptable time
        assert execution_time < 2.0  # Allow more time for multiple operations
    
    @pytest.mark.asyncio
    async def test_async_query_service_list_operations_performance(self):
        """Benchmark AsyncQueryService list operations performance."""
        mock_session = MagicMock()
        # Mock query results with 1000 items
        mock_items = [MagicMock() for _ in range(1000)]
        mock_scalars_result = MagicMock()
        mock_scalars_result.all.return_value = mock_items
        mock_execute_result = MagicMock()
        mock_execute_result.scalars.return_value = mock_scalars_result
        mock_execute_result.count.return_value = 1000  # Total count
        mock_session.execute.return_value = mock_execute_result
        
        service = AsyncQueryService(mock_session)
        
        start_time = time.time()
        from algobet.services.dto import TournamentFilter, TeamFilter
        tournament_filter = TournamentFilter(name=None, limit=1000)
        team_filter = TeamFilter(name=None, limit=1000)
        tournaments = await service.list_tournaments(tournament_filter)
        teams = await service.list_teams(team_filter)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert len(tournaments.tournaments) == 1000
        assert len(teams.teams) == 1000
        # Assert that the operation completes within an acceptable time
        assert execution_time < 2.0  # Allow more time for multiple operations
    
    def test_model_management_service_list_performance(self):
        """Benchmark ModelManagementService list operations performance."""
        mock_session = MagicMock()
        # Mock query results with 1000 items
        mock_items = [MagicMock() for _ in range(100)]
        mock_session.query.return_value.order_by.return_value.all.return_value = mock_items
        mock_active_model = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_active_model
        
        service = ModelManagementService(mock_session)
        
        start_time = time.time()
        from algobet.services.dto import ModelListRequest
        request = ModelListRequest(include_inactive=True)
        models = service.list_models(request)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert len(models.models) == 100
        # Assert that the operation completes within an acceptable time
        assert execution_time < 1.0
    
    @pytest.mark.asyncio
    async def test_async_model_management_service_list_performance(self):
        """Benchmark AsyncModelManagementService list operations performance."""
        mock_session = MagicMock()
        # Mock query results with 1000 items
        mock_items = [MagicMock() for _ in range(100)]
        mock_query_result = MagicMock()
        mock_query_result.all.return_value = mock_items
        mock_session.query.return_value.order_by.return_value = mock_query_result
        mock_active_model = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_active_model
        
        service = AsyncModelManagementService(mock_session)
        
        start_time = time.time()
        from algobet.services.dto import ModelListRequest
        request = ModelListRequest(include_inactive=True)
        models = await service.list_models(request)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert len(models.models) == 100
        # Assert that the operation completes within an acceptable time
        assert execution_time < 1.0


def test_comparison_sync_vs_async_performance():
    """Compare performance between sync and async implementations."""
    # This is more of a conceptual test showing how performance could be compared
    # In a real scenario, we'd need actual database connections to make meaningful comparisons
    
    mock_session_sync = MagicMock()
    sync_service = DatabaseService(mock_session_sync)
    
    mock_session_async = MagicMock()
    async_service = AsyncDatabaseService(mock_session_async)
    
    # Mock the responses for fair comparison
    mock_session_sync.query.return_value.filter.return_value.filter.return_value.scalar.return_value = 1000
    mock_session_async.execute.return_value.scalar.return_value = 1000
    
    # Time sync operation
    sync_start = time.time()
    from algobet.services.dto import DatabaseStatsRequest
    request = DatabaseStatsRequest()
    sync_stats = sync_service.get_stats(request)
    sync_end = time.time()
    sync_time = sync_end - sync_start
    
    # Time async operation
    async def run_async_op():
        return await async_service.get_stats(request)
    
    async_start = time.time()
    async_stats = asyncio.run(run_async_op())
    async_end = time.time()
    async_time = async_end - async_start
    
    # Both should return similar results
    assert sync_stats.tournaments_count == async_stats.tournaments_count
    assert sync_stats.teams_count == async_stats.teams_count
    assert sync_stats.matches_count == async_stats.matches_count
    
    # Performance comparison would depend on actual implementation and load
    # For this test, just ensure both complete in reasonable time
    assert sync_time < 1.0
    assert async_time < 1.0