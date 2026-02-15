"""Unit tests for database service classes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from algobet.services.database_service import DatabaseService
from algobet.services.async_database_service import AsyncDatabaseService
from algobet.models import Tournament, Team, Match


class TestDatabaseService:
    """Test cases for the DatabaseService class."""
    
    def test_database_service_initialization(self):
        """Test DatabaseService initialization."""
        mock_session = MagicMock()
        service = DatabaseService(mock_session)
        
        assert service.session == mock_session
    
    def test_get_stats(self):
        """Test DatabaseService get_stats method."""
        mock_session = MagicMock()
        # Mock scalar return values
        mock_session.query.return_value.scalar.return_value = 5  # tournament count
        
        service = DatabaseService(mock_session)
        
        from algobet.services.dto import DatabaseStatsRequest
        request = DatabaseStatsRequest()
        stats = service.get_stats(request)
        
        assert hasattr(stats, 'tournaments_count')
        assert hasattr(stats, 'teams_count')
        assert hasattr(stats, 'matches_count')
    
    def test_init_db(self):
        """Test DatabaseService init_db method."""
        mock_session = MagicMock()
        service = DatabaseService(mock_session)
        
        from algobet.services.dto import DatabaseInitRequest
        request = DatabaseInitRequest(drop_existing=False)
        
        with patch('algobet.services.database_service.create_db_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_metadata = MagicMock()
            with patch('algobet.services.database_service.Base.metadata', mock_metadata):
                response = service.initialize(request)
                
                assert response.success is True
                assert hasattr(response, 'tables_created')
    
    def test_reset_db(self):
        """Test DatabaseService reset_db method."""
        mock_session = MagicMock()
        service = DatabaseService(mock_session)
        
        from algobet.services.dto import DatabaseResetRequest
        request = DatabaseResetRequest(confirm=True)
        
        with patch('algobet.services.database_service.create_db_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_metadata = MagicMock()
            with patch('algobet.services.database_service.Base.metadata', mock_metadata):
                response = service.reset(request)
                
                assert response.success is True
                assert hasattr(response, 'tables_dropped')
                assert hasattr(response, 'tables_created')


class TestAsyncDatabaseService:
    """Test cases for the AsyncDatabaseService class."""
    
    @pytest.mark.asyncio
    async def test_async_database_service_initialization(self):
        """Test AsyncDatabaseService initialization."""
        mock_session = AsyncMock(spec=AsyncSession)
        service = AsyncDatabaseService(mock_session)
        
        assert service.session == mock_session
    
    @pytest.mark.asyncio
    async def test_async_get_stats(self):
        """Test AsyncDatabaseService get_stats method."""
        mock_session = AsyncMock(spec=AsyncSession)
        # Mock execute return values
        mock_result = AsyncMock()
        mock_result.scalar.return_value = 5  # tournament count
        mock_session.execute.return_value = mock_result
        
        service = AsyncDatabaseService(mock_session)
        
        from algobet.services.dto import DatabaseStatsRequest
        request = DatabaseStatsRequest()
        stats = await service.get_stats(request)
        
        assert hasattr(stats, 'tournaments_count')
        assert hasattr(stats, 'teams_count')
        assert hasattr(stats, 'matches_count')
        # Verify execute was called
        assert mock_session.execute.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_async_init_db(self):
        """Test AsyncDatabaseService init_db method."""
        mock_session = AsyncMock(spec=AsyncSession)
        service = AsyncDatabaseService(mock_session)
        
        from algobet.services.dto import DatabaseInitRequest
        request = DatabaseInitRequest(drop_existing=False)
        
        with patch('algobet.services.async_database_service.create_async_db_engine') as mock_create_engine:
            mock_engine = AsyncMock()
            mock_create_engine.return_value = mock_engine
            
            # Create a proper async context manager mock
            mock_conn = AsyncMock()
            mock_conn.run_sync = AsyncMock()
            
            # Create an async context manager mock with proper __aenter__ and __aexit__
            mock_async_context_manager = AsyncMock()
            mock_async_context_manager.__aenter__.return_value = mock_conn
            mock_async_context_manager.__aexit__.return_value = None
            
            # Mock the begin method to return the async context manager
            mock_engine.begin.return_value = mock_async_context_manager
            
            response = await service.initialize(request)
            
            # Verify that the engine creation and metadata operations happened
            mock_create_engine.assert_called_once()
            assert response.success is True
            assert hasattr(response, 'tables_created')
    
    @pytest.mark.asyncio
    async def test_async_reset_db(self):
        """Test AsyncDatabaseService reset_db method."""
        mock_session = AsyncMock(spec=AsyncSession)
        service = AsyncDatabaseService(mock_session)
        
        from algobet.services.dto import DatabaseResetRequest
        request = DatabaseResetRequest(confirm=True)
        
        with patch('algobet.services.async_database_service.create_async_db_engine') as mock_create_engine:
            mock_engine = AsyncMock()
            mock_create_engine.return_value = mock_engine
            
            # Create a proper async context manager mock
            mock_conn = AsyncMock()
            mock_conn.run_sync = AsyncMock()
            
            # Create an async context manager mock with proper __aenter__ and __aexit__
            mock_async_context_manager = AsyncMock()
            mock_async_context_manager.__aenter__.return_value = mock_conn
            mock_async_context_manager.__aexit__.return_value = None
            
            # Mock the begin method to return the async context manager
            mock_engine.begin.return_value = mock_async_context_manager
            
            response = await service.reset(request)
            
            # Verify that the engine creation and metadata operations happened
            mock_create_engine.assert_called_once()
            assert response.success is True
            assert hasattr(response, 'tables_dropped')
            assert hasattr(response, 'tables_created')