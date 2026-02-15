"""Unit tests for base service classes."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from algobet.services.base import BaseService
from algobet.services.async_base import AsyncBaseService


class TestBaseService:
    """Test cases for the BaseService class."""
    
    def test_base_service_initialization(self):
        """Test BaseService initialization with session."""
        mock_session = MagicMock()
        service = BaseService(mock_session)
        
        assert service.session == mock_session
    
    def test_base_service_commit(self):
        """Test BaseService commit method."""
        mock_session = MagicMock()
        service = BaseService(mock_session)
        
        service.commit()
        
        mock_session.commit.assert_called_once()
    
    def test_base_service_rollback(self):
        """Test BaseService rollback method."""
        mock_session = MagicMock()
        service = BaseService(mock_session)
        
        service.rollback()
        
        mock_session.rollback.assert_called_once()


class TestAsyncBaseService:
    """Test cases for the AsyncBaseService class."""
    
    @pytest.mark.asyncio
    async def test_async_base_service_initialization(self):
        """Test AsyncBaseService initialization with async session."""
        mock_session = AsyncMock()
        service = AsyncBaseService(mock_session)
        
        assert service.session == mock_session
    
    @pytest.mark.asyncio
    async def test_async_base_service_commit(self):
        """Test AsyncBaseService commit method."""
        mock_session = AsyncMock()
        service = AsyncBaseService(mock_session)
        
        await service.commit()
        
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_base_service_rollback(self):
        """Test AsyncBaseService rollback method."""
        mock_session = AsyncMock()
        service = AsyncBaseService(mock_session)
        
        await service.rollback()
        
        mock_session.rollback.assert_called_once()