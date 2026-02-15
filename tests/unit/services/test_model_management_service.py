"""Unit tests for model management service classes."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from algobet.services.async_model_management_service import AsyncModelManagementService
from algobet.services.model_management_service import ModelManagementService


class TestModelManagementService:
    """Test cases for the ModelManagementService class."""

    def test_model_management_service_initialization(self):
        """Test ModelManagementService initialization."""
        mock_session = MagicMock()
        service = ModelManagementService(mock_session)

        assert service.session == mock_session

    def test_list_models(self):
        """Test ModelManagementService list_models method."""
        mock_session = MagicMock()
        mock_models = [MagicMock(), MagicMock()]
        mock_session.query.return_value.order_by.return_value.all.return_value = (
            mock_models
        )
        mock_active_model = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_active_model
        )

        service = ModelManagementService(mock_session)

        from algobet.services.dto import ModelListRequest

        request = ModelListRequest(include_inactive=True)
        response = service.list_models(request)

        assert hasattr(response, "models")
        assert hasattr(response, "active_model_version")
        assert len(response.models) == 2

    def test_activate_model(self):
        """Test ModelManagementService activate_model method."""
        mock_session = MagicMock()
        mock_current_active = MagicMock()
        mock_target_model = MagicMock()
        mock_session.query.return_value.filter.return_value.first.side_effect = [
            mock_current_active,
            mock_target_model,
        ]

        service = ModelManagementService(mock_session)

        from algobet.services.dto import ModelActivateRequest

        request = ModelActivateRequest(version="v1.0.0")
        response = service.activate_model(request)

        assert hasattr(response, "success")
        assert response.success is True
        assert hasattr(response, "new_active_version")

    def test_get_model_info(self):
        """Test ModelManagementService get_model_info method."""
        mock_session = MagicMock()
        mock_model = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_model
        )

        service = ModelManagementService(mock_session)

        from algobet.services.dto import ModelInfoRequest

        request = ModelInfoRequest(version="v1.0.0")
        response = service.get_model_info(request)

        assert hasattr(response, "version")
        assert hasattr(response, "metrics")
        assert hasattr(response, "hyperparameters")


class TestAsyncModelManagementService:
    """Test cases for the AsyncModelManagementService class."""

    @pytest.mark.asyncio
    async def test_async_model_management_service_initialization(self):
        """Test AsyncModelManagementService initialization."""
        mock_session = AsyncMock()
        service = AsyncModelManagementService(mock_session)

        assert service.session == mock_session

    @pytest.mark.asyncio
    async def test_async_list_models(self):
        """Test AsyncModelManagementService list_models method."""
        mock_session = AsyncMock()
        mock_models = [MagicMock(), MagicMock()]
        mock_query_result = AsyncMock()
        mock_query_result.all.return_value = mock_models
        mock_session.query.return_value.order_by.return_value = mock_query_result
        mock_active_model = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_active_model
        )

        service = AsyncModelManagementService(mock_session)

        from algobet.services.dto import ModelListRequest

        request = ModelListRequest(include_inactive=True)
        response = await service.list_models(request)

        assert hasattr(response, "models")
        assert hasattr(response, "active_model_version")
        assert len(response.models) == 2

    @pytest.mark.asyncio
    async def test_async_activate_model(self):
        """Test AsyncModelManagementService activate_model method."""
        mock_session = AsyncMock()
        mock_current_active = MagicMock()
        mock_target_model = MagicMock()
        mock_session.query.return_value.filter.return_value.first.side_effect = [
            mock_current_active,
            mock_target_model,
        ]

        service = AsyncModelManagementService(mock_session)

        from algobet.services.dto import ModelActivateRequest

        request = ModelActivateRequest(version="v1.0.0")
        response = await service.activate_model(request)

        assert hasattr(response, "success")
        assert response.success is True
        assert hasattr(response, "new_active_version")

    @pytest.mark.asyncio
    async def test_async_get_model_info(self):
        """Test AsyncModelManagementService get_model_info method."""
        mock_session = AsyncMock()
        mock_model = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_model
        )

        service = AsyncModelManagementService(mock_session)

        from algobet.services.dto import ModelInfoRequest

        request = ModelInfoRequest(version="v1.0.0")
        response = await service.get_model_info(request)

        assert hasattr(response, "version")
        assert hasattr(response, "metrics")
        assert hasattr(response, "hyperparameters")
