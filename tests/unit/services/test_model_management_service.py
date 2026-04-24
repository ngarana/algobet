"""Unit tests for model management service classes."""

from datetime import datetime
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

        model_1 = MagicMock()
        model_1.version = "v2.0.0"
        model_1.created_at = datetime(2026, 1, 2)
        model_1.metrics = {"accuracy": 0.78}
        model_1.is_active = False
        model_1.algorithm = "xgboost"
        model_1.hyperparameters = {"feature_names": ["f1", "f2"]}

        model_2 = MagicMock()
        model_2.version = "v1.0.0"
        model_2.created_at = datetime(2026, 1, 1)
        model_2.metrics = {"accuracy": 0.75}
        model_2.is_active = True
        model_2.algorithm = "lightgbm"
        model_2.hyperparameters = {"feature_names": ["f1"]}

        list_result = AsyncMock()
        list_scalars = AsyncMock()
        list_scalars.all.return_value = [model_1, model_2]
        list_result.scalars.return_value = list_scalars

        active_result = AsyncMock()
        active_result.scalar_one_or_none.return_value = model_2

        mock_session.execute.side_effect = [list_result, active_result]

        service = AsyncModelManagementService(mock_session)

        from algobet.services.dto import ModelListRequest

        request = ModelListRequest(include_inactive=True)
        response = await service.list_models(request)

        assert hasattr(response, "models")
        assert hasattr(response, "active_model_version")
        assert len(response.models) == 2
        assert response.active_model_version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_async_activate_model(self):
        """Test AsyncModelManagementService activate_model method."""
        mock_session = AsyncMock()

        mock_current_active = MagicMock()
        mock_current_active.version = "v0.9.0"
        mock_target_model = MagicMock()

        current_result = AsyncMock()
        current_result.scalar_one_or_none.return_value = mock_current_active

        target_result = AsyncMock()
        target_result.scalar_one_or_none.return_value = mock_target_model

        deactivate_result = AsyncMock()

        mock_session.execute.side_effect = [
            current_result,
            target_result,
            deactivate_result,
        ]

        service = AsyncModelManagementService(mock_session)

        from algobet.services.dto import ModelActivateRequest

        request = ModelActivateRequest(version="v1.0.0")
        response = await service.activate_model(request)

        assert hasattr(response, "success")
        assert response.success is True
        assert hasattr(response, "new_active_version")
        assert response.previous_active_version == "v0.9.0"

    @pytest.mark.asyncio
    async def test_async_get_model_info(self):
        """Test AsyncModelManagementService get_model_info method."""
        mock_session = AsyncMock()

        mock_model = MagicMock()
        mock_model.version = "v1.0.0"
        mock_model.created_at = datetime(2026, 1, 1)
        mock_model.algorithm = "xgboost"
        mock_model.hyperparameters = {"feature_names": ["f1", "f2"], "max_depth": 6}
        mock_model.metrics = {"training_samples": 1000, "accuracy": 0.8}
        mock_model.is_active = True

        model_result = AsyncMock()
        model_result.scalar_one_or_none.return_value = mock_model
        mock_session.execute.return_value = model_result

        service = AsyncModelManagementService(mock_session)

        from algobet.services.dto import ModelInfoRequest

        request = ModelInfoRequest(version="v1.0.0")
        response = await service.get_model_info(request)

        assert hasattr(response, "version")
        assert hasattr(response, "metrics")
        assert hasattr(response, "hyperparameters")
