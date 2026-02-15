"""Async model management service for ML model version operations.

This module provides async business logic for managing ML model versions,
including listing, activating, and retrieving detailed model information.
"""

from __future__ import annotations

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from algobet.exceptions import (
    DatabaseQueryError,
    ModelError,
    ModelNotFoundError,
    NoActiveModelError,
)
from algobet.logging_config import get_logger
from algobet.models import ModelVersion
from algobet.services.async_base import AsyncBaseService
from algobet.services.dto import (
    ModelActivateRequest,
    ModelActivateResponse,
    ModelDetailResponse,
    ModelInfo,
    ModelInfoRequest,
    ModelListRequest,
    ModelListResponse,
)


class AsyncModelManagementService(AsyncBaseService[AsyncSession]):
    """Async service for managing ML model versions.

    Provides methods for:
    - Listing available model versions
    - Activating a specific model version
    - Getting detailed model information

    Attributes:
        session: SQLAlchemy async database session for queries
        logger: Logger instance for this service
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with an async database session.

        Args:
            session: SQLAlchemy async database session
        """
        super().__init__(session)
        self.logger = get_logger("services.async_model_management")

    async def list_models(self, request: ModelListRequest) -> ModelListResponse:
        """List available model versions asynchronously.

        Retrieves all model versions from the database, optionally filtering
        to only active models. Returns the list with the currently active
        model version identifier.

        Args:
            request: Request with include_inactive flag

        Returns:
            ModelListResponse with list of models and active version

        Raises:
            DatabaseQueryError: If query fails
        """
        self.logger.info(
            "Listing models",
            extra={
                "operation": "list_models",
                "include_inactive": request.include_inactive,
            },
        )

        try:
            # Build query
            query = select(ModelVersion)

            # Filter to active only if requested
            if not request.include_inactive:
                query = query.where(ModelVersion.is_active == True)

            # Order by created_at descending
            query = query.order_by(desc(ModelVersion.created_at))

            # Execute query
            result = await self.session.execute(query)
            models = result.scalars().all()

            # Get the currently active model version
            active_query = select(ModelVersion).where(ModelVersion.is_active == True)
            active_result = await self.session.execute(active_query)
            active_model = active_result.scalar_one_or_none()
            active_version = active_model.version if active_model else None

            # Convert to DTOs
            model_infos = [
                ModelInfo(
                    version=m.version,
                    created_at=m.created_at,
                    metrics=m.metrics or {},
                    is_active=m.is_active,
                    model_type=m.algorithm,
                    features_count=len(m.hyperparameters.get("feature_names", []))
                    if m.hyperparameters
                    else 0,
                )
                for m in models
            ]

            response = ModelListResponse(
                models=model_infos,
                active_model_version=active_version,
            )

            self.logger.info(
                "Models listed successfully",
                extra={
                    "operation": "list_models",
                    "count": len(model_infos),
                    "active_version": active_version,
                },
            )

            return response

        except ModelNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to list models",
                extra={"operation": "list_models", "error": str(e)},
            )
            raise DatabaseQueryError(
                f"Failed to list models: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    async def activate_model(
        self, request: ModelActivateRequest
    ) -> ModelActivateResponse:
        """Activate a specific model version asynchronously.

        Sets the specified model version as the active model by:
        1. Setting all models to is_active=False
        2. Setting the target model to is_active=True
        3. Committing the changes

        Args:
            request: Request with version to activate

        Returns:
            ModelActivateResponse with success status

        Raises:
            ModelNotFoundError: If model version doesn't exist
            DatabaseQueryError: If query fails
        """
        self.logger.info(
            "Activating model",
            extra={"operation": "activate_model", "version": request.version},
        )

        try:
            # Get the currently active model
            current_query = select(ModelVersion).where(ModelVersion.is_active == True)
            current_result = await self.session.execute(current_query)
            current_active = current_result.scalar_one_or_none()
            previous_version = current_active.version if current_active else None

            # Find the target model
            target_query = select(ModelVersion).where(
                ModelVersion.version == request.version
            )
            target_result = await self.session.execute(target_query)
            target_model = target_result.scalar_one_or_none()

            if not target_model:
                self.logger.warning(
                    "Model not found",
                    extra={"operation": "activate_model", "version": request.version},
                )
                raise ModelNotFoundError(
                    f"Model version '{request.version}' not found.",
                    details={"version": request.version},
                )

            # Deactivate all models
            deactivate_stmt = update(ModelVersion).values(is_active=False)
            await self.session.execute(deactivate_stmt)

            # Activate the target model
            target_model.is_active = True

            # Commit changes
            await self.commit()

            response = ModelActivateResponse(
                success=True,
                previous_active_version=previous_version,
                new_active_version=request.version,
                message=f"Model '{request.version}' activated successfully.",
            )

            self.logger.info(
                "Model activated successfully",
                extra={
                    "operation": "activate_model",
                    "version": request.version,
                    "previous_version": previous_version,
                },
            )

            return response

        except ModelNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to activate model",
                extra={
                    "operation": "activate_model",
                    "version": request.version,
                    "error": str(e),
                },
            )
            raise ModelError(
                f"Failed to activate model '{request.version}': {e}",
                details={"version": request.version, "error_type": type(e).__name__},
            ) from e

    async def get_model_info(self, request: ModelInfoRequest) -> ModelDetailResponse:
        """Get detailed information about a model asynchronously.

        Retrieves complete model metadata including version, algorithm,
        metrics, hyperparameters, and feature information.

        Args:
            request: Request with model version

        Returns:
            ModelDetailResponse with full model details

        Raises:
            ModelNotFoundError: If model doesn't exist
            DatabaseQueryError: If query fails
        """
        self.logger.info(
            "Getting model info",
            extra={"operation": "get_model_info", "version": request.version},
        )

        try:
            # Find the model by version
            query = select(ModelVersion).where(ModelVersion.version == request.version)
            result = await self.session.execute(query)
            model = result.scalar_one_or_none()

            if not model:
                self.logger.warning(
                    "Model not found",
                    extra={"operation": "get_model_info", "version": request.version},
                )
                raise ModelNotFoundError(
                    f"Model version '{request.version}' not found.",
                    details={"version": request.version},
                )

            # Extract feature names from hyperparameters if available
            feature_names: list[str] = []
            if model.hyperparameters and "feature_names" in model.hyperparameters:
                feature_names = model.hyperparameters.get("feature_names", [])

            # Extract training data size from metrics if available
            training_data_size = 0
            if model.metrics and "training_samples" in model.metrics:
                training_data_size = model.metrics.get("training_samples", 0)

            response = ModelDetailResponse(
                version=model.version,
                created_at=model.created_at,
                model_type=model.algorithm,
                features=feature_names,
                metrics=model.metrics or {},
                hyperparameters=model.hyperparameters or {},
                training_data_size=training_data_size,
                is_active=model.is_active,
            )

            self.logger.info(
                "Model info retrieved successfully",
                extra={
                    "operation": "get_model_info",
                    "version": request.version,
                    "is_active": model.is_active,
                },
            )

            return response

        except ModelNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to get model info",
                extra={
                    "operation": "get_model_info",
                    "version": request.version,
                    "error": str(e),
                },
            )
            raise DatabaseQueryError(
                f"Failed to get model info for '{request.version}': {e}",
                details={"version": request.version, "error_type": type(e).__name__},
            ) from e

    async def get_active_model(self) -> ModelVersion:
        """Get the currently active model asynchronously.

        Returns:
            The active ModelVersion instance

        Raises:
            NoActiveModelError: If no model is currently active
        """
        self.logger.info(
            "Getting active model",
            extra={"operation": "get_active_model"},
        )

        try:
            query = select(ModelVersion).where(ModelVersion.is_active == True)
            result = await self.session.execute(query)
            active_model = result.scalar_one_or_none()

            if not active_model:
                self.logger.warning(
                    "No active model found",
                    extra={"operation": "get_active_model"},
                )
                raise NoActiveModelError()

            self.logger.info(
                "Active model retrieved",
                extra={
                    "operation": "get_active_model",
                    "version": active_model.version,
                },
            )

            return active_model

        except NoActiveModelError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to get active model",
                extra={"operation": "get_active_model", "error": str(e)},
            )
            raise DatabaseQueryError(
                f"Failed to get active model: {e}",
                details={"error_type": type(e).__name__},
            ) from e
