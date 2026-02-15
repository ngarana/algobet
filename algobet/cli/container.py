"""Dependency Injection container for AlgoBet CLI.

This module provides a DI container using dependency-injector library
for managing service dependencies and lifecycle.

Usage:
    from algobet.cli.container import Container
    
    container = Container()
    container.init_resources()
    
    # Get services
    db_service = container.database_service()
    query_service = container.query_service()
"""

from __future__ import annotations

import logging
from typing import Iterator

from dependency_injector import containers, providers
from sqlalchemy.orm import Session, sessionmaker

from algobet.config import get_config
from algobet.database import create_db_engine, session_scope
from algobet.services.analysis_service import AnalysisService
from algobet.services.database_service import DatabaseService
from algobet.services.model_management_service import ModelManagementService
from algobet.services.prediction_service import PredictionService
from algobet.services.query_service import QueryService
from algobet.services.scheduler_service import SchedulerService
from algobet.services.scraping_service import ScrapingService

logger = logging.getLogger(__name__)


class Container(containers.DeclarativeContainer):
    """Main DI container for AlgoBet services.

    This container manages all service dependencies and provides
    factory providers for creating service instances with proper
    dependency injection.

    Configuration is loaded from environment variables via the
    algobet.config module.

    Example:
        >>> container = Container()
        >>> container.init_resources()
        >>> with container.session_factory() as session:
        ...     db_service = container.database_service(session=session)
        ...     stats = db_service.get_stats()
    """

    # Configuration provider
    config = providers.Configuration()

    # Wire configuration from environment
    wired_config = providers.Callable(lambda: get_config())

    # Database engine (singleton)
    engine = providers.Singleton(create_db_engine)

    # Session factory (singleton)
    session_factory = providers.Singleton(
        sessionmaker,
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    # Session provider - yields a session for use in context managers
    # This is a factory that creates new sessions
    @staticmethod
    def get_session() -> Iterator[Session]:
        """Get a database session using the session_scope context manager.

        This is a convenience method for getting sessions outside
        of the container's normal lifecycle management.

        Yields:
            Session: A database session that will be automatically closed.
        """
        with session_scope() as session:
            yield session

    # Service factories - each takes a session as a parameter
    # This allows services to be created with a session from the current context

    database_service = providers.Factory(
        DatabaseService,
    )

    query_service = providers.Factory(
        QueryService,
    )

    model_management_service = providers.Factory(
        ModelManagementService,
    )

    analysis_service = providers.Factory(
        AnalysisService,
    )

    prediction_service = providers.Factory(
        PredictionService,
    )

    scheduler_service = providers.Factory(
        SchedulerService,
    )

    scraping_service = providers.Factory(
        ScrapingService,
    )


# Global container instance
_container: Container | None = None


def get_container() -> Container:
    """Get the global DI container instance.

    Creates the container on first access and initializes resources.

    Returns:
        Container: The global DI container instance.
    """
    global _container
    if _container is None:
        _container = Container()
        _container.init_resources()
        logger.debug("DI container initialized")
    return _container


def reset_container() -> None:
    """Reset the global DI container.

    This is useful for testing or when configuration changes.
    """
    global _container
    if _container is not None:
        # Clean up resources if needed
        _container = None
        logger.debug("DI container reset")


class ServiceLocator:
    """Service locator for CLI commands.

    Provides a simple interface for CLI commands to access services
    without managing sessions directly. Uses the global container.

    Example:
        >>> with ServiceLocator.session() as session:
        ...     service = ServiceLocator.database_service(session)
        ...     stats = service.get_stats()
    """

    _container: Container | None = None

    @classmethod
    def get_container(cls) -> Container:
        """Get the DI container.

        Returns:
            Container: The DI container instance.
        """
        if cls._container is None:
            cls._container = get_container()
        return cls._container

    @classmethod
    def session(cls) -> Iterator[Session]:
        """Get a database session.

        Yields:
            Session: A database session.
        """
        with session_scope() as session:
            yield session

    @classmethod
    def database_service(cls, session: Session) -> DatabaseService:
        """Create a DatabaseService instance.

        Args:
            session: Database session.

        Returns:
            DatabaseService: A new service instance.
        """
        return DatabaseService(session)

    @classmethod
    def query_service(cls, session: Session) -> QueryService:
        """Create a QueryService instance.

        Args:
            session: Database session.

        Returns:
            QueryService: A new service instance.
        """
        return QueryService(session)

    @classmethod
    def model_management_service(cls, session: Session) -> ModelManagementService:
        """Create a ModelManagementService instance.

        Args:
            session: Database session.

        Returns:
            ModelManagementService: A new service instance.
        """
        return ModelManagementService(session)

    @classmethod
    def analysis_service(cls, session: Session) -> AnalysisService:
        """Create an AnalysisService instance.

        Args:
            session: Database session.

        Returns:
            AnalysisService: A new service instance.
        """
        return AnalysisService(session)

    @classmethod
    def prediction_service(cls, session: Session) -> PredictionService:
        """Create a PredictionService instance.

        Args:
            session: Database session.

        Returns:
            PredictionService: A new service instance.
        """
        return PredictionService(session)

    @classmethod
    def scheduler_service(cls, session: Session) -> SchedulerService:
        """Create a SchedulerService instance.

        Args:
            session: Database session.

        Returns:
            SchedulerService: A new service instance.
        """
        return SchedulerService(session)

    @classmethod
    def scraping_service(cls, session: Session) -> ScrapingService:
        """Create a ScrapingService instance.

        Args:
            session: Database session.

        Returns:
            ScrapingService: A new service instance.
        """
        return ScrapingService(session)