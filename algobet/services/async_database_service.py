"""Async database service for database management operations.

This module provides async business logic for database initialization,
statistics gathering, and reset operations.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from algobet.database import create_async_db_engine
from algobet.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    DatabaseQueryError,
    ValidationError,
)
from algobet.logging_config import get_logger
from algobet.models import (
    Base,
    Match,
    ModelVersion,
    ScheduledTask,
    Season,
    Team,
    Tournament,
)
from algobet.services.async_base import AsyncBaseService
from algobet.services.dto import (
    DatabaseInitRequest,
    DatabaseInitResponse,
    DatabaseResetRequest,
    DatabaseResetResponse,
    DatabaseStatsRequest,
    DatabaseStatsResponse,
)


class AsyncDatabaseService(AsyncBaseService[AsyncSession]):
    """Async service for database management operations.

    Provides methods for:
    - Getting database statistics
    - Initializing database schema
    - Resetting database (drop and recreate tables)

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
        self.logger = get_logger("services.async_database")

    async def get_stats(self, request: DatabaseStatsRequest) -> DatabaseStatsResponse:
        """Get database statistics asynchronously.

        Retrieves counts for all major entities in the database including
        tournaments, seasons, teams, matches (total, finished, scheduled),
        model versions, and scheduled tasks.

        Args:
            request: Request for database statistics (empty request)

        Returns:
            DatabaseStatsResponse with counts for all entities

        Raises:
            DatabaseConnectionError: If database is not accessible
            DatabaseQueryError: If query execution fails
        """
        self.logger.info(
            "Getting database statistics", extra={"operation": "get_stats"}
        )

        try:
            # Query counts for all entities
            tournaments_query = select(func.count(Tournament.id))
            t_result = await self.session.execute(tournaments_query)
            tournaments_count = t_result.scalar() or 0

            seasons_query = select(func.count(Season.id))
            s_result = await self.session.execute(seasons_query)
            seasons_count = s_result.scalar() or 0

            teams_query = select(func.count(Team.id))
            te_result = await self.session.execute(teams_query)
            teams_count = te_result.scalar() or 0

            # Match statistics with status breakdown
            matches_query = select(func.count(Match.id))
            m_result = await self.session.execute(matches_query)
            matches_count = m_result.scalar() or 0

            finished_query = select(func.count(Match.id)).where(
                Match.status == "FINISHED"
            )
            f_result = await self.session.execute(finished_query)
            finished_matches_count = f_result.scalar() or 0

            scheduled_query = select(func.count(Match.id)).where(
                Match.status == "SCHEDULED"
            )
            sc_result = await self.session.execute(scheduled_query)
            scheduled_matches_count = sc_result.scalar() or 0

            # Model versions and scheduled tasks
            mv_query = select(func.count(ModelVersion.id))
            mv_result = await self.session.execute(mv_query)
            model_versions_count = mv_result.scalar() or 0

            st_query = select(func.count(ScheduledTask.id))
            st_result = await self.session.execute(st_query)
            scheduled_tasks_count = st_result.scalar() or 0

            response = DatabaseStatsResponse(
                tournaments_count=tournaments_count,
                seasons_count=seasons_count,
                teams_count=teams_count,
                matches_count=matches_count,
                finished_matches_count=finished_matches_count,
                scheduled_matches_count=scheduled_matches_count,
                model_versions_count=model_versions_count,
                scheduled_tasks_count=scheduled_tasks_count,
            )

            self.logger.info(
                "Database statistics retrieved",
                extra={
                    "operation": "get_stats",
                    "tournaments": response.tournaments_count,
                    "matches": response.matches_count,
                },
            )

            return response

        except DatabaseQueryError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to retrieve database statistics",
                extra={"operation": "get_stats", "error": str(e)},
            )
            raise DatabaseQueryError(
                f"Failed to retrieve database statistics: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    async def initialize(self, request: DatabaseInitRequest) -> DatabaseInitResponse:
        """Initialize database schema asynchronously.

        Creates all tables defined in the SQLAlchemy metadata.
        If drop_existing is True, drops all tables first.

        Args:
            request: Request with drop_existing flag

        Returns:
            DatabaseInitResponse with success status and table count

        Raises:
            DatabaseConnectionError: If database is not accessible
            DatabaseError: If initialization fails
        """
        self.logger.info(
            "Initializing database",
            extra={"operation": "initialize", "drop_existing": request.drop_existing},
        )

        try:
            engine = create_async_db_engine()

            async with engine.begin() as conn:
                if request.drop_existing:
                    self.logger.info("Dropping existing tables")
                    await conn.run_sync(Base.metadata.drop_all)

                await conn.run_sync(Base.metadata.create_all)

            # Count the number of tables created
            tables_created = len(Base.metadata.tables)

            await engine.dispose()

            response = DatabaseInitResponse(
                success=True,
                tables_created=tables_created,
                message="Database initialized successfully",
            )

            self.logger.info(
                "Database initialized successfully",
                extra={"operation": "initialize", "tables_created": tables_created},
            )

            return response

        except DatabaseConnectionError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to initialize database",
                extra={"operation": "initialize", "error": str(e)},
            )
            raise DatabaseConnectionError(
                f"Failed to initialize database: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    async def reset(self, request: DatabaseResetRequest) -> DatabaseResetResponse:
        """Reset database asynchronously (drop and recreate all tables).

        WARNING: This destroys all data. Must be explicitly confirmed.

        Args:
            request: Request with confirm flag (must be True to proceed)

        Returns:
            DatabaseResetResponse with success status

        Raises:
            ValidationError: If confirm is False
            DatabaseConnectionError: If database is not accessible
            DatabaseError: If reset fails
        """
        self.logger.info(
            "Reset database requested",
            extra={"operation": "reset", "confirmed": request.confirm},
        )

        if not request.confirm:
            self.logger.warning(
                "Database reset aborted: confirmation required",
                extra={"operation": "reset"},
            )
            raise ValidationError(
                "Database reset requires confirmation. Set confirm=True to proceed.",
                details={"operation": "reset"},
            )

        try:
            engine = create_async_db_engine()

            # Count tables before dropping
            tables_dropped = len(Base.metadata.tables)

            async with engine.begin() as conn:
                # Drop all tables
                self.logger.info("Dropping all tables", extra={"operation": "reset"})
                await conn.run_sync(Base.metadata.drop_all)

                # Recreate all tables
                self.logger.info("Creating tables", extra={"operation": "reset"})
                await conn.run_sync(Base.metadata.create_all)

            # Count tables created
            tables_created = len(Base.metadata.tables)

            await engine.dispose()

            response = DatabaseResetResponse(
                success=True,
                tables_dropped=tables_dropped,
                tables_created=tables_created,
                message="Database reset successfully",
            )

            self.logger.info(
                "Database reset completed",
                extra={
                    "operation": "reset",
                    "tables_dropped": tables_dropped,
                    "tables_created": tables_created,
                },
            )

            return response

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to reset database",
                extra={"operation": "reset", "error": str(e)},
            )
            raise DatabaseError(
                f"Failed to reset database: {e}",
                details={"error_type": type(e).__name__},
            ) from e
