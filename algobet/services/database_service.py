"""Database service for database management operations.

This module provides business logic for database initialization,
statistics gathering, and reset operations.
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from algobet.database import create_db_engine
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
from algobet.services.base import BaseService
from algobet.services.dto import (
    DatabaseInitRequest,
    DatabaseInitResponse,
    DatabaseResetRequest,
    DatabaseResetResponse,
    DatabaseStatsRequest,
    DatabaseStatsResponse,
)


class DatabaseService(BaseService[Session]):
    """Service for database management operations.

    Provides methods for:
    - Getting database statistics
    - Initializing database schema
    - Resetting database (drop and recreate tables)

    Attributes:
        session: SQLAlchemy database session for queries
        logger: Logger instance for this service
    """

    def __init__(self, session: Session) -> None:
        """Initialize the service with a database session.

        Args:
            session: SQLAlchemy database session
        """
        super().__init__(session)
        self.logger = get_logger("services.database")

    def get_stats(self, request: DatabaseStatsRequest) -> DatabaseStatsResponse:
        """Get database statistics.

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
            tournaments_count = self.session.query(func.count(Tournament.id)).scalar()
            seasons_count = self.session.query(func.count(Season.id)).scalar()
            teams_count = self.session.query(func.count(Team.id)).scalar()

            # Match statistics with status breakdown
            matches_count = self.session.query(func.count(Match.id)).scalar()
            finished_matches_count = (
                self.session.query(func.count(Match.id))
                .filter(Match.status == "FINISHED")
                .scalar()
            )
            scheduled_matches_count = (
                self.session.query(func.count(Match.id))
                .filter(Match.status == "SCHEDULED")
                .scalar()
            )

            # Model versions and scheduled tasks
            model_versions_count = self.session.query(
                func.count(ModelVersion.id)
            ).scalar()
            scheduled_tasks_count = self.session.query(
                func.count(ScheduledTask.id)
            ).scalar()

            response = DatabaseStatsResponse(
                tournaments_count=tournaments_count or 0,
                seasons_count=seasons_count or 0,
                teams_count=teams_count or 0,
                matches_count=matches_count or 0,
                finished_matches_count=finished_matches_count or 0,
                scheduled_matches_count=scheduled_matches_count or 0,
                model_versions_count=model_versions_count or 0,
                scheduled_tasks_count=scheduled_tasks_count or 0,
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

    def initialize(self, request: DatabaseInitRequest) -> DatabaseInitResponse:
        """Initialize database schema.

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
            engine = create_db_engine()

            if request.drop_existing:
                self.logger.info("Dropping existing tables")
                Base.metadata.drop_all(bind=engine)

            Base.metadata.create_all(bind=engine)

            # Count the number of tables created
            tables_created = len(Base.metadata.tables)

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

    def reset(self, request: DatabaseResetRequest) -> DatabaseResetResponse:
        """Reset database (drop and recreate all tables).

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
            engine = create_db_engine()

            # Count tables before dropping
            tables_dropped = len(Base.metadata.tables)

            # Drop all tables
            self.logger.info("Dropping all tables", extra={"operation": "reset"})
            Base.metadata.drop_all(bind=engine)

            # Recreate all tables
            self.logger.info("Creating tables", extra={"operation": "reset"})
            Base.metadata.create_all(bind=engine)

            # Count tables created
            tables_created = len(Base.metadata.tables)

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
