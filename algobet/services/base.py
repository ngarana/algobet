"""Base service class with common patterns."""

from abc import ABC
from typing import Generic, TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


class BaseService(ABC, Generic[T]):
    """Base class for all services.

    Provides common functionality like session management and logging.
    """

    def __init__(self, session: Session) -> None:
        """Initialize the service with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def commit(self) -> None:
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.session.rollback()
