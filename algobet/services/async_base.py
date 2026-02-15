"""Async base service class with common patterns."""

from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class AsyncBaseService(ABC, Generic[T]):
    """Async base class for all services.

    Provides common functionality like session management and logging
    for async operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with an async database session.

        Args:
            session: SQLAlchemy async database session
        """
        self.session = session

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()

    async def refresh(self, obj: T) -> None:
        """Refresh an object from the database.

        Args:
            obj: The object to refresh.
        """
        await self.session.refresh(obj)

    async def flush(self) -> None:
        """Flush pending changes to the database."""
        await self.session.flush()
