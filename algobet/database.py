"""Database connection and session management.

This module provides both synchronous and asynchronous database
connection and session management using SQLAlchemy.

Sync Usage:
    with session_scope() as session:
        matches = session.query(Match).all()

Async Usage:
    async with async_session_scope() as session:
        result = await session.execute(select(Match))
        matches = result.scalars().all()
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from algobet.models import Base

# Load environment variables
load_dotenv()


def get_db_url() -> str:
    """Get database URL from environment variables."""
    user = os.getenv("POSTGRES_USER", "algobet")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    db = os.getenv("POSTGRES_DB", "football")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_async_db_url() -> str:
    """Get async database URL from environment variables.

    Uses asyncpg driver for async connections.
    """
    user = os.getenv("POSTGRES_USER", "algobet")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    db = os.getenv("POSTGRES_DB", "football")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


def create_db_engine() -> Engine:
    """Create synchronous database engine."""
    return create_engine(get_db_url())


def create_async_db_engine() -> AsyncEngine:
    """Create asynchronous database engine."""
    return create_async_engine(
        get_async_db_url(),
        echo=False,
        pool_pre_ping=True,
    )


def init_db() -> None:
    """Initialize the database tables."""
    engine = create_db_engine()
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")


async def async_init_db() -> None:
    """Initialize the database tables asynchronously."""
    engine = create_async_db_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("Database tables created successfully.")


def get_session() -> Session:
    """Get a new synchronous database session."""
    engine = create_db_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


async def get_async_session() -> AsyncSession:
    """Get a new asynchronous database session."""
    engine = create_async_db_engine()
    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return AsyncSessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as session:
            session.add(obj)
            # Auto-commits on exit, rolls back on exception
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def async_session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async transactional scope around a series of operations.

    Usage:
        async with async_session_scope() as session:
            session.add(obj)
            # Auto-commits on exit, rolls back on exception
    """
    session = await get_async_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# Singleton async engine for application-wide use
_async_engine: AsyncEngine | None = None


def get_shared_async_engine() -> AsyncEngine:
    """Get the shared async engine, creating it if necessary.

    Use this for long-running applications to reuse connections.
    """
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_db_engine()
    return _async_engine


async def close_async_engine() -> None:
    """Close the shared async engine.

    Call this during application shutdown.
    """
    global _async_engine
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
