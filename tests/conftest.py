"""Test configuration and fixtures for AlgoBet API tests."""

# Patch JSONB to TEXT for SQLite compatibility BEFORE importing models
from sqlalchemy import TEXT
from sqlalchemy.dialects import postgresql  # type: ignore[assignment,misc]

# Monkey-patch JSONB for SQLite compatibility
postgresql.JSONB = TEXT  # type: ignore[assignment,misc]

import contextlib
import os
import tempfile
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, close_all_sessions, sessionmaker

from algobet.api.dependencies import get_db
from algobet.api.main import app
from algobet.models import Base


@pytest.fixture(scope="session")
def test_db_file() -> Generator[str, None, None]:
    """Create a temporary database file for the test session."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Clean up after all tests
    with contextlib.suppress(OSError):
        os.unlink(path)


@pytest.fixture(scope="session")
def test_engine(test_db_file: str) -> Generator[Engine, None, None]:
    """Create a test database engine for the test session."""
    # Use file-based SQLite for table persistence
    TEST_DATABASE_URL = f"sqlite:///{test_db_file}"
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        echo=False,
    )

    # Create tables immediately
    Base.metadata.create_all(bind=engine)

    yield engine

    Base.metadata.drop_all(bind=engine)
    close_all_sessions()


@pytest.fixture
def test_session(test_engine: Engine) -> Generator[Session, None, None]:
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
        session.rollback()  # Rollback after each test to keep tables clean
    finally:
        session.close()


@pytest.fixture
def test_client(test_session: Session) -> Generator[TestClient, None, None]:
    """Create a test client with database override."""

    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield test_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        # Clean up all test data by deleting from tables
        # Delete in reverse order of dependencies
        try:
            test_session.query(Base.metadata.tables["predictions"]).delete(
                synchronize_session=False
            )
            test_session.query(Base.metadata.tables["model_versions"]).delete(
                synchronize_session=False
            )
            test_session.query(Base.metadata.tables["matches"]).delete(
                synchronize_session=False
            )
            test_session.query(Base.metadata.tables["teams"]).delete(
                synchronize_session=False
            )
            test_session.query(Base.metadata.tables["seasons"]).delete(
                synchronize_session=False
            )
            test_session.query(Base.metadata.tables["tournaments"]).delete(
                synchronize_session=False
            )
            test_session.commit()
        except Exception:
            # If cleanup fails, just rollback
            test_session.rollback()
            raise
