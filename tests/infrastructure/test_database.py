"""Unit tests for database connection and session management.

Tests for the database infrastructure module which handles:
- Database URL configuration
- Engine creation (sync and async)
- Session management (sync and async)
- Transaction scopes (session_scope, async_session_scope)
- Shared engine singleton
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from algobet.infrastructure.database import (
    async_init_db,
    async_session_scope,
    close_async_engine,
    create_async_db_engine,
    create_db_engine,
    get_async_db_url,
    get_async_session,
    get_db_url,
    get_session,
    get_shared_async_engine,
    init_db,
    session_scope,
)
from algobet.infrastructure.models import Base

# =============================================================================
# Test Database URL Functions
# =============================================================================


class TestDatabaseURLFunctions:
    """Test database URL generation functions."""

    def teardown_method(self):
        """Clean up environment variables after each test."""
        # Remove any test environment variables
        for key in list(os.environ.keys()):
            if key.startswith("POSTGRES_"):
                del os.environ[key]

    def test_get_db_url_defaults(self):
        """Test get_db_url uses default values."""
        url = get_db_url()

        assert url == "postgresql+psycopg2://algobet:password@localhost:5432/football"

    def test_get_async_db_url_defaults(self):
        """Test get_async_db_url uses default values."""
        url = get_async_db_url()

        assert url == "postgresql+asyncpg://algobet:password@localhost:5432/football"

    def test_get_db_url_from_environment(self):
        """Test get_db_url reads from environment variables."""
        os.environ["POSTGRES_USER"] = "testuser"
        os.environ["POSTGRES_PASSWORD"] = "testpass"
        os.environ["POSTGRES_DB"] = "testdb"
        os.environ["POSTGRES_HOST"] = "testhost"
        os.environ["POSTGRES_PORT"] = "5433"

        url = get_db_url()

        assert url == "postgresql+psycopg2://testuser:testpass@testhost:5433/testdb"

    def test_get_async_db_url_from_environment(self):
        """Test get_async_db_url reads from environment variables."""
        os.environ["POSTGRES_USER"] = "asyncuser"
        os.environ["POSTGRES_PASSWORD"] = "asyncpass"
        os.environ["POSTGRES_DB"] = "asyncdb"
        os.environ["POSTGRES_HOST"] = "asynchost"
        os.environ["POSTGRES_PORT"] = "5434"

        url = get_async_db_url()

        assert url == "postgresql+asyncpg://asyncuser:asyncpass@asynchost:5434/asyncdb"

    def test_get_db_url_partial_environment(self):
        """Test get_db_url uses mix of env and defaults."""
        os.environ["POSTGRES_USER"] = "customuser"
        # Other values should use defaults

        url = get_db_url()

        assert (
            url == "postgresql+psycopg2://customuser:password@localhost:5432/football"
        )

    def test_get_db_url_special_characters(self):
        """Test get_db_url handles special characters in password."""
        os.environ["POSTGRES_PASSWORD"] = "p@ss!word#123"

        url = get_db_url()

        assert "p@ss!word#123" in url


# =============================================================================
# Test Engine Creation
# =============================================================================


class TestEngineCreation:
    """Test engine creation functions."""

    @patch("algobet.infrastructure.database.create_engine")
    def test_create_db_engine(self, mock_create_engine):
        """Test create_db_engine creates SQLAlchemy engine."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        result = create_db_engine()

        assert result == mock_engine
        mock_create_engine.assert_called_once()
        # Verify it's called with the correct URL
        call_args = mock_create_engine.call_args[0][0]
        assert call_args.startswith("postgresql+psycopg2://")

    @patch("algobet.infrastructure.database.create_async_engine")
    def test_create_async_db_engine(self, mock_create_async_engine):
        """Test create_async_db_engine creates async engine."""
        mock_async_engine = MagicMock(spec=AsyncEngine)
        mock_create_async_engine.return_value = mock_async_engine

        result = create_async_db_engine()

        assert result == mock_async_engine
        mock_create_async_engine.assert_called_once()
        # Verify it's called with correct URL and options
        call_args = mock_create_async_engine.call_args
        assert call_args[0][0].startswith("postgresql+asyncpg://")
        assert call_args[1]["echo"] is False
        assert call_args[1]["pool_pre_ping"] is True

    @patch("algobet.infrastructure.database.create_engine")
    def test_create_db_engine_url(self, mock_create_engine):
        """Test create_db_engine uses correct database URL."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Set custom environment
        os.environ["POSTGRES_USER"] = "urltest"
        create_db_engine()

        call_url = mock_create_engine.call_args[0][0]
        assert "urltest" in call_url


# =============================================================================
# Test Database Initialization
# =============================================================================


class TestDatabaseInitialization:
    """Test database initialization functions."""

    @patch("algobet.infrastructure.database.create_db_engine")
    @patch("algobet.infrastructure.database.Base")
    def test_init_db(self, mock_base, mock_create_engine):
        """Test init_db creates all tables."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_metadata = MagicMock()
        mock_base.metadata = mock_metadata

        init_db()

        mock_create_engine.assert_called_once()
        mock_metadata.create_all.assert_called_once_with(mock_engine)

    @patch("algobet.infrastructure.database.create_db_engine")
    @patch("algobet.infrastructure.database.Base")
    def test_init_db_prints_success(self, mock_base, mock_create_engine, capsys):
        """Test init_db prints success message."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        init_db()

        captured = capsys.readouterr()
        assert "Database tables created successfully" in captured.out

    @patch("algobet.infrastructure.database.create_async_db_engine")
    @patch("algobet.infrastructure.database.Base")
    @pytest.mark.asyncio
    async def test_async_init_db(self, mock_base, mock_create_async_engine):
        """Test async_init_db creates tables asynchronously."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_create_async_engine.return_value = mock_engine
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        await async_init_db()

        mock_create_async_engine.assert_called_once()
        mock_conn.run_sync.assert_called_once()
        mock_engine.dispose.assert_called_once()

    @patch("algobet.infrastructure.database.create_async_db_engine")
    @patch("algobet.infrastructure.database.Base")
    @pytest.mark.asyncio
    async def test_async_init_db_prints_success(
        self, mock_base, mock_create_async_engine, capsys
    ):
        """Test async_init_db prints success message."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_create_async_engine.return_value = mock_engine
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        await async_init_db()

        captured = capsys.readouterr()
        assert "Database tables created successfully" in captured.out


# =============================================================================
# Test Session Creation
# =============================================================================


class TestSessionCreation:
    """Test session creation functions."""

    @patch("algobet.infrastructure.database.create_db_engine")
    @patch("algobet.infrastructure.database.sessionmaker")
    def test_get_session(self, mock_sessionmaker, mock_create_engine):
        """Test get_session creates and returns a session."""
        from sqlalchemy.orm import Session

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session = MagicMock(spec=Session)
        mock_sessionmaker.return_value.return_value = mock_session

        result = get_session()

        assert result == mock_session
        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()
        # Verify sessionmaker is configured correctly
        call_kwargs = mock_sessionmaker.call_args[1]
        assert call_kwargs["bind"] == mock_engine

    @patch("algobet.infrastructure.database.create_async_db_engine")
    @patch("algobet.infrastructure.database.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_get_async_session(
        self, mock_async_sessionmaker, mock_create_async_engine
    ):
        """Test get_async_session creates and returns an async session."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_create_async_engine.return_value = mock_engine
        mock_session = AsyncMock(spec=AsyncSession)
        mock_async_sessionmaker.return_value.return_value = mock_session

        result = await get_async_session()

        assert result == mock_session
        mock_create_async_engine.assert_called_once()
        mock_async_sessionmaker.assert_called_once()
        # Verify async_sessionmaker is configured correctly
        call_kwargs = mock_async_sessionmaker.call_args[1]
        assert call_kwargs["bind"] == mock_engine
        assert call_kwargs["class_"] == AsyncSession
        assert call_kwargs["expire_on_commit"] is False


# =============================================================================
# Test Session Scope (Sync)
# =============================================================================


class TestSessionScope:
    """Test synchronous session_scope context manager."""

    @patch("algobet.infrastructure.database.get_session")
    def test_session_scope_success(self, mock_get_session):
        """Test session_scope commits on successful exit."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        with session_scope() as session:
            session.add("test_object")

        mock_get_session.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    @patch("algobet.infrastructure.database.get_session")
    def test_session_scope_rollback_on_exception(self, mock_get_session):
        """Test session_scope rolls back on exception."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        with pytest.raises(ValueError), session_scope() as session:
            session.add("test_object")
            raise ValueError("Test error")

        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()

    @patch("algobet.infrastructure.database.get_session")
    def test_session_scope_always_closes(self, mock_get_session):
        """Test session_scope always closes session."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        try:
            with session_scope():
                raise Exception("Any exception")
        except Exception:
            pass

        mock_session.close.assert_called_once()

    @patch("algobet.infrastructure.database.get_session")
    def test_session_scope_yields_session(self, mock_get_session):
        """Test session_scope yields the session object."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        with session_scope() as session:
            assert session == mock_session


# =============================================================================
# Test Async Session Scope
# =============================================================================


class TestAsyncSessionScope:
    """Test asynchronous async_session_scope context manager."""

    @patch("algobet.infrastructure.database.get_async_session")
    @pytest.mark.asyncio
    async def test_async_session_scope_success(self, mock_get_async_session):
        """Test async_session_scope commits on successful exit."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_get_async_session.return_value = mock_session

        async with async_session_scope():
            # Simulate async operation
            pass

        mock_get_async_session.assert_called_once()
        mock_session.commit.assert_awaited_once()
        mock_session.close.assert_awaited_once()
        mock_session.rollback.assert_not_awaited()

    @patch("algobet.infrastructure.database.get_async_session")
    @pytest.mark.asyncio
    async def test_async_session_scope_rollback_on_exception(
        self, mock_get_async_session
    ):
        """Test async_session_scope rolls back on exception."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_get_async_session.return_value = mock_session

        with pytest.raises(ValueError):
            async with async_session_scope():
                raise ValueError("Test async error")

        mock_session.rollback.assert_awaited_once()
        mock_session.commit.assert_not_awaited()
        mock_session.close.assert_awaited_once()

    @patch("algobet.infrastructure.database.get_async_session")
    @pytest.mark.asyncio
    async def test_async_session_scope_always_closes(self, mock_get_async_session):
        """Test async_session_scope always closes session."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_get_async_session.return_value = mock_session

        try:
            async with async_session_scope():
                raise Exception("Any async exception")
        except Exception:
            pass

        mock_session.close.assert_awaited_once()

    @patch("algobet.infrastructure.database.get_async_session")
    @pytest.mark.asyncio
    async def test_async_session_scope_yields_session(self, mock_get_async_session):
        """Test async_session_scope yields the session object."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_get_async_session.return_value = mock_session

        async with async_session_scope() as session:
            assert session == mock_session


# =============================================================================
# Test Shared Async Engine (Singleton)
# =============================================================================


class TestSharedAsyncEngine:
    """Test shared async engine singleton pattern."""

    def setup_module(self):
        """Reset singleton before tests."""
        import algobet.infrastructure.database as db_module

        db_module._async_engine = None

    def teardown_module(self):
        """Reset singleton after all tests."""
        import algobet.infrastructure.database as db_module

        db_module._async_engine = None

    @patch("algobet.infrastructure.database.create_async_db_engine")
    def test_get_shared_async_engine_creates_once(self, mock_create_async_engine):
        """Test get_shared_async_engine creates engine only once."""
        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_async_engine.return_value = mock_engine

        # First call should create engine
        result1 = get_shared_async_engine()

        # Second call should reuse existing engine
        result2 = get_shared_async_engine()

        assert result1 == result2
        assert result1 == mock_engine
        mock_create_async_engine.assert_called_once()

    @patch("algobet.infrastructure.database.create_async_db_engine")
    def test_get_shared_async_engine_returns_engine(self, mock_create_async_engine):
        """Test get_shared_async_engine returns AsyncEngine."""
        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_async_engine.return_value = mock_engine

        result = get_shared_async_engine()

        assert isinstance(result, AsyncEngine) or hasattr(result, "dispose")

    @patch("algobet.infrastructure.database.create_async_db_engine")
    @pytest.mark.asyncio
    async def test_close_async_engine(self, mock_create_async_engine):
        """Test close_async_engine disposes and resets engine."""
        import algobet.infrastructure.database as db_module

        # Reset first
        db_module._async_engine = None

        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_engine.dispose = AsyncMock()
        mock_create_async_engine.return_value = mock_engine

        # Create engine (this sets the singleton)
        get_shared_async_engine()

        # Close it
        await close_async_engine()

        # Verify dispose was called on the engine
        mock_engine.dispose.assert_awaited_once()

    @patch("algobet.infrastructure.database.create_async_db_engine")
    @pytest.mark.asyncio
    async def test_close_async_engine_resets_singleton(self, mock_create_async_engine):
        """Test close_async_engine resets the singleton."""
        import algobet.infrastructure.database as db_module

        # Reset first
        db_module._async_engine = None

        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_engine.dispose = AsyncMock()
        mock_create_async_engine.return_value = mock_engine

        # Create and close engine
        get_shared_async_engine()
        await close_async_engine()

        # Verify singleton was reset
        assert db_module._async_engine is None

    @pytest.mark.asyncio
    async def test_close_async_engine_when_none(self):
        """Test close_async_engine handles None engine gracefully."""
        import algobet.infrastructure.database as db_module

        # Ensure engine is None
        db_module._async_engine = None

        # Should not raise
        await close_async_engine()

        # Verify engine is still None
        assert db_module._async_engine is None


# =============================================================================
# Test Integration with Real SQLite
# =============================================================================


class TestDatabaseIntegration:
    """Integration tests with real SQLite database."""

    @pytest.fixture
    def sqlite_db_url(self, tmp_path):
        """Create temporary SQLite database for testing."""
        db_file = tmp_path / "test.db"
        return f"sqlite:///{db_file}"

    @patch("algobet.infrastructure.database.get_db_url")
    def test_session_scope_with_real_db(self, mock_get_db_url, sqlite_db_url):
        """Test session_scope works with real database."""
        from sqlalchemy import create_engine

        mock_get_db_url.return_value = sqlite_db_url

        # Create tables
        engine = create_engine(sqlite_db_url)
        Base.metadata.create_all(engine)

        with session_scope() as session:
            # Execute a simple query
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @patch("algobet.infrastructure.database.get_db_url")
    def test_session_scope_rollback_integration(self, mock_get_db_url, sqlite_db_url):
        """Test session_scope rollback with real database."""
        from sqlalchemy import create_engine

        mock_get_db_url.return_value = sqlite_db_url

        # Create tables
        engine = create_engine(sqlite_db_url)
        Base.metadata.create_all(engine)

        # First session creates data
        with session_scope() as session:
            session.execute(text("CREATE TABLE IF NOT EXISTS test (id INTEGER)"))
            session.execute(text("INSERT INTO test VALUES (1)"))

        # Second session fails, should rollback
        try:
            with session_scope() as session:
                session.execute(text("INSERT INTO test VALUES (2)"))
                raise ValueError("Force rollback")
        except ValueError:
            pass

        # Verify rollback - only first insert should exist
        with session_scope() as session:
            result = session.execute(text("SELECT COUNT(*) FROM test"))
            assert result.scalar() == 1


# =============================================================================
# Test Environment Variable Edge Cases
# =============================================================================


class TestDatabaseEnvironmentEdgeCases:
    """Test database environment variable edge cases."""

    def teardown_method(self):
        """Clean up environment variables."""
        for key in list(os.environ.keys()):
            if key.startswith("POSTGRES_"):
                del os.environ[key]

    def test_empty_password(self):
        """Test handling empty password."""
        os.environ["POSTGRES_PASSWORD"] = ""

        url = get_db_url()

        assert ":@" in url or ":@localhost" in url

    def test_ipv6_host(self):
        """Test handling IPv6 host."""
        os.environ["POSTGRES_HOST"] = "::1"

        url = get_db_url()

        assert "::1" in url

    def test_large_port_number(self):
        """Test handling large port number."""
        os.environ["POSTGRES_PORT"] = "65535"

        url = get_db_url()

        assert ":65535" in url

    def test_unicode_database_name(self):
        """Test handling unicode database name."""
        os.environ["POSTGRES_DB"] = "testdb_测试"

        url = get_db_url()

        assert "testdb_测试" in url


# =============================================================================
# Test Async Engine Configuration
# =============================================================================


class TestAsyncEngineConfiguration:
    """Test async engine configuration options."""

    @patch("algobet.infrastructure.database.create_async_engine")
    def test_async_engine_echo_false(self, mock_create_async_engine):
        """Test async engine has echo=False."""
        mock_engine = MagicMock()
        mock_create_async_engine.return_value = mock_engine

        create_async_db_engine()

        call_kwargs = mock_create_async_engine.call_args[1]
        assert call_kwargs["echo"] is False

    @patch("algobet.infrastructure.database.create_async_engine")
    def test_async_engine_pool_pre_ping(self, mock_create_async_engine):
        """Test async engine has pool_pre_ping=True."""
        mock_engine = MagicMock()
        mock_create_async_engine.return_value = mock_engine

        create_async_db_engine()

        call_kwargs = mock_create_async_engine.call_args[1]
        assert call_kwargs["pool_pre_ping"] is True


# =============================================================================
# Test Session Configuration
# =============================================================================


class TestSessionConfiguration:
    """Test session configuration options."""

    @patch("algobet.infrastructure.database.create_async_db_engine")
    @patch("algobet.infrastructure.database.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_async_session_expire_on_commit_false(
        self, mock_async_sessionmaker, mock_create_async_engine
    ):
        """Test async session has expire_on_commit=False."""
        mock_engine = AsyncMock()
        mock_create_async_engine.return_value = mock_engine
        mock_session = AsyncMock()
        mock_async_sessionmaker.return_value.return_value = mock_session

        await get_async_session()

        call_kwargs = mock_async_sessionmaker.call_args[1]
        assert call_kwargs["expire_on_commit"] is False

    @patch("algobet.infrastructure.database.create_async_db_engine")
    @patch("algobet.infrastructure.database.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_async_session_class_setting(
        self, mock_async_sessionmaker, mock_create_async_engine
    ):
        """Test async session uses AsyncSession class."""
        mock_engine = AsyncMock()
        mock_create_async_engine.return_value = mock_engine
        mock_session = AsyncMock()
        mock_async_sessionmaker.return_value.return_value = mock_session

        await get_async_session()

        call_kwargs = mock_async_sessionmaker.call_args[1]
        assert call_kwargs["class_"] == AsyncSession
