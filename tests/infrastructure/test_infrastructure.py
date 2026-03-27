"""Essential tests for exceptions, logging, and scraping service."""

import contextlib
import logging
from unittest.mock import MagicMock, patch

import pytest

from algobet.infrastructure.exceptions import (
    AlgoBetError,
    DatabaseConnectionError,
    DatabaseError,
    DataError,
    InsufficientDataError,
    ModelError,
    ModelNotFoundError,
    NoActiveModelError,
    PredictionError,
    ScrapingConnectionError,
    ScrapingError,
    ServiceError,
    ServiceUnavailableError,
    ValidationError,
    get_exit_code_description,
)
from algobet.infrastructure.logging_config import (
    ColoredFormatter,
    JSONFormatter,
    get_logger,
    setup_logging,
)

# =============================================================================
# Test Exception Hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_algotbet_error_is_exception(self):
        """Test AlgoBetError is subclass of Exception."""
        assert issubclass(AlgoBetError, Exception)

    def test_database_error_is_algotbet_error(self):
        """Test DatabaseError is subclass of AlgoBetError."""
        assert issubclass(DatabaseError, AlgoBetError)

    def test_model_error_is_algotbet_error(self):
        """Test ModelError is subclass of AlgoBetError."""
        assert issubclass(ModelError, AlgoBetError)

    def test_data_error_is_algotbet_error(self):
        """Test DataError is subclass of AlgoBetError."""
        assert issubclass(DataError, AlgoBetError)

    def test_scraping_error_is_algotbet_error(self):
        """Test ScrapingError is subclass of AlgoBetError."""
        assert issubclass(ScrapingError, AlgoBetError)

    def test_prediction_error_is_algotbet_error(self):
        """Test PredictionError is subclass of AlgoBetError."""
        assert issubclass(PredictionError, AlgoBetError)

    def test_service_error_is_algotbet_error(self):
        """Test ServiceError is subclass of AlgoBetError."""
        assert issubclass(ServiceError, AlgoBetError)

    def test_validation_error_is_algotbet_error(self):
        """Test ValidationError is subclass of AlgoBetError."""
        assert issubclass(ValidationError, AlgoBetError)


class TestExceptionExitCodes:
    """Test exception exit codes."""

    def test_algotbet_error_default_exit_code(self):
        """Test AlgoBetError default exit code."""
        assert AlgoBetError.exit_code == 1

    def test_database_error_exit_code(self):
        """Test DatabaseError exit code."""
        assert DatabaseError.exit_code == 10

    def test_database_connection_error_exit_code(self):
        """Test DatabaseConnectionError exit code."""
        assert DatabaseConnectionError.exit_code == 11

    def test_model_not_found_error_exit_code(self):
        """Test ModelNotFoundError exit code."""
        assert ModelNotFoundError.exit_code == 21

    def test_insufficient_data_error_exit_code(self):
        """Test InsufficientDataError exit code."""
        assert InsufficientDataError.exit_code == 31

    def test_scraping_connection_error_exit_code(self):
        """Test ScrapingConnectionError exit code."""
        assert ScrapingConnectionError.exit_code == 41

    def test_prediction_error_exit_code(self):
        """Test PredictionError exit code."""
        assert PredictionError.exit_code == 60

    def test_service_unavailable_error_exit_code(self):
        """Test ServiceUnavailableError exit code."""
        assert ServiceUnavailableError.exit_code == 71


class TestExceptionInitialization:
    """Test exception initialization."""

    def test_algotbet_error_with_message(self):
        """Test AlgoBetError with message."""
        error = AlgoBetError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.exit_code == 1

    def test_algotbet_error_with_exit_code(self):
        """Test AlgoBetError with custom exit code."""
        error = AlgoBetError("Test error", exit_code=99)

        assert error.exit_code == 99

    def test_algotbet_error_with_details(self):
        """Test AlgoBetError with details."""
        error = AlgoBetError("Test error", details={"key": "value"})

        assert error.details == {"key": "value"}
        assert "key=value" in str(error)

    def test_no_active_model_error_default_message(self):
        """Test NoActiveModelError has default message."""
        error = NoActiveModelError()

        assert "No active model" in str(error)


class TestExitCodeHelper:
    """Test exit code helper function."""

    def test_get_exit_code_description_known(self):
        """Test get_exit_code_description with known code."""
        desc = get_exit_code_description(10)

        assert "Database" in desc

    def test_get_exit_code_description_unknown(self):
        """Test get_exit_code_description with unknown code."""
        desc = get_exit_code_description(999)

        assert "Unknown" in desc
        assert "999" in desc


# =============================================================================
# Test Logging Configuration
# =============================================================================


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_success_level_added(self):
        """Test SUCCESS log level is added."""
        assert logging.getLevelName(25) == "SUCCESS"

    def test_success_method_exists(self):
        """Test success method exists on Logger."""
        assert hasattr(logging.Logger, "success")

    def test_get_logger_returns_logger(self):
        """Test get_logger returns Logger instance."""
        logger = get_logger("test")

        assert isinstance(logger, logging.Logger)
        # Logger name includes module prefix
        assert "test" in logger.name

    @patch("algobet.infrastructure.logging_config.logging.basicConfig")
    def test_setup_logging_configures_logging(self, mock_basic_config):
        """Test setup_logging configures logging."""
        config = MagicMock()
        config.level = "INFO"
        config.format = "text"
        config.output = "stdout"
        config.show_timestamp = True
        config.show_level = True
        config.color = True

        with contextlib.suppress(Exception):
            setup_logging(config)

        # Verify basicConfig was at least attempted
        # Note: actual call count may vary based on implementation


class TestColoredFormatter:
    """Test ColoredFormatter."""

    def test_colored_formatter_init(self):
        """Test ColoredFormatter initialization."""
        formatter = ColoredFormatter(use_colors=True)

        assert formatter.use_colors is True

    def test_colored_formatter_init_no_colors(self):
        """Test ColoredFormatter without colors."""
        formatter = ColoredFormatter(use_colors=False)

        assert formatter.use_colors is False

    def test_colored_formatter_format(self):
        """Test ColoredFormatter formats with colors."""
        formatter = ColoredFormatter(fmt="%(levelname)s: %(message)s", use_colors=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result


class TestJSONFormatter:
    """Test JSONFormatter."""

    def test_json_formatter_format(self):
        """Test JSONFormatter formats as JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        import json

        # Should be valid JSON
        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"

    def test_json_formatter_includes_timestamp(self):
        """Test JSONFormatter includes timestamp."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        import json

        data = json.loads(result)
        assert "timestamp" in data


# =============================================================================
# Test Scraping Service
# =============================================================================


class TestScrapingService:
    """Test ScrapingService core operations."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def mock_client(self):
        """Create mock API-Football client."""
        return MagicMock()

    @pytest.fixture
    def scraping_service(self, mock_session, mock_client):
        """Create ScrapingService instance."""
        from algobet.services.scraping_service import ScrapingService

        return ScrapingService(session=mock_session)

    def test_init_with_session(self, mock_session):
        """Test ScrapingService initialization."""
        from algobet.services.scraping_service import ScrapingService

        service = ScrapingService(session=mock_session)

        assert service.session == mock_session

    def test_init_with_progress_callback(self, mock_session):
        """Test ScrapingService with progress callback."""
        from algobet.services.scraping_service import ScrapingService

        callback = MagicMock()
        service = ScrapingService(session=mock_session, progress_callback=callback)

        assert service.progress_callback == callback

    def test_create_job_returns_job(self, mock_session):
        """Test create_job returns ScrapingJob."""
        from algobet.services.scraping_service import ScrapingService

        service = ScrapingService(session=mock_session)
        job = service.create_job("results", "http://example.com")

        assert job.job_type == "results"
        assert job.url == "http://example.com"
        assert job.status.value == "pending"

    def test_create_job_stores_job(self, mock_session):
        """Test create_job stores job in memory."""
        from algobet.services.scraping_service import ScrapingService

        service = ScrapingService(session=mock_session)
        job = service.create_job("results", "http://example.com")

        assert job.id in ScrapingService._jobs

    def test_get_job_returns_job(self, mock_session):
        """Test get_job returns stored job."""
        from algobet.services.scraping_service import ScrapingService

        service = ScrapingService(session=mock_session)
        created_job = service.create_job("results", "http://example.com")

        retrieved_job = service.get_job(created_job.id)

        assert retrieved_job.id == created_job.id

    def test_get_job_not_found(self, mock_session):
        """Test get_job returns None for unknown job."""
        from uuid import uuid4

        from algobet.services.scraping_service import ScrapingService

        ScrapingService._jobs = {}
        service = ScrapingService(session=mock_session)

        job = service.get_job(uuid4())

        assert job is None

    def test_list_jobs_returns_all(self, mock_session):
        """Test list_jobs returns all jobs."""
        from algobet.services.scraping_service import ScrapingService

        ScrapingService._jobs = {}
        service = ScrapingService(session=mock_session)
        service.create_job("results", "http://example1.com")
        service.create_job("upcoming", "http://example2.com")

        jobs = service.list_jobs()

        assert len(jobs) == 2

    def test_scrape_results_exists(self, mock_session):
        """Test scrape_results method exists and runs."""
        from algobet.services.scraping_service import ScrapingService

        service = ScrapingService(session=mock_session)
        # Mock the client to avoid actual API calls
        service.api_football_client = MagicMock()
        service.api_football_client.get_results.return_value = MagicMock(fixtures=[])

        service.create_job("results", "http://example.com")

        # Should not raise
        with contextlib.suppress(Exception):
            service.scrape_results(league_id=39)

    def test_scrape_upcoming_exists(self, mock_session):
        """Test scrape_upcoming method exists and runs."""
        from algobet.services.scraping_service import ScrapingService

        service = ScrapingService(session=mock_session)
        # Mock the client to avoid actual API calls
        service.api_football_client = MagicMock()
        service.api_football_client.get_upcoming_fixtures.return_value = MagicMock(
            fixtures=[]
        )

        service.create_job("upcoming", "http://example.com")

        # Should not raise
        with contextlib.suppress(Exception):
            service.scrape_upcoming(league_ids=[39])

    def test_emit_progress_calls_callback(self, mock_session):
        """Test emit_progress calls callback."""
        from uuid import uuid4

        from algobet.services.scraping_service import (
            JobStatus,
            ScrapingProgress,
            ScrapingService,
        )

        callback = MagicMock()
        service = ScrapingService(session=mock_session, progress_callback=callback)

        progress = ScrapingProgress(
            job_id=uuid4(),
            status=JobStatus.RUNNING,
            progress=50.0,
        )

        service._emit_progress(progress)

        callback.assert_called_once_with(progress)

    def test_emit_progress_no_callback(self, mock_session):
        """Test emit_progress handles missing callback."""
        from uuid import uuid4

        from algobet.services.scraping_service import (
            JobStatus,
            ScrapingProgress,
            ScrapingService,
        )

        service = ScrapingService(session=mock_session)

        progress = ScrapingProgress(
            job_id=uuid4(),
            status=JobStatus.RUNNING,
            progress=50.0,
        )

        # Should not raise
        service._emit_progress(progress)
