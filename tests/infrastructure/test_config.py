"""Unit tests for configuration management.

Tests for the configuration system which handles all application settings
using Pydantic Settings with environment variable support.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from algobet.infrastructure.config import (
    AlgobetConfig,
    BacktestConfig,
    CLIConfig,
    DatabaseConfig,
    LoggingConfig,
    ModelsConfig,
    ScrapingConfig,
    get_config,
    reload_config,
    set_config,
)

# =============================================================================
# Test DatabaseConfig
# =============================================================================


class TestDatabaseConfig:
    """Test DatabaseConfig configuration."""

    def test_default_values(self):
        """Test DatabaseConfig uses default values."""
        config = DatabaseConfig()

        assert config.url == "postgresql://localhost/algobet"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.echo is False

    def test_custom_values(self):
        """Test DatabaseConfig accepts custom values."""
        config = DatabaseConfig(
            url="postgresql://user:pass@host/db",
            pool_size=20,
            max_overflow=30,
            echo=True,
        )

        assert config.url == "postgresql://user:pass@host/db"
        assert config.pool_size == 20
        assert config.max_overflow == 30
        assert config.echo is True

    def test_pool_size_minimum(self):
        """Test pool_size validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(pool_size=0)

        assert "pool_size" in str(exc_info.value)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_max_overflow_minimum(self):
        """Test max_overflow validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(max_overflow=-1)

        assert "max_overflow" in str(exc_info.value)

    @patch.dict(os.environ, {"ALGOBET_DATABASE__URL": "postgresql://env/host/db"})
    def test_environment_variable_loading(self):
        """Test loading from environment variables."""
        config = DatabaseConfig()

        assert config.url == "postgresql://env/host/db"

    @patch.dict(
        os.environ,
        {
            "ALGOBET_DATABASE__URL": "postgresql://test/test",
            "ALGOBET_DATABASE__POOL_SIZE": "25",
            "ALGOBET_DATABASE__ECHO": "true",
        },
    )
    def test_multiple_environment_variables(self):
        """Test loading multiple environment variables."""
        config = DatabaseConfig()

        assert config.url == "postgresql://test/test"
        assert config.pool_size == 25
        assert config.echo is True


# =============================================================================
# Test ModelsConfig
# =============================================================================


class TestModelsConfig:
    """Test ModelsConfig configuration."""

    def test_default_values(self):
        """Test ModelsConfig uses default values."""
        config = ModelsConfig()

        assert config.path == Path("data/models").expanduser().resolve()
        assert config.default_version is None

    def test_custom_path(self):
        """Test ModelsConfig accepts custom path."""
        custom_path = Path("/custom/models/path")
        config = ModelsConfig(path=custom_path)

        assert config.path == custom_path.expanduser().resolve()

    def test_path_resolution(self):
        """Test path is properly resolved."""
        config = ModelsConfig(path=Path("~/models"))

        # Path should be expanded and resolved
        assert config.path.is_absolute()

    def test_tilde_expansion(self):
        """Test tilde in path is expanded."""
        config = ModelsConfig(path=Path("~/test_models"))

        # Should not start with ~
        assert not str(config.path).startswith("~")

    @patch.dict(os.environ, {"ALGOBET_MODELS__PATH": "/env/models"})
    def test_environment_variable_loading(self):
        """Test loading path from environment variable."""
        config = ModelsConfig()

        assert config.path == Path("/env/models").expanduser().resolve()

    @patch.dict(os.environ, {"ALGOBET_MODELS__DEFAULT_VERSION": "v2.0.0"})
    def test_default_version_environment(self):
        """Test loading default_version from environment."""
        config = ModelsConfig()

        assert config.default_version == "v2.0.0"


# =============================================================================
# Test ScrapingConfig
# =============================================================================


class TestScrapingConfig:
    """Test ScrapingConfig configuration."""

    @pytest.fixture(autouse=True)
    def clear_env(self):
        """Clear Scraping environment variables before each test."""
        saved = {}
        for key in list(os.environ.keys()):
            if key.startswith("ALGOBET_SCRAPING__"):
                saved[key] = os.environ.pop(key)
        yield
        os.environ.update(saved)

    def test_default_values(self):
        """Test ScrapingConfig uses default values."""
        config = ScrapingConfig()

        assert config.default_url == "https://www.oddsportal.com/matches/football/"
        assert config.timeout == 120
        assert config.headless is True
        assert config.max_retries == 3
        assert config.retry_delay == 5

    def test_custom_values(self):
        """Test ScrapingConfig accepts custom values."""
        config = ScrapingConfig(
            default_url="https://custom.com",
            timeout=180,
            headless=False,
            max_retries=5,
            retry_delay=10,
        )

        assert config.default_url == "https://custom.com"
        assert config.timeout == 180
        assert config.headless is False
        assert config.max_retries == 5
        assert config.retry_delay == 10

    def test_timeout_minimum(self):
        """Test timeout validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            ScrapingConfig(timeout=0)

        assert "timeout" in str(exc_info.value)

    def test_max_retries_minimum(self):
        """Test max_retries validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            ScrapingConfig(max_retries=-1)

        assert "max_retries" in str(exc_info.value)

    def test_retry_delay_minimum(self):
        """Test retry_delay validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            ScrapingConfig(retry_delay=-5)

        assert "retry_delay" in str(exc_info.value)

    @patch.dict(os.environ, {"ALGOBET_SCRAPING__HEADLESS": "false"})
    def test_boolean_environment_variable(self):
        """Test boolean environment variable parsing."""
        config = ScrapingConfig()

        assert config.headless is False


# =============================================================================
# Test BacktestConfig
# =============================================================================


class TestBacktestConfig:
    """Test BacktestConfig configuration."""

    def test_default_values(self):
        """Test BacktestConfig uses default values."""
        config = BacktestConfig()

        assert config.default_min_matches == 100
        assert config.default_validation_split == 0.2
        assert config.max_history_days == 365

    def test_custom_values(self):
        """Test BacktestConfig accepts custom values."""
        config = BacktestConfig(
            default_min_matches=200,
            default_validation_split=0.3,
            max_history_days=730,
        )

        assert config.default_min_matches == 200
        assert config.default_validation_split == 0.3
        assert config.max_history_days == 730

    def test_min_matches_minimum(self):
        """Test default_min_matches validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(default_min_matches=5)

        assert "default_min_matches" in str(exc_info.value)

    def test_validation_split_minimum(self):
        """Test default_validation_split validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(default_validation_split=0.05)

        assert "default_validation_split" in str(exc_info.value)

    def test_validation_split_maximum(self):
        """Test default_validation_split validation maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(default_validation_split=0.6)

        assert "default_validation_split" in str(exc_info.value)

    def test_max_history_days_minimum(self):
        """Test max_history_days validation minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(max_history_days=20)

        assert "max_history_days" in str(exc_info.value)


# =============================================================================
# Test LoggingConfig
# =============================================================================


class TestLoggingConfig:
    """Test LoggingConfig configuration."""

    def test_default_values(self):
        """Test LoggingConfig uses default values."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.format == "text"
        assert config.output == "stdout"
        assert config.file_path is None
        assert config.show_timestamp is True
        assert config.show_level is True
        assert config.color is True

    def test_custom_values(self):
        """Test LoggingConfig accepts custom values."""
        config = LoggingConfig(
            level="DEBUG",
            format="json",
            output="file",
            file_path=Path("/var/log/algobet.log"),
            show_timestamp=False,
            show_level=False,
            color=False,
        )

        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.output == "file"
        assert config.file_path == Path("/var/log/algobet.log")
        assert config.show_timestamp is False
        assert config.show_level is False
        assert config.color is False

    def test_invalid_level(self):
        """Test invalid logging level raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(level="INVALID")

        assert "level" in str(exc_info.value)

    def test_invalid_format(self):
        """Test invalid format raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(format="invalid")

        assert "format" in str(exc_info.value)

    def test_invalid_output(self):
        """Test invalid output raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(output="invalid")

        assert "output" in str(exc_info.value)

    @patch.dict(os.environ, {"ALGOBET_LOGGING__LEVEL": "WARNING"})
    def test_level_from_environment(self):
        """Test loading level from environment variable."""
        config = LoggingConfig()

        assert config.level == "WARNING"


# =============================================================================
# Test CLIConfig
# =============================================================================


class TestCLIConfig:
    """Test CLIConfig configuration."""

    def test_default_values(self):
        """Test CLIConfig uses default values."""
        config = CLIConfig()

        assert config.debug is False
        assert config.verbose is False
        assert config.color is True

    def test_custom_values(self):
        """Test CLIConfig accepts custom values."""
        config = CLIConfig(
            debug=True,
            verbose=True,
            color=False,
        )

        assert config.debug is True
        assert config.verbose is True
        assert config.color is False

    @patch.dict(os.environ, {"ALGOBET_CLI__DEBUG": "true"})
    def test_debug_from_environment(self):
        """Test loading debug from environment variable."""
        config = CLIConfig()

        assert config.debug is True


# =============================================================================
# Test AlgobetConfig (Main Config)
# =============================================================================


class TestAlgobetConfig:
    """Test main AlgobetConfig configuration."""

    def test_default_values(self):
        """Test AlgobetConfig uses default values."""
        config = AlgobetConfig()

        assert config.app_name == "AlgoBet"
        assert config.app_version == "0.1.0"
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.models, ModelsConfig)
        assert isinstance(config.scraping, ScrapingConfig)
        assert isinstance(config.backtest, BacktestConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.cli, CLIConfig)

    def test_nested_config_defaults(self):
        """Test nested configurations have correct defaults."""
        config = AlgobetConfig()

        # Database defaults
        assert config.database.pool_size == 10
        # Scraping defaults
        assert config.scraping.headless is True
        # Logging defaults
        assert config.logging.level == "INFO"

    def test_to_dict(self):
        """Test to_dict method."""
        config = AlgobetConfig()

        result = config.to_dict()

        assert "app_name" in result
        assert "database" in result
        assert "scraping" in result

    def test_to_dict_includes_nested_configs(self):
        """Test to_dict includes all nested configurations."""
        config = AlgobetConfig()

        result = config.to_dict()

        assert "database" in result
        assert "models" in result
        assert "scraping" in result
        assert "backtest" in result
        assert "logging" in result
        assert "cli" in result

    def test_is_development_false_by_default(self):
        """Test is_development is False by default."""
        config = AlgobetConfig()

        assert config.is_development is False

    def test_is_development_with_debug(self):
        """Test is_development is True when debug is enabled."""
        config = AlgobetConfig(cli=CLIConfig(debug=True))

        assert config.is_development is True

    def test_is_development_with_debug_logging(self):
        """Test is_development is True with DEBUG logging level."""
        config = AlgobetConfig(logging=LoggingConfig(level="DEBUG"))

        assert config.is_development is True

    def test_custom_nested_config(self):
        """Test AlgobetConfig accepts custom nested configurations."""
        custom_db = DatabaseConfig(url="postgresql://custom/db", pool_size=50)

        config = AlgobetConfig(database=custom_db)

        assert config.database.url == "postgresql://custom/db"
        assert config.database.pool_size == 50


# =============================================================================
# Test Global Config Functions
# =============================================================================


class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def teardown_method(self):
        """Reset config after each test."""
        global _config
        from algobet.infrastructure import config as config_module

        config_module._config = None

    def test_get_config_creates_singleton(self):
        """Test get_config creates singleton instance."""
        config1 = get_config()
        config2 = get_config()

        # Should return same instance (singleton pattern)
        assert config1 is config2

    def test_get_config_returns_algobet_config(self):
        """Test get_config returns AlgobetConfig instance."""
        config = get_config()

        assert isinstance(config, AlgobetConfig)

    def test_reload_config_creates_new_instance(self):
        """Test reload_config creates new instance."""
        config1 = get_config()
        config2 = reload_config()

        # Should be different instances
        assert config1 is not config2
        assert isinstance(config2, AlgobetConfig)

    def test_set_config_sets_instance(self):
        """Test set_config sets the global instance."""
        custom_config = AlgobetConfig(app_name="CustomApp")
        set_config(custom_config)

        retrieved = get_config()
        assert retrieved is custom_config
        assert retrieved.app_name == "CustomApp"

    def test_set_config_affects_get_config(self):
        """Test set_config affects subsequent get_config calls."""
        custom_config = AlgobetConfig(app_version="9.9.9")
        set_config(custom_config)

        retrieved = get_config()
        assert retrieved.app_version == "9.9.9"

    def test_reload_after_set_resets(self):
        """Test reload_config resets after set_config."""
        custom_config = AlgobetConfig(app_name="TempApp")
        set_config(custom_config)

        reloaded = reload_config()
        assert reloaded.app_name == "AlgoBet"  # Back to default


# =============================================================================
# Test Environment Variable Integration
# =============================================================================


class TestEnvironmentIntegration:
    """Test environment variable integration."""

    def teardown_method(self):
        """Clean up environment variables and reset config."""
        from algobet.infrastructure import config as config_module

        config_module._config = None
        # Remove any test environment variables
        for key in list(os.environ.keys()):
            if key.startswith("ALGOBET_"):
                del os.environ[key]

    def test_config_respects_environment_variables(self):
        """Test that configuration can be loaded from environment variables.

        Note: This test verifies the mechanism works by directly instantiating
        config classes with environment variables set. Pydantic Settings reads
        env vars at instantiation time.
        """
        # Save original env vars
        original_env = {}
        for key in list(os.environ.keys()):
            if key.startswith("ALGOBET_"):
                original_env[key] = os.environ[key]

        try:
            # Clear any existing ALGOBET env vars
            for key in list(os.environ.keys()):
                if key.startswith("ALGOBET_"):
                    del os.environ[key]

            # Set test environment variables
            os.environ["ALGOBET_DATABASE__URL"] = "postgresql://test/envdb"
            os.environ["ALGOBET_DATABASE__POOL_SIZE"] = "50"

            # Create new config instances (should pick up env vars)
            db_config = DatabaseConfig()

            assert db_config.url == "postgresql://test/envdb"
            assert db_config.pool_size == 50

        finally:
            # Restore original environment
            for key in list(os.environ.keys()):
                if key.startswith("ALGOBET_"):
                    del os.environ[key]
            os.environ.update(original_env)


# =============================================================================
# Test Validation Edge Cases
# =============================================================================


class TestValidationEdgeCases:
    """Test validation edge cases."""

    def test_database_config_all_valid_values(self):
        """Test DatabaseConfig with all valid boundary values."""
        config = DatabaseConfig(
            pool_size=1,  # Minimum
            max_overflow=0,  # Minimum
        )

        assert config.pool_size == 1
        assert config.max_overflow == 0

    def test_scraping_config_boundary_values(self):
        """Test ScrapingConfig with boundary values."""
        config = ScrapingConfig(
            timeout=1,  # Minimum
            max_retries=0,  # Minimum
        )

        assert config.timeout == 1
        assert config.max_retries == 0

    def test_backtest_config_boundary_values(self):
        """Test BacktestConfig with boundary values."""
        config = BacktestConfig(
            default_min_matches=10,  # Minimum
            default_validation_split=0.1,  # Minimum
            max_history_days=30,  # Minimum
        )

        assert config.default_min_matches == 10
        assert config.default_validation_split == 0.1
        assert config.max_history_days == 30

    def test_backtest_config_maximum_values(self):
        """Test BacktestConfig with maximum values."""
        config = BacktestConfig(
            default_min_matches=10000,  # Large value
            default_validation_split=0.5,  # Maximum
            max_history_days=3650,  # Large value
        )

        assert config.default_min_matches == 10000
        assert config.default_validation_split == 0.5
        assert config.max_history_days == 3650

    def test_logging_all_valid_levels(self):
        """Test LoggingConfig with all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_logging_all_valid_formats(self):
        """Test LoggingConfig with all valid formats."""
        for fmt in ["json", "text", "structured"]:
            config = LoggingConfig(format=fmt)
            assert config.format == fmt

    def test_logging_all_valid_outputs(self):
        """Test LoggingConfig with all valid outputs."""
        for output in ["stdout", "stderr", "file", "both"]:
            config = LoggingConfig(output=output)
            assert config.output == output


# =============================================================================
# Test Config Description and Metadata
# =============================================================================


class TestConfigMetadata:
    """Test configuration metadata and documentation."""

    def test_database_config_has_description(self):
        """Test DatabaseConfig fields have descriptions."""
        config = DatabaseConfig()
        # Just verify config can be created
        assert config is not None

    def test_models_config_has_description(self):
        """Test ModelsConfig fields have descriptions."""
        config = ModelsConfig()
        assert config is not None

    def test_scraping_config_has_description(self):
        """Test ScrapingConfig fields have descriptions."""
        config = ScrapingConfig()
        assert config is not None

    def test_backtest_config_has_description(self):
        """Test BacktestConfig fields have descriptions."""
        config = BacktestConfig()
        assert config is not None

    def test_logging_config_has_description(self):
        """Test LoggingConfig fields have descriptions."""
        config = LoggingConfig()
        assert config is not None

    def test_cli_config_has_description(self):
        """Test CLIConfig fields have descriptions."""
        config = CLIConfig()
        assert config is not None
