"""AlgoBet configuration management.

This module provides centralized configuration using Pydantic Settings.
Configuration can be loaded from:
- Environment variables
- .env file
- YAML config file
- Default values
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_prefix="ALGOBET_DATABASE__")

    url: str = Field(
        default="postgresql://localhost/algobet",
        description="Database connection URL",
    )
    pool_size: int = Field(default=10, ge=1, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, description="Max overflow connections")
    echo: bool = Field(default=False, description="Echo SQL statements")


class ModelsConfig(BaseSettings):
    """Model management configuration."""

    model_config = SettingsConfigDict(env_prefix="ALGOBET_MODELS__")

    path: Path = Field(
        default=Path("data/models"),
        description="Path to model storage directory",
    )
    default_version: str | None = Field(
        default=None,
        description="Default model version (None = use active model)",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Ensure path is resolved."""
        return v.expanduser().resolve()


class ScrapingConfig(BaseSettings):
    """Web scraping configuration."""

    model_config = SettingsConfigDict(env_prefix="ALGOBET_SCRAPING__")

    default_url: str = Field(
        default="https://www.oddsportal.com/matches/football/",
        description="Default URL for upcoming matches",
    )
    timeout: int = Field(default=120, ge=1, description="Page load timeout in seconds")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts")
    retry_delay: int = Field(default=5, ge=0, description="Delay between retries")


class BacktestConfig(BaseSettings):
    """Backtesting configuration."""

    model_config = SettingsConfigDict(env_prefix="ALGOBET_BACKTEST__")

    default_min_matches: int = Field(
        default=100, ge=10, description="Minimum matches required"
    )
    default_validation_split: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation data fraction"
    )
    max_history_days: int = Field(
        default=365, ge=30, description="Max days of historical data"
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="ALGOBET_LOGGING__")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    format: Literal["json", "text", "structured"] = Field(
        default="text",
        description="Log output format",
    )
    output: Literal["stdout", "stderr", "file", "both"] = Field(
        default="stdout",
        description="Log output destination",
    )
    file_path: Path | None = Field(
        default=None,
        description="Log file path (when output is file or both)",
    )
    show_timestamp: bool = Field(default=True, description="Show timestamps")
    show_level: bool = Field(default=True, description="Show log level")
    color: bool = Field(default=True, description="Use colored output")


class CLIConfig(BaseSettings):
    """CLI-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="ALGOBET_CLI__")

    debug: bool = Field(default=False, description="Enable debug mode")
    verbose: bool = Field(default=False, description="Enable verbose output")
    color: bool = Field(default=True, description="Enable colored output")


class AlgobetConfig(BaseSettings):
    """Main AlgoBet configuration.

    Loads configuration from environment variables with prefix ALGOBET__
    or from a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Application info
    app_name: str = Field(default="AlgoBet", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "database": self.database.model_dump(),
            "models": self.models.model_dump(),
            "scraping": self.scraping.model_dump(),
            "backtest": self.backtest.model_dump(),
            "logging": self.logging.model_dump(),
            "cli": self.cli.model_dump(),
        }

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.cli.debug or self.logging.level == "DEBUG"


# Global configuration instance
_config: AlgobetConfig | None = None


def get_config() -> AlgobetConfig:
    """Get the global configuration instance.

    Returns:
        AlgobetConfig: The configuration singleton
    """
    global _config
    if _config is None:
        _config = AlgobetConfig()
    return _config


def reload_config() -> AlgobetConfig:
    """Reload configuration from environment.

    Useful for testing or when config files change.

    Returns:
        AlgobetConfig: The reloaded configuration
    """
    global _config
    _config = AlgobetConfig()
    return _config


def set_config(config: AlgobetConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: Configuration instance to set
    """
    global _config
    _config = config


# Convenience exports
config = get_config()
