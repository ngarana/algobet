"""AlgoBet custom exceptions.

Provides a hierarchy of custom exceptions with proper exit codes
for CLI and API error handling.
"""

from __future__ import annotations

from typing import Any


class AlgoBetError(Exception):
    """Base exception for all AlgoBet errors.

    Attributes:
        message: Human-readable error message
        exit_code: Exit code for CLI applications
        details: Additional error details for debugging
    """

    exit_code: int = 1

    def __init__(
        self,
        message: str,
        *,
        exit_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            exit_code: Override default exit code
            details: Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        if exit_code is not None:
            self.exit_code = exit_code
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# =============================================================================
# Database Errors (exit_code: 10-19)
# =============================================================================


class DatabaseError(AlgoBetError):
    """Database operation failed."""

    exit_code = 10


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database."""

    exit_code = 11


class DatabaseQueryError(DatabaseError):
    """Database query failed."""

    exit_code = 12


class DatabaseMigrationError(DatabaseError):
    """Database migration failed."""

    exit_code = 13


# =============================================================================
# Model Errors (exit_code: 20-29)
# =============================================================================


class ModelError(AlgoBetError):
    """Model-related error."""

    exit_code = 20


class ModelNotFoundError(ModelError):
    """Requested model not found."""

    exit_code = 21


class ModelLoadError(ModelError):
    """Failed to load model."""

    exit_code = 22


class ModelSaveError(ModelError):
    """Failed to save model."""

    exit_code = 23


class NoActiveModelError(ModelError):
    """No active model available."""

    exit_code = 24

    def __init__(self, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            "No active model found. Train or activate a model first.", details=details
        )


class ModelValidationError(ModelError):
    """Model validation failed."""

    exit_code = 25


# =============================================================================
# Data Errors (exit_code: 30-39)
# =============================================================================


class DataError(AlgoBetError):
    """Data-related error."""

    exit_code = 30


class InsufficientDataError(DataError):
    """Not enough data for operation."""

    exit_code = 31


class DataNotFoundError(DataError):
    """Requested data not found."""

    exit_code = 32


class DataValidationError(DataError):
    """Data validation failed."""

    exit_code = 33


class DataImportError(DataError):
    """Data import failed."""

    exit_code = 34


class DataExportError(DataError):
    """Data export failed."""

    exit_code = 35


# =============================================================================
# Scraping Errors (exit_code: 40-49)
# =============================================================================


class ScrapingError(AlgoBetError):
    """Web scraping operation failed."""

    exit_code = 40


class ScrapingConnectionError(ScrapingError):
    """Failed to connect to scraping target."""

    exit_code = 41


class ScrapingTimeoutError(ScrapingError):
    """Scraping operation timed out."""

    exit_code = 42


class ScrapingParseError(ScrapingError):
    """Failed to parse scraped data."""

    exit_code = 43


class ScrapingBlockedError(ScrapingError):
    """Scraping blocked by target site."""

    exit_code = 44


# =============================================================================
# Configuration Errors (exit_code: 50-59)
# =============================================================================


class ConfigurationError(AlgoBetError):
    """Configuration error."""

    exit_code = 50


class ConfigFileNotFoundError(ConfigurationError):
    """Configuration file not found."""

    exit_code = 51


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""

    exit_code = 52


class ConfigParseError(ConfigurationError):
    """Failed to parse configuration."""

    exit_code = 53


# =============================================================================
# Prediction Errors (exit_code: 60-69)
# =============================================================================


class PredictionError(AlgoBetError):
    """Prediction operation failed."""

    exit_code = 60


class PredictionValidationError(PredictionError):
    """Prediction validation failed."""

    exit_code = 61


class PredictionFeatureError(PredictionError):
    """Feature generation for prediction failed."""

    exit_code = 62


class PredictionCalibrationError(PredictionError):
    """Model calibration failed."""

    exit_code = 63


# =============================================================================
# Service Errors (exit_code: 70-79)
# =============================================================================


class ServiceError(AlgoBetError):
    """Service operation failed."""

    exit_code = 70


class ServiceUnavailableError(ServiceError):
    """Service is unavailable."""

    exit_code = 71


class ServiceTimeoutError(ServiceError):
    """Service operation timed out."""

    exit_code = 72


# =============================================================================
# Validation Errors (exit_code: 80-89)
# =============================================================================


class ValidationError(AlgoBetError):
    """General validation error."""

    exit_code = 80


class InputValidationError(ValidationError):
    """User input validation failed."""

    exit_code = 81


class ParameterValidationError(ValidationError):
    """Parameter validation failed."""

    exit_code = 82


# =============================================================================
# Exit Code Reference
# =============================================================================

EXIT_CODES = {
    # General
    1: "General error",
    # Database (10-19)
    10: "Database error",
    11: "Database connection failed",
    12: "Database query failed",
    13: "Database migration failed",
    # Model (20-29)
    20: "Model error",
    21: "Model not found",
    22: "Failed to load model",
    23: "Failed to save model",
    24: "No active model",
    25: "Model validation failed",
    # Data (30-39)
    30: "Data error",
    31: "Insufficient data",
    32: "Data not found",
    33: "Data validation failed",
    34: "Data import failed",
    35: "Data export failed",
    # Scraping (40-49)
    40: "Scraping error",
    41: "Scraping connection failed",
    42: "Scraping timeout",
    43: "Scraping parse error",
    44: "Scraping blocked",
    # Configuration (50-59)
    50: "Configuration error",
    51: "Config file not found",
    52: "Config validation failed",
    53: "Config parse error",
    # Prediction (60-69)
    60: "Prediction error",
    61: "Prediction validation failed",
    62: "Feature generation failed",
    63: "Calibration failed",
    # Service (70-79)
    70: "Service error",
    71: "Service unavailable",
    72: "Service timeout",
    # Validation (80-89)
    80: "Validation error",
    81: "Input validation failed",
    82: "Parameter validation failed",
}


def get_exit_code_description(code: int) -> str:
    """Get human-readable description for exit code.

    Args:
        code: Exit code

    Returns:
        Description of the exit code
    """
    return EXIT_CODES.get(code, f"Unknown exit code: {code}")
