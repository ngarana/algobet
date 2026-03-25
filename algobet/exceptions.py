"""Re-export of exceptions from infrastructure for backward compatibility.

This module provides a single import point for all custom exceptions in the AlgoBet system.
For new code, consider importing directly from algobet.infrastructure.exceptions
to follow the feature-root architecture.
"""

from algobet.infrastructure.exceptions import (
    AlgoBetError,
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    DataError,
    DataExportError,
    DataImportError,
    DataNotFoundError,
    DataValidationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseMigrationError,
    DatabaseQueryError,
    EXIT_CODES,
    get_exit_code_description,
    InputValidationError,
    InsufficientDataError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    ModelSaveError,
    ModelValidationError,
    NoActiveModelError,
    ParameterValidationError,
    PredictionCalibrationError,
    PredictionError,
    PredictionFeatureError,
    PredictionValidationError,
    ScrapingBlockedError,
    ScrapingConnectionError,
    ScrapingError,
    ScrapingParseError,
    ScrapingTimeoutError,
    ServiceError,
    ServiceTimeoutError,
    ServiceUnavailableError,
    ValidationError,
)

__all__ = [
    # Base
    "AlgoBetError",
    # Database
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseMigrationError",
    # Model
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelSaveError",
    "NoActiveModelError",
    "ModelValidationError",
    # Data
    "DataError",
    "InsufficientDataError",
    "DataNotFoundError",
    "DataValidationError",
    "DataImportError",
    "DataExportError",
    # Scraping
    "ScrapingError",
    "ScrapingConnectionError",
    "ScrapingTimeoutError",
    "ScrapingParseError",
    "ScrapingBlockedError",
    # Configuration
    "ConfigurationError",
    "ConfigFileNotFoundError",
    "ConfigParseError",
    "ConfigValidationError",
    # Prediction
    "PredictionError",
    "PredictionValidationError",
    "PredictionFeatureError",
    "PredictionCalibrationError",
    # Service
    "ServiceError",
    "ServiceUnavailableError",
    "ServiceTimeoutError",
    # Validation
    "ValidationError",
    "InputValidationError",
    "ParameterValidationError",
    # Utilities
    "EXIT_CODES",
    "get_exit_code_description",
]
