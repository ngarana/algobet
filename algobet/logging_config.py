"""Re-export of logging configuration from infrastructure for backward compatibility.

This module provides a single import point for logging utilities in the AlgoBet system.
For new code, consider importing directly from algobet.infrastructure.logging_config
to follow the feature-root architecture.
"""

from algobet.infrastructure.logging_config import (
    ColoredFormatter,
    JSONFormatter,
    LogContext,
    StructuredFormatter,
    get_logger,
    setup_logging,
    success,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "LogContext",
    "success",
    "ColoredFormatter",
    "JSONFormatter",
    "StructuredFormatter",
]
