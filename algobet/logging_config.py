"""AlgoBet logging configuration.

Provides structured logging with multiple output formats:
- text: Human-readable format for development
- json: Machine-readable format for production
- structured: Colored, structured text format

Includes a custom SUCCESS log level for successful operations.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from algobet.config import AlgobetConfig, LoggingConfig


# Add custom SUCCESS log level between INFO and WARNING
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success(self, message: str, *args: Any, **kwargs: Any) -> None:
    """Log a success message."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Add success method to Logger class
logging.Logger.success = success  # type: ignore


class ColoredFormatter(logging.Formatter):
    """Formatter with color support for terminal output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "SUCCESS": "\033[92m",  # Bright Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_colors: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        # Save original levelname
        original_levelname = record.levelname

        if self.use_colors:
            levelname = original_levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )

        result = super().format(record)

        # Restore original levelname to avoid side effects
        record.levelname = original_levelname

        return result


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        command = getattr(record, "command", None)
        if command:
            log_data["command"] = command
        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms:
            log_data["duration_ms"] = duration_ms
        extra = getattr(record, "extra", None)
        if extra:
            log_data.update(extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class StructuredFormatter(logging.Formatter):
    """Structured text formatter with visual separation."""

    LEVEL_SYMBOLS = {
        "DEBUG": "🔍",
        "INFO": "ℹ️",
        "SUCCESS": "✓",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔥",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in structured text format."""
        symbol = self.LEVEL_SYMBOLS.get(record.levelname, "•")
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Build message
        parts = [f"[{timestamp}] {symbol} {record.getMessage()}"]

        # Add extra context if present
        command = getattr(record, "command", None)
        if command:
            parts.append(f"[{command}]")
        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms:
            parts.append(f"({duration_ms:.0f}ms)")

        return " ".join(parts)


def get_formatter(config: LoggingConfig) -> logging.Formatter:
    """Get the appropriate formatter based on configuration.

    Args:
        config: Logging configuration

    Returns:
        logging.Formatter: Configured formatter
    """
    if config.format == "json":
        return JSONFormatter()
    elif config.format == "structured":
        return StructuredFormatter()
    else:
        # Text format with optional colors
        fmt = "%(message)s"
        if config.show_timestamp:
            fmt = "[%(asctime)s] " + fmt
        if config.show_level and config.format != "structured":
            fmt = "%(levelname)s: " + fmt

        return ColoredFormatter(
            fmt=fmt,
            datefmt="%H:%M:%S",
            use_colors=config.color,
        )


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Set up logging with the specified configuration.

    Args:
        config: Logging configuration. If None, uses default config.

    Returns:
        logging.Logger: The configured root logger
    """
    if config is None:
        from algobet.config import get_config

        config = get_config().logging

    # Get root logger
    logger = logging.getLogger("algobet")
    logger.setLevel(getattr(logging, config.level))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = get_formatter(config)

    # Console handler
    if config.output in ("stdout", "stderr", "both"):
        stream = sys.stdout if config.output == "stdout" else sys.stderr
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if config.output in ("file", "both") and config.file_path:
        from logging.handlers import RotatingFileHandler

        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setFormatter(JSONFormatter())  # Always use JSON for files
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name. If None, returns the root algobet logger.

    Returns:
        logging.Logger: Configured logger instance
    """
    if name:
        return logging.getLogger(f"algobet.{name}")
    return logging.getLogger("algobet")


class LogContext:
    """Context manager for adding context to logs.

    Example:
        with LogContext(command="db.init"):
            logger.info("Initializing database")
            # Logs will include command="db.init"
    """

    def __init__(self, **kwargs: Any):
        self.context = kwargs
        self.logger = get_logger()

    def __enter__(self) -> LogContext:
        """Add context to logger."""
        # Store original filters
        self.original_filters = list(self.logger.filters)

        # Capture context in closure
        context = self.context

        # Add context filter
        class ContextFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                for key, value in context.items():
                    setattr(record, key, value)
                return True

        self.logger.addFilter(ContextFilter())
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Remove context from logger."""
        # Restore original filters
        self.logger.filters = self.original_filters


# Module-level logger
logger = get_logger()
