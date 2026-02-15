"""CLI logging utilities.

Provides integration between the logging system and Click CLI commands.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import click

from algobet.config import get_config
from algobet.logging_config import LogContext, get_logger, setup_logging

F = TypeVar("F", bound=Callable[..., Any])


def init_logging() -> None:
    """Initialize logging for CLI commands.

    This should be called once when the CLI starts.
    """
    config = get_config()
    setup_logging(config.logging)


def log_command(func: F) -> F:
    """Decorator to add logging context to a CLI command.

    Automatically adds command name to log records and handles errors.

    Example:
        @cli.command()
        @log_command
        def my_command():
            logger.info("Doing something")
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get command name from the function
        command_name = func.__name__.replace("_", " ")

        # Set up logging context
        with LogContext(command=command_name):
            logger = get_logger("cli")
            logger.debug(f"Executing command: {command_name}")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Command completed: {command_name}")
                return result
            except Exception as e:
                logger.error(f"Command failed: {command_name} - {e}")
                raise

    return wrapper  # type: ignore


class EchoHandler(logging.Handler):
    """Logging handler that outputs via Click's echo function.

    This ensures proper handling of Unicode, colors, and TTY detection.
    """

    def __init__(self, level: int = logging.INFO, use_stderr: bool = False):
        super().__init__(level)
        self.use_stderr = use_stderr

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record via Click."""
        try:
            msg = self.format(record)

            # Choose output stream
            err = self.use_stderr or record.levelno >= logging.WARNING

            # Add emoji based on level for structured format
            if hasattr(record, "levelname"):
                emoji = {
                    "DEBUG": "🔍",
                    "INFO": "ℹ️",
                    "SUCCESS": "✓",
                    "WARNING": "⚠️",
                    "ERROR": "❌",
                    "CRITICAL": "🔥",
                }.get(
                    record.levelname.replace("\x1b[0m", "").replace("\x1b[92m", ""), ""
                )
                if emoji and not msg.startswith(emoji):
                    msg = f"{emoji} {msg}"

            click.echo(msg, err=err)
        except Exception:
            self.handleError(record)


def get_cli_logger(name: str | None = None) -> Any:
    """Get a logger configured for CLI output.

    This logger will output via Click's echo function instead of
    standard stream handlers.

    Args:
        name: Logger name suffix

    Returns:
        Logger with Click echo handler
    """
    logger_name = f"algobet.cli.{name}" if name else "algobet.cli"
    logger = get_logger(logger_name)

    # Add Click echo handler if not already present
    has_echo_handler = any(isinstance(h, EchoHandler) for h in logger.handlers)

    if not has_echo_handler:
        echo_handler = EchoHandler()
        logger.addHandler(echo_handler)

    return logger


# Convenience functions for common logging patterns


def success(message: str) -> None:
    """Log a success message."""
    logger = get_cli_logger()
    logger.success(message)


def info(message: str) -> None:
    """Log an info message."""
    logger = get_cli_logger()
    logger.info(message)


def warning(message: str) -> None:
    """Log a warning message."""
    logger = get_cli_logger()
    logger.warning(message)


def error(message: str) -> None:
    """Log an error message."""
    logger = get_cli_logger()
    logger.error(message)


def debug(message: str) -> None:
    """Log a debug message."""
    logger = get_cli_logger()
    logger.debug(message)


# Module-level convenience logger
cli_logger = get_cli_logger()
