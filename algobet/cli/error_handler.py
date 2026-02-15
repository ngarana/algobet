"""CLI error handling framework.

Provides centralized error handling for CLI commands with:
- Exception-to-exit-code mapping
- User-friendly error messages
- Debug mode with full stack traces
- Structured error logging
"""

from __future__ import annotations

import functools
import sys
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

import click

from algobet.cli.logger import debug, error
from algobet.config import get_config
from algobet.exceptions import AlgoBetError, get_exit_code_description

F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(func: F) -> F:
    """Decorator for automatic error handling in CLI commands.

    Catches exceptions, logs them appropriately, and exits with proper code.
    Supports debug mode for full stack traces.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with error handling

    Example:
        @cli.command()
        @handle_errors
        def my_command():
            raise DatabaseError("Connection failed")
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        config = get_config()
        is_debug = config.cli.debug

        try:
            return func(*args, **kwargs)

        except click.exceptions.Exit:
            # Re-raise Click's exit exceptions without modification
            raise

        except click.exceptions.Abort:
            # User aborted (Ctrl+C, etc.)
            error("Operation aborted by user")
            sys.exit(130)  # Standard exit code for Ctrl+C

        except AlgoBetError as e:
            # Handle our custom exceptions
            _handle_algobet_error(e, is_debug)

        except KeyboardInterrupt:
            # Handle Ctrl+C
            error("\nOperation interrupted by user")
            sys.exit(130)

        except Exception as e:
            # Handle unexpected exceptions
            _handle_unexpected_error(e, is_debug)

    return wrapper  # type: ignore


def _handle_algobet_error(e: AlgoBetError, is_debug: bool) -> None:
    """Handle AlgoBetError exceptions.

    Args:
        e: The exception
        is_debug: Whether debug mode is enabled
    """
    # Log the error
    error(f"{e.message}")

    # Log details if present
    if e.details:
        debug(f"Error details: {e.details}")

    # Show stack trace in debug mode
    if is_debug:
        error("\nStack trace:")
        traceback.print_exc()

    # Exit with appropriate code
    sys.exit(e.exit_code)


def _handle_unexpected_error(e: Exception, is_debug: bool) -> None:
    """Handle unexpected exceptions.

    Args:
        e: The exception
        is_debug: Whether debug mode is enabled
    """
    error(f"Unexpected error: {e}")

    if is_debug:
        error("\nFull stack trace:")
        traceback.print_exc()
    else:
        error("\nRun with --debug flag to see full stack trace")

    sys.exit(1)


class ErrorHandler:
    """Context manager for error handling.

    Alternative to decorator for cases where you need more control.

    Example:
        with ErrorHandler():
            risky_operation()
    """

    def __init__(self, *, suppress: bool = False) -> None:
        """Initialize error handler.

        Args:
            suppress: If True, don't exit on error (just log)
        """
        self.suppress = suppress
        self.config = get_config()

    def __enter__(self) -> ErrorHandler:
        """Enter context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context and handle any exception.

        Returns:
            True if exception was handled, False otherwise
        """
        if exc_val is None:
            return False

        is_debug = self.config.cli.debug

        if isinstance(exc_val, AlgoBetError):
            _handle_algobet_error(exc_val, is_debug)
            return not self.suppress
        elif isinstance(exc_val, Exception):
            _handle_unexpected_error(exc_val, is_debug)
            return not self.suppress

        return False


def format_error_message(e: AlgoBetError, include_details: bool = True) -> str:
    """Format an error message for display.

    Args:
        e: The exception
        include_details: Whether to include error details

    Returns:
        Formatted error message
    """
    parts = [f"Error: {e.message}"]

    if include_details and e.details:
        parts.append("\nDetails:")
        for key, value in e.details.items():
            parts.append(f"  {key}: {value}")

    parts.append(
        f"\nExit code: {e.exit_code} ({get_exit_code_description(e.exit_code)})"
    )

    return "\n".join(parts)


def exit_with_error(
    message: str, *, exit_code: int = 1, details: dict | None = None
) -> None:
    """Exit the application with an error.

    Convenience function for simple error exits.

    Args:
        message: Error message
        exit_code: Exit code
        details: Optional error details
    """
    error(message)

    if details:
        debug(f"Details: {details}")

    sys.exit(exit_code)


def confirm_or_exit(message: str, *, exit_code: int = 1) -> None:
    """Prompt for confirmation or exit.

    Args:
        message: Confirmation message
        exit_code: Exit code if user declines
    """
    if not click.confirm(message):
        error("Operation cancelled by user")
        sys.exit(exit_code)
