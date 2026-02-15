"""Async CLI runner for Click commands.

This module provides utilities for running async functions from Click commands,
including a decorator that handles event loop management automatically.

Usage:
    from algobet.cli.async_runner import click_async

    @click.command()
    @click_async
    async def my_command():
        await some_async_operation()

The @click_async decorator handles:
- Creating and managing the event loop
- Proper cleanup on exit
- Integration with Click's exception handling
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import click

T = TypeVar("T")


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop.

    This handles different Python versions and environments:
    - Python 3.10+: Use get_running_loop() or new_event_loop()
    - Python 3.7-3.9: Use get_event_loop() or new_event_loop()

    Returns:
        AbstractEventLoop: The event loop to use.
    """
    try:
        # Check if there's already a running loop
        return asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one
        return asyncio.new_event_loop()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    This is a wrapper around asyncio.run() that handles:
    - Existing event loops (nesting)
    - Proper cleanup
    - Exception handling

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        # Check if there's already a running event loop
        asyncio.get_running_loop()
        # If we get here, we're inside an async context
        # Use nest_asyncio to allow nesting (for testing)
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(coro)
    except RuntimeError:
        # No running loop, we can use asyncio.run directly
        return asyncio.run(coro)


def click_async(
    f: Callable[..., Coroutine[Any, Any, T]] | None = None,
    *,
    timeout: float | None = None,
) -> (
    Callable[..., T]
    | Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., T]]
):
    """Decorator to make a Click command async.

    This decorator wraps an async Click command function and handles
    the event loop management automatically.

    Args:
        f: The async function to wrap.
        timeout: Optional timeout in seconds for the command.

    Returns:
        A synchronous function that can be used as a Click command.

    Example:
        @click.command()
        @click.option("--name", default="World")
        @click_async
        async def hello(name: str):
            await asyncio.sleep(0.1)
            click.echo(f"Hello, {name}!")
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            coro = func(*args, **kwargs)

            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout=timeout)

            try:
                return run_async(coro)
            except asyncio.TimeoutError:
                raise click.ClickException(
                    f"Command timed out after {timeout} seconds"
                ) from None
            except Exception:
                # Let Click's exception handling deal with it
                raise

        return wrapper

    if f is None:
        # Called with arguments: @click_async(timeout=30)
        return decorator
    else:
        # Called without arguments: @click_async
        return decorator(f)


def click_async_pass_context(
    f: Callable[..., Coroutine[Any, Any, T]] | None = None,
    *,
    timeout: float | None = None,
) -> (
    Callable[..., T]
    | Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., T]]
):
    """Decorator for async Click commands that need the context.

    This is similar to @click_async but ensures the Click context
    is properly passed through.

    Args:
        f: The async function to wrap.
        timeout: Optional timeout in seconds.

    Returns:
        A synchronous function that can be used as a Click command.

    Example:
        @click.command()
        @click.pass_context
        @click_async_pass_context
        async def my_command(ctx: click.Context):
            debug = ctx.obj.get("debug", False)
            await do_something(debug)
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
        @functools.wraps(func)
        @click.pass_context  # type: ignore[arg-type]
        def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> T:
            coro = func(ctx, *args, **kwargs)

            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout=timeout)

            try:
                return run_async(coro)
            except asyncio.TimeoutError:
                raise click.ClickException(
                    f"Command timed out after {timeout} seconds"
                ) from None
            except Exception:
                raise

        return wrapper

    if f is None:
        return decorator
    else:
        return decorator(f)


class AsyncCommand(click.Command):
    """A Click command that supports async callbacks.

    This is an alternative to using the @click_async decorator
    when you need more control over the command class.

    Example:
        @click.command(cls=AsyncCommand)
        @click.option("--url")
        async def scrape(url: str):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    click.echo(data)
    """

    def invoke(self, ctx: click.Context) -> Any:
        """Invoke the command, handling async callbacks."""
        callback = self.callback

        if callback is None:
            return None

        if asyncio.iscoroutinefunction(callback):
            # Get the callback with parameters resolved
            coro = ctx.invoke(callback)

            # Run the coroutine
            try:
                return run_async(coro)
            except Exception as e:
                # Handle the exception through Click's error handling
                ctx.fail(str(e))
        else:
            # Synchronous callback, use normal invocation
            return super().invoke(ctx)


class AsyncGroup(click.Group):
    """A Click group that supports async commands.

    This group can contain both sync and async commands.

    Example:
        @click.group(cls=AsyncGroup)
        def cli():
            pass

        @cli.command(cls=AsyncCommand)
        async def async_cmd():
            await do_something()

        @cli.command()
        def sync_cmd():
            do_something()
    """

    def invoke(self, ctx: click.Context) -> Any:
        """Invoke the group, handling async callbacks."""
        # Groups can have async callbacks too
        if asyncio.iscoroutinefunction(self.callback):
            coro = ctx.invoke(self.callback)
            return run_async(coro)
        return super().invoke(ctx)


# Convenience function for running multiple async tasks
async def gather_with_concurrency(
    *coros: Coroutine[Any, Any, T],
    limit: int = 10,
) -> list[T]:
    """Run multiple coroutines with a concurrency limit.

    Args:
        *coros: Coroutines to run.
        limit: Maximum number of concurrent tasks.

    Returns:
        List of results in the same order as input coroutines.
    """
    semaphore = asyncio.Semaphore(limit)

    async def run_with_semaphore(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[run_with_semaphore(c) for c in coros],
        return_exceptions=False,
    )


# Utility for running async code in sync context with proper cleanup
class AsyncRunner:
    """Context manager for running async code in sync context.

    Example:
        with AsyncRunner() as runner:
            result = runner.run(my_async_function())
            # Or run multiple
            results = runner.run_all([
                async_func1(),
                async_func2(),
            ])
    """

    def __init__(self, timeout: float | None = None):
        """Initialize the runner.

        Args:
            timeout: Default timeout for operations.
        """
        self.timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None

    def __enter__(self) -> AsyncRunner:
        """Enter the context and create an event loop."""
        self._loop = get_event_loop()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and clean up."""
        if self._loop is not None:
            # Run any pending tasks
            pending = asyncio.all_tasks(self._loop)
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending))
            self._loop.close()
            self._loop = None

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a single coroutine.

        Args:
            coro: The coroutine to run.

        Returns:
            The result of the coroutine.
        """
        if self._loop is None:
            raise RuntimeError("AsyncRunner not in context")

        if self.timeout is not None:
            coro = asyncio.wait_for(coro, timeout=self.timeout)

        return self._loop.run_until_complete(coro)

    def run_all(
        self, coros: list[Coroutine[Any, Any, T]], concurrency: int = 10
    ) -> list[T]:
        """Run multiple coroutines with concurrency limit.

        Args:
            coros: List of coroutines to run.
            concurrency: Maximum concurrent tasks.

        Returns:
            List of results.
        """
        return self.run(gather_with_concurrency(*coros, limit=concurrency))
