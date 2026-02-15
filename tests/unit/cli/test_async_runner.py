"""Unit tests for async runner module."""

import asyncio

import pytest

from algobet.cli.async_runner import AsyncRunner, click_async, run_async


class TestAsyncRunner:
    """Test cases for the async runner functionality."""

    def test_click_async_decorator_with_async_function_only(self):
        """Test that click_async decorator is meant for async functions."""

        # The decorator is designed for async functions only
        async def async_func():
            return "async result"

        decorated_func = click_async(async_func)
        # The decorated function should be different from the original
        assert decorated_func is not async_func
        # Calling it should return the expected result
        # (after running the async function)
        result = decorated_func()
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_click_async_decorator_async_function(self):
        """Test click_async decorator with an async function."""

        async def async_func():
            return "async result"

        decorated_func = click_async(async_func)

        # The decorated function should be callable and return the expected result
        result = decorated_func()  # Note: click_async makes it sync
        assert result == "async result"

    def test_run_async_utility(self):
        """Test run_async utility function."""

        async def test_coroutine():
            await asyncio.sleep(0.01)  # Small delay to simulate async work
            return "completed"

        result = run_async(test_coroutine())
        assert result == "completed"

    def test_async_runner_context_manager(self):
        """Test AsyncRunner context manager."""

        async def test_coroutine():
            await asyncio.sleep(0.01)
            return "completed inside runner"

        with AsyncRunner():
            result = run_async(test_coroutine())
            assert result == "completed inside runner"


def test_click_async_with_real_async_function():
    """Integration test for click_async with a real async function."""

    async def sample_async_command():
        # Simulate some async work
        await asyncio.sleep(0.001)
        return "command executed"

    # Apply the decorator
    decorated_command = click_async(sample_async_command)

    # Call the decorated function (it's now sync due to decorator)
    result = decorated_command()

    assert result == "command executed"
