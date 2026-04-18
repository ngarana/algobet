"""CLI package for AlgoBet development tools."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import to avoid circular import issues when running as module."""
    if name == "cli":
        from algobet.cli.dev_tools import cli

        return cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cli"]
