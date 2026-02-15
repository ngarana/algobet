"""Development tools CLI for AlgoBet.

This module provides a unified CLI for development and administrative tasks.
Commands are organized into logical groups:
- db: Database management (init, reset, stats)
- list: Query/list commands (tournaments, teams, upcoming matches)
- model: Model management (list, delete)
- analyze: Prediction analysis (backtest, value-bets, calibrate)
- async-db: Async database management commands
- async-list: Async query/list commands
"""

from __future__ import annotations

import click

from algobet.cli.commands.analyze import analyze_cli
from algobet.cli.commands.async_db import async_db_cli
from algobet.cli.commands.async_query import async_list_cli
from algobet.cli.commands.db import db_cli
from algobet.cli.commands.models import model_cli
from algobet.cli.commands.query import list_cli
from algobet.cli.commands.train import train_cli
from algobet.cli.container import get_container
from algobet.cli.logger import init_logging
from algobet.scheduler.worker import cli as scheduler_cli

# Initialize logging on module import
init_logging()


@click.group()
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (shows stack traces, verbose output)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def cli(
    ctx: click.Context, debug: bool, verbose: bool, config_file: str | None
) -> None:
    """AlgoBet Development Tools.

    Provides administrative and development commands for managing
    the AlgoBet database, models, and predictions.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store options in context
    ctx.obj["debug"] = debug
    ctx.obj["verbose"] = verbose

    # Initialize DI container
    get_container()

    # Reload config if config file specified
    if config_file:
        # TODO: Implement config file loading
        click.echo(f"Loading config from: {config_file}")


# Register command groups (sync)
cli.add_command(db_cli)
cli.add_command(list_cli)
cli.add_command(model_cli)
cli.add_command(analyze_cli)
cli.add_command(train_cli)

# Register async command groups
cli.add_command(async_db_cli)
cli.add_command(async_list_cli)

# Register scheduler worker
cli.add_command(scheduler_cli, name="scheduler")


if __name__ == "__main__":
    cli()
