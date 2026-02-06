"""Standalone scheduler worker process for running scheduled tasks."""

import signal
import sys
from datetime import datetime
from typing import Any

import click

from algobet.database import session_scope
from algobet.services.scheduler_service import SchedulerService


class SchedulerWorker:
    """Worker process that executes scheduled tasks using APScheduler.

    This worker starts the APScheduler instance and loads all active tasks
    from the database for automated execution.
    """

    def __init__(self) -> None:
        self.running = False

    def start_scheduler(self) -> None:
        """Start the APScheduler and load all active tasks."""
        try:
            click.echo("Starting APScheduler...")
            SchedulerService.start_scheduler()

            click.echo("Loading active tasks from database...")
            SchedulerService.load_all_active_tasks()

            # Display loaded tasks
            with session_scope() as session:
                scheduler = SchedulerService(session)
                active_tasks = scheduler.get_active_schedules()
                click.echo(f"Loaded {len(active_tasks)} active tasks:")
                for task in active_tasks:
                    last_run = "Never"
                    last_execution = scheduler.get_last_execution(task.id)
                    if last_execution:
                        last_run = last_execution.started_at.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    click.echo(
                        f"  - {task.name} (cron: {task.cron_expression}, "
                        f"last: {last_run})"
                    )

            click.echo("\nScheduler is running. Press Ctrl+C to stop.\n")

        except Exception as e:
            click.echo(f"Failed to start scheduler: {e}", err=True)
            raise

    def shutdown_scheduler(self) -> None:
        """Shutdown the APScheduler gracefully."""
        try:
            click.echo("\nShutting down scheduler...")
            SchedulerService.shutdown_scheduler()
            click.echo("Scheduler stopped.")
        except Exception as e:
            click.echo(f"Error shutting down scheduler: {e}", err=True)

    def run_forever(self) -> None:
        """Run the worker continuously with APScheduler."""
        self.running = True

        # Start scheduler
        self.start_scheduler()

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum: int, frame: Any) -> None:
            self.running = False
            self.shutdown_scheduler()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep the process alive - APScheduler runs in background threads
        try:
            while self.running:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            self.shutdown_scheduler()

    def run_once(self) -> None:
        """Run a single check of all active tasks (for testing/one-shot execution)."""
        with session_scope() as session:
            scheduler = SchedulerService(session)
            active_tasks = scheduler.get_active_schedules()

            click.echo(f"\nChecking {len(active_tasks)} active tasks...\n")

            for task in active_tasks:
                try:
                    last_execution = scheduler.get_last_execution(task.id)
                    if last_execution:
                        click.echo(f"Task: {task.name}")
                        click.echo(f"  Last run: {last_execution.started_at}")
                        click.echo(f"  Status: {last_execution.status}")
                        if last_execution.duration:
                            click.echo(f"  Duration: {last_execution.duration:.2f}s")
                    else:
                        click.echo(f"Task: {task.name} (never executed)")
                    click.echo()

                except Exception as e:
                    click.echo(f"Error checking task '{task.name}': {e}", err=True)

    def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        self.shutdown_scheduler()


@click.group()
def cli() -> None:
    """AlgoBet Scheduler Worker."""
    pass


@cli.command()
@click.option(
    "--interval",
    "-i",
    default=60,
    help="Check interval in seconds (default: 60)",
)
@click.option(
    "--once",
    is_flag=True,
    help="Run once and exit",
)
def run(interval: int, once: bool) -> None:
    """Run the scheduler worker."""
    worker = SchedulerWorker()

    if once:
        worker.run_once()
    else:
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum: int, frame: Any) -> None:
            worker.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        worker.run_forever()


@cli.command()
def check() -> None:
    """Check which tasks would be run now without executing them."""
    with session_scope() as session:
        scheduler = SchedulerService(session)
        active_tasks = scheduler.get_active_schedules()

        click.echo("\nActive Scheduled Tasks:\n")

        for task in active_tasks:
            should_run = False
            last_run = "Never"

            try:
                last_execution = scheduler.get_last_execution(task.id)
                if last_execution:
                    from croniter import croniter

                    cron = croniter(task.cron_expression, last_execution.started_at)
                    next_run = cron.get_next(datetime)
                    should_run = datetime.now() >= next_run
                    last_run = last_execution.started_at.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

            status = "READY" if should_run else "WAITING"
            click.echo(f"  {task.name}")
            click.echo(f"    Status: {status}")
            click.echo(f"    Cron: {task.cron_expression}")
            click.echo(f"    Last run: {last_run}")
            click.echo()


if __name__ == "__main__":
    cli()
