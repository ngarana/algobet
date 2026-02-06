"""Standalone scheduler worker process for running scheduled tasks."""

import signal
import sys
import time
from datetime import datetime
from typing import Any

import click

from algobet.database import session_scope
from algobet.services.scheduler_service import SchedulerService


class SchedulerWorker:
    """Worker process that executes scheduled tasks based on cron expressions."""

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.running = False
        self.last_check: dict[int, float] = {}

    def should_run_task(self, task: Any, scheduler: SchedulerService) -> bool:
        """Check if a task should be run based on its cron expression."""
        from croniter import croniter

        last_execution = scheduler.get_last_execution(task.id)
        if not last_execution:
            # Never run before, should run
            return True

        cron = croniter(task.cron_expression, last_execution.started_at)
        next_run = cron.get_next(datetime)

        # Run if next run time has passed
        return datetime.now() >= next_run  # type: ignore

    def run_once(self) -> None:
        """Check and execute all active scheduled tasks once."""
        with session_scope() as session:
            scheduler = SchedulerService(session)
            active_tasks = scheduler.get_active_schedules()

            for task in active_tasks:
                try:
                    if self.should_run_task(task, scheduler):
                        click.echo(f"\n[{datetime.now()}] Executing task: {task.name}")
                        execution = scheduler.execute_task(task.id)

                        if execution.status == "completed":
                            click.echo(f"✓ Completed in {execution.duration:.2f}s")
                            if execution.result:
                                click.echo(f"  Result: {execution.result}")
                        else:
                            click.echo(f"✗ Failed: {execution.error_message}", err=True)

                except Exception as e:
                    click.echo(f"✗ Error executing task '{task.name}': {e}", err=True)

    def run_forever(self) -> None:
        """Run the worker continuously."""
        self.running = True
        click.echo(f"Scheduler worker started (check interval: {self.check_interval}s)")
        click.echo("Press Ctrl+C to stop\n")

        try:
            while self.running:
                self.run_once()
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            click.echo("\n\nShutting down scheduler worker...")
            self.running = False

    def stop(self) -> None:
        """Stop the worker."""
        self.running = False


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
    worker = SchedulerWorker(check_interval=interval)

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
