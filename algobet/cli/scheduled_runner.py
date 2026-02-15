"""CLI runner for scheduled tasks to be used with cron."""

from __future__ import annotations

import sys

import click

from algobet.database import session_scope
from algobet.services.scheduler_service import SchedulerService
from algobet.services.scheduler_tasks import register_default_tasks
from algobet.services.scraping_service import ScrapingService

# Register task definitions with the scheduler
register_default_tasks()


@click.group()
def cli() -> None:
    """AlgoBet Scheduled Task Runner."""
    pass


@cli.command()
@click.argument("task_name")
def run(task_name: str) -> None:
    """Run a scheduled task by name."""
    with session_scope() as session:
        scheduler = SchedulerService(session)
        task = scheduler.get_schedule_by_name(task_name)

        if not task:
            click.echo(f"Error: Scheduled task '{task_name}' not found", err=True)
            sys.exit(1)

        if not task.is_active:
            click.echo(f"Error: Scheduled task '{task_name}' is not active", err=True)
            sys.exit(1)

        click.echo(f"Executing task: {task.name} ({task.task_type})")
        click.echo(f"Parameters: {task.parameters}")

        try:
            execution = scheduler.execute_task(task.id)

            if execution.status == "completed":
                click.echo("✓ Task completed successfully")
                if execution.result:
                    click.echo(f"Result: {execution.result}")
            else:
                click.echo(f"✗ Task failed: {execution.error_message}", err=True)
                sys.exit(1)

        except Exception as e:
            click.echo(f"✗ Error executing task: {e}", err=True)
            sys.exit(1)


@cli.command()
def list_tasks() -> None:
    """List all available scheduled tasks."""
    with session_scope() as session:
        scheduler = SchedulerService(session)
        tasks = scheduler.list_schedules()

        if not tasks:
            click.echo("No scheduled tasks found.")
            return

        click.echo("\nScheduled Tasks:\n")
        for task in tasks:
            status = "✓ Active" if task.is_active else "✗ Inactive"
            click.echo(f"  {task.name} [{status}]")
            click.echo(f"    Type: {task.task_type}")
            click.echo(f"    Cron: {task.cron_expression}")
            if task.description:
                click.echo(f"    Description: {task.description}")

            last_exec = scheduler.get_last_execution(task.id)
            if last_exec:
                status_icon = "✓" if last_exec.status == "completed" else "✗"
                click.echo(
                    f"    Last run: {last_exec.started_at} "
                    f"({status_icon} {last_exec.status})"
                )
            click.echo()


@cli.command()
@click.argument("task_name")
def status(task_name: str) -> None:
    """Show status and execution history for a task."""
    with session_scope() as session:
        scheduler = SchedulerService(session)
        task = scheduler.get_schedule_by_name(task_name)

        if not task:
            click.echo(f"Error: Scheduled task '{task_name}' not found", err=True)
            sys.exit(1)

        click.echo(f"\nTask: {task.name}")
        click.echo(f"Type: {task.task_type}")
        click.echo(f"Status: {'Active' if task.is_active else 'Inactive'}")
        click.echo(f"Cron: {task.cron_expression}")
        if task.description:
            click.echo(f"Description: {task.description}")
        click.echo(f"Parameters: {task.parameters}")
        click.echo(f"Created: {task.created_at}")
        click.echo(f"Updated: {task.updated_at}")

        # Execution history
        history = scheduler.get_execution_history(task.id, limit=10)
        if history:
            click.echo(f"\nExecution History (last {len(history)} runs):\n")
            for exec in history:
                status_icon = "✓" if exec.status == "completed" else "✗"
                duration = f"{exec.duration:.2f}s" if exec.duration else "N/A"
                click.echo(
                    f"  {status_icon} {exec.started_at} ({exec.status}, {duration})"
                )
                if exec.result:
                    click.echo(f"    Result: {exec.result}")
                if exec.error_message:
                    click.echo(f"    Error: {exec.error_message}")
        else:
            click.echo("\nNo execution history.")


@cli.command()
@click.argument("task_name")
@click.option("--quick", is_flag=True, help="Quick scrape for upcoming matches only")
def quick_scrape(task_name: str, quick: bool) -> None:
    """Quick scrape operation for immediate execution."""
    with session_scope() as session:
        scraping_service = ScrapingService(session)

        if quick:
            click.echo("Quick scrape: fetching upcoming matches...")
            progress = scraping_service.scrape_upcoming(
                url="https://www.oddsportal.com/matches/football/",
                headless=True,
            )
            click.echo(f"✓ Scraped {progress.matches_scraped} matches")
            click.echo(f"✓ Saved {progress.matches_saved} matches")
        else:
            click.echo("Quick scrape requires --quick flag for upcoming matches")
            sys.exit(1)


if __name__ == "__main__":
    cli()
