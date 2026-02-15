"""CLI runner for scheduled tasks to be used with cron."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from algobet.database import session_scope
from algobet.services.prediction_service import PredictionService
from algobet.services.scheduler_service import SchedulerService, TaskDefinition
from algobet.services.scraping_service import ScrapingService


def execute_scrape_upcoming(session: Any, parameters: dict[str, Any]) -> dict[str, Any]:
    """Execute upcoming matches scraping task."""
    scraping_service = ScrapingService(session)
    progress = scraping_service.scrape_upcoming(
        url=parameters.get("url", "https://www.oddsportal.com/matches/football/"),
        headless=parameters.get("headless", True),
    )
    return {
        "status": "completed",
        "matches_scraped": progress.matches_scraped,
        "matches_saved": progress.matches_saved,
    }


def execute_scrape_results(session: Any, parameters: dict[str, Any]) -> dict[str, Any]:
    """Execute results scraping task."""
    scraping_service = ScrapingService(session)
    progress = scraping_service.scrape_results(
        url=parameters["url"],
        max_pages=parameters.get("max_pages"),
        headless=parameters.get("headless", True),
    )
    return {
        "status": "completed",
        "pages_scraped": progress.current_page,
        "matches_scraped": progress.matches_scraped,
        "matches_saved": progress.matches_saved,
    }


def execute_generate_predictions(
    session: Any, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Execute prediction generation task."""
    models_path = Path(parameters.get("models_path", "data/models"))
    prediction_service = PredictionService(session, models_path=models_path)

    predictions = prediction_service.predict_upcoming(
        days_ahead=parameters.get("days_ahead", 7),
        tournament_name=parameters.get("tournament_name"),
        min_confidence=parameters.get("min_confidence", 0.0),
        model_version=parameters.get("model_version"),
    )

    # Save predictions
    saved = prediction_service.save_predictions(predictions)

    return {
        "status": "completed",
        "predictions_generated": len(predictions),
        "predictions_saved": len(saved),
    }


# Register task definitions with the scheduler
SchedulerService.register_task(
    TaskDefinition(
        name="Scrape Upcoming Matches",
        task_type="scrape_upcoming",
        description="Scrape upcoming matches from OddsPortal",
        default_parameters={
            "url": "https://www.oddsportal.com/matches/football/",
            "headless": True,
        },
        execute=execute_scrape_upcoming,
    )
)

SchedulerService.register_task(
    TaskDefinition(
        name="Scrape Historical Results",
        task_type="scrape_results",
        description="Scrape historical match results from OddsPortal",
        default_parameters={
            "url": "",
            "max_pages": None,
            "headless": True,
        },
        execute=execute_scrape_results,
    )
)

SchedulerService.register_task(
    TaskDefinition(
        name="Generate Predictions",
        task_type="generate_predictions",
        description="Generate AI predictions for upcoming matches",
        default_parameters={
            "days_ahead": 7,
            "tournament_name": None,
            "min_confidence": 0.0,
            "model_version": None,
            "models_path": "data/models",
        },
        execute=execute_generate_predictions,
    )
)


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
