"""Seed default scheduled tasks into the database."""

from __future__ import annotations

import click

from algobet.database import session_scope
from algobet.services.scheduler_service import SchedulerService
from algobet.services.scheduler_tasks import register_default_tasks

DEFAULT_SCHEDULES = [
    {
        "name": "daily_upcoming_scrape_morning",
        "task_type": "scrape_upcoming",
        "cron_expression": "0 6 * * *",
        "parameters": {
            "url": "https://www.oddsportal.com/matches/football/",
            "headless": True,
        },
        "description": "Daily upcoming matches scrape at 6 AM",
        "is_active": True,
    },
    {
        "name": "daily_upcoming_scrape_evening",
        "task_type": "scrape_upcoming",
        "cron_expression": "0 18 * * *",
        "parameters": {
            "url": "https://www.oddsportal.com/matches/football/",
            "headless": True,
        },
        "description": "Daily upcoming matches scrape at 6 PM",
        "is_active": True,
    },
    {
        "name": "daily_predictions",
        "task_type": "generate_predictions",
        "cron_expression": "0 7 * * *",
        "parameters": {
            "days_ahead": 7,
            "tournament_name": None,
            "min_confidence": 0.0,
            "model_version": None,
            "models_path": "data/models",
        },
        "description": "Daily predictions at 7 AM",
        "is_active": True,
    },
    {
        "name": "weekly_results_scrape",
        "task_type": "scrape_results",
        "cron_expression": "0 3 * * 1",
        "parameters": {
            "url": "https://www.oddsportal.com/football/england/premier-league/results/",
            "max_pages": None,
            "headless": True,
        },
        "description": "Weekly results scrape on Monday at 3 AM",
        "is_active": True,
    },
]


@click.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be created without creating",
)
@click.option("--force", is_flag=True, help="Overwrite existing schedules")
def seed_schedules(dry_run: bool, force: bool) -> None:
    """Seed default scheduled tasks into the database."""
    # Ensure task types are registered
    register_default_tasks()

    with session_scope() as session:
        scheduler = SchedulerService(session)

        click.echo("\n" + "=" * 60)
        click.echo("AlgoBet Default Scheduled Tasks")
        click.echo("=" * 60 + "\n")

        created_count = 0
        skipped_count = 0
        updated_count = 0

        for schedule_config in DEFAULT_SCHEDULES:
            name: str = str(schedule_config["name"])
            existing = scheduler.get_schedule_by_name(name)

            if existing:
                if force:
                    if dry_run:
                        click.echo(f"[UPDATE] {name}")
                    else:
                        scheduler.update_schedule(
                            existing.id,
                            cron_expression=str(schedule_config.get("cron_expression")),
                            parameters=schedule_config.get("parameters"),  # type: ignore
                            description=str(schedule_config.get("description")),
                            is_active=bool(schedule_config.get("is_active")),
                        )
                        click.echo(f"✓ Updated: {name}")
                        updated_count += 1
                else:
                    click.echo(f"- Skipped (already exists): {name}")
                    skipped_count += 1
            else:
                if dry_run:
                    click.echo(f"[CREATE] {name}")
                else:
                    scheduler.create_schedule(
                        name=str(schedule_config["name"]),
                        task_type=str(schedule_config["task_type"]),
                        cron_expression=str(schedule_config["cron_expression"]),
                        parameters=schedule_config.get("parameters"),  # type: ignore
                        description=str(schedule_config["description"]),
                        is_active=bool(schedule_config["is_active"]),
                    )
                    click.echo(f"✓ Created: {name}")
                    created_count += 1

            click.echo(f"  Type: {schedule_config['task_type']}")
            click.echo(f"  Cron: {schedule_config['cron_expression']}")
            click.echo(f"  Description: {schedule_config['description']}")
            click.echo()

        click.echo("=" * 60)
        if dry_run:
            click.echo("DRY RUN - No changes made")
        else:
            click.echo(f"Created: {created_count}")
            click.echo(f"Updated: {updated_count}")
            click.echo(f"Skipped: {skipped_count}")
        click.echo("=" * 60 + "\n")


if __name__ == "__main__":
    seed_schedules()
