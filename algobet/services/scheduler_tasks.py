"""Registration of task types for the scheduler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from algobet.importers.soccerdata_importer import SoccerDataImporter
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


def execute_import_fbref(session: Any, parameters: dict[str, Any]) -> dict[str, Any]:
    """Execute FBref import task via soccerdata."""
    league = parameters.get("league", "ENG-Premier League")
    season = parameters.get("season")
    no_cache = parameters.get("no_cache", False)

    if not season:
        return {
            "status": "failed",
            "error": "Season parameter is required (e.g., '2425', '2024')",
        }

    importer = SoccerDataImporter(session)
    result = importer.import_schedule(league=league, season=season, no_cache=no_cache)

    return {
        "status": "completed" if result.success else "failed",
        "league": league,
        "season": season,
        "matches_imported": result.progress.matches_created,
        "matches_skipped": result.progress.matches_skipped,
        "errors": result.progress.errors,
    }


def execute_enrich_stats(session: Any, parameters: dict[str, Any]) -> dict[str, Any]:
    """Execute soccerdata stats enrichment (Understat + ESPN)."""
    league = parameters.get("league", "ENG-Premier League")
    season = parameters.get("season", "2024")

    importer = SoccerDataImporter(session)
    result = importer.enrich_all(league=league, season=season)

    return {
        "status": "completed",
        "league": league,
        "season": season,
        **result,
    }


def register_default_tasks() -> None:
    """Register all default task types with the SchedulerService."""
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

    SchedulerService.register_task(
        TaskDefinition(
            name="Import from FBref",
            task_type="import_fbref",
            description="Import match schedules from FBref via soccerdata",
            default_parameters={
                "league": "ENG-Premier League",
                "season": None,
                "no_cache": False,
            },
            execute=execute_import_fbref,
        )
    )

    SchedulerService.register_task(
        TaskDefinition(
            name="Enrich Match Statistics",
            task_type="enrich_stats",
            description="Enrich matches with xG (Understat) and player stats (ESPN)",
            default_parameters={
                "league": "ENG-Premier League",
                "season": "2024",
            },
            execute=execute_enrich_stats,
        )
    )
