"""Command-line interface for AlgoBet prediction model management."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

import click
import numpy as np
from sqlalchemy import select
from tabulate import tabulate

from algobet.database import session_scope
from algobet.models import Match, Tournament
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.form_features import FormCalculator
from algobet.predictions.models.registry import ModelRegistry


@click.group()
def predictions():
    """AlgoBet prediction engine model management commands."""
    pass


@dataclass
class PredictionResult:
    """Result of a match prediction."""
    match_id: int
    match_date: datetime
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    model_version: str


@dataclass
class ValueBet:
    """Result of a value betting opportunity."""
    match_date: datetime
    home_team: str
    away_team: str
    outcome: str
    probability: float
    odds: float
    expected_value: float
    stake: float
    potential_profit: float


def _load_model(
    registry: ModelRegistry, model_version: Optional[str] = None
) -> tuple[Any, str]:
    """Load model from registry.

    Args:
        registry: ModelRegistry instance
        model_version: Optional specific version ID

    Returns:
        Tuple of (model object, version string)
    """
    if model_version:
        model = registry.load_model(model_version)
        return model, model_version
    else:
        model, metadata = registry.get_active_model()
        return model, metadata.version


def _query_matches(
    session,
    match_ids: Optional[list[int]] = None,
    tournament_name: Optional[str] = None,
    days_ahead: int = 7,
    status: str = "SCHEDULED"
) -> list[Match]:
    """Query matches based on filters.

    Args:
        session: Database session
        match_ids: Optional list of specific match IDs
        tournament_name: Optional tournament name filter
        days_ahead: Number of days ahead to look
        status: Match status filter

    Returns:
        List of Match objects
    """
    if match_ids:
        stmt = select(Match).where(Match.id.in_(match_ids))
    else:
        stmt = select(Match).where(Match.status == status)
        max_date = datetime.now() + timedelta(days=days_ahead)
        stmt = stmt.where(Match.match_date <= max_date)

        if tournament_name:
            stmt = stmt.join(Tournament).where(Tournament.name == tournament_name)

    stmt = stmt.order_by(Match.match_date)
    result = session.execute(stmt)
    return list(result.scalars().all())


def _generate_features(
    match: Match, repo: MatchRepository, calc: FormCalculator
) -> dict[str, float]:
    """Generate features for a match.

    Args:
        match: Match object
        repo: MatchRepository instance
        calc: FormCalculator instance

    Returns:
        Dictionary of features
    """
    home_form = calc.calculate_recent_form(
        team_id=match.home_team_id,
        reference_date=match.match_date,
        n_matches=5
    )
    away_form = calc.calculate_recent_form(
        team_id=match.away_team_id,
        reference_date=match.match_date,
        n_matches=5
    )
    home_goals = calc.calculate_goals_scored(
        team_id=match.home_team_id,
        reference_date=match.match_date,
        n_matches=5
    )
    away_goals = calc.calculate_goals_scored(
        team_id=match.away_team_id,
        reference_date=match.match_date,
        n_matches=5
    )
    home_conceded = calc.calculate_goals_conceded(
        team_id=match.home_team_id,
        reference_date=match.match_date,
        n_matches=5
    )
    away_conceded = calc.calculate_goals_conceded(
        team_id=match.away_team_id,
        reference_date=match.match_date,
        n_matches=5
    )

    return {
        "home_form": home_form,
        "away_form": away_form,
        "home_goals_scored": home_goals,
        "away_goals_scored": away_goals,
        "home_goals_conceded": home_conceded,
        "away_goals_conceded": away_conceded,
    }


def _get_prediction(
    model: Any, features: dict[str, float]
) -> tuple[str, float]:
    """Get prediction from model.

    Args:
        model: Loaded model object
        features: Feature dictionary

    Returns:
        Tuple of (predicted_outcome, confidence)
    """
    feature_array = np.array([list(features.values())])
    try:
        probs = model.predict_proba(feature_array)[0]
    except AttributeError:
        probs = model.predict(feature_array)[0]

    outcomes = ["HOME", "DRAW", "AWAY"]
    max_idx = int(np.argmax(probs))
    confidence = float(probs[max_idx])
    return outcomes[max_idx], confidence


def _output_predictions(
    predictions: list[PredictionResult],
    output_format: str = "table",
    output_file: Optional[str] = None
) -> None:
    """Output predictions in specified format.

    Args:
        predictions: List of prediction results
        output_format: Format (table, json, csv)
        output_file: Optional output file path
    """
    if output_format == "json":
        data = [
            {
                "match_id": p.match_id,
                "match_date": p.match_date.isoformat(),
                "home_team": p.home_team,
                "away_team": p.away_team,
                "predicted_outcome": p.predicted_outcome,
                "confidence": p.confidence,
                "model_version": p.model_version,
            }
            for p in predictions
        ]
        json_str = json.dumps(data, indent=2)
        if output_file:
            Path(output_file).write_text(json_str)
            click.echo(f"Predictions saved to {output_file}")
        else:
            click.echo(json_str)
    elif output_format == "csv":
        rows = [
            [
                p.match_id,
                p.match_date.isoformat(),
                p.home_team,
                p.away_team,
                p.predicted_outcome,
                p.confidence,
                p.model_version,
            ]
            for p in predictions
        ]
        if output_file:
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["match_id", "match_date", "home_team", "away_team",
                     "predicted_outcome", "confidence", "model_version"]
                )
                writer.writerows(rows)
            click.echo(f"Predictions saved to {output_file}")
        else:
            for row in rows:
                click.echo(",".join(str(x) for x in row))
    else:
        table_data = [
            [
                p.match_id,
                p.match_date.strftime("%Y-%m-%d %H:%M"),
                p.home_team,
                p.away_team,
                p.predicted_outcome,
                f"{p.confidence:.3f}",
                p.model_version,
            ]
            for p in predictions
        ]
        headers = ["ID", "Date", "Home", "Away", "Prediction", "Confidence", "Model"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@predictions.command("predict")
@click.option(
    "--match-id",
    multiple=True,
    type=int,
    help="Specific match IDs (can be specified multiple times)",
)
@click.option("--tournament", type=str, help="Filter by tournament name")
@click.option("--days", type=int, default=7, help="Days ahead to predict")
@click.option("--model-version", type=str, help="Specific model version ID")
@click.option("--min-confidence", type=float, default=0.0, help="Minimum confidence threshold")
@click.option("--output", type=str, help="Output file path")
@click.option(
    "--format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def predict(
    match_ids: tuple[int, ...],
    tournament: Optional[str],
    days: int,
    model_version: Optional[str],
    min_confidence: float,
    output: Optional[str],
    format: str,
) -> None:
    """Generate predictions for specific matches."""
    with session_scope() as session:
        registry = ModelRegistry(storage_path=Path("data/models"), session=session)

        try:
            model, version = _load_model(registry, model_version)
        except ValueError as e:
            click.echo(f"Error loading model: {e}", err=True)
            raise click.Abort()

        matches = _query_matches(
            session,
            match_ids=list(match_ids) if match_ids else None,
            tournament_name=tournament,
            days_ahead=days,
            status="SCHEDULED"
        )

        if not matches:
            click.echo("No matches found matching the criteria.")
            return

        repo = MatchRepository(session)
        calc = FormCalculator(repo)
        predictions = []

        for match in matches:
            features = _generate_features(match, repo, calc)
            outcome, confidence = _get_prediction(model, features)

            if confidence >= min_confidence:
                predictions.append(
                    PredictionResult(
                        match_id=match.id,
                        match_date=match.match_date,
                        home_team=match.home_team.name,
                        away_team=match.away_team.name,
                        predicted_outcome=outcome,
                        confidence=confidence,
                        model_version=version,
                    )
                )

        if not predictions:
            click.echo("No predictions meet the minimum confidence threshold.")
            return

        _output_predictions(predictions, format, output)


@predictions.command("upcoming")
@click.option("--tournament", type=str, help="Filter by tournament name")
@click.option("--days", type=int, default=7, help="Days ahead to look")
@click.option("--model-version", type=str, help="Specific model version ID")
def upcoming(
    tournament: Optional[str],
    days: int,
    model_version: Optional[str],
) -> None:
    """List upcoming matches with predictions."""
    with session_scope() as session:
        registry = ModelRegistry(storage_path=Path("data/models"), session=session)

        try:
            model, version = _load_model(registry, model_version)
        except ValueError as e:
            click.echo(f"Error loading model: {e}", err=True)
            raise click.Abort()

        matches = _query_matches(
            session,
            tournament_name=tournament,
            days_ahead=days,
            status="SCHEDULED"
        )

        if not matches:
            click.echo("No upcoming matches found.")
            return

        repo = MatchRepository(session)
        calc = FormCalculator(repo)
        predictions = []

        for match in matches:
            features = _generate_features(match, repo, calc)
            outcome, confidence = _get_prediction(model, features)

            predictions.append(
                PredictionResult(
                    match_id=match.id,
                    match_date=match.match_date,
                    home_team=match.home_team.name,
                    away_team=match.away_team.name,
                    predicted_outcome=outcome,
                    confidence=confidence,
                    model_version=version,
                )
            )

        table_data = [
            [
                p.match_date.strftime("%Y-%m-%d %H:%M"),
                p.home_team,
                p.away_team,
                p.predicted_outcome,
                f"{p.confidence:.3f}",
            ]
            for p in predictions
        ]
        headers = ["Date", "Home Team", "Away Team", "Prediction", "Confidence"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
        click.echo(f"\nShowing {len(predictions)} upcoming match(es)")


@predictions.command("list-models")
@click.option(
    "--model-type",
    type=click.Choice(["xgboost", "lightgbm", "random_forest", "ensemble"]),
    help="Filter by model algorithm type",
)
@click.option("--limit", type=int, default=20, help="Maximum number of models to show")
@click.option("--active-only", is_flag=True, help="Show only active models")
def list_models(model_type: Optional[str], limit: int, active_only: bool):
    """List registered models with filtering options."""
    with session_scope() as session:
        registry = ModelRegistry(storage_path=Path("data/models"), session=session)

        models = list(registry.list_models(model_type=model_type, active_only=active_only))

        if not models:
            click.echo("No models found matching the criteria.")
            return

        # Apply limit
        models = models[:limit]

        # Prepare table data
        table_data = []
        for model in models:
            # Format key metrics
            metrics_str = ""
            if model.metrics:
                key_metrics = ["accuracy", "log_loss", "precision", "recall", "f1"]
                metric_parts = []
                for key in key_metrics:
                    if key in model.metrics:
                        metric_parts.append(f"{key}={model.metrics[key]:.3f}")
                metrics_str = ", ".join(metric_parts)

            table_data.append([
                model.version,
                model.model_id.split("_")[0] if "_" in model.model_id else model.model_id,
                model.model_type,
                model.created_at.strftime("%Y-%m-%d %H:%M"),
                "Yes" if model.is_production else "No",
                metrics_str,
            ])

        headers = ["Version", "Name", "Algorithm", "Created", "Active", "Metrics"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
        click.echo(f"\nShowing {len(models)} model(s)")


@predictions.command("activate")
@click.argument("version_id")
def activate_model(version_id: str):
    """Activate a specific model version."""
    with session_scope() as session:
        registry = ModelRegistry(storage_path=Path("data/models"), session=session)

        try:
            registry.activate_model(version_id)
            click.echo(f"Model version '{version_id}' activated successfully.")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            raise click.Abort()


@predictions.command("delete-model")
@click.argument("version_id")
def delete_model(version_id: str):
    """Delete a model version from registry and disk."""
    if not click.confirm(f"Are you sure you want to delete model version '{version_id}'?"):
        click.echo("Deletion cancelled.")
        return

    with session_scope() as session:
        registry = ModelRegistry(storage_path=Path("data/models"), session=session)

        try:
            registry.delete_model(version_id)
            click.echo(f"Model version '{version_id}' deleted successfully.")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            raise click.Abort()


if __name__ == "__main__":
    predictions()