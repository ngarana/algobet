"""Prediction analysis CLI commands."""

from __future__ import annotations

import json

import click

from algobet.cli.error_handler import handle_errors
from algobet.cli.presenters import display_value_bets
from algobet.database import session_scope
from algobet.services import AnalysisService
from algobet.services.dto import (
    BacktestRequest,
    CalibrateRequest,
    ValueBetsRequest,
)


@click.group(name="analyze")
def analyze_cli() -> None:
    """Prediction analysis and evaluation commands."""
    pass


@analyze_cli.command(name="backtest")
@click.option(
    "--model-version",
    "-m",
    help="Model version to backtest (default: active model)",
)
@click.option(
    "--min-matches",
    type=int,
    default=100,
    help="Minimum matches required for backtest",
)
@click.option(
    "--validation-split",
    type=float,
    default=0.2,
    help="Fraction of data for validation",
)
@handle_errors
def backtest(
    model_version: str | None,
    min_matches: int,
    validation_split: float,
) -> None:
    """Run historical backtest on model predictions."""
    with session_scope() as session:
        service = AnalysisService(session)
        request = BacktestRequest(
            min_matches=min_matches,
            validation_split=validation_split,
            model_version=model_version,
        )

        click.echo("\nRunning backtest...")
        response = service.run_backtest(request)

        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("BACKTEST RESULTS")
        click.echo("=" * 60)
        click.echo(f"\nModel: {response.model_version}")
        click.echo(f"Total matches: {response.total_matches:,}")
        click.echo(f"Training matches: {response.training_matches:,}")
        click.echo(f"Validation matches: {response.validation_matches:,}")
        click.echo(f"Execution time: {response.execution_time_seconds:.2f}s")

        # Display metrics
        click.echo("\n📊 Metrics:")
        click.echo("-" * 40)
        for name, value in response.metrics.items():
            if isinstance(value, float):
                if "accuracy" in name.lower() or "roi" in name.lower():
                    click.echo(f"  {name:20s}: {value:.2%}")
                else:
                    click.echo(f"  {name:20s}: {value:.4f}")
            else:
                click.echo(f"  {name:20s}: {value}")

        click.echo("\n" + "=" * 60)


@analyze_cli.command(name="value-bets")
@click.option(
    "--min-ev",
    type=float,
    default=0.05,
    help="Minimum expected value threshold",
)
@click.option(
    "--model-version",
    "-m",
    help="Model version to use",
)
@click.option(
    "--max-matches",
    type=int,
    default=20,
    help="Maximum number of value bets to display",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@handle_errors
def value_bets(
    min_ev: float,
    model_version: str | None,
    max_matches: int,
    output_format: str,
) -> None:
    """Find value betting opportunities."""
    with session_scope() as session:
        service = AnalysisService(session)
        request = ValueBetsRequest(
            min_edge=min_ev,
            model_version=model_version,
            limit=max_matches,
        )

        click.echo("\nFinding value bets...")
        response = service.find_value_bets(request)

        if not response.value_bets:
            click.echo(f"\nNo value bets found with edge ≥ {min_ev:.1%}")
            return

        # Convert to format expected by presenter
        value_bets_list = []
        for vb in response.value_bets:
            outcome_name_map = {
                "HOME_WIN": "Home Win",
                "DRAW": "Draw",
                "AWAY_WIN": "Away Win",
            }
            value_bets_list.append(
                {
                    "match_id": vb.match_id,
                    "date": (
                        vb.match_date.strftime("%Y-%m-%d %H:%M")
                        if vb.match_date
                        else "TBD"
                    ),
                    "home_team": vb.home_team,
                    "away_team": vb.away_team,
                    "outcome": vb.bet_type,
                    "outcome_name": outcome_name_map.get(vb.bet_type, vb.bet_type),
                    "predicted_prob": vb.model_probability,
                    "market_odds": vb.market_odds or 0.0,
                    "expected_value": vb.expected_value,
                    "edge": vb.edge,
                    "kelly_fraction": _calculate_kelly(
                        vb.model_probability, vb.market_odds
                    ),
                }
            )

        # Output results
        if output_format == "json":
            click.echo(json.dumps(value_bets_list, indent=2))
        else:
            display_value_bets(value_bets_list, min_ev)


def _calculate_kelly(probability: float, odds: float | None) -> float:
    """Calculate Kelly fraction for a bet."""
    if odds is None or odds <= 1.0:
        return 0.0
    b = odds - 1
    if b <= 0:
        return 0.0
    kelly = (probability * b - (1 - probability)) / b
    return max(0, kelly)


@analyze_cli.command(name="calibrate")
@click.option(
    "--model-version",
    "-m",
    help="Model version to calibrate",
)
@click.option(
    "--method",
    type=click.Choice(["isotonic", "sigmoid"]),
    default="isotonic",
    help="Calibration method",
)
@handle_errors
def calibrate(
    model_version: str | None,
    method: str,
) -> None:
    """Calibrate model probabilities."""
    with session_scope() as session:
        service = AnalysisService(session)
        request = CalibrateRequest(
            model_version=model_version,
            method=method,
        )

        click.echo(f"\nCalibrating model using {method} method...")
        response = service.calibrate_model(request)

        # Display results
        click.echo("\n" + "=" * 50)
        click.echo("CALIBRATION RESULTS")
        click.echo("=" * 50)
        click.echo(f"\nNew model version: {response.model_version}")
        click.echo(f"Calibration method: {response.calibration_method}")
        click.echo("\n📊 Calibration Improvement:")
        click.echo("-" * 40)
        click.echo(f"  Before (Brier score): {response.before_calibration_score:.4f}")
        click.echo(f"  After (Brier score):  {response.after_calibration_score:.4f}")
        click.echo(f"  Improvement: {response.improvement:+.4f}")
        click.echo("\n✓ Calibrated model saved and activated")
        click.echo("=" * 50 + "\n")
