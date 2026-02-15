"""ML model training CLI commands."""

from __future__ import annotations

import click

from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import success
from algobet.database import session_scope


@click.group(name="train")
def train_cli() -> None:
    """ML model training commands."""
    pass


@train_cli.command(name="run")
@click.option(
    "--model-type",
    "-t",
    type=click.Choice(["xgboost", "lightgbm", "random_forest"]),
    default="xgboost",
    help="Type of model to train",
)
@click.option(
    "--tune",
    is_flag=True,
    help="Enable hyperparameter tuning with Optuna",
)
@click.option(
    "--ensemble",
    is_flag=True,
    help="Train ensemble of multiple model types",
)
@click.option(
    "--ensemble-types",
    multiple=True,
    default=["xgboost", "lightgbm"],
    help="Model types to include in ensemble (repeatable)",
)
@click.option(
    "--description",
    "-d",
    help="Description for the trained model",
)
@click.option(
    "--models-path",
    type=click.Path(),
    default="data/models",
    help="Path to store trained models",
)
@handle_errors
def run_training(
    model_type: str,
    tune: bool,
    ensemble: bool,
    ensemble_types: tuple[str, ...],
    description: str | None,
    models_path: str,
) -> None:
    """Train a prediction model on historical match data."""
    from pathlib import Path

    from algobet.predictions.training.pipeline import TrainingConfig, TrainingPipeline

    config = TrainingConfig(
        model_type=model_type,
        use_ensemble=ensemble,
        ensemble_types=list(ensemble_types),
        tune_hyperparameters=tune,
        description=description or f"CLI trained {model_type} model",
    )

    click.echo("\n" + "=" * 60)
    click.echo("MODEL TRAINING")
    click.echo("=" * 60)
    click.echo(f"\n  Model type:    {model_type}")
    click.echo(f"  Ensemble:      {ensemble}")
    click.echo(f"  Tuning:        {tune}")
    click.echo(f"  Models path:   {models_path}")
    click.echo()

    with session_scope() as session:
        pipeline = TrainingPipeline(
            config=config,
            session=session,
            models_path=Path(models_path),
        )

        click.echo("Training in progress...")
        result = pipeline.run()

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("TRAINING COMPLETE")
    click.echo("=" * 60)
    click.echo(f"\n  Model version:   {result.model_version}")
    click.echo(f"  Model type:      {result.model_type}")
    click.echo(f"  Num features:    {result.num_features}")
    click.echo(f"  Duration:        {result.training_duration_seconds:.1f}s")

    click.echo("\n📊 Test Metrics:")
    click.echo("-" * 40)
    for name, value in result.test_metrics.items():
        if isinstance(value, float):
            if "accuracy" in name.lower() or "roi" in name.lower():
                click.echo(f"  {name:25s}: {value:.2%}")
            else:
                click.echo(f"  {name:25s}: {value:.4f}")
        else:
            click.echo(f"  {name:25s}: {value}")

    if result.feature_importance:
        click.echo("\n🔑 Top 10 Features:")
        click.echo("-" * 40)
        sorted_features = sorted(
            result.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for name, importance in sorted_features[:10]:
            bar = "█" * int(importance * 50)
            click.echo(f"  {name:30s}: {importance:.4f} {bar}")

    success(f"\nModel {result.model_version} saved to {models_path}")
    click.echo("=" * 60 + "\n")
