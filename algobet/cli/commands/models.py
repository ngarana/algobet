"""Model management CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from algobet.cli.error_handler import handle_errors
from algobet.cli.logger import success
from algobet.database import session_scope
from algobet.exceptions import ModelLoadError, ModelNotFoundError
from algobet.models import ModelVersion
from algobet.services import ModelManagementService
from algobet.services.dto import ModelListRequest


@click.group(name="model")
def model_cli() -> None:
    """Model management commands."""
    pass


@model_cli.command(name="delete")
@click.argument("model_id", type=int)
@handle_errors
def delete_model(model_id: int) -> None:
    """Delete a model version from the database."""
    with session_scope() as session:
        try:
            model = (
                session.query(ModelVersion).filter(ModelVersion.id == model_id).first()
            )
            if not model:
                raise ModelNotFoundError(
                    f"Model version with ID {model_id} not found.",
                    details={"model_id": model_id},
                )

            # Delete model file
            models_path = Path("data/models")
            model_file = models_path / f"{model.version}.pkl"
            if model_file.exists():
                model_file.unlink()
                success(f"Deleted model file: {model_file}")

            # Delete database record
            session.delete(model)
            success(f"Deleted model version: {model.version}")
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelLoadError(
                f"Failed to delete model: {e}",
                details={"model_id": model_id, "error_type": type(e).__name__},
            ) from e


@model_cli.command(name="list")
@handle_errors
def list_models() -> None:
    """List all model versions."""
    with session_scope() as session:
        service = ModelManagementService(session)
        request = ModelListRequest(include_inactive=True)
        response = service.list_models(request)

        if not response.models:
            raise ModelNotFoundError("No models found.")

        click.echo(f"\nFound {len(response.models)} model(s):\n")
        for m in response.models:
            status = "✓ Active" if m.is_active else " "
            click.echo(f"  [{status}] {m.version}")
            click.echo(f"    Type: {m.model_type}")
            click.echo(f"    Created: {m.created_at}")
            if m.metrics:
                click.echo(f"    Metrics: {m.metrics}")
            click.echo()


@model_cli.command(name="activate")
@click.argument("version")
@handle_errors
def activate_model(version: str) -> None:
    """Activate a specific model version."""
    from algobet.services.dto import ModelActivateRequest

    with session_scope() as session:
        service = ModelManagementService(session)
        request = ModelActivateRequest(version=version)
        response = service.activate_model(request)

        if response.previous_active_version:
            click.echo(
                f"Deactivated model: {response.previous_active_version}"
            )
        success(response.message)


@model_cli.command(name="info")
@click.argument("version")
@handle_errors
def model_info(version: str) -> None:
    """Show detailed information about a model."""
    from algobet.services.dto import ModelInfoRequest

    with session_scope() as session:
        service = ModelManagementService(session)
        request = ModelInfoRequest(version=version)
        response = service.get_model_info(request)

        click.echo(f"\n{'=' * 50}")
        click.echo(f"Model: {response.version}")
        click.echo(f"{'=' * 50}")
        click.echo(f"  Type: {response.model_type}")
        click.echo(f"  Created: {response.created_at}")
        click.echo(f"  Active: {'Yes' if response.is_active else 'No'}")
        click.echo(f"  Training samples: {response.training_data_size}")

        if response.features:
            click.echo(f"\n  Features ({len(response.features)}):")
            for feature in response.features[:10]:  # Show first 10
                click.echo(f"    - {feature}")
            if len(response.features) > 10:
                click.echo(f"    ... and {len(response.features) - 10} more")

        if response.metrics:
            click.echo("\n  Metrics:")
            for name, value in response.metrics.items():
                if isinstance(value, float):
                    click.echo(f"    {name}: {value:.4f}")
                else:
                    click.echo(f"    {name}: {value}")

        if response.hyperparameters:
            click.echo("\n  Hyperparameters:")
            for name, value in response.hyperparameters.items():
                if name != "feature_names":  # Skip feature names, shown above
                    click.echo(f"    {name}: {value}")

        click.echo()
