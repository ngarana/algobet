"""API router for model registry endpoints."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import ModelVersionResponse, PaginatedResponse
from algobet.models import ModelVersion
from algobet.predictions.models.registry import ModelRegistry

router = APIRouter()


@router.get("", response_model=PaginatedResponse[ModelVersionResponse])
def list_models(
    algorithm: str | None = Query(None, description="Filter by algorithm type"),
    active_only: bool = Query(False, description="Return only active models"),
    db: Session = Depends(get_db),
) -> list[ModelVersionResponse]:
    """List all model versions.

    Returns all model versions in the registry, optionally filtered.

    Args:
        algorithm: Filter by algorithm type (e.g., 'xgboost', 'random_forest')
        active_only: If True, only return active models

    Returns:
        List of model versions
    """
    registry = ModelRegistry(storage_path=Path("data/models"), session=db)

    models = list(registry.list_models(model_type=algorithm, active_only=active_only))

    # Convert ModelMetadata to ModelVersionResponse
    items = []
    for metadata in models:
        # Get the DB record to access all fields
        db_model = (
            db.query(ModelVersion)
            .filter(ModelVersion.version == metadata.version)
            .first()
        )

        if db_model:
            items.append(ModelVersionResponse.model_validate(db_model))

    return PaginatedResponse(
        items=items,
        total=len(items),
        limit=50,
        offset=0,
    )


@router.get("/active", response_model=ModelVersionResponse | None)
def get_active_model(
    db: Session = Depends(get_db),
) -> ModelVersionResponse | None:
    """Get the currently active model version.

    Returns the model version that is currently set as active.

    Returns:
        Active model version or None if no active model
    """
    registry = ModelRegistry(storage_path=Path("data/models"), session=db)

    try:
        _, metadata = registry.get_active_model()

        # Get the DB record to return full response
        db_model = (
            db.query(ModelVersion)
            .filter(ModelVersion.version == metadata.version)
            .first()
        )

        if db_model:
            return ModelVersionResponse.model_validate(db_model)
        return None
    except ValueError:
        # No active model set
        return None


@router.get("/{model_id}", response_model=ModelVersionResponse)
def get_model(
    model_id: int,
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    """Get details for a specific model version.

    Args:
        model_id: ID of the model version

    Returns:
        Model version details

    Raises:
        HTTPException: If model not found (404)
    """
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelVersionResponse.model_validate(model)


@router.post("/{model_id}/activate", response_model=dict[str, Any])
def activate_model(
    model_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Activate a model version.

    Sets the specified model version as the active model.

    Args:
        model_id: ID of the model version to activate

    Returns:
        Confirmation of activation

    Raises:
        HTTPException: If model not found (404)
    """
    # First check if model exists
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Use registry to activate
    registry = ModelRegistry(storage_path=Path("data/models"), session=db)
    registry.activate_model(model.version)

    return {
        "message": f"Model {model_id} activated successfully",
        "model_id": model_id,
        "version": model.version,
    }


@router.delete("/{model_id}", response_model=dict[str, Any])
def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Delete a model version.

    Permanently deletes a model version from the registry.

    Args:
        model_id: ID of the model version to delete

    Returns:
        Confirmation of deletion

    Raises:
        HTTPException: If model not found (404)
    """
    # First check if model exists
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Use registry to delete
    registry = ModelRegistry(storage_path=Path("data/models"), session=db)
    registry.delete_model(model.version)

    return {
        "message": f"Model {model_id} deleted successfully",
        "model_id": model_id,
        "version": model.version,
    }


@router.get("/{model_id}/metrics")
def get_model_metrics(
    model_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get detailed metrics for a model version.

    Returns comprehensive performance metrics for the specified model.

    Args:
        model_id: ID of the model version

    Returns:
        Detailed model metrics

    Raises:
        HTTPException: If model not found (404)
    """
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return {
        "model_id": model.id,
        "name": model.name,
        "version": model.version,
        "algorithm": model.algorithm,
        "accuracy": model.accuracy,
        "metrics": model.metrics or {},
        "hyperparameters": model.hyperparameters or {},
        "feature_schema_version": model.feature_schema_version,
        "created_at": model.created_at.isoformat(),
        "is_active": model.is_active,
    }
