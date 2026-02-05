"""Model registry for managing trained ML model versions."""

import contextlib
import json
import pickle
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import ModelVersion


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    model_id: str
    version: str
    model_type: str
    created_at: datetime
    metrics: dict[str, float]
    feature_schema_version: str
    artifact_path: Path
    is_production: bool = False
    tags: dict[str, str] | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["artifact_path"] = str(self.artifact_path)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["artifact_path"] = Path(data["artifact_path"])
        return cls(**data)


class ModelRegistry:
    """Registry for managing ML model lifecycle and versioning.

    Handles saving models to disk, registering metadata in the database,
    and managing active/production model versions.
    """

    def __init__(self, storage_path: Path, session: Session) -> None:
        """Initialize model registry.

        Args:
            storage_path: Directory path for storing model artifacts
            session: SQLAlchemy database session
        """
        self.storage_path = Path(storage_path)
        self.session = session
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        name: str,
        metrics: dict[str, float],
        model_type: str = "xgboost",
        feature_schema_version: str = "v1.0",
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Save model to disk and register in database.

        Args:
            model: The trained model object to save
            name: Human-readable name for the model
            metrics: Dictionary of performance metrics (accuracy, log_loss, etc.)
            model_type: Type of model algorithm (xgboost, random_forest, etc.)
            feature_schema_version: Version of feature schema used
            description: Optional description of the model
            tags: Optional dictionary of tags/labels

        Returns:
            The version string assigned to the model
        """
        # Generate version string with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{model_type}_{timestamp}"

        # Create version directory
        version_dir = self.storage_path / model_type / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifact using pickle
        artifact_path = version_dir / "model.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata to JSON file
        metadata = ModelMetadata(
            model_id=f"{name}_{version}",
            version=version,
            model_type=model_type,
            created_at=datetime.now(),
            metrics=metrics,
            feature_schema_version=feature_schema_version,
            artifact_path=artifact_path,
            is_production=False,
            tags=tags or {},
            description=description,
        )

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Register in database
        db_version = ModelVersion(
            name=name,
            version=version,
            algorithm=model_type,
            accuracy=metrics.get("accuracy"),
            file_path=str(artifact_path),
            is_active=False,
            metrics=metrics,
            hyperparameters=None,  # Could be extended to save hyperparams
            feature_schema_version=feature_schema_version,
            description=description,
        )
        self.session.add(db_version)
        self.session.flush()

        return version

    def load_model(self, version_id: str) -> Any:
        """Load model from disk by version.

        Args:
            version_id: The version string of the model to load

        Returns:
            The loaded model object

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        # Query database to get file path
        stmt = select(ModelVersion).where(ModelVersion.version == version_id)
        result = self.session.execute(stmt)
        db_version = result.scalar_one_or_none()

        if db_version is None:
            raise FileNotFoundError(f"Model version {version_id} not found in registry")

        artifact_path = Path(db_version.file_path)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

        with open(artifact_path, "rb") as f:
            model = pickle.load(f)

        return model

    def get_active_model(self) -> tuple[Any, ModelMetadata]:
        """Get currently active/production model.

        Returns:
            Tuple of (model object, metadata)

        Raises:
            ValueError: If no active model is set
        """
        stmt = select(ModelVersion).where(ModelVersion.is_active)
        result = self.session.execute(stmt)
        db_version = result.scalar_one_or_none()

        if db_version is None:
            raise ValueError("No active model set in registry")

        # Load model
        model = self.load_model(db_version.version)

        # Load metadata
        metadata_path = Path(db_version.file_path).parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = ModelMetadata.from_dict(json.load(f))
        else:
            # Create metadata from DB record
            metadata = ModelMetadata(
                model_id=f"{db_version.name}_{db_version.version}",
                version=db_version.version,
                model_type=db_version.algorithm,
                created_at=db_version.created_at,
                metrics=db_version.metrics or {},
                feature_schema_version=db_version.feature_schema_version or "v1.0",
                artifact_path=Path(db_version.file_path),
                is_production=db_version.is_active,
                description=db_version.description,
            )

        return model, metadata

    def list_models(
        self, model_type: str | None = None, active_only: bool = False
    ) -> Iterator[ModelMetadata]:
        """List all registered model versions.

        Args:
            model_type: Optional filter by model algorithm type
            active_only: If True, only return the active model

        Returns:
            Iterator of ModelMetadata objects
        """
        stmt = select(ModelVersion)

        if model_type:
            stmt = stmt.where(ModelVersion.algorithm == model_type)
        if active_only:
            stmt = stmt.where(ModelVersion.is_active)

        stmt = stmt.order_by(ModelVersion.created_at.desc())
        result = self.session.execute(stmt)

        for db_version in result.scalars():
            yield ModelMetadata(
                model_id=f"{db_version.name}_{db_version.version}",
                version=db_version.version,
                model_type=db_version.algorithm,
                created_at=db_version.created_at,
                metrics=db_version.metrics or {},
                feature_schema_version=db_version.feature_schema_version or "v1.0",
                artifact_path=Path(db_version.file_path),
                is_production=db_version.is_active,
                description=db_version.description,
            )

    def activate_model(self, version_id: str) -> None:
        """Set a model version as the active/production model.

        Deactivates any currently active model.

        Args:
            version_id: Version string of model to activate
        """
        # Deactivate all models first
        self.session.query(ModelVersion).update({"is_active": False})

        # Activate specified model
        stmt = select(ModelVersion).where(ModelVersion.version == version_id)
        result = self.session.execute(stmt)
        db_version = result.scalar_one_or_none()

        if db_version is None:
            raise ValueError(f"Model version {version_id} not found")

        db_version.is_active = True
        self.session.flush()

    def delete_model(self, version_id: str) -> None:
        """Delete a model version from registry and disk.

        Args:
            version_id: Version string of model to delete
        """
        stmt = select(ModelVersion).where(ModelVersion.version == version_id)
        result = self.session.execute(stmt)
        db_version = result.scalar_one_or_none()

        if db_version is None:
            raise ValueError(f"Model version {version_id} not found")

        # Delete from database
        self.session.delete(db_version)

        # Delete files from disk
        artifact_path = Path(db_version.file_path)
        if artifact_path.exists():
            artifact_path.unlink()

        metadata_path = artifact_path.parent / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # Try to remove empty parent directory
        with contextlib.suppress(OSError):
            artifact_path.parent.rmdir()  # Directory not empty

        self.session.flush()
