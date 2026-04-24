"""Essential unit tests for ModelRegistry.

Focused tests covering critical model management operations:
- Save/load models
- Activate models
- List models
- Delete models
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from algobet.predictions.models.registry import ModelMetadata, ModelRegistry


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_to_dict(self):
        """Test metadata serialization to dict."""
        metadata = ModelMetadata(
            model_id="test_v1",
            version="v1.0.0",
            model_type="xgboost",
            created_at=datetime(2026, 1, 1, 12, 0),
            metrics={"accuracy": 0.75},
            feature_schema_version="v1.0",
            artifact_path=Path("/models/test.pkl"),
            is_production=True,
        )

        result = metadata.to_dict()

        assert result["model_id"] == "test_v1"
        assert result["version"] == "v1.0.0"
        assert result["created_at"] == "2026-01-01T12:00:00"
        assert result["artifact_path"] == "/models/test.pkl"

    def test_from_dict(self):
        """Test metadata deserialization from dict."""
        data = {
            "model_id": "test_v1",
            "version": "v1.0.0",
            "model_type": "xgboost",
            "created_at": "2026-01-01T12:00:00",
            "metrics": {"accuracy": 0.75},
            "feature_schema_version": "v1.0",
            "artifact_path": "/models/test.pkl",
            "is_production": True,
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.model_id == "test_v1"
        assert metadata.version == "v1.0.0"
        assert isinstance(metadata.created_at, datetime)
        assert metadata.artifact_path == Path("/models/test.pkl")


class TestModelRegistry:
    """Test ModelRegistry core operations."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        # Mock query interface
        query_mock = MagicMock()
        session.query.return_value = query_mock
        session.execute.return_value = MagicMock()
        return session

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        storage = tmp_path / "models"
        storage.mkdir()
        return storage

    @pytest.fixture
    def registry(self, mock_session, temp_storage):
        """Create ModelRegistry instance."""
        return ModelRegistry(storage_path=temp_storage, session=mock_session)

    def test_init_creates_storage_directory(self, tmp_path, mock_session):
        """Test registry creates storage directory on init."""
        storage = tmp_path / "new_models"

        registry = ModelRegistry(storage_path=storage, session=mock_session)

        assert storage.exists()
        assert registry.storage_path == storage

    def test_save_model_creates_directory(self, registry, temp_storage):
        """Test save_model creates version directory."""
        mock_model = {"weights": [1, 2, 3]}

        version = registry.save_model(
            model=mock_model,
            name="test_model",
            metrics={"accuracy": 0.75},
            model_type="xgboost",
        )

        version_dir = temp_storage / "xgboost" / version
        assert version_dir.exists()
        assert (version_dir / "model.pkl").exists()
        assert (version_dir / "metadata.json").exists()

    def test_save_model_saves_pickle(self, registry, temp_storage):
        """Test save_model saves model as pickle."""
        mock_model = {"weights": [1, 2, 3]}

        version = registry.save_model(
            model=mock_model,
            name="test_model",
            metrics={"accuracy": 0.75},
        )

        artifact_path = temp_storage / "xgboost" / version / "model.pkl"
        with open(artifact_path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded == mock_model

    def test_save_model_saves_metadata(self, registry, temp_storage):
        """Test save_model saves metadata JSON."""
        mock_model = {"test": "data"}

        version = registry.save_model(
            model=mock_model,
            name="test_model",
            metrics={"accuracy": 0.75, "log_loss": 0.5},
            model_type="xgboost",
            description="Test model",
        )

        metadata_path = temp_storage / "xgboost" / version / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["version"] == version
        assert metadata["model_type"] == "xgboost"
        assert metadata["metrics"]["accuracy"] == 0.75
        assert metadata["description"] == "Test model"

    def test_save_model_registers_in_database(self, registry, mock_session):
        """Test save_model creates database record."""
        mock_model = {"test": "data"}

        registry.save_model(
            model=mock_model,
            name="test_model",
            metrics={"accuracy": 0.75},
            model_type="xgboost",
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    def test_save_model_stores_hyperparameters(
        self, registry, mock_session, temp_storage
    ):
        """Test save_model persists hyperparameters in metadata and DB record."""
        mock_model = {"test": "data"}

        version = registry.save_model(
            model=mock_model,
            name="test_model",
            metrics={"accuracy": 0.75},
            model_type="xgboost",
            hyperparameters={"max_depth": 6, "feature_names": ["f1", "f2"]},
        )

        db_model = mock_session.add.call_args.args[0]
        assert db_model.hyperparameters is not None
        assert db_model.hyperparameters["max_depth"] == 6

        metadata_path = temp_storage / "xgboost" / version / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["hyperparameters"]["max_depth"] == 6

    def test_save_model_returns_version(self, registry):
        """Test save_model returns version string."""
        mock_model = {"test": "data"}

        version = registry.save_model(
            model=mock_model,
            name="test_model",
            metrics={"accuracy": 0.75},
        )

        assert version.startswith("xgboost_")
        assert len(version) > len("xgboost_")

    def test_load_model_returns_model(self, registry, temp_storage):
        """Test load_model returns saved model."""
        # Save a model first
        test_model = {"test": "data"}
        version = registry.save_model(
            model=test_model,
            name="test",
            metrics={},
        )

        # Mock database query result
        mock_result = MagicMock()
        mock_db_model = MagicMock()
        mock_db_model.file_path = str(temp_storage / "xgboost" / version / "model.pkl")
        mock_result.scalar_one_or_none.return_value = mock_db_model
        registry.session.execute.return_value = mock_result

        loaded = registry.load_model(version)

        assert loaded == test_model

    def test_load_model_not_found(self, registry):
        """Test load_model raises on missing model."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        registry.session.execute.return_value = mock_result

        with pytest.raises(FileNotFoundError) as exc_info:
            registry.load_model("nonexistent_v1")

        assert "not found in registry" in str(exc_info.value)

    def test_load_model_artifact_missing(self, registry, temp_storage):
        """Test load_model raises on missing artifact."""
        mock_result = MagicMock()
        mock_db_model = MagicMock()
        mock_db_model.file_path = str(temp_storage / "nonexistent_model.pkl")
        mock_result.scalar_one_or_none.return_value = mock_db_model
        registry.session.execute.return_value = mock_result

        with pytest.raises(FileNotFoundError) as exc_info:
            registry.load_model("test_v1")

        assert "artifact not found" in str(exc_info.value)

    def test_get_active_model_returns_model(self, registry, temp_storage):
        """Test get_active_model returns active model and metadata."""
        # Save a model
        test_model = {"active": True}
        version = registry.save_model(
            model=test_model,
            name="active_model",
            metrics={"accuracy": 0.8},
        )

        # Mock active model query
        mock_result = MagicMock()
        mock_db_model = MagicMock()
        mock_db_model.version = version
        mock_db_model.file_path = str(temp_storage / "xgboost" / version / "model.pkl")
        mock_db_model.algorithm = "xgboost"
        mock_db_model.metrics = {"accuracy": 0.8}
        mock_db_model.feature_schema_version = "v1.0"
        mock_db_model.is_active = True
        mock_db_model.name = "active_model"
        mock_db_model.description = None
        mock_result.scalar_one_or_none.return_value = mock_db_model
        registry.session.execute.return_value = mock_result

        model, metadata = registry.get_active_model()

        assert model == test_model
        assert metadata.version == version
        assert metadata.model_type == "xgboost"

    def test_get_active_model_no_active(self, registry):
        """Test get_active_model raises when no active model."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        registry.session.execute.return_value = mock_result

        with pytest.raises(ValueError) as exc_info:
            registry.get_active_model()

        assert "No active model" in str(exc_info.value)

    def test_list_models_returns_iterator(self, registry):
        """Test list_models returns iterator of metadata."""
        mock_result = MagicMock()
        mock_db_model = MagicMock()
        mock_db_model.name = "test"
        mock_db_model.version = "v1.0.0"
        mock_db_model.algorithm = "xgboost"
        mock_db_model.created_at = datetime.now()
        mock_db_model.metrics = {"accuracy": 0.75}
        mock_db_model.feature_schema_version = "v1.0"
        mock_db_model.file_path = "/model.pkl"
        mock_db_model.is_active = False
        mock_db_model.description = None
        mock_result.scalars.return_value = [mock_db_model]
        registry.session.execute.return_value = mock_result

        models = list(registry.list_models())

        assert len(models) == 1
        assert models[0].version == "v1.0.0"
        assert models[0].model_type == "xgboost"

    def test_activate_model_sets_active(self, registry):
        """Test activate_model sets is_active flag."""
        # Mock the two queries (deactivate all, then get model)
        mock_result = MagicMock()
        mock_db_model = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_db_model
        registry.session.execute.return_value = mock_result
        registry.session.query.return_value.update.return_value = None

        registry.activate_model("v1.0.0")

        assert mock_db_model.is_active is True
        registry.session.flush.assert_called()

    def test_activate_model_not_found(self, registry):
        """Test activate_model raises on missing model."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        registry.session.execute.return_value = mock_result

        with pytest.raises(ValueError) as exc_info:
            registry.activate_model("nonexistent_v1")

        assert "not found" in str(exc_info.value)

    def test_delete_model_removes_from_db(self, registry, temp_storage):
        """Test delete_model removes database record."""
        # Save a model
        version = registry.save_model(
            model={"test": "data"},
            name="test",
            metrics={},
        )

        # Mock the query result
        mock_result = MagicMock()
        mock_db_model = MagicMock()
        mock_db_model.file_path = str(temp_storage / "xgboost" / version / "model.pkl")
        mock_result.scalar_one_or_none.return_value = mock_db_model
        registry.session.execute.return_value = mock_result

        registry.delete_model(version)

        registry.session.delete.assert_called_once_with(mock_db_model)

    def test_delete_model_removes_files(self, registry, temp_storage):
        """Test delete_model removes model files."""
        version = registry.save_model(
            model={"test": "data"},
            name="test",
            metrics={},
        )

        mock_result = MagicMock()
        mock_db_model = MagicMock()
        mock_db_model.file_path = str(temp_storage / "xgboost" / version / "model.pkl")
        mock_result.scalar_one_or_none.return_value = mock_db_model
        registry.session.execute.return_value = mock_result

        registry.delete_model(version)

        assert not (temp_storage / "xgboost" / version / "model.pkl").exists()
        assert not (temp_storage / "xgboost" / version / "metadata.json").exists()

    def test_delete_model_not_found(self, registry):
        """Test delete_model raises on missing model."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        registry.session.execute.return_value = mock_result

        with pytest.raises(ValueError) as exc_info:
            registry.delete_model("nonexistent_v1")

        assert "not found" in str(exc_info.value)
