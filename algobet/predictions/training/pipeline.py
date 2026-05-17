"""Compatibility entry point for the training pipeline.

The training workflow is implemented through focused modules in this package.
This module keeps the stable public import path used by API routes, scripts,
and tests.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.pipeline import FeaturePipeline
from algobet.predictions.features.store import FeatureStore
from algobet.predictions.features.transformers import (
    create_tree_model_transformer_pipeline,
)
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.calibration import ProbabilityCalibrator
from algobet.predictions.training.classifiers import MatchPredictor
from algobet.predictions.training.collapse_recovery import CollapseRecoveryMixin
from algobet.predictions.training.config import (
    ALLOWED_FEATURE_GROUPS,
    LIGHTGBM_SEARCH_SPACE,
    MODEL_FEATURE_SCHEMA_VERSION,
    XGBOOST_SEARCH_SPACE,
    TrainingConfig,
    TrainingResult,
)
from algobet.predictions.training.data_preparation import DataPreparationMixin
from algobet.predictions.training.evaluation_pipeline import EvaluationPipelineMixin
from algobet.predictions.training.feature_selection import FeatureSelectionReport
from algobet.predictions.training.feature_selection_pipeline import (
    FeatureSelectionPipelineMixin,
)
from algobet.predictions.training.model_training import ModelTrainingMixin
from algobet.predictions.training.runner import PipelineRunnerMixin
from algobet.predictions.training.tuning_pipeline import TuningPipelineMixin


class TrainingPipeline(
    PipelineRunnerMixin,
    FeatureSelectionPipelineMixin,
    CollapseRecoveryMixin,
    DataPreparationMixin,
    TuningPipelineMixin,
    ModelTrainingMixin,
    EvaluationPipelineMixin,
):
    """End-to-end training pipeline for match prediction.

    Orchestrates the complete ML workflow through focused mixins:
    data preparation, feature selection, tuning, fitting, collapse recovery,
    calibration, evaluation, and model registration.
    """

    def __init__(
        self,
        config: TrainingConfig,
        session: Session,
        models_path: Path = Path("data/models"),
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        """Initialize training pipeline.

        Args:
            config: Training configuration
            session: Database session
            models_path: Path to store models
            feature_pipeline: Optional feature pipeline (default: create new)
        """
        self.config = config
        self.session = session
        self.models_path = Path(models_path)
        self.feature_pipeline_path: Path | None = None

        if feature_pipeline:
            self.feature_pipeline = feature_pipeline
        else:
            self.feature_pipeline = self._new_feature_pipeline()
        self.feature_pipeline.config.schema_version = config.feature_schema_version
        self.feature_store = FeatureStore(
            session=session,
            schema_version=config.feature_schema_version,
        )
        self.model_registry = ModelRegistry(
            storage_path=models_path,
            session=session,
        )
        self.repo = MatchRepository(session)

        # Internal state
        self._predictor: MatchPredictor | None = None
        self._calibrator: ProbabilityCalibrator | None = None
        self._X_train: NDArray[np.float64] | None = None
        self._y_train: NDArray[np.int64] | None = None
        self._X_val: NDArray[np.float64] | None = None
        self._y_val: NDArray[np.int64] | None = None
        self._X_test: NDArray[np.float64] | None = None
        self._y_test: NDArray[np.int64] | None = None
        self._train_df: pd.DataFrame | None = None
        self._val_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None
        self._train_raw_features: pd.DataFrame | None = None
        self._val_raw_features: pd.DataFrame | None = None
        self._test_raw_features: pd.DataFrame | None = None
        self._selected_feature_names: list[str] | None = None
        self._feature_selection_report: FeatureSelectionReport | None = None
        self._collapse_recovery: dict[str, Any] | None = None
        self._prepared_matches_df: pd.DataFrame | None = None
        self._prepared_splits: list[Any] = []

    def _new_feature_pipeline(self) -> FeaturePipeline:
        """Create a fresh feature pipeline matching the training config."""
        if self.config.feature_groups:
            pipeline = self._create_feature_pipeline_with_groups(
                self.config.feature_groups
            )
        else:
            pipeline = FeaturePipeline.create_default()

        if self._uses_native_missing_value_model():
            pipeline.transformers = create_tree_model_transformer_pipeline()
        pipeline.config.schema_version = self.config.feature_schema_version
        return pipeline

    def _uses_native_missing_value_model(self) -> bool:
        """Return True when every trained model can consume NaN values directly."""
        if self.config.use_ensemble:
            model_types = self.config.ensemble_types
        else:
            model_types = [self.config.model_type]
        return all(
            model_type in {"xgboost", "lightgbm", "dixon_coles", "hybrid_poisson"}
            for model_type in model_types
        )

    def save_training_config(self, path: Path) -> None:
        """Save training configuration to file.

        Args:
            path: Path to save config
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self.config)
        # Convert Path objects to strings
        config_dict["model_path"] = str(config_dict.get("model_path", ""))

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)


def train_model(
    session: Session,
    model_type: str = "xgboost",
    tune: bool = False,
    models_path: Path = Path("data/models"),
    description: str | None = None,
) -> TrainingResult:
    """Convenience function to train a model with default settings.

    Args:
        session: Database session
        model_type: Type of model to train
        tune: Whether to tune hyperparameters
        models_path: Path to store models
        description: Optional model description

    Returns:
        TrainingResult with model info and metrics
    """
    config = TrainingConfig(
        model_type=model_type,
        tune_hyperparameters=tune,
        description=description,
    )

    pipeline = TrainingPipeline(
        config=config,
        session=session,
        models_path=models_path,
    )

    return pipeline.run()


__all__ = [
    "MODEL_FEATURE_SCHEMA_VERSION",
    "ALLOWED_FEATURE_GROUPS",
    "XGBOOST_SEARCH_SPACE",
    "LIGHTGBM_SEARCH_SPACE",
    "TrainingConfig",
    "TrainingResult",
    "TrainingPipeline",
    "train_model",
]
