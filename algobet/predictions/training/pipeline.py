"""Training pipeline orchestrator for match prediction.

This module provides the main TrainingPipeline class that orchestrates
the complete ML training workflow from data preparation to model registration.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.pipeline import FeaturePipeline
from algobet.predictions.features.store import FeatureStore
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.calibration import (
    CalibratedPredictor,
    ProbabilityCalibrator,
    calculate_calibration_metrics,
)
from algobet.predictions.training.classifiers import (
    EnsemblePredictor,
    MatchPredictor,
    ModelConfig,
    create_predictor,
)
from algobet.predictions.training.split import (
    TemporalSplitter,
    encode_targets,
    get_class_weights,
)
from algobet.predictions.training.tuner import (
    HAS_OPTUNA,
    HyperparameterTuner,
    TuningConfig,
    TuningResult,
)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model settings
    model_type: str = "xgboost"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    use_ensemble: bool = False
    ensemble_types: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])

    # Feature settings
    feature_schema_version: str = "v1.0"
    use_feature_cache: bool = True

    # Split settings
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Tuning settings
    tune_hyperparameters: bool = False
    tuning_trials: int = 50

    # Calibration settings
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"

    # Training settings
    early_stopping_rounds: int = 50
    random_seed: int = 42

    # Output settings
    model_name: str = "match_predictor"
    description: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of a training run."""

    # Model info
    model_version: str
    model_type: str

    # Metrics
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]

    # Feature info
    feature_schema_version: str
    num_features: int
    feature_importance: dict[str, float] | None

    # Training info
    trained_at: datetime
    training_duration_seconds: float
    config: TrainingConfig

    # Optional tuning results
    tuning_result: TuningResult | None = None

    # Paths
    model_path: Path | None = None


class TrainingPipeline:
    """End-to-end training pipeline for match prediction.

    Orchestrates the complete ML workflow:
    1. Load and prepare data
    2. Generate features
    3. Split data temporally
    4. Tune hyperparameters (optional)
    5. Train model
    6. Calibrate probabilities (optional)
    7. Evaluate on test set
    8. Register model

    Example:
        >>> config = TrainingConfig(model_type="xgboost")
        >>> pipeline = TrainingPipeline(
        ...     config=config,
        ...     session=db_session,
        ...     models_path=Path("data/models"),
        ... )
        >>> result = pipeline.run()
        >>> print(f"Model version: {result.model_version}")
        >>> print(f"Test accuracy: {result.test_metrics['accuracy']}")
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

        # Initialize components
        self.feature_pipeline = feature_pipeline or FeaturePipeline.create_default()
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

    def run(self) -> TrainingResult:
        """Execute the complete training pipeline.

        Returns:
            TrainingResult with model info and metrics
        """
        import time

        start_time = time.time()

        # Step 1: Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data()

        # Step 2: Handle class imbalance
        class_weights = get_class_weights(y_train)

        # Step 3: Hyperparameter tuning (optional)
        tuning_result = None
        best_params = self.config.hyperparameters.copy()

        if self.config.tune_hyperparameters and HAS_OPTUNA:
            tuning_result = self._tune_hyperparameters(
                X_train, y_train, X_val, y_val, class_weights
            )
            best_params = tuning_result.best_params

        # Step 4: Train model
        predictor = self._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            best_params,
            class_weights,
        )

        # Step 5: Calibrate probabilities (optional)
        if self.config.calibrate_probabilities:
            self._calibrator = ProbabilityCalibrator(
                method=self.config.calibration_method,
            )
            val_probas = predictor.predict_proba(X_val)
            self._calibrator.fit(val_probas, y_val)

        # Step 6: Evaluate
        train_metrics = self._evaluate(predictor, X_train, y_train)
        val_metrics = self._evaluate(predictor, X_val, y_val)
        test_metrics = self._evaluate(predictor, X_test, y_test)

        # Step 7: Register model
        all_metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

        model_version = self.model_registry.save_model(
            model=CalibratedPredictor(predictor, self._calibrator)
            if self._calibrator
            else predictor,
            name=self.config.model_name,
            metrics=all_metrics,
            model_type=self.config.model_type,
            feature_schema_version=self.config.feature_schema_version,
            description=self.config.description,
            tags=self.config.tags,
        )

        # Compile result
        duration = time.time() - start_time

        return TrainingResult(
            model_version=model_version,
            model_type=self.config.model_type,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_schema_version=self.config.feature_schema_version,
            num_features=len(self.feature_pipeline.feature_names),
            feature_importance=predictor.feature_importance,
            trained_at=datetime.now(),
            training_duration_seconds=duration,
            config=self.config,
            tuning_result=tuning_result,
            model_path=self.models_path / self.config.model_type / model_version,
        )

    def _prepare_data(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        """Prepare training data from database.

        Steps:
            1. Load historical matches
            2. Generate raw features once for all matches
            3. Split by temporal indices
            4. Fit transformers on training subset only
            5. Transform all three subsets
            6. Save fitted pipeline to disk for inference
        """
        from algobet.predictions.features.pipeline import prepare_match_dataframe

        # Get historical matches
        matches = self.repo.get_historical_matches(require_results=True)

        if not matches:
            raise ValueError("No historical matches found for training")

        # Convert to DataFrame
        matches_df = prepare_match_dataframe(matches)

        # Add result column
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )

        # Split data temporally FIRST
        splitter = TemporalSplitter(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
        )

        splits = list(splitter.split(matches_df))
        split = splits[0]  # Single split

        # Encode targets
        y = encode_targets(matches_df["result"].values)
        y_train = y[split.train_indices]
        y_val = y[split.val_indices]
        y_test = y[split.test_indices]

        # Fit pipeline on training data only, then transform all subsets
        train_df = matches_df.iloc[split.train_indices]
        val_df = matches_df.iloc[split.val_indices]
        test_df = matches_df.iloc[split.test_indices]

        X_train = self.feature_pipeline.fit_transform(train_df, self.repo)
        X_val = self.feature_pipeline.transform(val_df, self.repo)
        X_test = self.feature_pipeline.transform(test_df, self.repo)

        # Cache raw features if enabled
        if self.config.use_feature_cache:
            try:
                raw_features = self.feature_pipeline.generate_raw(matches_df, self.repo)
                from algobet.predictions.features.store import features_to_store_format

                features_list = features_to_store_format(
                    raw_features,
                    schema_version=self.config.feature_schema_version,
                )
                self.feature_store.store_bulk(features_list)
            except Exception:
                pass  # Feature caching is best-effort

        # Save fitted pipeline for inference
        pipeline_dir = (
            self.models_path.parent / "pipelines" / self.config.feature_schema_version
        )
        self.feature_pipeline.save(pipeline_dir)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _tune_hyperparameters(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        class_weights: dict[int, float],
    ) -> TuningResult:
        """Run hyperparameter tuning."""
        tuning_config = TuningConfig(
            model_type=self.config.model_type,
            n_trials=self.config.tuning_trials,
        )

        tuner = HyperparameterTuner(
            model_type=self.config.model_type,
            config=tuning_config,
        )

        return tuner.tune(X_train, y_train, X_val, y_val, class_weights)

    def _train_model(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        hyperparameters: dict[str, Any],
        class_weights: dict[int, float],
    ) -> MatchPredictor:
        """Train the prediction model."""
        if self.config.use_ensemble:
            # Train ensemble of multiple model types
            predictors = []
            for model_type in self.config.ensemble_types:
                config = ModelConfig(
                    model_type=model_type,
                    hyperparameters=hyperparameters,
                    class_weights=class_weights,
                    random_seed=self.config.random_seed,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                )
                predictor = create_predictor(model_type, config)
                predictor.fit(X_train, y_train, X_val, y_val)
                predictors.append(predictor)

            return EnsemblePredictor(predictors=predictors)
        else:
            # Train single model
            config = ModelConfig(
                model_type=self.config.model_type,
                hyperparameters=hyperparameters,
                class_weights=class_weights,
                random_seed=self.config.random_seed,
                early_stopping_rounds=self.config.early_stopping_rounds,
            )

            predictor = create_predictor(self.config.model_type, config)
            predictor.set_feature_names(self.feature_pipeline.feature_names)
            predictor.fit(X_train, y_train, X_val, y_val)

            return predictor

    def _evaluate(
        self,
        predictor: MatchPredictor,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
        )

        # Get predictions
        probas = predictor.predict_proba(X)

        # Apply calibration if available
        if self._calibrator is not None:
            probas = self._calibrator.calibrate(probas)

        y_pred = np.argmax(probas, axis=1)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, probas),
            "precision_macro": precision_score(
                y, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        }

        # Add calibration metrics
        cal_metrics = calculate_calibration_metrics(y, probas)
        metrics.update(cal_metrics)

        return metrics

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
