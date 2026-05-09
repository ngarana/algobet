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
import pandas as pd
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.generators import (
    create_generators_by_names,
)
from algobet.predictions.features.pipeline import FeaturePipeline
from algobet.predictions.features.store import FeatureStore
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.acceleration import resolve_training_hyperparameters
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
    ExpandingWindowSplitter,
    SeasonAwareSplitter,
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

MODEL_FEATURE_SCHEMA_VERSION = "v2.0_odds_free"
ALLOWED_FEATURE_GROUPS = (
    "team_form",
    "head_to_head",
    "temporal",
    "standings",
    "enriched_stats",
)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model settings
    model_type: str = "xgboost"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    use_ensemble: bool = False
    ensemble_types: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])

    # Data range settings
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_matches: int | None = None

    # Filtering: tournament and team selection
    tournament_ids: list[int] | None = None
    team_ids: list[int] | None = None
    venue_filter: str | None = None  # "home", "away", "both"

    # Match quality filters
    min_total_goals: float | None = None
    max_total_goals: float | None = None

    # Feature settings
    feature_schema_version: str = MODEL_FEATURE_SCHEMA_VERSION
    use_feature_cache: bool = True

    # Feature group selection
    feature_groups: list[str] | None = None

    # Split settings
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_strategy: str = "temporal"  # "temporal", "expanding_window", "season_aware"
    gap_days: int = 0
    # Expanding window params
    min_train_size: int = 100
    ew_val_size: int = 50
    ew_test_size: int = 50
    step_size: int = 50
    # Season-aware params
    train_seasons: int = 3
    val_seasons: int = 1
    test_seasons: int = 1

    # Tuning settings
    tune_hyperparameters: bool = False
    tuning_trials: int = 50

    # Calibration settings
    calibrate_probabilities: bool = True
    calibration_method: str = "sigmoid"

    # Training settings
    early_stopping_rounds: int = 50
    random_seed: int = 42

    # Outcome balancing control (opt-in; disabled by default to preserve
    # calibrated probabilities)
    outcome_balance: bool = False

    # Degenerate model protection
    collapse_recovery: bool = True
    min_prediction_classes: int = 2

    # Feature importance pruning
    feature_selection: bool = False
    feature_selection_threshold: float = 0.01
    min_samples_per_feature: int | None = None

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
        self.feature_pipeline_path: Path | None = None

        # Initialize feature pipeline with optional feature group selection
        if feature_pipeline:
            self.feature_pipeline = feature_pipeline
        elif config.feature_groups:
            self.feature_pipeline = self._create_feature_pipeline_with_groups(
                config.feature_groups
            )
        else:
            self.feature_pipeline = FeaturePipeline.create_default()
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
        self._collapse_recovery: dict[str, Any] | None = None

    def _create_feature_pipeline_with_groups(
        self, feature_groups: list[str]
    ) -> FeaturePipeline:
        """Create a feature pipeline with only the specified feature groups.

        Args:
            feature_groups: List of feature group names to include

        Returns:
            FeaturePipeline configured with selected generators
        """
        unsupported_groups = sorted(set(feature_groups) - set(ALLOWED_FEATURE_GROUPS))
        if unsupported_groups:
            supported = ", ".join(ALLOWED_FEATURE_GROUPS)
            unsupported = ", ".join(unsupported_groups)
            raise ValueError(
                "Unsupported feature groups: "
                f"{unsupported}. Supported groups: {supported}"
            )

        if not feature_groups:
            return FeaturePipeline.create_default()

        return FeaturePipeline(generators=create_generators_by_names(feature_groups))

    def _choose_selected_feature_names(
        self,
        feature_names: list[str],
        feature_importance: dict[str, float] | None,
        n_samples: int,
    ) -> list[str]:
        """Choose a stable feature subset from normalized importance scores."""
        if not feature_names:
            raise ValueError("Cannot select features from an empty feature list")

        importance = {
            name: max(float((feature_importance or {}).get(name, 0.0)), 0.0)
            for name in feature_names
        }
        total_importance = sum(importance.values())
        if total_importance <= 0.0:
            normalized = {name: 1.0 / len(feature_names) for name in feature_names}
            selected = list(feature_names)
        else:
            normalized = {
                name: value / total_importance for name, value in importance.items()
            }
            selected = [
                name
                for name in feature_names
                if normalized[name] >= self.config.feature_selection_threshold
            ]
            if not selected:
                selected = [max(feature_names, key=lambda name: normalized[name])]

        if self.config.min_samples_per_feature:
            max_features = max(1, n_samples // self.config.min_samples_per_feature)
            top_selected = sorted(
                selected,
                key=lambda name: normalized[name],
                reverse=True,
            )[:max_features]
            top_selected_set = set(top_selected)
            selected = [name for name in feature_names if name in top_selected_set]

        return selected

    def _apply_feature_selection(
        self,
        X_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        X_test: NDArray[np.float64],
        y_train: NDArray[np.int64],
        y_val: NDArray[np.int64],
        class_weights: dict[int, float] | None,
        hyperparameters: dict[str, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Train a probe model, select important features, then refit transforms."""
        original_feature_names = list(self.feature_pipeline.feature_names)
        probe = self._train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters,
            class_weights=class_weights,
        )
        selected_feature_names = self._choose_selected_feature_names(
            feature_names=original_feature_names,
            feature_importance=probe.feature_importance,
            n_samples=len(y_train),
        )
        self._selected_feature_names = selected_feature_names

        # Verify no odds-derived features leaked into the selected subset
        forbidden_terms = ("odds", "implied_prob", "bookmaker", "favorite", "market")
        leaked = [
            name
            for name in selected_feature_names
            if any(term in name for term in forbidden_terms)
        ]
        if leaked:
            raise ValueError(
                f"Odds-derived features detected in selected subset: {leaked}"
            )

        if hasattr(self.feature_pipeline, "set_selected_features"):
            self.feature_pipeline.set_selected_features(selected_feature_names)

        if (
            self._train_raw_features is not None
            and self._val_raw_features is not None
            and self._test_raw_features is not None
            and hasattr(self.feature_pipeline, "fit_transform_raw_features")
            and hasattr(self.feature_pipeline, "transform_raw_features")
        ):
            X_train_selected = self.feature_pipeline.fit_transform_raw_features(
                self._train_raw_features,
                y_train,
            )
            X_val_selected = self.feature_pipeline.transform_raw_features(
                self._val_raw_features
            )
            X_test_selected = self.feature_pipeline.transform_raw_features(
                self._test_raw_features
            )
            return X_train_selected, X_val_selected, X_test_selected

        if self._train_df is None or self._val_df is None or self._test_df is None:
            selected_indices = [
                original_feature_names.index(name) for name in selected_feature_names
            ]
            return (
                X_train[:, selected_indices],
                X_val[:, selected_indices],
                X_test[:, selected_indices],
            )

        X_train_selected = self.feature_pipeline.fit_transform(
            self._train_df,
            self.repo,
            y_train,
        )
        X_val_selected = self.feature_pipeline.transform(self._val_df, self.repo)
        X_test_selected = self.feature_pipeline.transform(self._test_df, self.repo)

        return X_train_selected, X_val_selected, X_test_selected

    def _prediction_class_report(
        self,
        predictor: MatchPredictor | CalibratedPredictor,
        X: NDArray[np.float64],
    ) -> dict[str, Any] | None:
        """Summarize predicted-class diversity for model quality gates."""
        probas = predictor.predict_proba(X)
        if not isinstance(probas, np.ndarray) or probas.ndim != 2:
            return None

        predictions = np.argmax(probas, axis=1)
        counts = np.bincount(predictions, minlength=probas.shape[1])
        n_samples = int(len(predictions))
        max_share = float(counts.max() / n_samples) if n_samples else 0.0
        return {
            "num_classes": int(np.count_nonzero(counts)),
            "counts": [int(count) for count in counts.tolist()],
            "max_share": max_share,
            "mean_probabilities": [
                float(value) for value in probas.mean(axis=0).tolist()
            ],
        }

    def _is_prediction_collapsed(self, report: dict[str, Any] | None) -> bool:
        """Return True when validation predictions are too class-collapsed."""
        if report is None:
            return False
        return int(report["num_classes"]) < self.config.min_prediction_classes

    def _restore_full_feature_matrices(
        self,
        y_train: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
        """Refit transforms against full raw features after feature pruning."""
        if (
            self._train_raw_features is None
            or self._val_raw_features is None
            or self._test_raw_features is None
            or not hasattr(self.feature_pipeline, "clear_selected_features")
        ):
            return None

        self.feature_pipeline.clear_selected_features()
        self._selected_feature_names = None
        return (
            self.feature_pipeline.fit_transform_raw_features(
                self._train_raw_features,
                y_train,
            ),
            self.feature_pipeline.transform_raw_features(self._val_raw_features),
            self.feature_pipeline.transform_raw_features(self._test_raw_features),
        )

    def _collapse_recovery_hyperparameters(
        self,
        hyperparameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Relax XGBoost parameters enough to avoid class-prior collapse."""
        if self.config.model_type != "xgboost":
            return dict(hyperparameters)

        recovered = dict(hyperparameters)
        recovered["max_depth"] = max(int(recovered.get("max_depth", 3)), 3)
        recovered["learning_rate"] = max(
            float(recovered.get("learning_rate", 0.03)),
            0.03,
        )
        recovered["n_estimators"] = max(int(recovered.get("n_estimators", 600)), 600)
        recovered["min_child_weight"] = min(
            float(recovered.get("min_child_weight", 10)),
            5.0,
        )
        recovered["gamma"] = min(float(recovered.get("gamma", 1.0)), 0.5)
        recovered["reg_alpha"] = min(float(recovered.get("reg_alpha", 2.0)), 1.0)
        recovered["reg_lambda"] = min(float(recovered.get("reg_lambda", 10.0)), 5.0)
        recovered["subsample"] = max(float(recovered.get("subsample", 0.7)), 0.8)
        recovered["colsample_bytree"] = max(
            float(recovered.get("colsample_bytree", 0.5)),
            0.8,
        )
        return recovered

    def _train_with_collapse_recovery(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        X_test: NDArray[np.float64],
        hyperparameters: dict[str, Any],
        class_weights: dict[int, float] | None,
    ) -> tuple[
        MatchPredictor,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        dict[int, float] | None,
        dict[str, Any],
    ]:
        """Train the final model and recover from one-class collapse."""
        predictor = self._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            hyperparameters,
            class_weights,
        )
        initial_report = self._prediction_class_report(predictor, X_val)
        if not self._is_prediction_collapsed(initial_report):
            self._collapse_recovery = {
                "enabled": self.config.collapse_recovery,
                "triggered": False,
                "validation_predictions": initial_report,
            }
            return predictor, X_train, X_val, X_test, class_weights, hyperparameters

        if not self.config.collapse_recovery or self.config.model_type != "xgboost":
            raise ValueError(
                "Training produced a degenerate model: validation predictions "
                f"covered {initial_report['num_classes']} classes with counts "
                f"{initial_report['counts']}. Enable collapse recovery or adjust "
                "the model configuration."
            )

        recovery_notes: list[dict[str, Any]] = [
            {"stage": "initial", "validation_predictions": initial_report}
        ]
        recovered_weights = class_weights or get_class_weights(y_train)
        recovered_hyperparameters = dict(hyperparameters)

        restored = self._restore_full_feature_matrices(y_train)
        if restored is not None:
            X_train, X_val, X_test = restored
            recovery_notes.append(
                {
                    "stage": "restore_full_feature_set",
                    "num_features": len(self.feature_pipeline.feature_names),
                }
            )

        predictor = self._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            recovered_hyperparameters,
            recovered_weights,
        )
        recovered_report = self._prediction_class_report(predictor, X_val)
        if not self._is_prediction_collapsed(recovered_report):
            self._collapse_recovery = {
                "enabled": True,
                "triggered": True,
                "strategy": "class_weighted_full_features",
                "validation_predictions": recovered_report,
                "notes": recovery_notes,
            }
            return (
                predictor,
                X_train,
                X_val,
                X_test,
                recovered_weights,
                recovered_hyperparameters,
            )

        recovery_notes.append(
            {
                "stage": "class_weighted_full_features",
                "validation_predictions": recovered_report,
            }
        )
        recovered_hyperparameters = self._collapse_recovery_hyperparameters(
            hyperparameters
        )
        predictor = self._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            recovered_hyperparameters,
            recovered_weights,
        )
        relaxed_report = self._prediction_class_report(predictor, X_val)
        if not self._is_prediction_collapsed(relaxed_report):
            self._collapse_recovery = {
                "enabled": True,
                "triggered": True,
                "strategy": "class_weighted_relaxed_xgboost",
                "validation_predictions": relaxed_report,
                "notes": recovery_notes,
            }
            return (
                predictor,
                X_train,
                X_val,
                X_test,
                recovered_weights,
                recovered_hyperparameters,
            )

        raise ValueError(
            "Training produced a degenerate model even after recovery: validation "
            f"predictions covered {relaxed_report['num_classes']} classes with "
            f"counts {relaxed_report['counts']}. Refusing to save or activate it."
        )

    def run(self) -> TrainingResult:
        """Execute the complete training pipeline.

        Returns:
            TrainingResult with model info and metrics
        """
        import time

        start_time = time.time()

        # Step 1: Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data()

        # Step 2: Handle class imbalance (opt-in only)
        if self.config.outcome_balance:
            class_weights = get_class_weights(y_train)
        else:
            class_weights = None

        # Step 3: Optional feature selection using a probe model
        if self.config.feature_selection:
            X_train, X_val, X_test = self._apply_feature_selection(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                class_weights=class_weights,
                hyperparameters=self.config.hyperparameters.copy(),
            )

        # Step 4: Hyperparameter tuning (optional)
        tuning_result = None
        best_params = self.config.hyperparameters.copy()

        if self.config.tune_hyperparameters and HAS_OPTUNA:
            tuning_result = self._tune_hyperparameters(
                X_train, y_train, X_val, y_val, class_weights
            )
            best_params = tuning_result.best_params

        # Step 5: Train model, refusing to persist one-class collapses.
        (
            predictor,
            X_train,
            X_val,
            X_test,
            class_weights,
            best_params,
        ) = self._train_with_collapse_recovery(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            hyperparameters=best_params,
            class_weights=class_weights,
        )

        # Step 6: Calibrate probabilities (optional)
        if self.config.calibrate_probabilities:
            self._calibrator = ProbabilityCalibrator(
                method=self.config.calibration_method,
            )
            val_probas = predictor.predict_proba(X_val)
            self._calibrator.fit(val_probas, y_val)
            calibrated_predictor = CalibratedPredictor(predictor, self._calibrator)
            calibrated_report = self._prediction_class_report(
                calibrated_predictor,
                X_val,
            )
            if self._is_prediction_collapsed(calibrated_report):
                self._calibrator = None
                if self._collapse_recovery is None:
                    self._collapse_recovery = {}
                self._collapse_recovery["calibration_disabled"] = True
                self._collapse_recovery["calibrated_validation_predictions"] = (
                    calibrated_report
                )

        # Step 7: Evaluate
        # Train/val use raw probabilities (calibrator is fit on val, so
        # calibrating val again would leak data and make calibration metrics
        # look artificially perfect).
        train_metrics = self._evaluate(
            predictor, X_train, y_train, self._train_df, apply_calibration=False
        )
        val_metrics = self._evaluate(
            predictor, X_val, y_val, self._val_df, apply_calibration=False
        )
        test_metrics = self._evaluate(
            predictor, X_test, y_test, self._test_df, apply_calibration=True
        )

        # Step 8: Register model
        all_metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        if not self.config.use_ensemble:
            resolved_registry_hyperparameters = {
                **resolve_training_hyperparameters(
                    model_type=self.config.model_type,
                    hyperparameters=best_params,
                ),
                **predictor.effective_hyperparameters,
            }
        else:
            resolved_registry_hyperparameters = dict(best_params)
        model_hyperparameters: dict[str, Any] = {
            **resolved_registry_hyperparameters,
            "feature_names": self.feature_pipeline.feature_names,
            "random_seed": self.config.random_seed,
            "early_stopping_rounds": self.config.early_stopping_rounds,
            "use_ensemble": self.config.use_ensemble,
            "outcome_balance": class_weights is not None,
            "min_prediction_classes": self.config.min_prediction_classes,
        }
        if self._selected_feature_names is not None:
            model_hyperparameters["selected_feature_names"] = (
                self._selected_feature_names
            )
            model_hyperparameters["feature_selection"] = {
                "enabled": True,
                "threshold": self.config.feature_selection_threshold,
                "min_samples_per_feature": self.config.min_samples_per_feature,
                "num_selected": len(self._selected_feature_names),
            }
        if self.config.use_ensemble:
            model_hyperparameters["ensemble_types"] = self.config.ensemble_types
        if self.config.calibrate_probabilities:
            model_hyperparameters["calibration_method"] = self.config.calibration_method
            model_hyperparameters["calibration_enabled"] = self._calibrator is not None
        if self._collapse_recovery is not None:
            model_hyperparameters["collapse_recovery"] = self._collapse_recovery

        model_version = self.model_registry.save_model(
            model=CalibratedPredictor(predictor, self._calibrator)
            if self._calibrator
            else predictor,
            name=self.config.model_name,
            metrics=all_metrics,
            model_type=self.config.model_type,
            feature_schema_version=self.config.feature_schema_version,
            hyperparameters=model_hyperparameters,
            description=self.config.description,
            tags=self.config.tags,
        )
        model_dir = self.models_path / self.config.model_type / model_version
        self.feature_pipeline_path = model_dir / "feature_pipeline"
        self.feature_pipeline.save(self.feature_pipeline_path)
        model_hyperparameters["feature_pipeline_path"] = str(self.feature_pipeline_path)
        self.model_registry.update_model_hyperparameters(
            model_version=model_version,
            hyperparameters=model_hyperparameters,
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
            model_path=model_dir,
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
        6. Cache raw features for reproducibility
        """
        from algobet.predictions.features.pipeline import prepare_match_dataframe

        # Get historical matches with optional date and filter constraints
        matches = self.repo.get_historical_matches(
            min_date=self.config.start_date,
            max_date=self.config.end_date,
            tournament_ids=self.config.tournament_ids,
            team_ids=self.config.team_ids,
            require_results=True,
            min_total_goals=self.config.min_total_goals,
            max_total_goals=self.config.max_total_goals,
            venue_filter=self.config.venue_filter,
        )

        if not matches:
            raise ValueError("No historical matches found for training")

        # Check minimum matches requirement if specified
        min_matches = getattr(self.config, "min_matches", None)
        if min_matches and len(matches) < min_matches:
            raise ValueError(
                f"Insufficient matches: {len(matches)} < {min_matches}. "
                "Adjust date range or reduce minimum matches requirement."
            )

        # Convert to DataFrame
        matches_df = prepare_match_dataframe(matches)

        # Preload all team match history and H2H data into memory to avoid
        # per-match DB queries during feature generation (N+1 → 2 bulk queries)
        all_team_ids = list(
            set(
                matches_df["home_team_id"].tolist()
                + matches_df["away_team_id"].tolist()
            )
        )
        max_match_date = matches_df["match_date"].max()
        self.repo.preload_team_matches(all_team_ids, before_date=max_match_date)
        team_pairs = list(
            zip(
                matches_df["home_team_id"].tolist(),
                matches_df["away_team_id"].tolist(),
                strict=False,
            )
        )
        self.repo.preload_h2h_matches(team_pairs, before_date=max_match_date)

        # Preload standings for tournament-season pairs
        tournament_season_pairs = list(
            set(
                zip(
                    matches_df["tournament_id"].tolist(),
                    matches_df["season_id"].tolist(),
                    strict=False,
                )
            )
        )
        self.repo.preload_season_standings(
            tournament_season_pairs, before_date=max_match_date
        )

        # Add result column
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )

        # Split data using configured strategy
        if self.config.split_strategy == "expanding_window":
            splitter = ExpandingWindowSplitter(
                min_train_size=self.config.min_train_size,
                val_size=self.config.ew_val_size,
                test_size=self.config.ew_test_size,
                step_size=self.config.step_size,
            )
        elif self.config.split_strategy == "season_aware":
            splitter = SeasonAwareSplitter(
                train_seasons=self.config.train_seasons,
                val_seasons=self.config.val_seasons,
                test_seasons=self.config.test_seasons,
            )
        else:
            splitter = TemporalSplitter(
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                gap_days=self.config.gap_days,
            )

        splits = list(splitter.split(matches_df))
        split = splits[0]  # Use the first (or only) split

        # Encode targets
        y = encode_targets(matches_df["result"].values)
        y_train = y[split.train_indices]
        y_val = y[split.val_indices]
        y_test = y[split.test_indices]

        # Fit pipeline on training data only, then transform all subsets
        train_df = matches_df.iloc[split.train_indices]
        val_df = matches_df.iloc[split.val_indices]
        test_df = matches_df.iloc[split.test_indices]
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df

        X_train = self.feature_pipeline.fit_transform(train_df, self.repo)
        self._train_raw_features = self.feature_pipeline.last_raw_features
        X_val = self.feature_pipeline.transform(val_df, self.repo)
        self._val_raw_features = self.feature_pipeline.last_raw_features
        X_test = self.feature_pipeline.transform(test_df, self.repo)
        self._test_raw_features = self.feature_pipeline.last_raw_features

        # Cache raw features if enabled, reusing frames already produced while
        # fitting and transforming the split data.
        if self.config.use_feature_cache:
            try:
                import pandas as pd

                raw_frames = [
                    frame
                    for frame in (
                        self._train_raw_features,
                        self._val_raw_features,
                        self._test_raw_features,
                    )
                    if frame is not None
                ]
                raw_features = pd.concat(raw_frames) if raw_frames else pd.DataFrame()
                from algobet.predictions.features.store import features_to_store_format

                features_list = features_to_store_format(
                    raw_features,
                    schema_version=self.config.feature_schema_version,
                )
                self.feature_store.store_bulk(features_list)
            except Exception:
                pass  # Feature caching is best-effort

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _tune_hyperparameters(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        class_weights: dict[int, float] | None,
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
        y_val: NDArray[np.float64],
        hyperparameters: dict[str, Any],
        class_weights: dict[int, float] | None,
    ) -> MatchPredictor:
        """Train the prediction model."""
        if self.config.use_ensemble:
            # Train ensemble of multiple model types
            predictors = []
            for model_type in self.config.ensemble_types:
                model_hyperparameters = resolve_training_hyperparameters(
                    model_type=model_type,
                    hyperparameters=hyperparameters,
                )
                config = ModelConfig(
                    model_type=model_type,
                    hyperparameters=model_hyperparameters,
                    class_weights=class_weights,
                    random_seed=self.config.random_seed,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                )
                predictor = create_predictor(model_type, config)
                predictor.set_feature_names(self.feature_pipeline.feature_names)
                predictor.fit(X_train, y_train, X_val, y_val)
                predictors.append(predictor)

            return EnsemblePredictor(predictors=predictors)
        else:
            # Train single model
            model_hyperparameters = resolve_training_hyperparameters(
                model_type=self.config.model_type,
                hyperparameters=hyperparameters,
            )
            config = ModelConfig(
                model_type=self.config.model_type,
                hyperparameters=model_hyperparameters,
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
        matches_df: pd.DataFrame | None = None,
        apply_calibration: bool = True,
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            predictor: Fitted predictor
            X: Feature matrix
            y: True labels
            matches_df: Optional match metadata for market diagnostics
            apply_calibration: If False, report raw probabilities even when
                a calibrator exists. Used for train/val to avoid calibrator
                contamination (the calibrator is fit on val data).
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
        )

        # Get predictions
        probas = predictor.predict_proba(X)

        # Apply calibration only when explicitly requested.
        # We skip calibration for train/val because the calibrator was fit on
        # the validation set; applying it back to val would make calibration
        # metrics look artificially perfect (data leakage).
        if apply_calibration and self._calibrator is not None:
            probas = self._calibrator.calibrate(probas)

        y_pred = np.argmax(probas, axis=1)
        prediction_counts = np.bincount(y_pred, minlength=probas.shape[1])
        max_prediction_share = (
            float(prediction_counts.max() / len(y_pred)) if len(y_pred) else 0.0
        )

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, probas, labels=list(range(probas.shape[1]))),
            "precision_macro": precision_score(
                y, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "predicted_classes": float(np.count_nonzero(prediction_counts)),
            "max_prediction_share": max_prediction_share,
        }

        # Add calibration metrics
        cal_metrics = calculate_calibration_metrics(y, probas)
        metrics.update(cal_metrics)
        metrics.update(self._calculate_market_diagnostics(y, probas, matches_df))

        return metrics

    def _calculate_market_diagnostics(
        self,
        y: NDArray[np.int64],
        probas: NDArray[np.float64],
        matches_df: pd.DataFrame | None,
    ) -> dict[str, float]:
        """Compare model probabilities to implied odds for diagnostics only."""
        if matches_df is None:
            return {}

        required_columns = ["odds_home", "odds_draw", "odds_away"]
        if not all(column in matches_df.columns for column in required_columns):
            return {}

        odds = matches_df[required_columns].astype(float).to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(odds).all(axis=1) & (odds > 0).all(axis=1)
        if not np.any(valid_mask):
            return {"market_samples": 0.0}

        valid_odds = odds[valid_mask]
        implied = 1.0 / valid_odds
        implied = implied / implied.sum(axis=1, keepdims=True)

        y_valid = y[valid_mask]
        model_valid = probas[valid_mask]
        row_indices = np.arange(len(y_valid))

        market_log_loss = -np.log(
            np.clip(implied[row_indices, y_valid], 1e-12, 1.0)
        ).mean()
        market_favorite = np.argmax(implied, axis=1)
        model_favorite = np.argmax(model_valid, axis=1)

        return {
            "market_samples": float(len(y_valid)),
            "market_log_loss": float(market_log_loss),
            "market_favorite_accuracy": float(np.mean(market_favorite == y_valid)),
            "market_model_probability_mae": float(np.abs(model_valid - implied).mean()),
            "market_favorite_agreement": float(
                np.mean(model_favorite == market_favorite)
            ),
        }

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
