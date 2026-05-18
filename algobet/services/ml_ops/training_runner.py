"""Training use case for ML operations."""

import json
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from algobet.api.schemas.ml_operations import TrainModelRequest, TrainModelResponse
from algobet.infrastructure.logging_config import get_logger
from algobet.models import ModelVersion
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.pipeline import (
    MODEL_FEATURE_SCHEMA_VERSION,
    TrainingConfig,
    TrainingPipeline,
)

logger = get_logger("ml_operations")


class TrainingRunner:
    """Run this ML operation use case."""

    def run_training(
        self, request: TrainModelRequest, db: Session
    ) -> TrainModelResponse:
        """Train a prediction model and optionally activate it."""
        # Log the training request payload
        logger.info(
            json.dumps(
                {
                    "event": "training_request_received",
                    "model_type": request.model_type,
                    "tune_hyperparameters": request.tune_hyperparameters,
                    "description": request.description,
                    "activate": request.activate,
                    "start_date": str(request.start_date)
                    if request.start_date
                    else None,
                    "end_date": str(request.end_date) if request.end_date else None,
                    "min_matches": request.min_matches,
                    "tournament_ids": request.tournament_ids,
                    "team_ids": request.team_ids,
                    "venue_filter": request.venue_filter,
                    "train_ratio": request.train_ratio,
                    "val_ratio": request.val_ratio,
                    "test_ratio": request.test_ratio,
                    "random_seed": request.random_seed,
                    "early_stopping_rounds": request.early_stopping_rounds,
                    "calibrate_probabilities": request.calibrate_probabilities,
                    "calibration_method": request.calibration_method,
                    "use_ensemble": request.use_ensemble,
                    "split_strategy": request.split_strategy,
                    "outcome_balance_strength": request.outcome_balance_strength,
                    "production_lane": request.production_lane,
                    "taken_odds_snapshot": request.taken_odds_snapshot,
                    "closing_odds_required": request.closing_odds_required,
                    "min_expected_clv": request.min_expected_clv,
                    "min_positive_clv_probability": (
                        request.min_positive_clv_probability
                    ),
                }
            )
        )

        # Validate split ratios sum to 1.0
        total_ratio = request.train_ratio + request.val_ratio + request.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise HTTPException(
                status_code=400,
                detail=f"Split ratios must sum to 1.0, got {total_ratio:.2f}",
            )

        feature_groups = request.feature_groups
        if request.model_type == "market_mediation":
            feature_groups = list(feature_groups or [])
            if "market_mediation" not in feature_groups:
                feature_groups.append("market_mediation")

        config = TrainingConfig(
            model_type=request.model_type,
            tune_hyperparameters=request.tune_hyperparameters,
            description=request.description
            or f"Frontend trained {request.model_type} model",
            # Data range settings
            start_date=request.start_date,
            end_date=request.end_date,
            min_matches=request.min_matches,
            # Filtering: tournament and team selection
            tournament_ids=request.tournament_ids,
            team_ids=request.team_ids,
            venue_filter=request.venue_filter,
            # Match quality filters
            min_total_goals=request.min_total_goals,
            max_total_goals=request.max_total_goals,
            # Split ratios
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            # Training settings
            random_seed=request.random_seed,
            early_stopping_rounds=request.early_stopping_rounds,
            tuning_trials=request.tuning_trials,
            # Calibration settings
            calibrate_probabilities=request.calibrate_probabilities,
            calibration_method=request.calibration_method,
            # Outcome balancing
            outcome_balance=request.outcome_balance or False,
            outcome_balance_strength=request.outcome_balance_strength,
            production_lane=request.production_lane,
            taken_odds_snapshot=request.taken_odds_snapshot,
            closing_odds_required=request.closing_odds_required,
            min_expected_clv=request.min_expected_clv,
            min_positive_clv_probability=request.min_positive_clv_probability,
            # Feature importance pruning
            feature_selection=request.feature_selection,
            feature_selection_threshold=request.feature_selection_threshold,
            min_samples_per_feature=request.min_samples_per_feature,
            # Feature groups
            feature_groups=feature_groups,
            # Ensemble training
            use_ensemble=request.use_ensemble,
            ensemble_types=request.ensemble_types or ["xgboost", "lightgbm"],
            # Stacking ensemble
            use_stacking_ensemble=request.use_stacking_ensemble,
            stacking_base_models=request.stacking_base_models,
            stacking_meta_learner=request.stacking_meta_learner,
            stacking_n_folds=request.stacking_n_folds,
            # Split strategy
            split_strategy=request.split_strategy,
            gap_days=request.gap_days,
            min_train_size=request.min_train_size,
            ew_val_size=request.ew_val_size,
            ew_test_size=request.ew_test_size,
            step_size=request.step_size,
            train_seasons=request.train_seasons,
            val_seasons=request.val_seasons,
            test_seasons=request.test_seasons,
            # Model tags
            tags=request.tags,
            # Custom hyperparameters
            hyperparameters=request.hyperparameters,
            feature_schema_version=MODEL_FEATURE_SCHEMA_VERSION,
            min_market_mediation_features=15
            if request.model_type == "market_mediation"
            else 0,
        )

        try:
            pipeline = TrainingPipeline(
                config=config,
                session=db,
                models_path=Path("data/models"),
            )
            result = pipeline.run()
        except (ImportError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            logger.error(f"Training failed with traceback:\n{tb}")
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {e}",
            ) from e

        registry = ModelRegistry(storage_path=Path("data/models"), session=db)
        is_active = False
        if request.activate and self._passes_activation_gate(result):
            registry.activate_model(result.model_version)
            is_active = True
        elif request.activate:
            logger.warning(
                "Model %s was trained but not activated because it failed "
                "post-training quality gates.",
                result.model_version,
            )

        db_model = (
            db.query(ModelVersion)
            .filter(ModelVersion.version == result.model_version)
            .first()
        )

        # Convert numpy float32 values to Python floats for JSON serialization
        def _to_float_dict(d: dict[str, float] | None) -> dict[str, float] | None:
            if d is None:
                return None
            return {k: float(v) for k, v in d.items()}

        return TrainModelResponse(
            model_id=db_model.id if db_model else None,
            model_version=result.model_version,
            model_type=result.model_type,
            is_active=is_active,
            feature_schema_version=result.feature_schema_version,
            num_features=result.num_features,
            trained_at=result.trained_at.isoformat(),
            training_duration_seconds=result.training_duration_seconds,
            train_metrics={k: float(v) for k, v in result.train_metrics.items()},
            val_metrics={k: float(v) for k, v in result.val_metrics.items()},
            test_metrics={k: float(v) for k, v in result.test_metrics.items()},
            feature_importance=_to_float_dict(result.feature_importance),
            ensemble_weights=result.ensemble_weights,
            ensemble_validation_metrics=result.ensemble_validation_metrics,
            ensemble_types=config.ensemble_types if config.use_ensemble else None,
            stacking_metadata=(
                {
                    "base_models": config.stacking_base_models,
                    "meta_learner": config.stacking_meta_learner,
                    "n_folds": config.stacking_n_folds,
                }
                if config.use_stacking_ensemble
                else None
            ),
            calibration_metadata=(
                {
                    "method": config.calibration_method,
                    "enabled": config.calibrate_probabilities,
                    "cv_folds": config.calibration_cv_folds,
                }
                if config.calibrate_probabilities
                else None
            ),
        )

    def _passes_activation_gate(self, result: object) -> bool:
        """Prevent known-poor models from becoming active automatically."""
        test_metrics = getattr(result, "test_metrics", {}) or {}

        predicted_classes = float(test_metrics.get("predicted_classes", 3.0))
        if predicted_classes < 2.0:
            return False

        if getattr(result, "model_type", None) == "market_mediation":
            fold_count = float(
                test_metrics.get("market_mediation_walk_forward_fold_count", 0.0)
            )
            if fold_count >= 4.0:
                coverage = float(
                    test_metrics.get(
                        "market_mediation_walk_forward_coverage_min",
                        0.0,
                    )
                )
                selected_bets = float(
                    test_metrics.get(
                        "market_mediation_walk_forward_pooled_selected_bets",
                        0.0,
                    )
                )
                mean_clv = float(
                    test_metrics.get(
                        "market_mediation_walk_forward_pooled_mean_clv",
                        0.0,
                    )
                )
                lower_95 = float(
                    test_metrics.get(
                        "market_mediation_walk_forward_pooled_clv_lower_95",
                        0.0,
                    )
                )
                positive_folds = float(
                    test_metrics.get(
                        "market_mediation_walk_forward_positive_clv_folds",
                        0.0,
                    )
                )
                if positive_folds < 3.0:
                    return False
            else:
                coverage = float(
                    test_metrics.get("market_mediation_closing_coverage", 0.0)
                )
                selected_bets = float(
                    test_metrics.get("market_mediation_selected_bets", 0.0)
                )
                mean_clv = float(
                    test_metrics.get("market_mediation_selected_mean_clv", 0.0)
                )
                lower_95 = float(
                    test_metrics.get("market_mediation_selected_clv_lower_95", 0.0)
                )
            residual_ll = test_metrics.get(
                "final_split_market_mediation_residual_lane_log_loss",
                test_metrics.get(
                    "market_mediation_residual_lane_log_loss",
                    test_metrics.get("log_loss"),
                ),
            )
            opening_ll = test_metrics.get(
                "final_split_market_mediation_opening_market_log_loss",
                test_metrics.get(
                    "market_mediation_opening_market_log_loss",
                    test_metrics.get("market_log_loss"),
                ),
            )
            if coverage < 0.80 or selected_bets < 200.0:
                return False
            if mean_clv <= 0.0 or lower_95 <= 0.0:
                return False
            if residual_ll is None or opening_ll is None:
                return False
            return float(residual_ll) <= float(opening_ll)

        model_log_loss = test_metrics.get("log_loss")
        market_log_loss = test_metrics.get("market_log_loss")
        if model_log_loss is None or market_log_loss is None:
            return True

        return float(model_log_loss) <= float(market_log_loss)
