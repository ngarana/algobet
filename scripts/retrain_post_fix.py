#!/usr/bin/env python3
"""Retrain a fresh model with all fixes applied."""

import os
import sys

sys.path.insert(0, "/home/arch/Coding/algobet")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "algobet")
os.environ.setdefault("POSTGRES_PASSWORD", "password")
os.environ.setdefault("POSTGRES_DB", "football")

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

# Force import of all model modules so SQLAlchemy mappers resolve correctly
from algobet.infrastructure.database import get_session
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.pipeline import TrainingConfig, TrainingPipeline


def main():
    config = TrainingConfig(
        model_type="xgboost",
        description="EPL odds-free calibrated probability model (post-fix retrain)",
        tournament_ids=[359],
        feature_groups=[
            "team_form",
            "head_to_head",
            "temporal",
            "standings",
            "enriched_stats",
        ],
        feature_selection=True,
        feature_selection_threshold=0.005,
        min_samples_per_feature=40,
        min_matches=150,
        outcome_balance=False,
        tune_hyperparameters=False,
        calibrate_probabilities=True,
        calibration_method="sigmoid",
        hyperparameters={
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 1200,
            "min_child_weight": 10,
            "gamma": 1.0,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
        },
        tags={"model_scope": "epl", "odds_policy": "pure_model", "retrain": "post_fix"},
    )

    with get_session() as session:
        pipeline = TrainingPipeline(
            config=config,
            session=session,
            models_path=Path("data/models"),
        )
        print("Starting training...")
        result = pipeline.run()
        print(f"\nModel trained: {result.model_version}")
        print(f"  Type: {result.model_type}")
        print(f"  Features: {result.num_features}")
        print(f"  Duration: {result.training_duration_seconds:.1f}s")
        print("\nMetrics (raw / uncalibrated for train/val):")
        print(f"  Train accuracy: {result.train_metrics.get('accuracy'):.4f}")
        print(f"  Train log_loss: {result.train_metrics.get('log_loss'):.4f}")
        print(f"  Val accuracy:   {result.val_metrics.get('accuracy'):.4f}")
        print(f"  Val log_loss:   {result.val_metrics.get('log_loss'):.4f}")
        print(f"  Test accuracy:  {result.test_metrics.get('accuracy'):.4f}")
        print(f"  Test log_loss:  {result.test_metrics.get('log_loss'):.4f}")
        print("\nCalibration (test only, calibrated):")
        print(
            f"  Test ECE: {result.test_metrics.get('expected_calibration_error'):.4f}"
        )
        print(f"  Test MCE: {result.test_metrics.get('maximum_calibration_error'):.4f}")
        print("\nMarket diagnostics (test):")
        print(f"  Market log loss: {result.test_metrics.get('market_log_loss'):.4f}")
        print(
            f"  Market fav accuracy: {result.test_metrics.get('market_favorite_accuracy'):.4f}"
        )
        print(
            f"  Model-market prob MAE: {result.test_metrics.get('market_model_probability_mae'):.4f}"
        )
        print(
            f"  Market fav agreement: {result.test_metrics.get('market_favorite_agreement'):.4f}"
        )
        if result.feature_importance:
            top = sorted(
                result.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            print("\nTop 10 features:")
            for name, score in top:
                print(f"  {name}: {score:.4f}")

        # Activate the model
        registry = ModelRegistry(storage_path=Path("data/models"), session=session)
        registry.activate_model(result.model_version)
        print(f"\nActivated: {result.model_version}")

        return result.model_version


if __name__ == "__main__":
    main()
