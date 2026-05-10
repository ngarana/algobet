#!/usr/bin/env python3
"""Run a backtest for model 59 and store results."""

import os
import sys

sys.path.insert(0, "/home/arch/Coding/algobet")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "algobet")
os.environ.setdefault("POSTGRES_PASSWORD", "password")
os.environ.setdefault("POSTGRES_DB", "football")
from datetime import datetime
from pathlib import Path

import numpy as np

from algobet.infrastructure.database import get_session
from algobet.models import BacktestHistory, ModelVersion
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.evaluation import evaluate_predictions
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry


def run_backtest_for_model(model_version: str) -> dict:
    with get_session() as session:
        registry = ModelRegistry(storage_path=Path("data/models"), session=session)
        db_model = (
            session.query(ModelVersion)
            .filter(ModelVersion.version == model_version)
            .first()
        )
        if not db_model:
            raise ValueError(f"Model {model_version} not found in DB")
        model = registry.load_model(model_version)
        model_meta = next(
            (m for m in registry.list_models() if m.version == model_version),
            None,
        )
        # Query finished matches with odds
        from sqlalchemy import and_
        from sqlalchemy.orm import joinedload

        from algobet.matches.models import Match

        matches = (
            session.query(Match)
            .options(joinedload(Match.home_team), joinedload(Match.away_team))
            .filter(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None),
                    Match.odds_home.is_not(None),
                    Match.odds_draw.is_not(None),
                    Match.odds_away.is_not(None),
                )
            )
            .order_by(Match.match_date)
            .all()
        )
        print(f"Loaded {len(matches)} historical matches")
        matches_df = prepare_match_dataframe(matches)
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )
        # Load feature pipeline
        feature_pipeline = None
        pipeline_path = None
        if model_meta and model_meta.hyperparameters:
            pipeline_path = model_meta.hyperparameters.get("feature_pipeline_path")
        if pipeline_path:
            pipeline_path = Path(pipeline_path)
            if pipeline_path.exists() and (pipeline_path / "config.json").exists():
                try:
                    feature_pipeline = FeaturePipeline.load(pipeline_path)
                    print(f"Loaded feature pipeline from {pipeline_path}")
                except Exception as e:
                    print(f"Failed to load pipeline: {e}")
        if feature_pipeline is None:
            feature_pipeline = FeaturePipeline.create_default()
            print("Using default feature pipeline")
        # Temporal split: first 30% train, rest test
        train_size = int(len(matches) * 0.3)
        train_matches = matches_df.iloc[:train_size]
        test_matches = matches_df.iloc[train_size:]
        repo = MatchRepository(session)
        # Only fit if not already fitted from saved pipeline
        if not feature_pipeline.is_fitted:
            feature_pipeline.fit(train_matches, repo)
            print("Fitted feature pipeline on train split")
        else:
            print("Using pre-fitted feature pipeline")
        X_test = feature_pipeline.transform(test_matches, repo)
        odds = test_matches[["odds_home", "odds_draw", "odds_away"]].values
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        result_map = {"H": 0, "D": 1, "A": 2}
        y_true = test_matches["result"].map(result_map).values
        date_range = (
            str(test_matches["match_date"].min().date()),
            str(test_matches["match_date"].max().date()),
        )
        result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            odds=odds,
            model_version=model_version,
            date_range=date_range,
        )
        # Store backtest record
        full_metrics = {
            "classification": {
                "accuracy": result.classification.accuracy,
                "log_loss": result.classification.log_loss,
                "brier_score": result.classification.brier_score,
                "f1_macro": result.classification.f1_macro,
                "per_class_f1": result.classification.per_class_f1,
                "confusion_matrix": result.classification.confusion_matrix,
            },
            "betting": {
                "total_bets": result.betting.total_bets if result.betting else None,
                "win_rate": result.betting.win_rate if result.betting else None,
                "roi_percent": result.betting.roi_percent if result.betting else None,
                "sharpe_ratio": result.betting.sharpe_ratio if result.betting else None,
                "max_drawdown": result.betting.max_drawdown if result.betting else None,
            }
            if result.betting
            else None,
            "calibration": {
                "expected_calibration_error": result.expected_calibration_error,
                "maximum_calibration_error": result.maximum_calibration_error,
            },
        }
        backtest = BacktestHistory(
            model_version_id=db_model.id,
            min_matches=100,
            start_date=datetime(2000, 1, 1),
            end_date=datetime.now(),
            num_samples=result.num_samples,
            date_range_start=date_range[0],
            date_range_end=date_range[1],
            accuracy=result.classification.accuracy,
            log_loss=result.classification.log_loss,
            brier_score=result.classification.brier_score,
            f1_macro=result.classification.f1_macro,
            f1_weighted=result.classification.f1_weighted,
            precision_macro=result.classification.precision_macro,
            recall_macro=result.classification.recall_macro,
            top_2_accuracy=result.classification.top_2_accuracy,
            cohen_kappa=result.classification.cohen_kappa,
            total_bets=result.betting.total_bets if result.betting else None,
            win_rate=result.betting.win_rate if result.betting else None,
            roi_percent=result.betting.roi_percent if result.betting else None,
            profit_loss=result.betting.profit_loss if result.betting else None,
            sharpe_ratio=result.betting.sharpe_ratio if result.betting else None,
            max_drawdown=result.betting.max_drawdown if result.betting else None,
            expected_calibration_error=result.expected_calibration_error,
            maximum_calibration_error=result.maximum_calibration_error,
            full_metrics=full_metrics,
        )
        session.add(backtest)
        session.commit()
        print(f"\nBacktest stored (id={backtest.id})")
        print(f"  Samples: {result.num_samples}")
        print(f"  Accuracy: {result.classification.accuracy:.4f}")
        print(f"  Log Loss: {result.classification.log_loss:.4f}")
        print(f"  F1 Macro: {result.classification.f1_macro:.4f}")
        print(f"  ECE: {result.expected_calibration_error:.4f}")
        print(f"  MCE: {result.maximum_calibration_error:.4f}")
        if result.betting:
            print(f"  Total Bets: {result.betting.total_bets}")
            print(f"  Win Rate: {result.betting.win_rate:.4f}")
            print(f"  ROI%: {result.betting.roi_percent:.4f}")
            print(f"  Sharpe: {result.betting.sharpe_ratio:.4f}")
            print(f"  Max Drawdown: {result.betting.max_drawdown:.4f}")
        else:
            print("  No betting metrics generated")
        return {
            "accuracy": result.classification.accuracy,
            "log_loss": result.classification.log_loss,
            "f1_macro": result.classification.f1_macro,
            "ece": result.expected_calibration_error,
            "mce": result.maximum_calibration_error,
            "total_bets": result.betting.total_bets if result.betting else None,
            "win_rate": result.betting.win_rate if result.betting else None,
            "roi_percent": result.betting.roi_percent if result.betting else None,
        }


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    run_backtest_for_model("xgboost_20260508_120707")
