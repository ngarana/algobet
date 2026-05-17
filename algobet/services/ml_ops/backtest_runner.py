"""Backtest use case for ML operations."""

import contextlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

from algobet.api.schemas.ml_operations import (
    BacktestRequest,
    BacktestResultResponse,
    BettingMetricsResponse,
    ClassificationMetricsResponse,
)
from algobet.models import BacktestHistory, Match, ModelVersion
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.evaluation import evaluate_predictions
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry
from algobet.services.ml_ops.utils import convert_numpy_types


class BacktestRunner:
    """Run this ML operation use case."""

    @staticmethod
    def _load_saved_feature_pipeline(
        model_meta: Any | None,
        db_model: ModelVersion | None,
    ) -> FeaturePipeline:
        """Load the fitted feature pipeline saved with a model."""
        feature_pipeline = None
        has_hyperparameters = (
            model_meta
            and hasattr(model_meta, "hyperparameters")
            and model_meta.hyperparameters
        )
        pipeline_path = (
            model_meta.hyperparameters.get("feature_pipeline_path")  # type: ignore[union-attr]
            if has_hyperparameters
            else None
        )

        if pipeline_path:
            p_path = Path(pipeline_path)
            if p_path.exists() and (p_path / "config.json").exists():
                with contextlib.suppress(Exception):
                    feature_pipeline = FeaturePipeline.load(p_path)

        if feature_pipeline is None and db_model and db_model.file_path:
            rel_path = Path(db_model.file_path).parent / "feature_pipeline"
            if rel_path.exists() and (rel_path / "config.json").exists():
                feature_pipeline = FeaturePipeline.load(rel_path)

        if feature_pipeline is None or not feature_pipeline.is_fitted:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Could not load fitted feature pipeline. Backtest aborted "
                    "to prevent scaling drift."
                ),
            )

        return feature_pipeline

    def run_backtest(
        self, request: BacktestRequest, db: Session
    ) -> BacktestResultResponse:
        """Run a historical backtest on model predictions.

        Evaluates model performance on historical match data with
        comprehensive classification and betting metrics.

        Args:
            request: Backtest parameters

        Returns:
            BacktestResultResponse with evaluation metrics

        Raises:
            HTTPException: If model not found or insufficient data
        """
        # Get model
        registry = ModelRegistry(storage_path=Path("data/models"), session=db)

        try:
            if request.model_version:
                model = registry.load_model(request.model_version)
                model_meta = next(
                    (
                        m
                        for m in registry.list_models()
                        if m.version == request.model_version
                    ),
                    None,
                )
                # Get database record for backtest association
                db_model = (
                    db.query(ModelVersion)
                    .filter(ModelVersion.version == request.model_version)
                    .first()
                )
            else:
                model, model_meta = registry.get_active_model()
                db_model = (
                    db.query(ModelVersion)
                    .filter(ModelVersion.version == model_meta.version)
                    .first()
                )
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {e}",
            ) from None

        # Set default date range
        end_date = request.end_date or datetime.now()
        start_date = request.start_date or (end_date - timedelta(days=365))

        # Get historical matches with results
        query = db.query(Match).options(
            joinedload(Match.home_team), joinedload(Match.away_team)
        )
        filters = [
            Match.status == "FINISHED",
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
            Match.match_date >= start_date,
            Match.match_date <= end_date,
            Match.odds_home.is_not(None),
            Match.odds_draw.is_not(None),
            Match.odds_away.is_not(None),
        ]

        if request.tournament_id:
            filters.append(Match.tournament_id == request.tournament_id)

        matches = query.filter(and_(*filters)).order_by(Match.match_date).all()

        if len(matches) < request.min_matches:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient matches: {len(matches)} < {request.min_matches}",
            )

        # Prepare data
        matches_df = prepare_match_dataframe(matches)

        # Calculate results
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )

        # Load the saved feature pipeline (MANDATORY for backtest consistency).
        feature_pipeline = self._load_saved_feature_pipeline(model_meta, db_model)

        repo = MatchRepository(db)

        # Preload data into cache — standings has no DB fallback and returns
        # None for every match without this, causing all standings features to
        # be NaN and the model to output its base-rate prediction for all rows.
        team_ids = list(
            {m.home_team_id for m in matches} | {m.away_team_id for m in matches}
        )
        tournament_season_pairs = list(
            {
                (m.tournament_id, m.season_id)
                for m in matches
                if m.tournament_id is not None and m.season_id is not None
            }
        )
        repo.preload_team_matches(team_ids, before_date=end_date)
        repo.preload_h2h_matches(
            [(m.home_team_id, m.away_team_id) for m in matches],
            before_date=end_date,
        )
        repo.preload_season_standings(tournament_season_pairs, before_date=end_date)

        X_test = feature_pipeline.transform(matches_df, repo)

        # Get odds for betting simulation
        odds = matches_df[["odds_home", "odds_draw", "odds_away"]].values

        # Get predictions
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)

        # Encode true labels
        result_map = {"H": 0, "D": 1, "A": 2}
        y_true = matches_df["result"].map(result_map).values

        # Evaluate
        date_range = (
            str(matches_df["match_date"].min().date()),
            str(matches_df["match_date"].max().date()),
        )

        # Only one odds snapshot per match is persisted (Match.odds_*).
        # Treat it as the closing market price and route CLV through the
        # model-CLV path so the metric reflects model-vs-closing edge
        # rather than the structurally-zero dual-snapshot value.
        result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            odds=odds,
            model_version=model_meta.version if model_meta else "unknown",
            date_range=date_range,
            min_edge=request.min_edge,
            use_model_clv=True,
        )

        # Save backtest results to database
        if db_model:
            backtest_record = BacktestHistory(
                model_version_id=db_model.id,
                min_matches=request.min_matches,
                start_date=start_date,
                end_date=end_date,
                num_samples=result.num_samples,
                date_range_start=date_range[0],
                date_range_end=date_range[1],
                accuracy=convert_numpy_types(result.classification.accuracy),
                log_loss=convert_numpy_types(result.classification.log_loss),
                brier_score=convert_numpy_types(result.classification.brier_score),
                f1_macro=convert_numpy_types(result.classification.f1_macro),
                f1_weighted=convert_numpy_types(result.classification.f1_weighted),
                precision_macro=convert_numpy_types(
                    result.classification.precision_macro
                ),
                recall_macro=convert_numpy_types(result.classification.recall_macro),
                top_2_accuracy=convert_numpy_types(
                    result.classification.top_2_accuracy
                ),
                cohen_kappa=convert_numpy_types(result.classification.cohen_kappa),
                f1_home=convert_numpy_types(
                    result.classification.per_class_f1.get("H", 0.0)
                ),
                f1_draw=convert_numpy_types(
                    result.classification.per_class_f1.get("D", 0.0)
                ),
                f1_away=convert_numpy_types(
                    result.classification.per_class_f1.get("A", 0.0)
                ),
                total_bets=convert_numpy_types(result.betting.total_bets)
                if result.betting
                else None,
                win_rate=convert_numpy_types(result.betting.win_rate)
                if result.betting
                else None,
                roi_percent=convert_numpy_types(result.betting.roi_percent)
                if result.betting
                else None,
                profit_loss=convert_numpy_types(result.betting.profit_loss)
                if result.betting
                else None,
                sharpe_ratio=convert_numpy_types(result.betting.sharpe_ratio)
                if result.betting
                else None,
                max_drawdown=convert_numpy_types(result.betting.max_drawdown)
                if result.betting
                else None,
                expected_calibration_error=convert_numpy_types(
                    result.expected_calibration_error
                ),
                maximum_calibration_error=convert_numpy_types(
                    result.maximum_calibration_error
                ),
                full_metrics=convert_numpy_types(
                    {
                        "classification": {
                            "accuracy": result.classification.accuracy,
                            "log_loss": result.classification.log_loss,
                            "brier_score": result.classification.brier_score,
                            "f1_macro": result.classification.f1_macro,
                            "per_class_f1": result.classification.per_class_f1,
                            "confusion_matrix": result.classification.confusion_matrix,
                        },
                        "betting": {
                            "total_bets": result.betting.total_bets
                            if result.betting
                            else None,
                            "win_rate": result.betting.win_rate
                            if result.betting
                            else None,
                            "roi_percent": result.betting.roi_percent
                            if result.betting
                            else None,
                            "sharpe_ratio": result.betting.sharpe_ratio
                            if result.betting
                            else None,
                            "max_drawdown": result.betting.max_drawdown
                            if result.betting
                            else None,
                            "mean_clv": result.betting.mean_clv
                            if result.betting
                            else None,
                            "clv_hit_rate": result.betting.clv_hit_rate
                            if result.betting
                            else None,
                            "clv_weighted_roi": result.betting.clv_weighted_roi
                            if result.betting
                            else None,
                        }
                        if result.betting
                        else None,
                        "calibration": {
                            "expected_calibration_error": (
                                result.expected_calibration_error
                            ),
                            "maximum_calibration_error": (
                                result.maximum_calibration_error
                            ),
                        },
                        "outcome_accuracy": result.outcome_accuracy,
                    }
                ),
            )
            db.add(backtest_record)
            db.commit()
            db.refresh(backtest_record)

        # Build response
        return BacktestResultResponse(
            model_version=result.model_version,
            evaluated_at=result.evaluated_at,
            num_samples=result.num_samples,
            date_range=result.date_range,
            classification=ClassificationMetricsResponse(
                accuracy=result.classification.accuracy,
                log_loss=result.classification.log_loss,
                brier_score=result.classification.brier_score,
                precision_macro=result.classification.precision_macro,
                recall_macro=result.classification.recall_macro,
                f1_macro=result.classification.f1_macro,
                precision_weighted=result.classification.precision_weighted,
                recall_weighted=result.classification.recall_weighted,
                f1_weighted=result.classification.f1_weighted,
                per_class_precision=result.classification.per_class_precision,
                per_class_recall=result.classification.per_class_recall,
                per_class_f1=result.classification.per_class_f1,
                confusion_matrix=result.classification.confusion_matrix,
                top_2_accuracy=result.classification.top_2_accuracy,
                cohen_kappa=result.classification.cohen_kappa,
            ),
            betting=(
                BettingMetricsResponse(
                    total_bets=result.betting.total_bets,
                    winning_bets=result.betting.winning_bets,
                    losing_bets=result.betting.losing_bets,
                    total_stake=result.betting.total_stake,
                    total_return=result.betting.total_return,
                    profit_loss=result.betting.profit_loss,
                    roi_percent=result.betting.roi_percent,
                    yield_percent=result.betting.yield_percent,
                    sharpe_ratio=result.betting.sharpe_ratio,
                    max_drawdown=result.betting.max_drawdown,
                    win_rate=result.betting.win_rate,
                    average_winning_odds=result.betting.average_winning_odds,
                    average_losing_odds=result.betting.average_losing_odds,
                    average_kelly_fraction=result.betting.average_kelly_fraction,
                    optimal_kelly_fraction=result.betting.optimal_kelly_fraction,
                    mean_clv=result.betting.mean_clv,
                    clv_hit_rate=result.betting.clv_hit_rate,
                    clv_weighted_roi=result.betting.clv_weighted_roi,
                )
                if result.betting
                else None
            ),
            expected_calibration_error=result.expected_calibration_error,
            maximum_calibration_error=result.maximum_calibration_error,
            outcome_accuracy=result.outcome_accuracy,
        )
