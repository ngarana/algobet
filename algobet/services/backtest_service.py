"""Backtest service for running historical model evaluations."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

from algobet.exceptions import (
    InsufficientDataError,
    ModelNotFoundError,
    NoActiveModelError,
    PredictionError,
)
from algobet.logging_config import get_logger
from algobet.models import Match, ModelVersion
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry
from algobet.services.base import BaseService
from algobet.services.dto import BacktestRequest, BacktestResponse


class BacktestService(BaseService[Session]):
    """Service for running historical backtests on model predictions."""

    def __init__(
        self, session: Session, models_path: Path = Path("data/models")
    ) -> None:
        """Initialize the backtest service.

        Args:
            session: SQLAlchemy database session
            models_path: Path to model storage directory
        """
        super().__init__(session)
        self.logger = get_logger("services.backtest")
        self.models_path = models_path

    def _get_model(self, model_version: str | None) -> tuple[Any, ModelVersion | None]:
        """Load model from registry."""
        registry = ModelRegistry(storage_path=self.models_path, session=self.session)

        try:
            if model_version:
                model = registry.load_model(model_version)
                model_meta = None
                for m in registry.list_models():
                    if m.version == model_version:
                        model_meta = (
                            self.session.query(ModelVersion)
                            .filter(ModelVersion.version == model_version)
                            .first()
                        )
                        break
                return model, model_meta
            else:
                model, metadata = registry.get_active_model()
                model_meta = (
                    self.session.query(ModelVersion)
                    .filter(ModelVersion.version == metadata.version)
                    .first()
                )
                return model, model_meta
        except ValueError as e:
            if model_version:
                raise ModelNotFoundError(
                    f"Model version '{model_version}' not found.",
                    details={"version": model_version},
                ) from e
            raise NoActiveModelError(details={"error": str(e)}) from e
        except FileNotFoundError as e:
            if model_version:
                raise ModelNotFoundError(
                    f"Model version '{model_version}' not found.",
                    details={"version": model_version},
                ) from e
            raise NoActiveModelError(details={"error": str(e)}) from e

    def _load_matches(self, request: BacktestRequest) -> tuple[list[Match], int]:
        """Load and filter historical matches for backtesting."""

        query = self.session.query(Match).options(
            joinedload(Match.home_team), joinedload(Match.away_team)
        )
        filters = [
            Match.status == "FINISHED",
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
            Match.odds_home.is_not(None),
            Match.odds_draw.is_not(None),
            Match.odds_away.is_not(None),
        ]

        if request.tournament_id:
            filters.append(Match.tournament_id == request.tournament_id)

        matches = query.filter(and_(*filters)).order_by(Match.match_date).all()
        total_matches = len(matches)

        if total_matches < request.min_matches:
            self.logger.warning(
                "Insufficient matches for backtest",
                extra={
                    "operation": "run_backtest",
                    "total_matches": total_matches,
                    "min_matches": request.min_matches,
                },
            )
            raise InsufficientDataError(
                f"Insufficient matches: {total_matches} < "
                f"{request.min_matches} required.",
                details={
                    "total_matches": total_matches,
                    "min_matches": request.min_matches,
                },
            )

        self.logger.info(
            "Found matches for backtest",
            extra={"operation": "run_backtest", "total_matches": total_matches},
        )
        return matches, total_matches

    def _prepare_pipeline_and_data(
        self, matches: list[Match], model_meta: ModelVersion | None
    ) -> tuple[FeaturePipeline, MatchRepository, Any, Any]:
        """Prepare feature pipeline and split data for backtesting."""
        import contextlib

        matches_df = prepare_match_dataframe(matches)
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )

        repo = MatchRepository(self.session)

        feature_pipeline = None
        pipeline_path = None
        if model_meta and model_meta.hyperparameters:
            pipeline_path = model_meta.hyperparameters.get("feature_pipeline_path")

        if pipeline_path:
            pipeline_path = Path(pipeline_path)
            if pipeline_path.exists() and (pipeline_path / "config.json").exists():
                with contextlib.suppress(Exception):
                    feature_pipeline = FeaturePipeline.load(pipeline_path)

        if feature_pipeline is None or not feature_pipeline.is_fitted:
            raise PredictionError(
                "Could not load fitted feature pipeline. Backtest aborted to "
                "prevent preprocessing drift."
            )

        val_split = 0.2
        train_size = int(len(matches) * (1 - val_split))
        train_matches = matches_df.iloc[:train_size]
        test_matches = matches_df.iloc[train_size:]

        self.logger.debug(
            "Data split complete",
            extra={
                "operation": "run_backtest",
                "training_matches": len(train_matches),
                "validation_matches": len(test_matches),
            },
        )

        return feature_pipeline, repo, train_matches, test_matches

    def run(self, request: BacktestRequest) -> BacktestResponse:
        """Run a backtest on historical data.

        Args:
            request: Request with min_matches, validation_split, model_version

        Returns:
            BacktestResponse with metrics and statistics

        Raises:
            InsufficientDataError: If not enough matches
            NoActiveModelError: If no active model and no version specified
            ModelNotFoundError: If specified model version not found
            PredictionError: If prediction fails
        """
        self.logger.info(
            "Starting backtest",
            extra={
                "operation": "run_backtest",
                "min_matches": request.min_matches,
                "validation_split": request.validation_split,
                "model_version": request.model_version,
            },
        )

        start_time = time.time()

        try:
            model, model_meta = self._get_model(request.model_version)
            version = model_meta.version if model_meta else "unknown"

            self.logger.debug(
                "Model loaded for backtest",
                extra={"model_version": version},
            )

            matches, total_matches = self._load_matches(request)

            feature_pipeline, repo, train_matches, test_matches = (
                self._prepare_pipeline_and_data(matches, model_meta)
            )

            training_matches = len(train_matches)
            validation_matches = len(test_matches)

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

            from algobet.predictions.evaluation import evaluate_predictions

            # Single odds snapshot per match → treat as closing and use
            # model-CLV (see metrics.calculate_betting_metrics docstring).
            result = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                odds=odds,
                model_version=version,
                date_range=date_range,
                use_model_clv=True,
            )

            execution_time = time.time() - start_time

            metrics = {
                "accuracy": result.classification.accuracy,
                "log_loss": result.classification.log_loss,
                "brier_score": result.classification.brier_score,
                "home_f1": result.classification.per_class_f1.get("H", 0.0),
                "draw_f1": result.classification.per_class_f1.get("D", 0.0),
                "away_f1": result.classification.per_class_f1.get("A", 0.0),
            }

            if result.betting:
                metrics["roi_percent"] = result.betting.roi_percent
                metrics["win_rate"] = result.betting.win_rate

            self.logger.info(
                "Backtest completed successfully",
                extra={
                    "operation": "run_backtest",
                    "model_version": version,
                    "total_matches": total_matches,
                    "accuracy": metrics.get("accuracy"),
                    "execution_time_seconds": execution_time,
                },
            )

            return BacktestResponse(
                success=True,
                total_matches=total_matches,
                training_matches=training_matches,
                validation_matches=validation_matches,
                metrics=metrics,
                model_version=version,
                execution_time_seconds=execution_time,
            )

        except InsufficientDataError:
            raise
        except (NoActiveModelError, ModelNotFoundError):
            raise
        except Exception as e:
            self.logger.error(
                "Backtest failed",
                extra={
                    "operation": "run_backtest",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise PredictionError(
                f"Backtest failed: {e}",
                details={"error_type": type(e).__name__},
            ) from e
