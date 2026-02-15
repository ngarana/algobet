"""Analysis service for backtest, value bets, and calibration operations.

This module provides business logic for analysis operations, extracting
functionality from the CLI layer into a reusable service layer.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
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
from algobet.services.base import BaseService
from algobet.services.dto import (
    BacktestRequest,
    BacktestResponse,
    CalibrateRequest,
    CalibrateResponse,
    ValueBetInfo,
    ValueBetsRequest,
    ValueBetsResponse,
)


class AnalysisService(BaseService[Session]):
    """Service for analysis operations: backtest, value bets, calibration.

    This service provides business logic for:
    - Running historical backtests on model predictions
    - Finding value betting opportunities
    - Calibrating model probabilities

    Attributes:
        session: SQLAlchemy database session for queries
        logger: Logger instance for this service
        models_path: Path to model storage directory
    """

    def __init__(
        self, session: Session, models_path: Path = Path("data/models")
    ) -> None:
        """Initialize the service with a database session.

        Args:
            session: SQLAlchemy database session
            models_path: Path to model storage directory
        """
        super().__init__(session)
        self.logger = get_logger("services.analysis")
        self.models_path = models_path

    def _get_model(
        self, model_version: str | None
    ) -> tuple[Any, ModelVersion | None]:
        """Load model from registry.

        Args:
            model_version: Optional specific version ID

        Returns:
            Tuple of (model object, ModelVersion metadata)

        Raises:
            NoActiveModelError: If no active model and no version specified
            ModelNotFoundError: If specified model version not found
        """
        from algobet.predictions.models.registry import ModelRegistry

        registry = ModelRegistry(storage_path=self.models_path, session=self.session)

        try:
            if model_version:
                model = registry.load_model(model_version)
                # Get metadata
                model_meta = None
                for m in registry.list_models():
                    if m.version == model_version:
                        model_meta = self.session.query(ModelVersion).filter(
                            ModelVersion.version == model_version
                        ).first()
                        break
                return model, model_meta
            else:
                model, metadata = registry.get_active_model()
                # Get the database record
                model_meta = self.session.query(ModelVersion).filter(
                    ModelVersion.version == metadata.version
                ).first()
                return model, model_meta
        except ValueError as e:
            if model_version:
                raise ModelNotFoundError(
                    f"Model version '{model_version}' not found.",
                    details={"version": model_version},
                ) from e
            raise NoActiveModelError(
                details={"error": str(e)}
            ) from e
        except FileNotFoundError as e:
            if model_version:
                raise ModelNotFoundError(
                    f"Model version '{model_version}' not found.",
                    details={"version": model_version},
                ) from e
            raise NoActiveModelError(
                details={"error": str(e)}
            ) from e

    def run_backtest(self, request: BacktestRequest) -> BacktestResponse:
        """Run a backtest on historical data.

        Evaluates model performance on historical match data by:
        1. Loading finished matches with odds
        2. Splitting into training/validation sets
        3. Generating features and predictions
        4. Computing evaluation metrics

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
            # Load model
            model, model_meta = self._get_model(request.model_version)
            version = model_meta.version if model_meta else "unknown"

            self.logger.debug(
                "Model loaded for backtest",
                extra={"model_version": version},
            )

            # Get historical matches with results
            matches = (
                self.session.query(Match)
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

            total_matches = len(matches)

            # Check minimum matches requirement
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

            # Prepare data
            matches_df = prepare_match_dataframe(matches)
            matches_df["result"] = matches_df.apply(
                lambda m: "H"
                if m["home_score"] > m["away_score"]
                else ("A" if m["home_score"] < m["away_score"] else "D"),
                axis=1,
            )

            # Generate features
            repo = MatchRepository(self.session)
            feature_pipeline = FeaturePipeline.create_default()

            # Split data (temporal split - train on first portion)
            train_size = int(len(matches) * (1 - request.validation_split))
            train_matches = matches_df.iloc[:train_size]
            test_matches = matches_df.iloc[train_size:]

            training_matches = len(train_matches)
            validation_matches = len(test_matches)

            self.logger.debug(
                "Data split complete",
                extra={
                    "operation": "run_backtest",
                    "training_matches": training_matches,
                    "validation_matches": validation_matches,
                },
            )

            # Fit pipeline on training data
            feature_pipeline.fit(train_matches, repo)
            X_test = feature_pipeline.transform(test_matches, repo)

            # Get odds for betting simulation
            odds = test_matches[["odds_home", "odds_draw", "odds_away"]].values

            # Get predictions
            y_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_proba, axis=1)

            # Encode true labels
            result_map = {"H": 0, "D": 1, "A": 2}
            y_true = test_matches["result"].map(result_map).values

            # Calculate metrics
            from algobet.predictions.evaluation import evaluate_predictions

            date_range = (
                str(test_matches["match_date"].min().date()),
                str(test_matches["match_date"].max().date()),
            )

            result = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                odds=odds,
                model_version=version,
                date_range=date_range,
            )

            execution_time = time.time() - start_time

            # Build metrics dict from evaluation result
            metrics = {
                "accuracy": result.classification.accuracy,
                "log_loss": result.classification.log_loss,
                "brier_score": result.classification.brier_score,
                "home_accuracy": result.classification.home_accuracy,
                "draw_accuracy": result.classification.draw_accuracy,
                "away_accuracy": result.classification.away_accuracy,
            }

            if result.betting:
                metrics["roi"] = result.betting.roi
                metrics["betting_accuracy"] = result.betting.betting_accuracy

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

    def find_value_bets(self, request: ValueBetsRequest) -> ValueBetsResponse:
        """Find value bets based on model predictions.

        Identifies betting opportunities where the model's predicted probability
        exceeds the implied probability from market odds.

        Args:
            request: Request with min_edge, model_version, limit

        Returns:
            ValueBetsResponse with list of value bets

        Raises:
            NoActiveModelError: If no active model
            ModelNotFoundError: If specified model not found
            PredictionError: If prediction fails
            InsufficientDataError: If no upcoming matches with odds
        """
        self.logger.info(
            "Finding value bets",
            extra={
                "operation": "find_value_bets",
                "min_edge": request.min_edge,
                "model_version": request.model_version,
                "limit": request.limit,
            },
        )

        try:
            # Load model
            model, model_meta = self._get_model(request.model_version)
            version = model_meta.version if model_meta else "unknown"

            self.logger.debug(
                "Model loaded for value bets",
                extra={"model_version": version},
            )

            # Get upcoming matches with odds
            now = datetime.now()
            end_date = now + timedelta(days=7)

            matches = (
                self.session.query(Match)
                .options(
                    joinedload(Match.home_team),
                    joinedload(Match.away_team),
                    joinedload(Match.tournament),
                )
                .filter(
                    and_(
                        Match.status == "SCHEDULED",
                        Match.match_date >= now,
                        Match.match_date <= end_date,
                        Match.odds_home.is_not(None),
                        Match.odds_draw.is_not(None),
                        Match.odds_away.is_not(None),
                    )
                )
                .order_by(Match.match_date)
                .all()
            )

            if not matches:
                self.logger.info(
                    "No upcoming matches with odds found",
                    extra={"operation": "find_value_bets"},
                )
                # Return empty response instead of error for no matches
                return ValueBetsResponse(
                    value_bets=[],
                    model_version=version,
                    generated_at=datetime.now(),
                )

            self.logger.info(
                "Found upcoming matches for value bet analysis",
                extra={"operation": "find_value_bets", "match_count": len(matches)},
            )

            # Generate predictions
            repo = MatchRepository(self.session)
            feature_pipeline = FeaturePipeline.create_default()
            matches_df = prepare_match_dataframe(matches)

            # Fit pipeline on historical matches
            historical = (
                self.session.query(Match)
                .filter(
                    and_(
                        Match.status == "FINISHED",
                        Match.home_score.is_not(None),
                    )
                )
                .order_by(Match.match_date.desc())
                .limit(1000)
                .all()
            )

            if historical:
                hist_df = prepare_match_dataframe(historical)
                hist_df["result"] = hist_df.apply(
                    lambda m: "H"
                    if m["home_score"] > m["away_score"]
                    else ("A" if m["home_score"] < m["away_score"] else "D"),
                    axis=1,
                )
                feature_pipeline.fit(hist_df, repo)

            X = feature_pipeline.transform(matches_df, repo)
            y_proba = model.predict_proba(X)

            # Find value bets
            value_bets_list: list[ValueBetInfo] = []
            outcome_map = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}

            for i, match in enumerate(matches):
                match_odds = [match.odds_home, match.odds_draw, match.odds_away]
                probas = y_proba[i]

                for outcome_idx in range(3):
                    prob = probas[outcome_idx]
                    odds = match_odds[outcome_idx]

                    if odds is None or odds <= 1.0:
                        continue

                    # Calculate edge: model_prob - implied_prob
                    implied_prob = 1.0 / odds
                    edge = prob - implied_prob

                    if edge >= request.min_edge:
                        # Calculate expected value
                        expected_value = (prob * odds) - 1

                        value_bets_list.append(
                            ValueBetInfo(
                                match_id=match.id,
                                home_team=match.home_team.name,
                                away_team=match.away_team.name,
                                match_date=match.match_date,
                                bet_type=outcome_map[outcome_idx],
                                model_probability=prob,
                                market_odds=odds,
                                edge=edge,
                                expected_value=expected_value,
                            )
                        )

            # Sort by edge (descending) and limit
            value_bets_list.sort(key=lambda x: x.edge, reverse=True)
            value_bets_list = value_bets_list[: request.limit]

            self.logger.info(
                "Value bets found",
                extra={
                    "operation": "find_value_bets",
                    "value_bets_count": len(value_bets_list),
                    "model_version": version,
                },
            )

            return ValueBetsResponse(
                value_bets=value_bets_list,
                model_version=version,
                generated_at=datetime.now(),
            )

        except (NoActiveModelError, ModelNotFoundError):
            raise
        except Exception as e:
            self.logger.error(
                "Value bet finding failed",
                extra={
                    "operation": "find_value_bets",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise PredictionError(
                f"Value bet finding failed: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    def calibrate_model(self, request: CalibrateRequest) -> CalibrateResponse:
        """Calibrate model probabilities.

        Fits a calibrator (isotonic or sigmoid) to improve probability estimates.
        Uses historical match data for calibration.

        Args:
            request: Request with model_version, method

        Returns:
            CalibrateResponse with before/after scores

        Raises:
            ModelNotFoundError: If model not found
            NoActiveModelError: If no active model
            InsufficientDataError: If not enough data for calibration
            PredictionError: If calibration fails
        """
        self.logger.info(
            "Starting model calibration",
            extra={
                "operation": "calibrate_model",
                "model_version": request.model_version,
                "method": request.method,
            },
        )

        try:
            # Load model
            model, model_meta = self._get_model(request.model_version)
            version = model_meta.version if model_meta else "unknown"

            self.logger.debug(
                "Model loaded for calibration",
                extra={"model_version": version},
            )

            # Get historical matches
            matches = (
                self.session.query(Match)
                .filter(
                    and_(
                        Match.status == "FINISHED",
                        Match.home_score.is_not(None),
                        Match.away_score.is_not(None),
                    )
                )
                .order_by(Match.match_date.desc())
                .limit(2000)
                .all()
            )

            if len(matches) < 100:
                self.logger.warning(
                    "Insufficient matches for calibration",
                    extra={
                        "operation": "calibrate_model",
                        "match_count": len(matches),
                    },
                )
                raise InsufficientDataError(
                    f"Insufficient historical matches for calibration: "
                    f"{len(matches)} < 100 required.",
                    details={"match_count": len(matches)},
                )

            self.logger.info(
                "Found matches for calibration",
                extra={"operation": "calibrate_model", "match_count": len(matches)},
            )

            # Prepare data
            repo = MatchRepository(self.session)
            feature_pipeline = FeaturePipeline.create_default()

            matches_df = prepare_match_dataframe(matches)
            matches_df["result"] = matches_df.apply(
                lambda m: "H"
                if m["home_score"] > m["away_score"]
                else ("A" if m["home_score"] < m["away_score"] else "D"),
                axis=1,
            )
            matches_df = matches_df.sort_values("match_date")

            # Split data (use last 20% for validation)
            val_size = int(len(matches_df) * 0.2)
            train_df = matches_df.iloc[:-val_size] if val_size > 0 else matches_df
            val_df = matches_df.iloc[-val_size:] if val_size > 0 else matches_df

            self.logger.debug(
                "Data split for calibration",
                extra={
                    "operation": "calibrate_model",
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                },
            )

            # Fit pipeline and generate features
            feature_pipeline.fit(train_df, repo)
            X_val = feature_pipeline.transform(val_df, repo)

            # Encode targets
            result_map = {"H": 0, "D": 1, "A": 2}
            y_val = val_df["result"].map(result_map).values

            # Get predictions
            y_proba = model.predict_proba(X_val)

            # Calculate before calibration metrics
            from algobet.predictions.training.calibration import (
                CalibratedPredictor,
                ProbabilityCalibrator,
                calculate_calibration_metrics,
            )

            raw_metrics = calculate_calibration_metrics(y_val, y_proba)

            # Fit calibrator
            calibrator = ProbabilityCalibrator(method=request.method, n_classes=3)
            calibrator.fit(y_proba, y_val)

            # Get calibrated probabilities and metrics
            calibrated_proba = calibrator.calibrate(y_proba)
            cal_metrics = calculate_calibration_metrics(y_val, calibrated_proba)

            # Use Brier score as the main calibration metric
            before_score = raw_metrics["brier_score"]
            after_score = cal_metrics["brier_score"]
            improvement = before_score - after_score  # Lower is better for Brier

            self.logger.info(
                "Calibration completed",
                extra={
                    "operation": "calibrate_model",
                    "before_brier": before_score,
                    "after_brier": after_score,
                    "improvement": improvement,
                },
            )

            # Save calibrated model
            from algobet.predictions.models.registry import ModelRegistry

            registry = ModelRegistry(
                storage_path=self.models_path, session=self.session
            )

            calibrated_model = CalibratedPredictor(
                predictor=model, calibrator=calibrator
            )

            metrics = {
                "calibration_method": request.method,
                "calibration_ece": cal_metrics["expected_calibration_error"],
                "calibration_brier": cal_metrics["brier_score"],
                "base_model": version,
            }

            new_version = registry.save_model(
                model=calibrated_model,
                name="match_predictor_calibrated",
                metrics=metrics,
                model_type="calibrated",
                description=f"Calibrated version of {version} using {request.method}",
            )

            self.logger.info(
                "Calibrated model saved",
                extra={
                    "operation": "calibrate_model",
                    "new_version": new_version,
                },
            )

            return CalibrateResponse(
                success=True,
                model_version=new_version,
                calibration_method=request.method,
                before_calibration_score=before_score,
                after_calibration_score=after_score,
                improvement=improvement,
            )

        except (NoActiveModelError, ModelNotFoundError):
            raise
        except InsufficientDataError:
            raise
        except Exception as e:
            self.logger.error(
                "Calibration failed",
                extra={
                    "operation": "calibrate_model",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise PredictionError(
                f"Calibration failed: {e}",
                details={"error_type": type(e).__name__},
            ) from e
