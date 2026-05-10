"""Calibration service for improving model probability estimates."""

from pathlib import Path
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Session

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
from algobet.services.dto import CalibrateRequest, CalibrateResponse


class CalibrationService(BaseService[Session]):
    """Service for calibrating model probabilities."""

    def __init__(
        self, session: Session, models_path: Path = Path("data/models")
    ) -> None:
        """Initialize the calibration service.

        Args:
            session: SQLAlchemy database session
            models_path: Path to model storage directory
        """
        super().__init__(session)
        self.logger = get_logger("services.calibration")
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

    def _load_calibration_matches(self, limit: int = 2000) -> list[Match]:
        """Load historical matches for calibration."""
        return (
            self.session.query(Match)
            .filter(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None),
                )
            )
            .order_by(Match.match_date.desc())
            .limit(limit)
            .all()
        )

    def _split_for_calibration(self, matches_df: Any) -> tuple[Any, Any]:
        """Split data for calibration training and validation."""
        val_size = int(len(matches_df) * 0.2)
        train_df = matches_df.iloc[:-val_size] if val_size > 0 else matches_df
        val_df = matches_df.iloc[-val_size:] if val_size > 0 else matches_df
        return train_df, val_df

    def run(self, request: CalibrateRequest) -> CalibrateResponse:
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
            model, model_meta = self._get_model(request.model_version)
            version = model_meta.version if model_meta else "unknown"

            self.logger.debug(
                "Model loaded for calibration",
                extra={"model_version": version},
            )

            matches = self._load_calibration_matches()

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

            train_df, val_df = self._split_for_calibration(matches_df)

            self.logger.debug(
                "Data split for calibration",
                extra={
                    "operation": "calibrate_model",
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                },
            )

            feature_pipeline.fit(train_df, repo)
            X_val = feature_pipeline.transform(val_df, repo)

            result_map = {"H": 0, "D": 1, "A": 2}
            y_val = val_df["result"].map(result_map).values

            y_proba = model.predict_proba(X_val)

            from algobet.predictions.training.calibration import (
                CalibratedPredictor,
                ProbabilityCalibrator,
                calculate_calibration_metrics,
            )

            raw_metrics = calculate_calibration_metrics(y_val, y_proba)

            calibrator = ProbabilityCalibrator(method=request.method, n_classes=3)
            calibrator.fit(y_proba, y_val)

            calibrated_proba = calibrator.calibrate(y_proba)
            cal_metrics = calculate_calibration_metrics(y_val, calibrated_proba)

            before_score = raw_metrics["brier_score"]
            after_score = cal_metrics["brier_score"]
            improvement = before_score - after_score

            self.logger.info(
                "Calibration completed",
                extra={
                    "operation": "calibrate_model",
                    "before_brier": before_score,
                    "after_brier": after_score,
                    "improvement": improvement,
                },
            )

            registry = ModelRegistry(
                storage_path=self.models_path, session=self.session
            )

            calibrated_model = CalibratedPredictor(
                predictor=model, calibrator=calibrator
            )

            metrics: dict[str, float] = {
                "calibration_method": 1.0 if request.method == "isotonic" else 0.0,
                "calibration_ece": float(cal_metrics["expected_calibration_error"]),
                "calibration_brier": float(cal_metrics["brier_score"]),
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
