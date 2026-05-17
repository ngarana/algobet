"""Calibration use case for ML operations."""

from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import and_
from sqlalchemy.orm import Session

from algobet.api.schemas.ml_operations import (
    CalibrateRequest,
    CalibrateResultResponse,
    CalibrationMetricsResponse,
)
from algobet.models import Match
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.calibration import (
    ProbabilityCalibrator,
    calculate_calibration_metrics,
)


class CalibrationRunner:
    """Run this ML operation use case."""

    @staticmethod
    def _load_saved_feature_pipeline(model_meta: object | None) -> FeaturePipeline:
        """Load the fitted feature pipeline saved with a model."""
        pipeline_path = None
        hyperparameters = getattr(model_meta, "hyperparameters", None) or {}
        if hyperparameters:
            pipeline_path = hyperparameters.get("feature_pipeline_path")

        candidate_paths = []
        if pipeline_path:
            candidate_paths.append(Path(pipeline_path))

        artifact_path = getattr(model_meta, "artifact_path", None)
        if artifact_path:
            candidate_paths.append(Path(artifact_path).parent / "feature_pipeline")

        for candidate in candidate_paths:
            if candidate.exists() and (candidate / "config.json").exists():
                feature_pipeline = FeaturePipeline.load(candidate)
                if feature_pipeline.is_fitted:
                    return feature_pipeline

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load fitted feature pipeline. Calibration aborted to "
                "prevent preprocessing drift."
            ),
        )

    def run_calibrate(
        self, request: CalibrateRequest, db: Session
    ) -> CalibrateResultResponse:
        """Calibrate model probabilities.

        Fits a calibrator to improve probability estimates for
        better value betting accuracy.

        Args:
            request: Calibration parameters

        Returns:
            CalibrateResultResponse with before/after metrics

        Raises:
            HTTPException: If model not found or insufficient data
        """
        from algobet.predictions.training.calibration import CalibratedPredictor

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
                base_version = request.model_version
            else:
                model, model_meta = registry.get_active_model()
                base_version = model_meta.version if model_meta else "unknown"
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {e}",
            ) from None

        # Get historical matches for calibration
        matches = (
            db.query(Match)
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
            raise HTTPException(
                status_code=400,
                detail="Insufficient historical matches for calibration (< 100)",
            )

        # Prepare data with the saved fitted pipeline. Do not refit a default
        # pipeline here: that changes selected features and transformer state.
        repo = MatchRepository(db)
        feature_pipeline = self._load_saved_feature_pipeline(model_meta)
        model_hyperparameters = getattr(model_meta, "hyperparameters", None) or {}
        artifact_path = getattr(model_meta, "artifact_path", None)
        saved_pipeline_path = model_hyperparameters.get("feature_pipeline_path")
        if not saved_pipeline_path and artifact_path:
            saved_pipeline_path = str(Path(artifact_path).parent / "feature_pipeline")

        matches_df = prepare_match_dataframe(matches)
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )

        # Sort by date for temporal split
        matches_df = matches_df.sort_values("match_date")

        # Split into train/val for calibration
        val_size = int(len(matches_df) * request.validation_split)
        val_df = matches_df.iloc[-val_size:]

        team_ids = list(
            set(
                matches_df["home_team_id"].tolist()
                + matches_df["away_team_id"].tolist()
            )
        )
        max_match_date = matches_df["match_date"].max()
        repo.preload_team_matches(team_ids, before_date=max_match_date)
        repo.preload_h2h_matches(
            list(
                zip(
                    matches_df["home_team_id"].tolist(),
                    matches_df["away_team_id"].tolist(),
                    strict=False,
                )
            ),
            before_date=max_match_date,
        )
        repo.preload_season_standings(
            list(
                set(
                    zip(
                        matches_df["tournament_id"].tolist(),
                        matches_df["season_id"].tolist(),
                        strict=False,
                    )
                )
            ),
            before_date=max_match_date,
        )

        # Generate features
        X_val = feature_pipeline.transform(val_df, repo)

        # Encode targets
        result_map = {"H": 0, "D": 1, "A": 2}
        y_val = val_df["result"].map(result_map).values

        # Get raw predictions
        y_proba = model.predict_proba(X_val)

        # Calculate before metrics
        before_metrics = calculate_calibration_metrics(y_val, y_proba)

        # Fit calibrator
        calibrator = ProbabilityCalibrator(method=request.method, n_classes=3)
        calibrator.fit(y_proba, y_val)

        # Calculate after metrics
        calibrated_proba = calibrator.calibrate(y_proba)
        after_metrics = calculate_calibration_metrics(y_val, calibrated_proba)

        # Calculate improvement
        improvement = {
            "ece_improvement": before_metrics["expected_calibration_error"]
            - after_metrics["expected_calibration_error"],
            "brier_improvement": before_metrics["brier_score"]
            - after_metrics["brier_score"],
            "log_loss_improvement": before_metrics["log_loss"]
            - after_metrics["log_loss"],
        }

        # Save calibrated model
        calibrated_model = CalibratedPredictor(predictor=model, calibrator=calibrator)

        metrics: dict[str, float] = {
            "calibration_ece": float(after_metrics["expected_calibration_error"]),
            "calibration_brier": float(after_metrics["brier_score"]),
        }

        new_version = registry.save_model(
            model=calibrated_model,
            name="match_predictor_calibrated",
            metrics=metrics,
            model_type="calibrated",
            hyperparameters={
                "base_model_version": base_version,
                "calibration_method": request.method,
                "feature_names": feature_pipeline.feature_names,
                "feature_pipeline_path": str(saved_pipeline_path),
            },
            description=f"Calibrated version of {base_version} using {request.method}",
        )

        # Activate if requested
        is_active = False
        if request.activate:
            registry.activate_model(new_version)
            is_active = True

        return CalibrateResultResponse(
            base_model_version=base_version,
            calibrated_model_version=new_version,
            method=request.method,
            samples_used=len(val_df),
            before_metrics=CalibrationMetricsResponse(
                expected_calibration_error=before_metrics["expected_calibration_error"],
                maximum_calibration_error=before_metrics["maximum_calibration_error"],
                brier_score=before_metrics["brier_score"],
                log_loss=before_metrics["log_loss"],
            ),
            after_metrics=CalibrationMetricsResponse(
                expected_calibration_error=after_metrics["expected_calibration_error"],
                maximum_calibration_error=after_metrics["maximum_calibration_error"],
                brier_score=after_metrics["brier_score"],
                log_loss=after_metrics["log_loss"],
            ),
            improvement=improvement,
            is_active=is_active,
        )
