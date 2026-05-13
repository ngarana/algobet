"""Unified prediction service for generating match predictions."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import Match, ModelVersion, Prediction, Tournament
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.form_features import FormCalculator
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.pipeline import MODEL_FEATURE_SCHEMA_VERSION
from algobet.services.base import BaseService

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a match prediction."""

    match_id: int
    match_date: datetime
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    model_version: str
    prob_home: float
    prob_draw: float
    prob_away: float


class PredictionService(BaseService[Any]):
    """Service for generating and managing predictions.

    Loads the exact FeaturePipeline saved with a retrained model so prediction
    generation uses the same odds-free feature schema as training.
    """

    def __init__(
        self,
        session: Session,
        models_path: Path = Path("data/models"),
        pipelines_path: Path = Path("data/pipelines"),
        dc_model_path: Path | None = None,
    ) -> None:
        """Initialize the prediction service.

        Args:
            session: SQLAlchemy database session
            models_path: Path to model storage directory
            pipelines_path: Path to pipeline storage directory
            dc_model_path: Optional path to Dixon-Coles model for blending
        """
        super().__init__(session)
        self.registry = ModelRegistry(storage_path=models_path, session=session)
        self.repo = MatchRepository(session)
        self.calc = FormCalculator(self.repo)
        self.pipelines_path = Path(pipelines_path)
        self.dc_model_path = dc_model_path
        self._feature_pipeline: FeaturePipeline | None = None
        self._feature_pipeline_path: Path | None = None
        self._dc_model: Any = None
        self._draw_aware_calibrator: Any = None

    def _load_feature_pipeline(
        self,
        pipeline_path: Path,
    ) -> FeaturePipeline | None:
        """Load the fitted FeaturePipeline stored with a specific model.

        Args:
            pipeline_path: Stored pipeline directory for the model

        Returns:
            Loaded FeaturePipeline or None if not available
        """
        resolved_path = pipeline_path.resolve()
        if (
            self._feature_pipeline is not None
            and self._feature_pipeline_path == resolved_path
        ):
            return self._feature_pipeline

        if resolved_path.exists() and (resolved_path / "config.json").exists():
            try:
                self._feature_pipeline = FeaturePipeline.load(resolved_path)
                self._feature_pipeline_path = resolved_path
                logger.info(
                    "Loaded FeaturePipeline from %s (%d features)",
                    resolved_path,
                    len(self._feature_pipeline.feature_names),
                )
                return self._feature_pipeline
            except Exception as e:
                logger.warning(
                    "Failed to load FeaturePipeline from %s: %s. "
                    "Prediction generation requires retraining the model.",
                    resolved_path,
                    e,
                )
        return None

    def _load_dc_model(self) -> Any:
        """Load Dixon-Coles model for blending (lazy, cached)."""
        if self._dc_model is not None:
            return self._dc_model

        if self.dc_model_path is None:
            return None

        try:
            from algobet.predictions.training.classifiers import DixonColesPredictor

            self._dc_model = DixonColesPredictor.load(self.dc_model_path)
            logger.info("Loaded Dixon-Coles model from %s", self.dc_model_path)
            return self._dc_model
        except Exception as e:
            logger.warning("Failed to load Dixon-Coles model: %s", e)
            return None

    def _load_draw_aware_calibrator(self, model_dir: Path) -> Any:
        """Load DrawAwareCalibrator if available."""
        if self._draw_aware_calibrator is not None:
            return self._draw_aware_calibrator

        calibrator_path = model_dir / "draw_aware_calibrator.joblib"
        if not calibrator_path.exists():
            return None

        try:
            from algobet.predictions.training.calibration import DrawAwareCalibrator

            self._draw_aware_calibrator = DrawAwareCalibrator.load(calibrator_path)
            logger.info("Loaded DrawAwareCalibrator from %s", calibrator_path)
            return self._draw_aware_calibrator
        except Exception as e:
            logger.warning("Failed to load DrawAwareCalibrator: %s", e)
            return None

    def load_model(self, model_version: str | None = None) -> tuple[Any, str]:
        """Load model from registry.

        Args:
            model_version: Optional specific version ID

        Returns:
            Tuple of (model object, version string)
        """
        if model_version:
            stmt = select(ModelVersion).where(ModelVersion.version == model_version)
        else:
            stmt = select(ModelVersion).where(ModelVersion.is_active)

        db_model = self.session.execute(stmt).scalar_one_or_none()
        if db_model is None:
            if model_version:
                raise ValueError(f"Model version {model_version} not found")
            raise ValueError("No active model set in registry")

        if db_model.feature_schema_version != MODEL_FEATURE_SCHEMA_VERSION:
            actual_schema = db_model.feature_schema_version or "unknown"
            raise ValueError(
                f"Model {db_model.version} uses feature schema {actual_schema}. "
                f"Retrain it under {MODEL_FEATURE_SCHEMA_VERSION} before "
                "generating predictions."
            )

        hyperparameters = db_model.hyperparameters or {}
        pipeline_path_raw = hyperparameters.get("feature_pipeline_path")
        if not isinstance(pipeline_path_raw, str) or not pipeline_path_raw:
            raise ValueError(
                f"Model {db_model.version} is missing its saved feature pipeline. "
                f"Retrain it under {MODEL_FEATURE_SCHEMA_VERSION} before "
                "generating predictions."
            )

        pipeline_path = Path(pipeline_path_raw)
        pipeline = self._load_feature_pipeline(pipeline_path)
        if pipeline is None or not pipeline.is_fitted:
            raise ValueError(
                f"Model {db_model.version} does not have a usable saved feature "
                "pipeline. Retrain it before generating predictions."
            )

        if model_version:
            model = self.registry.load_model(model_version)
            return model, model_version

        model = self.registry.load_model(db_model.version)
        return model, db_model.version

    def generate_features(self, match: Match) -> dict[str, float]:
        """Generate ML features for a match using FormCalculator (basic, 6 features).

        This is the fallback method. Prefer generate_features_v2() when a
        fitted FeaturePipeline is available for richer feature generation.

        Args:
            match: Match object to generate features for

        Returns:
            Dictionary of feature name to value
        """
        return {
            "home_form": self.calc.calculate_recent_form(
                team_id=match.home_team_id, reference_date=match.match_date, n_matches=5
            ),
            "away_form": self.calc.calculate_recent_form(
                team_id=match.away_team_id, reference_date=match.match_date, n_matches=5
            ),
            "home_goals_scored": self.calc.calculate_goals_scored(
                team_id=match.home_team_id, reference_date=match.match_date, n_matches=5
            ),
            "away_goals_scored": self.calc.calculate_goals_scored(
                team_id=match.away_team_id, reference_date=match.match_date, n_matches=5
            ),
            "home_goals_conceded": self.calc.calculate_goals_conceded(
                team_id=match.home_team_id, reference_date=match.match_date, n_matches=5
            ),
            "away_goals_conceded": self.calc.calculate_goals_conceded(
                team_id=match.away_team_id, reference_date=match.match_date, n_matches=5
            ),
        }

    def generate_features_v2(self, match: Match) -> np.ndarray | None:
        """Generate ML features using the exact pipeline saved with the model.

        Args:
            match: Match object to generate features for

        Returns:
            Feature array (1, n_features) or None if no model pipeline is loaded
        """
        pipeline = self._feature_pipeline
        if pipeline is None or not pipeline.is_fitted:
            return None

        match_df = prepare_match_dataframe([match])
        try:
            features = pipeline.transform(match_df, self.repo)
            return features
        except Exception as e:
            logger.error(
                "FeaturePipeline transform failed for match %d: %s.",
                match.id,
                e,
            )
            raise ValueError(
                f"Saved feature pipeline failed for match {match.id}. "
                "Retrain the model before generating predictions."
            ) from e

    def get_prediction(
        self,
        model: Any,
        features: dict[str, float] | np.ndarray,
        draw_boost: float = 1.0,
        blend_factor: float | None = None,
        dc_probas: np.ndarray | None = None,
    ) -> tuple[str, float, dict[str, float]]:
        """Get prediction from model.

        Args:
            model: Loaded model object
            features: Feature dictionary or numpy array
            draw_boost: Draw prob multiplier (1.0 = no change, 1.5 = 50% boost)
            blend_factor: Manual blend weight for Dixon-Coles (overrides fitted)
            dc_probas: Dixon-Coles probabilities for blending

        Returns:
            Tuple of (predicted_outcome, confidence, probabilities)
        """
        if isinstance(features, dict):
            feature_array = np.array([list(features.values())])
        else:
            feature_array = features if features.ndim == 2 else features.reshape(1, -1)

        try:
            probs = model.predict_proba(feature_array)[0]
        except AttributeError:
            probs = model.predict(feature_array)[0]

        # Apply Dixon-Coles blend if available
        if dc_probas is not None:
            dc_probs_single = dc_probas[0] if dc_probas.ndim == 2 else dc_probas

            if blend_factor is not None:
                # Manual blend factor
                probs = (1 - blend_factor) * probs + blend_factor * dc_probs_single
            elif self._draw_aware_calibrator is not None:
                # Use fitted calibrator
                probs = self._draw_aware_calibrator.calibrate(
                    probs.reshape(1, -1), dc_probs_single.reshape(1, -1)
                )[0]

            # Renormalize
            probs = probs / probs.sum()
        elif draw_boost != 1.0:
            # Fallback to draw boost if no DC blend
            probs = self._apply_draw_boost(probs, draw_boost)

        outcomes = ["HOME", "DRAW", "AWAY"]
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        probabilities = {
            "home": float(probs[0]),
            "draw": float(probs[1]),
            "away": float(probs[2]),
        }

        return outcomes[max_idx], confidence, probabilities

    def _apply_draw_boost(self, probs: np.ndarray, boost_factor: float) -> np.ndarray:
        """Apply draw boost calibration to probabilities.

        Args:
            probs: Raw probabilities [home, draw, away]
            boost_factor: Multiplier for draw probability

        Returns:
            Calibrated probabilities with boosted draws
        """
        calibrated = probs.copy()
        calibrated[1] = calibrated[1] * boost_factor  # Boost draw column

        # Renormalize so probabilities sum to 1
        row_sum = calibrated.sum()
        if row_sum > 0:
            calibrated = calibrated / row_sum

        return calibrated

    def predict_match(
        self,
        match: Match,
        model_version: str | None = None,
    ) -> PredictionResult:
        """Generate prediction for a single match.

        Uses the exact saved FeaturePipeline for the selected model.

        Args:
            match: Match object to predict
            model_version: Optional specific model version

        Returns:
            PredictionResult with prediction details
        """
        model, version = self.load_model(model_version)

        v2_features = self.generate_features_v2(match)
        if v2_features is None:
            raise ValueError(
                f"Model {version} is missing its saved feature pipeline. "
                "Retrain it before generating predictions."
            )
        features: dict[str, float] | np.ndarray = v2_features

        outcome, confidence, probs = self.get_prediction(model, features)

        return PredictionResult(
            match_id=match.id,
            match_date=match.match_date,
            home_team=match.home_team.name,
            away_team=match.away_team.name,
            predicted_outcome=outcome,
            confidence=confidence,
            model_version=version,
            prob_home=probs["home"],
            prob_draw=probs["draw"],
            prob_away=probs["away"],
        )

    def query_matches(
        self,
        match_ids: list[int] | None = None,
        tournament_name: str | None = None,
        days_ahead: int = 7,
        status: str = "SCHEDULED",
    ) -> list[Match]:
        """Query matches based on filters.

        Args:
            match_ids: Optional list of specific match IDs
            tournament_name: Optional tournament name filter
            days_ahead: Number of days ahead to look
            status: Match status filter

        Returns:
            List of Match objects
        """
        if match_ids:
            stmt = select(Match).where(Match.id.in_(match_ids))
        else:
            stmt = select(Match).where(Match.status == status)
            max_date = datetime.now() + timedelta(days=days_ahead)
            stmt = stmt.where(Match.match_date <= max_date)

            if tournament_name:
                stmt = stmt.join(Tournament).where(Tournament.name == tournament_name)

        stmt = stmt.order_by(Match.match_date)
        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def predict_upcoming(
        self,
        days_ahead: int = 7,
        tournament_name: str | None = None,
        min_confidence: float = 0.0,
        model_version: str | None = None,
    ) -> list[PredictionResult]:
        """Generate predictions for upcoming matches.

        Args:
            days_ahead: Number of days ahead to look for matches
            tournament_name: Optional tournament filter
            min_confidence: Minimum confidence threshold
            model_version: Optional specific model version

        Returns:
            List of PredictionResult objects
        """
        matches = self.query_matches(
            tournament_name=tournament_name,
            days_ahead=days_ahead,
            status="SCHEDULED",
        )

        predictions = []
        for match in matches:
            result = self.predict_match(match, model_version)
            if result.confidence >= min_confidence:
                predictions.append(result)

        return predictions

    def save_predictions(self, predictions: list[PredictionResult]) -> list[Prediction]:
        """Save predictions to database.

        Args:
            predictions: List of PredictionResult to save

        Returns:
            List of created Prediction ORM objects
        """
        if not predictions:
            return []

        # Get model version ID
        model_version = self.session.execute(
            select(ModelVersion).where(
                ModelVersion.version == predictions[0].model_version
            )
        ).scalar_one()

        db_predictions = []
        for pred in predictions:
            db_pred = Prediction(
                match_id=pred.match_id,
                model_version_id=model_version.id,
                prob_home=pred.prob_home,
                prob_draw=pred.prob_draw,
                prob_away=pred.prob_away,
                predicted_outcome=pred.predicted_outcome[0],  # H, D, or A
                confidence=pred.confidence,
            )
            self.session.add(db_pred)
            db_predictions.append(db_pred)

        return db_predictions
