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

    Uses the full FeaturePipeline when a fitted pipeline is available on disk,
    otherwise falls back to the basic FormCalculator with 6 features.
    """

    def __init__(
        self,
        session: Session,
        models_path: Path = Path("data/models"),
        pipelines_path: Path = Path("data/pipelines"),
    ) -> None:
        """Initialize the prediction service.

        Args:
            session: SQLAlchemy database session
            models_path: Path to model storage directory
            pipelines_path: Path to pipeline storage directory
        """
        super().__init__(session)
        self.registry = ModelRegistry(storage_path=models_path, session=session)
        self.repo = MatchRepository(session)
        self.calc = FormCalculator(self.repo)
        self.pipelines_path = Path(pipelines_path)
        self._feature_pipeline: FeaturePipeline | None = None

    def _load_feature_pipeline(
        self,
        schema_version: str = "v1.0",
    ) -> FeaturePipeline | None:
        """Load a fitted FeaturePipeline from disk if available.

        Args:
            schema_version: Feature schema version to load

        Returns:
            Loaded FeaturePipeline or None if not available
        """
        if self._feature_pipeline is not None:
            return self._feature_pipeline

        pipeline_dir = self.pipelines_path / schema_version
        if pipeline_dir.exists() and (pipeline_dir / "config.json").exists():
            try:
                self._feature_pipeline = FeaturePipeline.load(pipeline_dir)
                logger.info(
                    "Loaded FeaturePipeline from %s (%d features)",
                    pipeline_dir,
                    len(self._feature_pipeline.feature_names),
                )
                return self._feature_pipeline
            except Exception as e:
                logger.warning(
                    "Failed to load FeaturePipeline from %s: %s. "
                    "Falling back to FormCalculator.",
                    pipeline_dir,
                    e,
                )
        return None

    def load_model(self, model_version: str | None = None) -> tuple[Any, str]:
        """Load model from registry.

        Args:
            model_version: Optional specific version ID

        Returns:
            Tuple of (model object, version string)
        """
        if model_version:
            model = self.registry.load_model(model_version)
            return model, model_version
        else:
            model, metadata = self.registry.get_active_model()
            return model, metadata.version

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
        """Generate ML features using the full FeaturePipeline.

        Produces ~50+ features including odds, temporal, tournament context,
        H2H, and multi-window form features. Returns None if no fitted
        pipeline is available, in which case callers should fall back to
        generate_features().

        Args:
            match: Match object to generate features for

        Returns:
            Feature array (1, n_features) or None if pipeline unavailable
        """
        pipeline = self._load_feature_pipeline()
        if pipeline is None or not pipeline.is_fitted:
            return None

        match_df = prepare_match_dataframe([match])
        try:
            features = pipeline.transform(match_df, self.repo)
            return features
        except Exception as e:
            logger.warning(
                "FeaturePipeline transform failed for match %d: %s. "
                "Falling back to FormCalculator.",
                match.id,
                e,
            )
            return None

    def get_prediction(
        self, model: Any, features: dict[str, float] | np.ndarray
    ) -> tuple[str, float, dict[str, float]]:
        """Get prediction from model.

        Args:
            model: Loaded model object
            features: Feature dictionary or numpy array

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

        outcomes = ["HOME", "DRAW", "AWAY"]
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        probabilities = {
            "home": float(probs[0]),
            "draw": float(probs[1]),
            "away": float(probs[2]),
        }

        return outcomes[max_idx], confidence, probabilities

    def predict_match(
        self,
        match: Match,
        model_version: str | None = None,
    ) -> PredictionResult:
        """Generate prediction for a single match.

        Attempts to use the full FeaturePipeline for richer predictions.
        Falls back to FormCalculator if the pipeline is not available.

        Args:
            match: Match object to predict
            model_version: Optional specific model version

        Returns:
            PredictionResult with prediction details
        """
        model, version = self.load_model(model_version)

        # Try full FeaturePipeline first, fall back to FormCalculator
        v2_features = self.generate_features_v2(match)
        if v2_features is not None:
            features: dict[str, float] | np.ndarray = v2_features
        else:
            features = self.generate_features(match)

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
