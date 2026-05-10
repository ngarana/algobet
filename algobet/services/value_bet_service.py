"""Value bet service for finding betting opportunities."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

from algobet.exceptions import (
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
from algobet.services.dto import ValueBetInfo, ValueBetsRequest, ValueBetsResponse


class ValueBetService(BaseService[Session]):
    """Service for finding value betting opportunities."""

    def __init__(
        self, session: Session, models_path: Path = Path("data/models")
    ) -> None:
        """Initialize the value bet service.

        Args:
            session: SQLAlchemy database session
            models_path: Path to model storage directory
        """
        super().__init__(session)
        self.logger = get_logger("services.value_bet")
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

    def _load_upcoming_matches_with_odds(self, days_ahead: int = 7) -> list[Match]:
        """Load upcoming matches that have odds data."""
        now = datetime.now()
        end_date = now + timedelta(days=days_ahead)

        return (
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

    def _get_historical_for_pipeline(self, limit: int = 1000) -> list[Match]:
        """Load historical matches for fitting feature pipeline."""
        return (
            self.session.query(Match)
            .filter(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                )
            )
            .order_by(Match.match_date.desc())
            .limit(limit)
            .all()
        )

    def _compute_value_bets(
        self,
        matches: list[Match],
        y_proba: Any,
        min_edge: float,
        limit: int,
    ) -> list[ValueBetInfo]:
        """Compute value bets from matches and predictions."""
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

                implied_prob = 1.0 / odds
                edge = prob - implied_prob

                if edge >= min_edge:
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

        value_bets_list.sort(key=lambda x: x.edge, reverse=True)
        return value_bets_list[:limit]

    def run(self, request: ValueBetsRequest) -> ValueBetsResponse:
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
            model, model_meta = self._get_model(request.model_version)
            version = model_meta.version if model_meta else "unknown"

            self.logger.debug(
                "Model loaded for value bets",
                extra={"model_version": version},
            )

            matches = self._load_upcoming_matches_with_odds()

            if not matches:
                self.logger.info(
                    "No upcoming matches with odds found",
                    extra={"operation": "find_value_bets"},
                )
                return ValueBetsResponse(
                    value_bets=[],
                    model_version=version,
                    generated_at=datetime.now(),
                )

            self.logger.info(
                "Found upcoming matches for value bet analysis",
                extra={"operation": "find_value_bets", "match_count": len(matches)},
            )

            repo = MatchRepository(self.session)
            feature_pipeline = FeaturePipeline.create_default()
            matches_df = prepare_match_dataframe(matches)

            historical = self._get_historical_for_pipeline()

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

            value_bets_list = self._compute_value_bets(
                matches, y_proba, request.min_edge, request.limit
            )

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
