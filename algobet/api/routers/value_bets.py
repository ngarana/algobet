"""API router for value bets endpoints."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import ValueBetResponse
from algobet.models import Match, ModelVersion, Prediction
from algobet.predictions.models.registry import ModelRegistry
from algobet.user_workflow.service import build_match_response

router = APIRouter()


def calculate_expected_value(predicted_probability: float, market_odds: float) -> float:
    """Calculate expected value for a bet.

    EV = (predicted_probability * market_odds) - 1

    Args:
        predicted_probability: Model's predicted probability (0-1)
        market_odds: Market odds for the outcome

    Returns:
        Expected value (positive = value bet)
    """
    return (predicted_probability * market_odds) - 1


def calculate_kelly_fraction(predicted_probability: float, market_odds: float) -> float:
    """Calculate Kelly criterion recommended stake fraction.

    Kelly fraction = (p * (b) - (1 - p)) / b
    where:
    - p = predicted probability
    - b = odds - 1 (net odds received on a win)

    Args:
        predicted_probability: Model's predicted probability (0-1)
        market_odds: Market odds for the outcome

    Returns:
        Kelly fraction (recommend staking this fraction of bankroll)
        Returns 0 if negative (don't bet)
    """
    if market_odds <= 1:
        return 0.0

    b = market_odds - 1  # Net odds
    p = predicted_probability

    kelly = (p * b - (1 - p)) / b

    return max(0.0, kelly)  # Don't recommend negative Kelly (don't bet)


def build_value_bets(
    db: Session,
    min_ev: float = 0.05,
    max_odds: float = 10.0,
    days: int = 7,
    model_version: str | None = None,
    min_confidence: float | None = None,
    max_matches: int = 20,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[ValueBetResponse]:
    """Build value-bet responses for a date window."""
    # Get the model version to use
    if model_version:
        model_record = (
            db.query(ModelVersion).filter(ModelVersion.version == model_version).first()
        )
    else:
        # Get active model
        registry = ModelRegistry(storage_path=Path("data/models"), session=db)
        try:
            _, metadata = registry.get_active_model()
            model_record = (
                db.query(ModelVersion)
                .filter(ModelVersion.version == metadata.version)
                .first()
            )
        except (ValueError, FileNotFoundError):
            # No active model
            return []

    if not model_record:
        return []

    # Get date range for upcoming matches
    now = start_date or datetime.now(timezone.utc)
    end = end_date or now + timedelta(days=days)

    # Query predictions for upcoming matches with the specified model
    predictions_query = (
        db.query(Prediction)
        .join(Match)
        .filter(
            and_(
                Prediction.model_version_id == model_record.id,
                Match.match_date >= now,
                Match.match_date <= end,
                Match.status == "SCHEDULED",
                Match.odds_home.isnot(None),
                Match.odds_draw.isnot(None),
                Match.odds_away.isnot(None),
            )
        )
    )

    if min_confidence:
        predictions_query = predictions_query.filter(
            Prediction.confidence >= min_confidence
        )

    predictions = predictions_query.all()

    value_bets = []

    for pred in predictions:
        match = pred.match

        # Determine which outcome to evaluate based on predicted outcome
        if pred.predicted_outcome == "H":
            market_odds = match.odds_home
            predicted_prob = pred.prob_home
        elif pred.predicted_outcome == "D":
            market_odds = match.odds_draw
            predicted_prob = pred.prob_draw
        else:  # "A"
            market_odds = match.odds_away
            predicted_prob = pred.prob_away

        # Skip if odds not available or exceed max_odds
        if market_odds is None or market_odds > max_odds:
            continue

        # Calculate expected value
        ev = calculate_expected_value(predicted_prob, market_odds)

        # Only include if EV exceeds threshold
        if ev >= min_ev:
            kelly = calculate_kelly_fraction(predicted_prob, market_odds)

            value_bets.append(
                ValueBetResponse(
                    match=build_match_response(match),
                    prediction_id=pred.id,
                    predicted_outcome=pred.predicted_outcome,
                    predicted_probability=predicted_prob,
                    market_odds=market_odds,
                    expected_value=ev,
                    kelly_fraction=kelly,
                    confidence=pred.confidence,
                )
            )

    # Sort by expected value (descending) and return top matches
    value_bets.sort(key=lambda x: x.expected_value, reverse=True)
    return value_bets[:max_matches]


@router.get("", response_model=list[ValueBetResponse])
def get_value_bets(
    min_ev: float = Query(0.05, ge=0, description="Minimum expected value"),
    max_odds: float = Query(10.0, ge=1.0, description="Maximum odds"),
    days: int = Query(7, ge=1, le=30, description="Days ahead to look for value bets"),
    model_version: str | None = Query(None, description="Model version to use"),
    min_confidence: float | None = Query(
        None, ge=0, le=1, description="Minimum confidence"
    ),
    max_matches: int = Query(20, ge=1, le=100, description="Maximum number of matches"),
    db: Session = Depends(get_db),
) -> list[ValueBetResponse]:
    """Find value betting opportunities.

    Identifies matches where the model's predicted probability suggests
    positive expected value compared to market odds.

    Args:
        min_ev: Minimum expected value threshold (default: 0.05 = 5%)
        max_odds: Maximum odds to consider (default: 10.0)
        days: Days ahead to look for value bets (default: 7)
        model_version: Specific model version to use (default: active model)
        min_confidence: Minimum confidence for predictions
        max_matches: Maximum number of value bets to return

    Returns:
        List of value betting opportunities sorted by expected value
    """
    return build_value_bets(
        db,
        min_ev=min_ev,
        max_odds=max_odds,
        days=days,
        model_version=model_version,
        min_confidence=min_confidence,
        max_matches=max_matches,
    )
