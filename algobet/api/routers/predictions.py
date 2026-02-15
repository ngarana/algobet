"""API router for prediction endpoints."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import (
    MatchStatus,
    ModelVersionResponse,
    PaginatedResponse,
    PredictionResponse,
    PredictionWithMatchResponse,
)
from algobet.api.schemas.match import MatchResponse as MR
from algobet.models import Match, ModelVersion, Prediction
from algobet.predictions.models.registry import ModelRegistry

router = APIRouter()


class GeneratePredictionsRequest(BaseModel):
    """Request body for generating predictions."""

    match_ids: list[int] | None = None
    model_version: str | None = None
    tournament_id: int | None = None
    days_ahead: int | None = None


@router.get("", response_model=PaginatedResponse[PredictionResponse])
def list_predictions(
    match_id: int | None = Query(None, description="Filter by match ID"),
    model_version_id: int | None = Query(
        None, description="Filter by model version ID"
    ),
    has_result: bool | None = Query(
        None, description="Filter by whether result is known"
    ),
    from_date: datetime | None = Query(
        None, description="Filter predictions from this date"
    ),
    to_date: datetime | None = Query(
        None, description="Filter predictions until this date"
    ),
    min_confidence: float | None = Query(
        None, ge=0, le=1, description="Minimum confidence score"
    ),
    db: Session = Depends(get_db),
) -> list[PredictionResponse]:
    """List predictions with filtering.

    Returns predictions matching the specified filters.
    """
    query = db.query(Prediction)

    if match_id:
        query = query.filter(Prediction.match_id == match_id)

    if model_version_id:
        query = query.filter(Prediction.model_version_id == model_version_id)

    if has_result is not None:
        if has_result:
            # Has result means the match has finished scores
            query = query.join(Match).filter(
                and_(
                    Match.status == MatchStatus.FINISHED,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                )
            )
        else:
            # No result means match is scheduled or live
            query = query.join(Match).filter(
                Match.status.in_([MatchStatus.SCHEDULED, MatchStatus.LIVE])
            )

    if from_date:
        query = query.filter(Prediction.predicted_at >= from_date)

    if to_date:
        query = query.filter(Prediction.predicted_at <= to_date)

    if min_confidence:
        query = query.filter(Prediction.confidence >= min_confidence)

    total = query.count()
    predictions = query.order_by(Prediction.predicted_at.desc()).all()

    items = []
    for pred in predictions:
        items.append(
            PredictionResponse(
                id=pred.id,
                match_id=pred.match_id,
                model_version_id=pred.model_version_id,
                prob_home=pred.prob_home,
                prob_draw=pred.prob_draw,
                prob_away=pred.prob_away,
                predicted_outcome=pred.predicted_outcome,
                confidence=pred.confidence,
                predicted_at=pred.predicted_at,
                actual_roi=pred.actual_roi,
                max_probability=pred.max_probability,
            )
        )

    return PaginatedResponse(
        items=items,
        total=total,
        limit=50,  # Default limit if not specified
        offset=0,
    )


@router.post("/generate", response_model=dict[str, Any])
def generate_predictions(
    request: GeneratePredictionsRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Generate predictions for upcoming matches.

    Creates predictions for specified matches or all upcoming matches in a tournament.

    Args:
        request: Prediction generation request with match IDs and options

    Returns:
        Summary of generated predictions
    """
    # Get the model to use
    registry = ModelRegistry(storage_path=Path("data/models"), session=db)

    try:
        if request.model_version:
            # Use specified model version
            model = registry.load_model(request.model_version)
            model_version_record = (
                db.query(ModelVersion)
                .filter(ModelVersion.version == request.model_version)
                .first()
            )
        else:
            # Use active model
            model, metadata = registry.get_active_model()
            model_version_record = (
                db.query(ModelVersion)
                .filter(ModelVersion.version == metadata.version)
                .first()
            )

        if not model_version_record:
            raise HTTPException(
                status_code=400, detail="No model available for predictions"
            )

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=f"Model error: {str(e)}") from e

    # Get matches to predict
    match_query = db.query(Match).filter(Match.status == MatchStatus.SCHEDULED)

    if request.match_ids:
        match_query = match_query.filter(Match.id.in_(request.match_ids))

    if request.tournament_id:
        match_query = match_query.filter(Match.tournament_id == request.tournament_id)

    if request.days_ahead:
        now = datetime.now(timezone.utc)
        end_date = now + timedelta(days=request.days_ahead)
        match_query = match_query.filter(
            and_(
                Match.match_date >= now,
                Match.match_date <= end_date,
            )
        )

    matches = match_query.all()

    # Generate predictions (placeholder implementation)
    # In a full implementation, this would use the ML model to make predictions
    generated_predictions = []

    for match in matches:
        # Check if prediction already exists for this match and model
        existing = (
            db.query(Prediction)
            .filter(
                and_(
                    Prediction.match_id == match.id,
                    Prediction.model_version_id == model_version_record.id,
                )
            )
            .first()
        )

        if existing:
            continue  # Skip if already predicted

        # Placeholder: Create a prediction with dummy probabilities
        # In production, this would call model.predict()
        prob_home = 0.40
        prob_draw = 0.30
        prob_away = 0.30

        # Determine predicted outcome
        probs: dict[str, float] = {"H": prob_home, "D": prob_draw, "A": prob_away}
        predicted_outcome = max(probs, key=lambda k: probs[k])
        confidence = probs[predicted_outcome]

        prediction = Prediction(
            match_id=match.id,
            model_version_id=model_version_record.id,
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
        )
        db.add(prediction)
        generated_predictions.append(prediction)

    db.flush()

    return {
        "generated": len(generated_predictions),
        "predictions": [pred.id for pred in generated_predictions],
        "model_version": model_version_record.version,
        "matches_processed": len(matches),
    }


@router.get("/upcoming", response_model=PaginatedResponse[PredictionWithMatchResponse])
def get_upcoming_predictions(
    days: int = Query(7, ge=1, le=30, description="Days ahead for predictions"),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get predictions for upcoming matches.

    Returns predictions for matches in the next N days.

    Args:
        days: Number of days ahead to look for predictions

    Returns:
        List of upcoming predictions
    """
    now = datetime.now(timezone.utc)
    end_date = now + timedelta(days=days)

    predictions = (
        db.query(Prediction)
        .join(Match)
        .filter(
            and_(
                Match.match_date >= now,
                Match.match_date <= end_date,
                Match.status == MatchStatus.SCHEDULED,
            )
        )
        .order_by(Match.match_date)
        .all()
    )

    total = len(predictions)
    items = []
    for pred in predictions:
        match = pred.match

        # Build MatchDetailResponse
        match_response = MR(
            id=match.id,
            tournament_id=match.tournament_id,
            season_id=match.season_id,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            match_date=match.match_date,
            home_score=match.home_score,
            away_score=match.away_score,
            status=match.status,
            odds_home=match.odds_home,
            odds_draw=match.odds_draw,
            odds_away=match.odds_away,
            num_bookmakers=match.num_bookmakers,
            created_at=match.created_at,
            updated_at=match.updated_at,
            result=None,  # Upcoming match has no result
        )

        # Build ModelVersionResponse
        model_version = ModelVersionResponse.model_validate(pred.model_version)

        items.append(
            {
                "id": pred.id,
                "match_id": pred.match_id,
                "model_version_id": pred.model_version_id,
                "prob_home": pred.prob_home,
                "prob_draw": pred.prob_draw,
                "prob_away": pred.prob_away,
                "predicted_outcome": pred.predicted_outcome,
                "confidence": pred.confidence,
                "predicted_at": pred.predicted_at,
                "actual_roi": pred.actual_roi,
                "max_probability": pred.max_probability,
                "match": match_response,
                "model_version": model_version.model_dump(),
            }
        )

    return PaginatedResponse(
        items=items,
        total=total,
        limit=len(items),
        offset=0,
    )


@router.get("/history", response_model=PaginatedResponse[PredictionResponse])
def get_prediction_history(
    from_date: datetime | None = Query(None, description="Start date"),
    to_date: datetime | None = Query(None, description="End date"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records"),
    db: Session = Depends(get_db),
) -> PaginatedResponse[PredictionResponse]:
    """Get prediction accuracy history.

    Returns historical prediction accuracy metrics over time.

    Args:
        from_date: Start date for history
        to_date: End date for history
        limit: Maximum number of records

    Returns:
        Prediction accuracy history data
    """
    query = (
        db.query(Prediction).join(Match).filter(Match.status == MatchStatus.FINISHED)
    )

    if from_date:
        query = query.filter(Prediction.predicted_at >= from_date)

    if to_date:
        query = query.filter(Prediction.predicted_at <= to_date)

    total = query.count()
    predictions = query.order_by(Prediction.predicted_at.desc()).limit(limit).all()

    items = []
    for pred in predictions:
        items.append(
            PredictionResponse(
                id=pred.id,
                match_id=pred.match_id,
                model_version_id=pred.model_version_id,
                prob_home=pred.prob_home,
                prob_draw=pred.prob_draw,
                prob_away=pred.prob_away,
                predicted_outcome=pred.predicted_outcome,
                confidence=pred.confidence,
                predicted_at=pred.predicted_at,
                actual_roi=pred.actual_roi,
                max_probability=pred.max_probability,
            )
        )

    return PaginatedResponse(
        items=items,
        total=total,
        limit=limit,
        offset=0,
    )
