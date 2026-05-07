"""API router for prediction endpoints."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

from algobet.api.dependencies import get_db
from algobet.api.schemas import (
    MatchStatus,
    ModelVersionResponse,
    PaginatedResponse,
    PredictionListItemResponse,
    PredictionMatchSummaryResponse,
)
from algobet.models import Match, Prediction
from algobet.services.prediction_service import PredictionResult, PredictionService

router = APIRouter()


class GeneratePredictionsRequest(BaseModel):
    """Request body for generating predictions."""

    match_ids: list[int] | None = None
    model_version: str | None = None
    tournament_id: int | None = None
    days_ahead: int | None = None


def _build_prediction_item(prediction: Prediction) -> PredictionListItemResponse:
    """Build a prediction list item with compact related entities."""
    match = prediction.match

    match_summary = (
        PredictionMatchSummaryResponse(
            id=match.id,
            match_date=match.match_date,
            status=match.status,
            home_team_name=match.home_team.name,
            away_team_name=match.away_team.name,
            tournament_name=match.tournament.name if match.tournament else None,
            season_name=match.season.name if match.season else None,
            home_score=match.home_score,
            away_score=match.away_score,
            odds_home=match.odds_home,
            odds_draw=match.odds_draw,
            odds_away=match.odds_away,
        )
        if match is not None
        else None
    )

    model_version = (
        ModelVersionResponse.model_validate(prediction.model_version)
        if prediction.model_version is not None
        else None
    )

    return PredictionListItemResponse(
        id=prediction.id,
        match_id=prediction.match_id,
        model_version_id=prediction.model_version_id,
        prob_home=prediction.prob_home,
        prob_draw=prediction.prob_draw,
        prob_away=prediction.prob_away,
        predicted_outcome=prediction.predicted_outcome,
        confidence=prediction.confidence,
        predicted_at=prediction.predicted_at,
        actual_roi=prediction.actual_roi,
        max_probability=prediction.max_probability,
        match=match_summary,
        model_version=model_version,
    )


class GeneratePredictionsResponse(BaseModel):
    """Summary of a prediction generation request."""

    generated: int
    prediction_ids: list[int]
    model_version: str
    matches_processed: int
    existing_predictions_skipped: int


@router.get("", response_model=PaginatedResponse[PredictionListItemResponse])
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
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    db: Session = Depends(get_db),
) -> PaginatedResponse[PredictionListItemResponse]:
    """List predictions with filtering.

    Returns predictions matching the specified filters.
    """
    query = db.query(Prediction).options(
        joinedload(Prediction.match).joinedload(Match.home_team),
        joinedload(Prediction.match).joinedload(Match.away_team),
        joinedload(Prediction.match).joinedload(Match.tournament),
        joinedload(Prediction.match).joinedload(Match.season),
        joinedload(Prediction.model_version),
    )

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
    predictions = (
        query.order_by(Prediction.predicted_at.desc()).offset(offset).limit(limit).all()
    )

    items = [_build_prediction_item(pred) for pred in predictions]

    return PaginatedResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/generate", response_model=GeneratePredictionsResponse)
def generate_predictions(
    request: GeneratePredictionsRequest,
    db: Session = Depends(get_db),
) -> GeneratePredictionsResponse:
    """Generate predictions for upcoming matches.

    Creates predictions for specified matches or all upcoming matches in a tournament.

    Args:
        request: Prediction generation request with match IDs and options

    Returns:
        Summary of generated predictions
    """
    service = PredictionService(db)

    try:
        model, resolved_version = service.load_model(request.model_version)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=f"Model error: {str(e)}") from e

    # Get matches to predict
    match_query = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.tournament),
            joinedload(Match.season),
        )
        .filter(Match.status == MatchStatus.SCHEDULED)
    )

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

    prediction_results: list[PredictionResult] = []
    skipped_predictions = 0

    for match in matches:
        # Check if prediction already exists for this match and model
        existing = (
            db.query(Prediction)
            .filter(
                and_(
                    Prediction.match_id == match.id,
                    Prediction.model_version.has(version=resolved_version),
                )
            )
            .first()
        )

        if existing:
            skipped_predictions += 1
            continue

        try:
            features = service.generate_features_v2(match)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Model error: {str(e)}",
            ) from e

        if features is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model error: Model {resolved_version} is missing its saved "
                    "feature pipeline. Retrain it before generating predictions."
                ),
            )

        outcome, confidence, probabilities = service.get_prediction(model, features)

        prediction_results.append(
            PredictionResult(
                match_id=match.id,
                match_date=match.match_date,
                home_team=match.home_team.name,
                away_team=match.away_team.name,
                predicted_outcome=outcome,
                confidence=confidence,
                model_version=resolved_version,
                prob_home=probabilities["home"],
                prob_draw=probabilities["draw"],
                prob_away=probabilities["away"],
            )
        )

    saved_predictions = service.save_predictions(prediction_results)
    db.flush()

    return GeneratePredictionsResponse(
        generated=len(saved_predictions),
        prediction_ids=[pred.id for pred in saved_predictions],
        model_version=resolved_version,
        matches_processed=len(matches),
        existing_predictions_skipped=skipped_predictions,
    )


@router.get("/upcoming", response_model=PaginatedResponse[PredictionListItemResponse])
def get_upcoming_predictions(
    days: int = Query(7, ge=1, le=30, description="Days ahead for predictions"),
    model_version_id: int | None = Query(
        None, description="Optional model version filter"
    ),
    db: Session = Depends(get_db),
) -> PaginatedResponse[PredictionListItemResponse]:
    """Get predictions for upcoming matches.

    Returns predictions for matches in the next N days.

    Args:
        days: Number of days ahead to look for predictions

    Returns:
        List of upcoming predictions
    """
    now = datetime.now(timezone.utc)
    end_date = now + timedelta(days=days)

    query = (
        db.query(Prediction)
        .options(
            joinedload(Prediction.match).joinedload(Match.home_team),
            joinedload(Prediction.match).joinedload(Match.away_team),
            joinedload(Prediction.match).joinedload(Match.tournament),
            joinedload(Prediction.match).joinedload(Match.season),
            joinedload(Prediction.model_version),
        )
        .join(Match)
        .filter(
            and_(
                Match.match_date >= now,
                Match.match_date <= end_date,
                Match.status == MatchStatus.SCHEDULED,
            )
        )
        .order_by(Match.match_date)
    )

    if model_version_id is not None:
        query = query.filter(Prediction.model_version_id == model_version_id)

    predictions = query.all()
    items = [_build_prediction_item(pred) for pred in predictions]

    return PaginatedResponse(
        items=items,
        total=len(items),
        limit=len(items),
        offset=0,
    )


@router.get("/history", response_model=PaginatedResponse[PredictionListItemResponse])
def get_prediction_history(
    model_version_id: int | None = Query(
        None, description="Optional model version filter"
    ),
    from_date: datetime | None = Query(None, description="Start date"),
    to_date: datetime | None = Query(None, description="End date"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records"),
    db: Session = Depends(get_db),
) -> PaginatedResponse[PredictionListItemResponse]:
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
        db.query(Prediction)
        .options(
            joinedload(Prediction.match).joinedload(Match.home_team),
            joinedload(Prediction.match).joinedload(Match.away_team),
            joinedload(Prediction.match).joinedload(Match.tournament),
            joinedload(Prediction.match).joinedload(Match.season),
            joinedload(Prediction.model_version),
        )
        .join(Match)
        .filter(Match.status == MatchStatus.FINISHED)
    )

    if model_version_id is not None:
        query = query.filter(Prediction.model_version_id == model_version_id)

    if from_date:
        query = query.filter(Prediction.predicted_at >= from_date)

    if to_date:
        query = query.filter(Prediction.predicted_at <= to_date)

    total = query.count()
    predictions = query.order_by(Prediction.predicted_at.desc()).limit(limit).all()

    items = [_build_prediction_item(pred) for pred in predictions]

    return PaginatedResponse(
        items=items,
        total=total,
        limit=limit,
        offset=0,
    )
