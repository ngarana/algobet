"""API router for match endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

from algobet.api.dependencies import get_db
from algobet.api.schemas import (
    MatchDetailResponse,
    MatchResponse,
    MatchStatus,
    PaginatedResponse,
    PredictionResponse,
)
from algobet.models import Match, Prediction
from algobet.predictions.data.queries import MatchRepository
from algobet.services.match_detail_builder import (
    _h2h_responses,
    build_match_detail_response,
)
from algobet.services.match_query_orchestrator import (
    MatchListFilters,
    MatchQueryOrchestrator,
)

router = APIRouter()


def _match_queries() -> MatchQueryOrchestrator:
    return MatchQueryOrchestrator()


@router.get("", response_model=PaginatedResponse[MatchResponse])
def list_matches(
    status: str | None = Query(
        None, description="Filter by status (SCHEDULED, FINISHED, LIVE)"
    ),
    tournament_id: int | None = Query(None, description="Filter by tournament ID"),
    season_id: int | None = Query(None, description="Filter by season ID"),
    team_id: int | None = Query(None, description="Filter by team ID"),
    from_date: datetime | None = Query(
        None, description="Filter matches from this date"
    ),
    to_date: datetime | None = Query(
        None, description="Filter matches until this date"
    ),
    days_ahead: int | None = Query(
        None, ge=1, le=365, description="Show matches in next N days"
    ),
    has_odds: bool | None = Query(None, description="Filter by odds availability"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of matches"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    orchestrator: MatchQueryOrchestrator = Depends(_match_queries),
) -> PaginatedResponse[MatchResponse]:
    """List matches with filtering.

    Supports filtering by status, tournament, season, team, date range, and
    odds availability.
    """
    return orchestrator.list_matches(
        db,
        MatchListFilters(
            status=status,
            tournament_id=tournament_id,
            season_id=season_id,
            team_id=team_id,
            from_date=from_date,
            to_date=to_date,
            days_ahead=days_ahead,
            has_odds=has_odds,
            limit=limit,
            offset=offset,
        ),
    )


@router.get("/upcoming", response_model=PaginatedResponse[MatchDetailResponse])
def get_upcoming_matches(
    tournament_id: int | None = Query(None, description="Filter by tournament ID"),
    limit: int = Query(100, ge=1, le=200, description="Maximum number of matches"),
    db: Session = Depends(get_db),
) -> PaginatedResponse[MatchDetailResponse]:
    """Get upcoming matches for today with team and tournament details.

    Returns matches scheduled for the next 24 hours with odds available,
    including team names and tournament info for the scraping UI.
    """
    now = datetime.now(timezone.utc)
    tomorrow = datetime.fromtimestamp(now.timestamp() + 24 * 3600)

    query = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.tournament),
            joinedload(Match.season),
        )
        .filter(
            and_(
                Match.status == MatchStatus.SCHEDULED,
                Match.match_date >= now,
                Match.match_date <= tomorrow,
                Match.odds_home.is_not(None),
                Match.odds_draw.is_not(None),
                Match.odds_away.is_not(None),
            )
        )
    )

    if tournament_id:
        query = query.filter(Match.tournament_id == tournament_id)

    total = query.count()
    matches = query.order_by(Match.match_date).limit(limit).all()

    return PaginatedResponse(
        items=[build_match_detail_response(db, match) for match in matches],
        total=total,
        limit=limit,
        offset=0,
    )


@router.get("/{match_id}", response_model=MatchDetailResponse)
def get_match(
    match_id: int,
    db: Session = Depends(get_db),
) -> MatchDetailResponse:
    """Get detailed information for a specific match.

    Includes tournament, season, teams, predictions, and H2H history.
    """
    match = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.tournament),
            joinedload(Match.season),
        )
        .filter(Match.id == match_id)
        .first()
    )
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")
    return build_match_detail_response(db, match)


@router.get("/{match_id}/preview")
def get_match_preview(
    match_id: int,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Get match preview with form analysis for both teams.

    Returns team form data and basic match information for prediction preview.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    repo = MatchRepository(db)
    from algobet.predictions.features.form_features import FormCalculator

    calc = FormCalculator(repo)

    # Get form for both teams
    home_form = calc.calculate_recent_form(match.home_team_id, match.match_date, 5)
    away_form = calc.calculate_recent_form(match.away_team_id, match.match_date, 5)

    return {
        "match": {
            "id": match.id,
            "home_team": match.home_team.name,
            "away_team": match.away_team.name,
            "match_date": match.match_date,
            "tournament": match.tournament.name,
        },
        "form": {
            "home": {"avg_points": round(home_form, 2)},
            "away": {"avg_points": round(away_form, 2)},
        },
    }


@router.get("/{match_id}/predictions")
def get_match_predictions(
    match_id: int,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Get predictions for a specific match.

    Returns all predictions for this match across different model versions.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    predictions = db.query(Prediction).filter(Prediction.match_id == match_id).all()
    prediction_responses = [
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
        for pred in predictions
    ]

    return {"match_id": match_id, "predictions": prediction_responses}


@router.get("/{match_id}/h2h", response_model=PaginatedResponse[MatchResponse])
def get_match_h2h(
    match_id: int,
    limit: int = Query(5, ge=1, le=20, description="Number of H2H matches"),
    db: Session = Depends(get_db),
) -> PaginatedResponse[MatchResponse]:
    """Get head-to-head history for a match.

    Returns previous meetings between the two teams.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    items = _h2h_responses(db, match, limit=limit)

    return PaginatedResponse(
        items=items,
        total=len(items),  # repo doesn't return count for h2h
        limit=limit,
        offset=0,
    )
