"""API router for match endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import (
    MatchDetailResponse,
    MatchResponse,
    MatchStatus,
    PredictionResponse,
)
from algobet.api.schemas.team import TeamResponse
from algobet.api.schemas.tournament import SeasonResponse, TournamentResponse
from algobet.models import Match, Prediction
from algobet.predictions.data.queries import MatchRepository

router = APIRouter()


@router.get("", response_model=list[MatchResponse])
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
) -> list[MatchResponse]:
    """List matches with filtering.

    Supports filtering by status, tournament, season, team, date range, and
    odds availability.
    """
    query = db.query(Match)

    if status:
        if status not in [
            MatchStatus.SCHEDULED,
            MatchStatus.FINISHED,
            MatchStatus.LIVE,
        ]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid status: {status}. " "Must be SCHEDULED, FINISHED, or LIVE"
                ),
            )
        query = query.filter(Match.status == status)

    if tournament_id:
        query = query.filter(Match.tournament_id == tournament_id)

    if season_id:
        query = query.filter(Match.season_id == season_id)

    if team_id:
        query = query.filter(
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id)
        )

    if from_date:
        query = query.filter(Match.match_date >= from_date)

    if to_date:
        query = query.filter(Match.match_date <= to_date)

    if days_ahead:
        now = datetime.utcnow()
        end_date = datetime.fromtimestamp(now.timestamp() + days_ahead * 86400)
        query = query.filter(
            and_(
                Match.match_date >= now,
                Match.match_date <= end_date,
            )
        )

    if has_odds is not None:
        if has_odds:
            query = query.filter(
                and_(
                    Match.odds_home.is_not(None),
                    Match.odds_draw.is_not(None),
                    Match.odds_away.is_not(None),
                )
            )
        else:
            query = query.filter(
                or_(
                    Match.odds_home.is_(None),
                    Match.odds_draw.is_(None),
                    Match.odds_away.is_(None),
                )
            )

    matches = query.order_by(Match.match_date).offset(offset).limit(limit).all()

    # Add computed result field
    result = []
    for match in matches:
        result_value = None
        if (
            match.status == MatchStatus.FINISHED
            and match.home_score is not None
            and match.away_score is not None
        ):
            if match.home_score > match.away_score:
                result_value = "H"
            elif match.home_score < match.away_score:
                result_value = "A"
            else:
                result_value = "D"

        result.append(
            MatchResponse(
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
                result=result_value,
            )
        )

    return result


@router.get("/{match_id}", response_model=MatchDetailResponse)
def get_match(
    match_id: int,
    db: Session = Depends(get_db),
) -> MatchDetailResponse:
    """Get detailed information for a specific match.

    Includes tournament, season, teams, predictions, and H2H history.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    # Calculate result
    result = None
    if (
        match.status == MatchStatus.FINISHED
        and match.home_score is not None
        and match.away_score is not None
    ):
        if match.home_score > match.away_score:
            result = "H"
        elif match.home_score < match.away_score:
            result = "A"
        else:
            result = "D"

    # Get H2H matches
    repo = MatchRepository(db)
    h2h_matches = repo.get_h2h_matches(
        match.home_team_id,
        match.away_team_id,
        limit=5,
        before_date=match.match_date,
    )

    # Convert H2H matches to response format
    h2h_responses = []
    for h2h_match in h2h_matches:
        h2h_result = None
        if h2h_match.home_score is not None and h2h_match.away_score is not None:
            if h2h_match.home_score > h2h_match.away_score:
                h2h_result = "H"
            elif h2h_match.home_score < h2h_match.away_score:
                h2h_result = "A"
            else:
                h2h_result = "D"

        h2h_responses.append(
            MatchResponse(
                id=h2h_match.id,
                tournament_id=h2h_match.tournament_id,
                season_id=h2h_match.season_id,
                home_team_id=h2h_match.home_team_id,
                away_team_id=h2h_match.away_team_id,
                match_date=h2h_match.match_date,
                home_score=h2h_match.home_score,
                away_score=h2h_match.away_score,
                status=h2h_match.status,
                odds_home=h2h_match.odds_home,
                odds_draw=h2h_match.odds_draw,
                odds_away=h2h_match.odds_away,
                num_bookmakers=h2h_match.num_bookmakers,
                created_at=h2h_match.created_at,
                updated_at=h2h_match.updated_at,
                result=h2h_result,
            )
        )

    # Load predictions for this match
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

    return MatchDetailResponse(
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
        result=result,
        tournament=TournamentResponse.model_validate(match.tournament),
        season=SeasonResponse.model_validate(match.season),
        home_team=TeamResponse.model_validate(match.home_team),
        away_team=TeamResponse.model_validate(match.away_team),
        predictions=prediction_responses,
        h2h_matches=h2h_responses,
    )


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


@router.get("/{match_id}/h2h", response_model=list[MatchResponse])
def get_match_h2h(
    match_id: int,
    limit: int = Query(5, ge=1, le=20, description="Number of H2H matches"),
    db: Session = Depends(get_db),
) -> list[MatchResponse]:
    """Get head-to-head history for a match.

    Returns previous meetings between the two teams.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    repo = MatchRepository(db)
    h2h_matches = repo.get_h2h_matches(
        match.home_team_id,
        match.away_team_id,
        limit=limit,
        before_date=match.match_date,
    )

    # Convert to response format
    result = []
    for h2h_match in h2h_matches:
        h2h_result = None
        if (
            h2h_match.status == MatchStatus.FINISHED
            and h2h_match.home_score is not None
            and h2h_match.away_score is not None
        ):
            if h2h_match.home_score > h2h_match.away_score:
                h2h_result = "H"
            elif h2h_match.home_score < h2h_match.away_score:
                h2h_result = "A"
            else:
                h2h_result = "D"

        result.append(
            MatchResponse(
                id=h2h_match.id,
                tournament_id=h2h_match.tournament_id,
                season_id=h2h_match.season_id,
                home_team_id=h2h_match.home_team_id,
                away_team_id=h2h_match.away_team_id,
                match_date=h2h_match.match_date,
                home_score=h2h_match.home_score,
                away_score=h2h_match.away_score,
                status=h2h_match.status,
                odds_home=h2h_match.odds_home,
                odds_draw=h2h_match.odds_draw,
                odds_away=h2h_match.odds_away,
                num_bookmakers=h2h_match.num_bookmakers,
                created_at=h2h_match.created_at,
                updated_at=h2h_match.updated_at,
                result=h2h_result,
            )
        )

    return result
