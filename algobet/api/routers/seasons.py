"""API router for season endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import MatchResponse, MatchStatus
from algobet.models import Match, Season

router = APIRouter()


@router.get("/{season_id}/matches", response_model=list[MatchResponse])
def get_season_matches(
    season_id: int,
    status: str = Query(
        None, description="Filter by status (SCHEDULED, FINISHED, LIVE)"
    ),
    team_id: int = Query(None, description="Filter by team ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of matches"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> list[MatchResponse]:
    """Get all matches for a season.

    Args:
        season_id: ID of the season
        status: Optional filter by match status
        team_id: Optional filter by team ID (home or away)
        limit: Maximum number of matches to return
        offset: Offset for pagination

    Returns:
        List of matches for the season

    Raises:
        HTTPException: If season not found (404) or invalid status (400)
    """
    # Verify season exists
    season = db.query(Season).filter(Season.id == season_id).first()
    if not season:
        raise HTTPException(status_code=404, detail=f"Season {season_id} not found")

    # Build query
    query = db.query(Match).filter(Match.season_id == season_id)

    # Apply status filter if provided
    if status:
        valid_statuses = [MatchStatus.SCHEDULED, MatchStatus.FINISHED, MatchStatus.LIVE]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of {valid_statuses}",
            )
        query = query.filter(Match.status == status)

    # Apply team filter if provided
    if team_id:
        from sqlalchemy import or_

        query = query.filter(
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id)
        )

    # Get matches with pagination
    matches = query.order_by(Match.match_date).offset(offset).limit(limit).all()

    # Convert to response format with computed result field
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
