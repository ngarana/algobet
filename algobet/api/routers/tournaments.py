"""API router for tournament endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import SeasonResponse, TournamentResponse
from algobet.models import Season, Tournament

router = APIRouter()


@router.get("", response_model=list[TournamentResponse])
def list_tournaments(
    limit: int = Query(100, ge=1, le=100, description="Maximum number of tournaments"),
    db: Session = Depends(get_db),
) -> list[TournamentResponse]:
    """List all tournaments.

    Returns a list of all tournaments in the database, optionally limited.
    """
    stmt = db.query(Tournament).order_by(Tournament.name).limit(limit)
    tournaments = stmt.all()
    return [TournamentResponse.model_validate(t) for t in tournaments]


@router.get("/{tournament_id}", response_model=TournamentResponse)
def get_tournament(
    tournament_id: int,
    db: Session = Depends(get_db),
) -> TournamentResponse:
    """Get details for a specific tournament.

    Args:
        tournament_id: ID of the tournament

    Returns:
        Tournament details

    Raises:
        HTTPException: If tournament not found (404)
    """
    tournament = db.query(Tournament).filter(Tournament.id == tournament_id).first()
    if not tournament:
        raise HTTPException(
            status_code=404, detail=f"Tournament {tournament_id} not found"
        )
    return TournamentResponse.model_validate(tournament)


@router.get("/{tournament_id}/seasons", response_model=list[SeasonResponse])
def get_tournament_seasons(
    tournament_id: int,
    db: Session = Depends(get_db),
) -> list[SeasonResponse]:
    """Get all seasons for a tournament.

    Args:
        tournament_id: ID of the tournament

    Returns:
        List of seasons for the tournament

    Raises:
        HTTPException: If tournament not found (404)
    """
    # Verify tournament exists
    tournament = db.query(Tournament).filter(Tournament.id == tournament_id).first()
    if not tournament:
        raise HTTPException(
            status_code=404, detail=f"Tournament {tournament_id} not found"
        )

    seasons = (
        db.query(Season)
        .filter(Season.tournament_id == tournament_id)
        .order_by(Season.start_year.desc(), Season.end_year.desc())
        .all()
    )
    return [SeasonResponse.model_validate(s) for s in seasons]
