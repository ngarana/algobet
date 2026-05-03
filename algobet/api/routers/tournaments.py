"""API router for tournament endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_cache.decorator import cache
from sqlalchemy import case, func, or_
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import SeasonResponse, TournamentResponse
from algobet.models import Season, Tournament

router = APIRouter()


def _normalize_search_term(value: str | None) -> str | None:
    """Normalize user-entered search text for predictive search queries."""
    if value is None:
        return None

    normalized = " ".join(value.split()).strip()
    return normalized or None


@router.get("", response_model=list[TournamentResponse])
@cache(expire=300)  # type: ignore[misc]
def list_tournaments(
    search: str | None = Query(
        None, description="Search by tournament name or country"
    ),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of tournaments"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> list[TournamentResponse]:
    """List all tournaments.

    Returns a list of all tournaments in the database, optionally limited.
    """
    normalized_search = _normalize_search_term(search)
    query = db.query(Tournament)

    if normalized_search:
        lowered_search = normalized_search.lower()
        contains_pattern = f"%{lowered_search}%"
        prefix_pattern = f"{lowered_search}%"
        lowered_name = func.lower(Tournament.name)
        lowered_country = func.lower(Tournament.country)

        query = query.filter(
            or_(
                lowered_name.like(contains_pattern),
                lowered_country.like(contains_pattern),
            )
        ).order_by(
            case((lowered_name.like(prefix_pattern), 0), else_=1),
            case((lowered_country.like(prefix_pattern), 0), else_=1),
            func.length(Tournament.name),
            Tournament.name,
        )
    else:
        query = query.order_by(Tournament.name)

    tournaments = query.offset(offset).limit(limit).all()
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
