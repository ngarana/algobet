"""API router for team endpoints."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas import FormBreakdown, TeamResponse
from algobet.models import Team
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.form_features import FormCalculator

router = APIRouter()


@router.get("", response_model=list[TeamResponse])
def list_teams(
    search: str | None = Query(None, description="Search by team name"),
    tournament_id: int | None = Query(None, description="Filter by tournament ID"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of teams"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> list[TeamResponse]:
    """List teams with optional search and filter.

    Args:
        search: Optional search string for team name (case-insensitive)
        tournament_id: Optional filter by tournament ID
        limit: Maximum number of teams to return
        offset: Offset for pagination

    Returns:
        List of teams matching the criteria
    """
    query = db.query(Team)

    if search:
        query = query.filter(Team.name.ilike(f"%{search}%"))

    if tournament_id:
        # Filter to teams that have matches in this tournament
        from algobet.models import Match

        query = query.filter(
            Team.id.in_(
                db.query(Match.home_team_id)
                .filter(Match.tournament_id == tournament_id)
                .union(
                    db.query(Match.away_team_id).filter(
                        Match.tournament_id == tournament_id
                    )
                )
            )
        )

    teams = query.order_by(Team.name).offset(offset).limit(limit).all()
    return [TeamResponse.model_validate(t) for t in teams]


@router.get("/{team_id}", response_model=TeamResponse)
def get_team(
    team_id: int,
    db: Session = Depends(get_db),
) -> TeamResponse:
    """Get details for a specific team.

    Args:
        team_id: ID of the team

    Returns:
        Team details

    Raises:
        HTTPException: If team not found (404)
    """
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    return TeamResponse.model_validate(team)


@router.get("/{team_id}/form", response_model=FormBreakdown)
def get_team_form(
    team_id: int,
    n_matches: int = Query(5, ge=1, le=20, description="Number of recent matches"),
    reference_date: datetime | None = Query(
        None, description="Reference date (default: now)"
    ),
    db: Session = Depends(get_db),
) -> FormBreakdown:
    """Get form breakdown for a team.

    Computes win/draw/loss rates and goal statistics from recent matches.

    Args:
        team_id: ID of the team
        n_matches: Number of recent matches to analyze
        reference_date: Date up to which to analyze (default: current time)

    Returns:
        Form breakdown with statistics

    Raises:
        HTTPException: If team not found (404)
    """
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    repo = MatchRepository(db)
    calc = FormCalculator(repo)

    if reference_date is None:
        reference_date = datetime.utcnow()

    # Calculate form metrics
    avg_points = calc.calculate_recent_form(team_id, reference_date, n_matches)
    avg_goals_for = calc.calculate_goals_scored(team_id, reference_date, n_matches)
    avg_goals_against = calc.calculate_goals_conceded(
        team_id, reference_date, n_matches
    )

    # Calculate rates from avg_points
    # avg_points = 3*win_rate + 1*draw_rate + 0*loss_rate
    # And: win_rate + draw_rate + loss_rate = 1
    # From these: win_rate = (avg_points - 1) / 2, draw_rate = 2 - 2*win_rate,
    # loss_rate = 1 - win_rate - draw_rate

    win_rate = min(1.0, max(0.0, (avg_points - 1) / 2)) if avg_points >= 1 else 0.0

    draw_rate = min(1.0, max(0.0, 2 - 2 * win_rate))
    loss_rate = 1.0 - win_rate - draw_rate

    return FormBreakdown(
        avg_points=round(avg_points, 2),
        win_rate=round(win_rate, 2),
        draw_rate=round(draw_rate, 2),
        loss_rate=round(loss_rate, 2),
        avg_goals_for=round(avg_goals_for, 2),
        avg_goals_against=round(avg_goals_against, 2),
    )


@router.get("/{team_id}/matches")
def get_team_matches(
    team_id: int,
    venue: str = Query("all", regex="^(home|away|all)$", description="Venue filter"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of matches"),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get match history for a team.

    Args:
        team_id: ID of the team
        venue: Filter by venue - 'home', 'away', or 'all'
        limit: Maximum number of matches to return

    Returns:
        List of team's recent matches

    Raises:
        HTTPException: If team not found (404)
    """
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    repo = MatchRepository(db)

    matches = repo.get_team_matches(
        team_id=team_id,
        home_only=(venue == "home"),
        away_only=(venue == "away"),
        limit=limit,
    )

    return [
        {
            "id": m.id,
            "tournament_id": m.tournament_id,
            "season_id": m.season_id,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "match_date": m.match_date,
            "home_score": m.home_score,
            "away_score": m.away_score,
            "status": m.status,
        }
        for m in matches
    ]
