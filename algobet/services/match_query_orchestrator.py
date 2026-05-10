"""Query orchestration for match list endpoints."""

from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session, joinedload

from algobet.api.schemas import MatchResponse, MatchStatus, PaginatedResponse
from algobet.models import Match
from algobet.user_workflow.service import build_match_response


@dataclass(frozen=True)
class MatchListFilters:
    """Validated filter payload for listing matches."""

    status: str | None = None
    tournament_id: int | None = None
    season_id: int | None = None
    team_id: int | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    days_ahead: int | None = None
    has_odds: bool | None = None
    limit: int = 50
    offset: int = 0


class MatchQueryOrchestrator:
    """Build match queries outside the FastAPI router layer."""

    def list_matches(
        self,
        db: Session,
        filters: MatchListFilters,
    ) -> PaginatedResponse[MatchResponse]:
        query = db.query(Match).options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.tournament),
            joinedload(Match.season),
        )

        if filters.status:
            if filters.status not in {
                MatchStatus.SCHEDULED,
                MatchStatus.FINISHED,
                MatchStatus.LIVE,
            }:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid status: {filters.status}. "
                        "Must be SCHEDULED, FINISHED, or LIVE"
                    ),
                )
            query = query.filter(Match.status == filters.status)

        if filters.tournament_id:
            query = query.filter(Match.tournament_id == filters.tournament_id)

        if filters.season_id:
            query = query.filter(Match.season_id == filters.season_id)

        if filters.team_id:
            query = query.filter(
                or_(
                    Match.home_team_id == filters.team_id,
                    Match.away_team_id == filters.team_id,
                )
            )

        if filters.from_date:
            query = query.filter(Match.match_date >= filters.from_date)

        if filters.to_date:
            query = query.filter(Match.match_date <= filters.to_date)

        if filters.days_ahead:
            now = datetime.now(timezone.utc)
            end_date = datetime.fromtimestamp(
                now.timestamp() + filters.days_ahead * 86400
            )
            query = query.filter(
                and_(
                    Match.match_date >= now,
                    Match.match_date <= end_date,
                )
            )

        if filters.has_odds is not None:
            odds_available = and_(
                Match.odds_home.is_not(None),
                Match.odds_draw.is_not(None),
                Match.odds_away.is_not(None),
            )
            if filters.has_odds:
                query = query.filter(odds_available)
            else:
                query = query.filter(
                    or_(
                        Match.odds_home.is_(None),
                        Match.odds_draw.is_(None),
                        Match.odds_away.is_(None),
                    )
                )

        total = query.count()
        matches = (
            query.order_by(Match.match_date)
            .offset(filters.offset)
            .limit(filters.limit)
            .all()
        )

        return PaginatedResponse(
            items=[build_match_response(match) for match in matches],
            total=total,
            limit=filters.limit,
            offset=filters.offset,
        )
