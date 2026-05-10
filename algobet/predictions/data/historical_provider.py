"""Historical match provider for the MatchRepository facade."""

from datetime import datetime

from sqlalchemy.orm import Session

from algobet.matches.models import Match
from algobet.predictions.data.match_query_builder import MatchQueryBuilder


class HistoricalMatchProvider:
    """Fetch historical matches using a dedicated query builder."""

    def __init__(
        self,
        session: Session,
        query_builder: MatchQueryBuilder | None = None,
    ) -> None:
        self.session = session
        self.query_builder = query_builder or MatchQueryBuilder()

    def get_historical_matches(
        self,
        min_date: datetime | None = None,
        max_date: datetime | None = None,
        tournament_id: int | None = None,
        tournament_ids: list[int] | None = None,
        team_ids: list[int] | None = None,
        require_results: bool = True,
        require_odds: bool | None = None,
        min_total_goals: float | None = None,
        max_total_goals: float | None = None,
        venue_filter: str | None = None,
        require_enriched_stats: bool = False,
    ) -> list[Match]:
        stmt = self.query_builder.historical_matches(
            min_date=min_date,
            max_date=max_date,
            tournament_id=tournament_id,
            tournament_ids=tournament_ids,
            team_ids=team_ids,
            require_results=require_results,
            require_odds=require_odds,
            min_total_goals=min_total_goals,
            max_total_goals=max_total_goals,
            venue_filter=venue_filter,
            require_enriched_stats=require_enriched_stats,
        )
        result = self.session.execute(stmt)
        return list(result.scalars().all())
