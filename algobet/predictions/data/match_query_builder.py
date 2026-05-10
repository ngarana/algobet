"""SQL construction helpers for match repository queries."""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, or_, select

from algobet.matches.models import Match, MatchStatistics, PlayerMatchStats


class MatchQueryBuilder:
    """Build SQLAlchemy statements for match data access."""

    def historical_matches(
        self,
        *,
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
    ) -> Any:
        """Build the historical match statement used by training and backtests."""
        stmt = select(Match)

        if min_date:
            stmt = stmt.where(Match.match_date >= min_date)
        if max_date:
            stmt = stmt.where(Match.match_date <= max_date)
        if tournament_id:
            stmt = stmt.where(Match.tournament_id == tournament_id)
        if tournament_ids:
            stmt = stmt.where(Match.tournament_id.in_(tournament_ids))
        if team_ids:
            if venue_filter == "home":
                stmt = stmt.where(Match.home_team_id.in_(team_ids))
            elif venue_filter == "away":
                stmt = stmt.where(Match.away_team_id.in_(team_ids))
            else:
                stmt = stmt.where(
                    or_(
                        Match.home_team_id.in_(team_ids),
                        Match.away_team_id.in_(team_ids),
                    )
                )
        if require_odds:
            stmt = stmt.where(
                and_(
                    Match.odds_home.is_not(None),
                    Match.odds_draw.is_not(None),
                    Match.odds_away.is_not(None),
                )
            )
        if require_results:
            stmt = stmt.where(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None),
                )
            )
        if require_enriched_stats:
            has_understat = (
                select(MatchStatistics.match_id)
                .where(
                    and_(
                        MatchStatistics.match_id == Match.id,
                        MatchStatistics.home_xg.is_not(None),
                        MatchStatistics.away_xg.is_not(None),
                    )
                )
                .exists()
            )
            has_player_stats = (
                select(PlayerMatchStats.match_id)
                .where(PlayerMatchStats.match_id == Match.id)
                .exists()
            )
            stmt = stmt.where(and_(has_understat, has_player_stats))
        if min_total_goals is not None:
            stmt = stmt.where((Match.home_score + Match.away_score) >= min_total_goals)
        if max_total_goals is not None:
            stmt = stmt.where((Match.home_score + Match.away_score) <= max_total_goals)

        return stmt.order_by(Match.match_date)
