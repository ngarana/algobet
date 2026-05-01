"""SQL queries and repository for match data access."""

from datetime import datetime

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session

from algobet.matches.models import Match


class MatchRepository:
    """Repository for querying match data from the database.

    Provides methods for extracting historical match data needed for
    feature engineering and model training.
    """

    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

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
    ) -> list[Match]:
        """Get matches for training within a date range with advanced filtering.

        Args:
            min_date: Optional start date filter
            max_date: Optional end date filter
            tournament_id: Optional single tournament filter (deprecated, prefer tournament_ids)
            tournament_ids: Optional list of tournament IDs to include
            team_ids: Optional list of team IDs (matches where either home or away team is in list)
            require_results: If True, only return finished matches with scores
            require_odds: If True, only return matches with odds available
            min_total_goals: Optional minimum total goals filter (home_score + away_score)
            max_total_goals: Optional maximum total goals filter
            venue_filter: Optional venue filter - "home", "away", or "both" (default)

        Returns:
            List of Match objects ordered by date
        """
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
            # Match where either home or away team is in the list
            if venue_filter == "home":
                stmt = stmt.where(Match.home_team_id.in_(team_ids))
            elif venue_filter == "away":
                stmt = stmt.where(Match.away_team_id.in_(team_ids))
            else:  # "both" or None
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
        # Goals filters must be applied after results are required (need scores)
        if min_total_goals is not None:
            stmt = stmt.where((Match.home_score + Match.away_score) >= min_total_goals)
        if max_total_goals is not None:
            stmt = stmt.where((Match.home_score + Match.away_score) <= max_total_goals)

        stmt = stmt.order_by(Match.match_date)
        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def get_team_matches(
        self,
        team_id: int,
        before_date: datetime | None = None,
        limit: int = 10,
        home_only: bool = False,
        away_only: bool = False,
    ) -> list[Match]:
        """Get team's recent matches before a given date.

        Args:
            team_id: ID of the team
            before_date: Only return matches before this date
            limit: Maximum number of matches to return
            home_only: If True, only return home matches
            away_only: If True, only return away matches

        Returns:
            List of Match objects ordered by date (most recent first)
        """
        # Build venue filter
        if home_only:
            venue_filter = Match.home_team_id == team_id
        elif away_only:
            venue_filter = Match.away_team_id == team_id
        else:
            venue_filter = or_(
                Match.home_team_id == team_id, Match.away_team_id == team_id
            )

        stmt = select(Match).where(venue_filter)

        if before_date:
            stmt = stmt.where(Match.match_date < before_date)

        # Only include finished matches
        stmt = stmt.where(
            and_(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
            )
        )

        stmt = stmt.order_by(desc(Match.match_date)).limit(limit)
        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def get_h2h_matches(
        self,
        team1_id: int,
        team2_id: int,
        limit: int = 5,
        before_date: datetime | None = None,
    ) -> list[Match]:
        """Get head-to-head history between two teams.

        Args:
            team1_id: ID of first team
            team2_id: ID of second team
            limit: Maximum number of matches to return
            before_date: Only return matches before this date

        Returns:
            List of Match objects ordered by date (most recent first)
        """
        # H2H matches where these two teams played each other
        stmt = select(Match).where(
            or_(
                and_(Match.home_team_id == team1_id, Match.away_team_id == team2_id),
                and_(Match.home_team_id == team2_id, Match.away_team_id == team1_id),
            )
        )

        if before_date:
            stmt = stmt.where(Match.match_date < before_date)

        # Only include finished matches
        stmt = stmt.where(
            and_(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
            )
        )

        stmt = stmt.order_by(desc(Match.match_date)).limit(limit)
        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def get_match_count(self, team_id: int, before_date: datetime) -> int:
        """Get count of matches played by a team before a given date.

        Args:
            team_id: ID of the team
            before_date: Count matches before this date

        Returns:
            Number of matches played
        """
        stmt = select(func.count(Match.id)).where(
            and_(
                or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                Match.match_date < before_date,
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
            )
        )
        result = self.session.execute(stmt)
        return result.scalar() or 0
