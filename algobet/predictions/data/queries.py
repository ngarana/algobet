"""SQL queries and repository for match data access."""

from datetime import datetime
from typing import Optional

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import Session

from algobet.models import Match, Team


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
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        tournament_id: Optional[int] = None,
        require_results: bool = True
    ) -> list[Match]:
        """Get matches for training within a date range.
        
        Args:
            min_date: Optional start date filter
            max_date: Optional end date filter
            tournament_id: Optional tournament filter
            require_results: If True, only return finished matches with scores
            
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
        if require_results:
            stmt = stmt.where(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None)
                )
            )
        
        stmt = stmt.order_by(Match.match_date)
        result = self.session.execute(stmt)
        return list(result.scalars().all())
    
    def get_team_matches(
        self,
        team_id: int,
        before_date: Optional[datetime] = None,
        limit: int = 10,
        home_only: bool = False,
        away_only: bool = False
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
                Match.home_team_id == team_id,
                Match.away_team_id == team_id
            )
        
        stmt = select(Match).where(venue_filter)
        
        if before_date:
            stmt = stmt.where(Match.match_date < before_date)
        
        # Only include finished matches
        stmt = stmt.where(
            and_(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                Match.away_score.is_not(None)
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
        before_date: Optional[datetime] = None
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
                and_(
                    Match.home_team_id == team1_id,
                    Match.away_team_id == team2_id
                ),
                and_(
                    Match.home_team_id == team2_id,
                    Match.away_team_id == team1_id
                )
            )
        )
        
        if before_date:
            stmt = stmt.where(Match.match_date < before_date)
        
        # Only include finished matches
        stmt = stmt.where(
            and_(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                Match.away_score.is_not(None)
            )
        )
        
        stmt = stmt.order_by(desc(Match.match_date)).limit(limit)
        result = self.session.execute(stmt)
        return list(result.scalars().all())
    
    def get_match_count(
        self,
        team_id: int,
        before_date: datetime
    ) -> int:
        """Get count of matches played by a team before a given date.
        
        Args:
            team_id: ID of the team
            before_date: Count matches before this date
            
        Returns:
            Number of matches played
        """
        stmt = select(func.count(Match.id)).where(
            and_(
                or_(
                    Match.home_team_id == team_id,
                    Match.away_team_id == team_id
                ),
                Match.match_date < before_date,
                Match.status == "FINISHED",
                Match.home_score.is_not(None)
            )
        )
        result = self.session.execute(stmt)
        return result.scalar() or 0