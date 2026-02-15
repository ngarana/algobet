"""Query service for database entity queries.

This module provides business logic for querying tournaments, seasons,
teams, and matches from the database.
"""

from __future__ import annotations

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from algobet.exceptions import DatabaseQueryError, DataError, DataNotFoundError
from algobet.logging_config import get_logger
from algobet.models import Match, Season, Team, Tournament
from algobet.services.base import BaseService
from algobet.services.dto import (
    MatchFilter,
    MatchInfo,
    MatchListResponse,
    SeasonFilter,
    SeasonInfo,
    SeasonListResponse,
    TeamFilter,
    TeamInfo,
    TeamListResponse,
    TournamentFilter,
    TournamentInfo,
    TournamentListResponse,
)


class QueryService(BaseService[Session]):
    """Service for querying database entities.

    Provides methods for:
    - Listing tournaments with optional name filtering
    - Listing seasons with tournament/season name filtering
    - Listing teams with optional name filtering
    - Listing matches with comprehensive filtering options

    Attributes:
        session: SQLAlchemy database session for queries
        logger: Logger instance for this service
    """

    def __init__(self, session: Session) -> None:
        """Initialize the service with a database session.

        Args:
            session: SQLAlchemy database session
        """
        super().__init__(session)
        self.logger = get_logger("services.query")

    def list_tournaments(self, filter: TournamentFilter) -> TournamentListResponse:
        """List tournaments with optional filtering.

        Args:
            filter: Filter criteria for tournaments

        Returns:
            TournamentListResponse with list of tournaments

        Raises:
            DatabaseQueryError: If query execution fails
        """
        self.logger.info(
            "Listing tournaments",
            extra={"operation": "list_tournaments", "filter_name": filter.name},
        )

        try:
            query = self.session.query(Tournament)

            # Apply name filter if provided (case-insensitive)
            if filter.name:
                query = query.filter(Tournament.name.ilike(f"%{filter.name}%"))

            # Order by country and name
            query = query.order_by(Tournament.country, Tournament.name)

            # Apply limit
            query = query.limit(filter.limit)

            tournaments = query.all()

            # Build response with seasons count
            tournament_infos = []
            for t in tournaments:
                seasons_count = (
                    self.session.query(func.count(Season.id))
                    .filter(Season.tournament_id == t.id)
                    .scalar()
                    or 0
                )
                tournament_infos.append(
                    TournamentInfo(
                        id=t.id,
                        name=t.name,
                        url_slug=t.url_slug,
                        seasons_count=seasons_count,
                    )
                )

            self.logger.info(
                "Tournaments retrieved",
                extra={
                    "operation": "list_tournaments",
                    "count": len(tournament_infos),
                },
            )

            return TournamentListResponse(tournaments=tournament_infos)

        except DataError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to list tournaments",
                extra={"operation": "list_tournaments", "error": str(e)},
            )
            raise DatabaseQueryError(
                f"Failed to list tournaments: {e}",
                details={"error_type": type(e).__name__, "filter": filter.name},
            ) from e

    def list_seasons(self, filter: SeasonFilter) -> SeasonListResponse:
        """List seasons with optional filtering.

        Args:
            filter: Filter criteria for seasons

        Returns:
            SeasonListResponse with list of seasons

        Raises:
            DataNotFoundError: If tournament not found when filtering by tournament
            DatabaseQueryError: If query execution fails
        """
        self.logger.info(
            "Listing seasons",
            extra={
                "operation": "list_seasons",
                "tournament_name": filter.tournament_name,
                "season_name": filter.season_name,
            },
        )

        try:
            query = self.session.query(Season).join(Tournament)

            # Apply tournament name filter if provided
            if filter.tournament_name:
                tournament = (
                    self.session.query(Tournament)
                    .filter(Tournament.name.ilike(f"%{filter.tournament_name}%"))
                    .first()
                )
                if not tournament:
                    raise DataNotFoundError(
                        f"Tournament not found: {filter.tournament_name}",
                        details={"tournament_name": filter.tournament_name},
                    )
                query = query.filter(Season.tournament_id == tournament.id)

            # Apply season name filter if provided
            if filter.season_name:
                query = query.filter(Season.name.ilike(f"%{filter.season_name}%"))

            # Order by tournament name and season name
            query = query.order_by(Tournament.name, Season.name)

            seasons = query.all()

            # Build response with tournament name and matches count
            season_infos = []
            for s in seasons:
                matches_count = (
                    self.session.query(func.count(Match.id))
                    .filter(Match.season_id == s.id)
                    .scalar()
                    or 0
                )
                season_infos.append(
                    SeasonInfo(
                        id=s.id,
                        name=s.name,
                        tournament_name=s.tournament.name,
                        matches_count=matches_count,
                    )
                )

            self.logger.info(
                "Seasons retrieved",
                extra={"operation": "list_seasons", "count": len(season_infos)},
            )

            return SeasonListResponse(seasons=season_infos)

        except DataNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to list seasons",
                extra={"operation": "list_seasons", "error": str(e)},
            )
            raise DatabaseQueryError(
                f"Failed to list seasons: {e}",
                details={
                    "error_type": type(e).__name__,
                    "tournament_name": filter.tournament_name,
                    "season_name": filter.season_name,
                },
            ) from e

    def list_teams(self, filter: TeamFilter) -> TeamListResponse:
        """List teams with optional filtering.

        Args:
            filter: Filter criteria for teams

        Returns:
            TeamListResponse with list of teams
        """
        self.logger.info(
            "Listing teams",
            extra={"operation": "list_teams", "filter_name": filter.name},
        )

        try:
            query = self.session.query(Team)

            # Apply name filter if provided (case-insensitive)
            if filter.name:
                query = query.filter(Team.name.ilike(f"%{filter.name}%"))

            # Order by name
            query = query.order_by(Team.name)

            # Apply limit
            query = query.limit(filter.limit)

            teams = query.all()

            # Build response with matches played count
            team_infos = []
            for t in teams:
                home_matches = (
                    self.session.query(func.count(Match.id))
                    .filter(Match.home_team_id == t.id)
                    .scalar()
                    or 0
                )
                away_matches = (
                    self.session.query(func.count(Match.id))
                    .filter(Match.away_team_id == t.id)
                    .scalar()
                    or 0
                )
                team_infos.append(
                    TeamInfo(
                        id=t.id,
                        name=t.name,
                        matches_played=home_matches + away_matches,
                    )
                )

            self.logger.info(
                "Teams retrieved",
                extra={"operation": "list_teams", "count": len(team_infos)},
            )

            return TeamListResponse(teams=team_infos)

        except Exception as e:
            self.logger.error(
                "Failed to list teams",
                extra={"operation": "list_teams", "error": str(e)},
            )
            raise DatabaseQueryError(
                f"Failed to list teams: {e}",
                details={"error_type": type(e).__name__, "filter": filter.name},
            ) from e

    def list_matches(self, filter: MatchFilter) -> MatchListResponse:
        """List matches with optional filtering.

        Args:
            filter: Filter criteria for matches

        Returns:
            MatchListResponse with list of matches

        Raises:
            DataNotFoundError: If tournament/season/team not found when filtering
            DatabaseQueryError: If query execution fails
        """
        self.logger.info(
            "Listing matches",
            extra={
                "operation": "list_matches",
                "tournament_name": filter.tournament_name,
                "season_name": filter.season_name,
                "team_name": filter.team_name,
                "status": filter.status,
            },
        )

        try:
            query = self.session.query(Match).join(Tournament).join(Season)

            # Apply tournament name filter if provided
            if filter.tournament_name:
                tournament = (
                    self.session.query(Tournament)
                    .filter(Tournament.name.ilike(f"%{filter.tournament_name}%"))
                    .first()
                )
                if not tournament:
                    raise DataNotFoundError(
                        f"Tournament not found: {filter.tournament_name}",
                        details={"tournament_name": filter.tournament_name},
                    )
                query = query.filter(Match.tournament_id == tournament.id)

            # Apply season name filter if provided
            if filter.season_name:
                season = (
                    self.session.query(Season)
                    .filter(Season.name.ilike(f"%{filter.season_name}%"))
                    .first()
                )
                if not season:
                    raise DataNotFoundError(
                        f"Season not found: {filter.season_name}",
                        details={"season_name": filter.season_name},
                    )
                query = query.filter(Match.season_id == season.id)

            # Apply team name filter if provided (home or away)
            if filter.team_name:
                team = (
                    self.session.query(Team)
                    .filter(Team.name.ilike(f"%{filter.team_name}%"))
                    .first()
                )
                if not team:
                    raise DataNotFoundError(
                        f"Team not found: {filter.team_name}",
                        details={"team_name": filter.team_name},
                    )
                query = query.filter(
                    or_(Match.home_team_id == team.id, Match.away_team_id == team.id)
                )

            # Apply status filter if provided
            if filter.status:
                query = query.filter(Match.status == filter.status)

            # Get total count before applying limit
            total_count = query.count()

            # Order by match date (most recent first)
            query = query.order_by(Match.match_date.desc())

            # Apply limit
            query = query.limit(filter.limit)

            matches = query.all()

            # Build response with team names
            match_infos = []
            for m in matches:
                match_infos.append(
                    MatchInfo(
                        id=m.id,
                        home_team=m.home_team.name,
                        away_team=m.away_team.name,
                        match_date=m.match_date,
                        status=m.status,
                        home_score=m.home_score,
                        away_score=m.away_score,
                        tournament_name=m.tournament.name,
                        season_name=m.season.name,
                    )
                )

            self.logger.info(
                "Matches retrieved",
                extra={
                    "operation": "list_matches",
                    "count": len(match_infos),
                    "total_count": total_count,
                },
            )

            return MatchListResponse(matches=match_infos, total_count=total_count)

        except DataNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to list matches",
                extra={"operation": "list_matches", "error": str(e)},
            )
            raise DatabaseQueryError(
                f"Failed to list matches: {e}",
                details={
                    "error_type": type(e).__name__,
                    "tournament_name": filter.tournament_name,
                    "season_name": filter.season_name,
                    "team_name": filter.team_name,
                    "status": filter.status,
                },
            ) from e
