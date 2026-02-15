"""Async query service for database entity queries.

This module provides async business logic for querying tournaments, seasons,
teams, and matches from the database.
"""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from algobet.exceptions import DatabaseQueryError, DataError, DataNotFoundError
from algobet.logging_config import get_logger
from algobet.models import Match, Season, Team, Tournament
from algobet.services.async_base import AsyncBaseService
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


class AsyncQueryService(AsyncBaseService[AsyncSession]):
    """Async service for querying database entities.

    Provides methods for:
    - Listing tournaments with optional name filtering
    - Listing seasons with tournament/season name filtering
    - Listing teams with optional name filtering
    - Listing matches with comprehensive filtering options

    Attributes:
        session: SQLAlchemy async database session for queries
        logger: Logger instance for this service
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service with an async database session.

        Args:
            session: SQLAlchemy async database session
        """
        super().__init__(session)
        self.logger = get_logger("services.async_query")

    async def list_tournaments(
        self, filter: TournamentFilter
    ) -> TournamentListResponse:
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
            query = select(Tournament)

            # Apply name filter if provided (case-insensitive)
            if filter.name:
                query = query.where(Tournament.name.ilike(f"%{filter.name}%"))

            # Order by country and name
            query = query.order_by(Tournament.country, Tournament.name)

            # Apply limit
            query = query.limit(filter.limit)

            result = await self.session.execute(query)
            tournaments = result.scalars().all()

            # Build response with seasons count
            tournament_infos = []
            for t in tournaments:
                seasons_count_query = select(func.count(Season.id)).where(
                    Season.tournament_id == t.id
                )
                seasons_result = await self.session.execute(seasons_count_query)
                seasons_count = seasons_result.scalar() or 0

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

    async def list_seasons(self, filter: SeasonFilter) -> SeasonListResponse:
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
            query = select(Season).join(Tournament)

            # Apply tournament name filter if provided
            if filter.tournament_name:
                tournament_query = select(Tournament).where(
                    Tournament.name.ilike(f"%{filter.tournament_name}%")
                )
                t_result = await self.session.execute(tournament_query)
                tournament = t_result.scalar_one_or_none()

                if not tournament:
                    raise DataNotFoundError(
                        f"Tournament not found: {filter.tournament_name}",
                        details={"tournament_name": filter.tournament_name},
                    )
                query = query.where(Season.tournament_id == tournament.id)

            # Apply season name filter if provided
            if filter.season_name:
                query = query.where(Season.name.ilike(f"%{filter.season_name}%"))

            # Order by tournament name and season name
            query = query.order_by(Tournament.name, Season.name)

            result = await self.session.execute(query)
            seasons = result.scalars().all()

            # Build response with tournament name and matches count
            season_infos = []
            for s in seasons:
                matches_count_query = select(func.count(Match.id)).where(
                    Match.season_id == s.id
                )
                m_result = await self.session.execute(matches_count_query)
                matches_count = m_result.scalar() or 0

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

    async def list_teams(self, filter: TeamFilter) -> TeamListResponse:
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
            query = select(Team)

            # Apply name filter if provided (case-insensitive)
            if filter.name:
                query = query.where(Team.name.ilike(f"%{filter.name}%"))

            # Order by name
            query = query.order_by(Team.name)

            # Apply limit
            query = query.limit(filter.limit)

            result = await self.session.execute(query)
            teams = result.scalars().all()

            # Build response with matches played count
            team_infos = []
            for t in teams:
                home_matches_query = select(func.count(Match.id)).where(
                    Match.home_team_id == t.id
                )
                hm_result = await self.session.execute(home_matches_query)
                home_matches = hm_result.scalar() or 0

                away_matches_query = select(func.count(Match.id)).where(
                    Match.away_team_id == t.id
                )
                am_result = await self.session.execute(away_matches_query)
                away_matches = am_result.scalar() or 0

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

    async def list_matches(self, filter: MatchFilter) -> MatchListResponse:
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
            query = select(Match).options(
                joinedload(Match.tournament),
                joinedload(Match.season),
                joinedload(Match.home_team),
                joinedload(Match.away_team),
            )

            # Apply tournament name filter if provided
            if filter.tournament_name:
                t_query = select(Tournament).where(
                    Tournament.name.ilike(f"%{filter.tournament_name}%")
                )
                t_result = await self.session.execute(t_query)
                tournament = t_result.scalar_one_or_none()

                if not tournament:
                    raise DataNotFoundError(
                        f"Tournament not found: {filter.tournament_name}",
                        details={"tournament_name": filter.tournament_name},
                    )
                query = query.where(Match.tournament_id == tournament.id)

            # Apply season name filter if provided
            if filter.season_name:
                s_query = select(Season).where(
                    Season.name.ilike(f"%{filter.season_name}%")
                )
                s_result = await self.session.execute(s_query)
                season = s_result.scalar_one_or_none()

                if not season:
                    raise DataNotFoundError(
                        f"Season not found: {filter.season_name}",
                        details={"season_name": filter.season_name},
                    )
                query = query.where(Match.season_id == season.id)

            # Apply team name filter if provided (home or away)
            if filter.team_name:
                team_query = select(Team).where(
                    Team.name.ilike(f"%{filter.team_name}%")
                )
                team_result = await self.session.execute(team_query)
                team = team_result.scalar_one_or_none()

                if not team:
                    raise DataNotFoundError(
                        f"Team not found: {filter.team_name}",
                        details={"team_name": filter.team_name},
                    )
                query = query.where(
                    or_(Match.home_team_id == team.id, Match.away_team_id == team.id)
                )

            # Apply status filter if provided
            if filter.status:
                query = query.where(Match.status == filter.status)

            # Get total count before applying limit
            count_query = select(func.count()).select_from(query.subquery())
            count_result = await self.session.execute(count_query)
            total_count = count_result.scalar() or 0

            # Order by match date (most recent first)
            query = query.order_by(Match.match_date.desc())

            # Apply limit
            query = query.limit(filter.limit)

            result = await self.session.execute(query)
            matches = result.scalars().unique().all()

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
