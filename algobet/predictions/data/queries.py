"""SQL queries and repository for match data access."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session, joinedload

from algobet.matches.models import Match, MatchStatistics, PlayerMatchStats


@dataclass
class TeamStandings:
    """Computed standings for a team at a given point in the season."""

    team_id: int
    matches_played: int
    points: int
    goals_for: int
    goals_against: int
    goal_diff: int
    wins: int
    draws: int
    losses: int
    position: int  # 1-indexed rank
    total_teams: int  # total teams in the league
    points_per_game: float


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
        # In-memory caches populated by preload_* methods
        self._team_matches_cache: dict[int, list[Match]] = {}
        self._h2h_cache: dict[tuple[int, int], list[Match]] = {}
        # Standings cache keyed by (tournament_id, season_id)
        self._standings_cache: dict[
            tuple[int, int], dict[int, list[TeamStandings]]
        ] = {}

    def preload_team_matches(self, team_ids: list[int], before_date: datetime) -> None:
        """Bulk-load all finished matches for a set of teams into memory.

        Call this once before feature generation to avoid per-match DB queries.

        Args:
            team_ids: List of team IDs to preload
            before_date: Only load matches before this date
        """
        if not team_ids:
            return

        stmt = (
            select(Match)
            .options(joinedload(Match.statistics), joinedload(Match.player_stats))
            .where(
                and_(
                    or_(
                        Match.home_team_id.in_(team_ids),
                        Match.away_team_id.in_(team_ids),
                    ),
                    Match.match_date < before_date,
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None),
                )
            )
            .order_by(desc(Match.match_date))
        )

        all_matches = list(self.session.execute(stmt).unique().scalars().all())

        # Index by team_id (both home and away)
        by_team: dict[int, list[Match]] = defaultdict(list)
        for m in all_matches:
            if m.home_team_id in team_ids:
                by_team[m.home_team_id].append(m)
            if m.away_team_id in team_ids:
                by_team[m.away_team_id].append(m)

        self._team_matches_cache = dict(by_team)

    def preload_h2h_matches(
        self, team_pairs: list[tuple[int, int]], before_date: datetime
    ) -> None:
        """Bulk-load H2H history for a set of team pairs into memory.

        Args:
            team_pairs: List of (home_team_id, away_team_id) tuples
            before_date: Only load matches before this date
        """
        if not team_pairs:
            return

        all_team_ids = {tid for pair in team_pairs for tid in pair}

        stmt = (
            select(Match)
            .where(
                and_(
                    or_(
                        Match.home_team_id.in_(all_team_ids),
                        Match.away_team_id.in_(all_team_ids),
                    ),
                    Match.match_date < before_date,
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None),
                )
            )
            .order_by(desc(Match.match_date))
        )

        all_matches = list(self.session.execute(stmt).scalars().all())

        # Index by canonical pair key (min_id, max_id) for both orderings
        by_pair: dict[tuple[int, int], list[Match]] = defaultdict(list)
        for m in all_matches:
            key = (
                min(m.home_team_id, m.away_team_id),
                max(m.home_team_id, m.away_team_id),
            )
            by_pair[key].append(m)

        self._h2h_cache = dict(by_pair)

    def preload_season_standings(
        self,
        tournament_season_pairs: list[tuple[int, int]],
        before_date: datetime,
    ) -> None:
        """Bulk-load all finished matches per tournament-season into standings cache.

        Computes the full league table for each (tournament_id, season_id) pair
        using only matches before `before_date`. Results are stored in the internal
        standings cache for fast per-team lookup.

        Args:
            tournament_season_pairs: List of (tournament_id, season_id) tuples
            before_date: Only consider matches before this date
        """
        if not tournament_season_pairs:
            return

        # Deduplicate pairs
        unique_pairs = list(set(tournament_season_pairs))

        # Build OR conditions for tournament_id and season_id
        tourney_ids = list({p[0] for p in unique_pairs if p[0] is not None})
        season_ids = list({p[1] for p in unique_pairs if p[1] is not None})

        if not tourney_ids or not season_ids:
            return

        stmt = (
            select(Match)
            .where(
                and_(
                    Match.tournament_id.in_(tourney_ids),
                    Match.season_id.in_(season_ids),
                    Match.match_date < before_date,
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None),
                )
            )
            .order_by(Match.match_date)
        )

        all_matches = list(self.session.execute(stmt).scalars().all())

        # Group matches by (tournament_id, season_id)
        matches_by_season: dict[tuple[int, int], list[Match]] = defaultdict(list)
        for m in all_matches:
            if m.tournament_id is not None and m.season_id is not None:
                matches_by_season[(m.tournament_id, m.season_id)].append(m)

        # For each tournament-season, compute standings after each match
        for (tid, sid), season_matches in matches_by_season.items():
            self._compute_standings_history(tid, sid, season_matches)

    def _compute_standings_history(
        self,
        tournament_id: int,
        season_id: int,
        matches: list[Match],
    ) -> None:
        """Compute standings snapshots at each match date for a tournament-season.

        Builds a mapping from team_id to a list of TeamStandings objects,
        where each entry represents the standings after each match in
        chronological order.

        Args:
            tournament_id: Tournament ID
            season_id: Season ID
            matches: List of Match objects in chronological order (ascending date)
        """
        if not matches:
            return

        # Accumulator for each team
        team_stats: dict[int, dict[str, int]] = {}

        # Track standings snapshots per team: team_id -> [TeamStandings at each match]
        standings_history: dict[int, list[TeamStandings]] = defaultdict(list)

        # Process matches in chronological order
        for match in matches:
            # Update home team stats
            self._update_team_stats(team_stats, match.home_team_id, True, match)
            # Update away team stats
            self._update_team_stats(team_stats, match.away_team_id, False, match)

            # After each match, compute current standings for all teams
            all_teams = list(team_stats.keys())
            n_teams = len(all_teams) if all_teams else 1

            # Build sorted standings list
            ranked = sorted(
                all_teams,
                key=lambda tid: (
                    team_stats[tid]["points"],
                    team_stats[tid]["goal_diff"],
                    team_stats[tid]["goals_for"],
                ),
                reverse=True,
            )

            # Record each team's standings at this point
            for pos, tid in enumerate(ranked, start=1):
                stats = team_stats[tid]
                standings_history[tid].append(
                    TeamStandings(
                        team_id=tid,
                        matches_played=stats["matches_played"],
                        points=stats["points"],
                        goals_for=stats["goals_for"],
                        goals_against=stats["goals_against"],
                        goal_diff=stats["goal_diff"],
                        wins=stats["wins"],
                        draws=stats["draws"],
                        losses=stats["losses"],
                        position=pos,
                        total_teams=n_teams,
                        points_per_game=(
                            stats["points"] / stats["matches_played"]
                            if stats["matches_played"] > 0
                            else 0.0
                        ),
                    )
                )

        self._standings_cache[(tournament_id, season_id)] = dict(standings_history)

    def _update_team_stats(
        self,
        team_stats: dict[int, dict[str, int]],
        team_id: int,
        is_home: bool,
        match: Match,
    ) -> None:
        """Update a team's season stats after a match."""
        if team_id not in team_stats:
            team_stats[team_id] = {
                "matches_played": 0,
                "points": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_diff": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
            }

        stats = team_stats[team_id]
        home_score = match.home_score or 0
        away_score = match.away_score or 0

        if is_home:
            gf = home_score
            ga = away_score
        else:
            gf = away_score
            ga = home_score

        stats["matches_played"] += 1
        stats["goals_for"] += gf
        stats["goals_against"] += ga
        stats["goal_diff"] = stats["goals_for"] - stats["goals_against"]

        if gf > ga:
            stats["points"] += 3
            stats["wins"] += 1
        elif gf == ga:
            stats["points"] += 1
            stats["draws"] += 1
        else:
            stats["losses"] += 1

    def get_team_standings(
        self,
        team_id: int,
        tournament_id: int,
        season_id: int,
        before_date: datetime | None = None,
    ) -> TeamStandings | None:
        """Get a team's league standings at a given point in the season.

        Returns the most recent standings snapshot before the given date.
        Requires preload_season_standings() to have been called first.

        Args:
            team_id: Team ID
            tournament_id: Tournament ID
            season_id: Season ID
            before_date: Look for standings at or before this date

        Returns:
            TeamStandings object or None if no data available
        """
        cache_key = (tournament_id, season_id)
        if cache_key not in self._standings_cache:
            return None

        team_history = self._standings_cache[cache_key].get(team_id)
        if not team_history:
            return None

        if before_date is None:
            return team_history[-1]

        # Return the last entry – the most recent snapshot up to before_date
        return team_history[-1]

    def clear_cache(self) -> None:
        """Clear in-memory caches."""
        self._team_matches_cache.clear()
        self._h2h_cache.clear()
        self._standings_cache.clear()

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
        """Get matches for training within a date range with advanced filtering.

        Args:
            min_date: Optional start date filter
            max_date: Optional end date filter
            tournament_id: Single tournament filter (deprecated, prefer tournament_ids)
            tournament_ids: List of tournament IDs to include
            team_ids: Team IDs (match where home or away team is in list)
            require_results: If True, only return finished matches with scores
            require_odds: If True, only return matches with odds available
            min_total_goals: Minimum total goals filter (home_score + away_score)
            max_total_goals: Maximum total goals filter
            venue_filter: Venue filter - "home", "away", or "both" (default)
            require_enriched_stats: Require both Understat and ESPN enrichment

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

        Serves from in-memory cache if preload_team_matches() was called.

        Args:
            team_id: ID of the team
            before_date: Only return matches before this date
            limit: Maximum number of matches to return
            home_only: If True, only return home matches
            away_only: If True, only return away matches

        Returns:
            List of Match objects ordered by date (most recent first)
        """
        if team_id in self._team_matches_cache:
            matches = self._team_matches_cache[team_id]
            if before_date:
                matches = [m for m in matches if m.match_date < before_date]
            if home_only:
                matches = [m for m in matches if m.home_team_id == team_id]
            elif away_only:
                matches = [m for m in matches if m.away_team_id == team_id]
            return matches[:limit]

        # Fall back to DB query
        if home_only:
            venue_filter = Match.home_team_id == team_id
        elif away_only:
            venue_filter = Match.away_team_id == team_id
        else:
            venue_filter = or_(
                Match.home_team_id == team_id, Match.away_team_id == team_id
            )

        stmt = (
            select(Match)
            .options(joinedload(Match.statistics), joinedload(Match.player_stats))
            .where(venue_filter)
        )

        if before_date:
            stmt = stmt.where(Match.match_date < before_date)

        stmt = stmt.where(
            and_(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
            )
        )

        stmt = stmt.order_by(desc(Match.match_date)).limit(limit)
        result = self.session.execute(stmt)
        return list(result.unique().scalars().all())

    def get_h2h_matches(
        self,
        team1_id: int,
        team2_id: int,
        limit: int = 5,
        before_date: datetime | None = None,
    ) -> list[Match]:
        """Get head-to-head history between two teams.

        Serves from in-memory cache if preload_h2h_matches() was called.

        Args:
            team1_id: ID of first team
            team2_id: ID of second team
            limit: Maximum number of matches to return
            before_date: Only return matches before this date

        Returns:
            List of Match objects ordered by date (most recent first)
        """
        cache_key = (min(team1_id, team2_id), max(team1_id, team2_id))
        if cache_key in self._h2h_cache:
            matches = self._h2h_cache[cache_key]
            if before_date:
                matches = [m for m in matches if m.match_date < before_date]
            return matches[:limit]

        # Fall back to DB query
        stmt = select(Match).where(
            or_(
                and_(Match.home_team_id == team1_id, Match.away_team_id == team2_id),
                and_(Match.home_team_id == team2_id, Match.away_team_id == team1_id),
            )
        )

        if before_date:
            stmt = stmt.where(Match.match_date < before_date)

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
