"""SQL queries and repository for match data access."""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session, joinedload

from algobet.matches.models import Match
from algobet.predictions.data.historical_provider import HistoricalMatchProvider


@dataclass
class TeamStandings:
    """Computed standings for a team at a given point in the season."""

    team_id: int
    as_of: datetime
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
        self._team_matches_asc_cache: dict[int, list[Match]] = {}
        self._team_match_dates_asc_cache: dict[int, list[datetime]] = {}
        self._h2h_cache: dict[tuple[int, int], list[Match]] = {}
        self._h2h_asc_cache: dict[tuple[int, int], list[Match]] = {}
        self._h2h_dates_asc_cache: dict[tuple[int, int], list[datetime]] = {}
        # Standings cache keyed by (tournament_id, season_id)
        self._standings_cache: dict[
            tuple[int, int], dict[int, list[TeamStandings]]
        ] = {}
        self._historical_provider = HistoricalMatchProvider(session)

    @staticmethod
    def _normalize_before_date(value: datetime | object | None) -> datetime | None:
        if value is not None and hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        return None

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
        self._team_matches_asc_cache = {
            team_id: list(reversed(matches)) for team_id, matches in by_team.items()
        }
        self._team_match_dates_asc_cache = {
            team_id: [match.match_date for match in matches]
            for team_id, matches in self._team_matches_asc_cache.items()
        }

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
        self._h2h_asc_cache = {
            pair: list(reversed(matches)) for pair, matches in by_pair.items()
        }
        self._h2h_dates_asc_cache = {
            pair: [match.match_date for match in matches]
            for pair, matches in self._h2h_asc_cache.items()
        }

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

        # Build exact (tournament_id, season_id) pair conditions to avoid
        # a Cartesian product that would load unintended cross-pair matches.
        pair_conditions = [
            and_(Match.tournament_id == tid, Match.season_id == sid)
            for tid, sid in unique_pairs
        ]

        stmt = (
            select(Match)
            .where(
                and_(
                    or_(*pair_conditions),
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

            # After each match, compute current standings for all teams.
            # Use the full set of teams that appear anywhere in the season
            # so that early-season snapshots have the correct league size.
            all_season_teams = {m.home_team_id for m in matches} | {
                m.away_team_id for m in matches
            }
            n_teams = len(all_season_teams) if all_season_teams else 1

            # Build sorted standings list (only teams that have played so far)
            ranked = sorted(
                team_stats.keys(),
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
                        as_of=match.match_date,
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

        eligible_snapshots = [
            standings for standings in team_history if standings.as_of < before_date
        ]
        if not eligible_snapshots:
            return None

        return eligible_snapshots[-1]

    def clear_cache(self) -> None:
        """Clear in-memory caches."""
        self._team_matches_cache.clear()
        self._team_matches_asc_cache.clear()
        self._team_match_dates_asc_cache.clear()
        self._h2h_cache.clear()
        self._h2h_asc_cache.clear()
        self._h2h_dates_asc_cache.clear()
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
        return self._historical_provider.get_historical_matches(
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
            before_date = self._normalize_before_date(before_date)
            asc_matches = self._team_matches_asc_cache.get(team_id, [])
            asc_dates = self._team_match_dates_asc_cache.get(team_id, [])
            end = (
                bisect_left(asc_dates, before_date) if before_date else len(asc_matches)
            )
            result: list[Match] = []
            for match in reversed(asc_matches[:end]):
                if home_only and match.home_team_id != team_id:
                    continue
                if away_only and match.away_team_id != team_id:
                    continue

                result.append(match)
                if len(result) >= limit:
                    break

            return result

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
        db_result = self.session.execute(stmt)
        return list(db_result.unique().scalars().all())

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
            before_date = self._normalize_before_date(before_date)
            asc_matches = self._h2h_asc_cache.get(cache_key, [])
            asc_dates = self._h2h_dates_asc_cache.get(cache_key, [])
            end = (
                bisect_left(asc_dates, before_date) if before_date else len(asc_matches)
            )
            result: list[Match] = []
            for match in reversed(asc_matches[:end]):
                result.append(match)
                if len(result) >= limit:
                    break
            return result

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
        db_result = self.session.execute(stmt)
        return list(db_result.scalars().all())

    def get_match_count(self, team_id: int, before_date: datetime) -> int:
        """Get count of matches played by a team before a given date.

        Args:
            team_id: ID of the team
            before_date: Count matches before this date

        Returns:
            Number of matches played
        """
        if team_id in self._team_matches_cache:
            normalized_date = self._normalize_before_date(before_date)
            if normalized_date is None:
                return 0
            dates = self._team_match_dates_asc_cache.get(team_id, [])
            return bisect_left(dates, normalized_date)

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
