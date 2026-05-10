"""Temporal feature generator."""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class TemporalFeatureGenerator(FeatureGenerator):
    """Generate temporal features from match dates.

    Produces features related to timing, rest days, fixture congestion,
    calendar effects, and season period (start, mid, end).
    """

    def __init__(
        self,
        include_rest_days: bool = True,
        include_fixture_density: bool = True,
        include_season_period: bool = True,
        season_length_days: int = 300,
    ) -> None:
        """Initialize temporal generator.

        Args:
            include_rest_days: Calculate rest days for each team
            include_fixture_density: Calculate fixture congestion
            include_season_period: Calculate season period features
            season_length_days: Expected season length in days (default: 300)
        """
        self.include_rest_days = include_rest_days
        self.include_fixture_density = include_fixture_density
        self.include_season_period = include_season_period
        self.season_length_days = season_length_days

    @property
    def name(self) -> str:
        return "temporal"

    @property
    def feature_names(self) -> list[str]:
        names = [
            "day_of_week",
            "month",
            "is_weekend",
            "days_from_season_start",
        ]

        if self.include_season_period:
            names.extend(
                [
                    "season_progress",
                    "is_season_start",
                    "is_season_mid",
                    "is_season_late",
                    "is_season_end",
                ]
            )

        if self.include_rest_days:
            names.extend(
                [
                    "home_rest_days",
                    "away_rest_days",
                    "rest_days_diff",
                ]
            )

        if self.include_fixture_density:
            names.extend(
                [
                    "home_matches_last_14_days",
                    "away_matches_last_14_days",
                ]
            )

        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate temporal features for each match.

        Args:
            matches: Match data
            repository: Repository for historical queries

        Returns:
            DataFrame with temporal features indexed by match id
        """
        features = []
        history_cache: dict[tuple[int, datetime], list[Any]] = {}
        history_limit = 20 if self.include_fixture_density else 1

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])
            needs_history = self.include_rest_days or self.include_fixture_density
            home_history = (
                self._get_cached_history(
                    repository,
                    history_cache,
                    home_team_id,
                    match_date,
                    history_limit,
                )
                if needs_history
                else []
            )
            away_history = (
                self._get_cached_history(
                    repository,
                    history_cache,
                    away_team_id,
                    match_date,
                    history_limit,
                )
                if needs_history
                else []
            )

            match_features: dict[str, Any] = {"match_id": match_id}

            # Calendar features
            match_features["day_of_week"] = match_date.dayofweek
            match_features["month"] = match_date.month
            match_features["is_weekend"] = float(match_date.dayofweek >= 5)

            # Approximate season start (August 1)
            season_start = datetime(
                match_date.year - 1 if match_date.month < 8 else match_date.year, 8, 1
            )
            days_from_start = (match_date - season_start).days
            match_features["days_from_season_start"] = days_from_start

            # Season period features
            if self.include_season_period:
                progress = min(1.0, days_from_start / self.season_length_days)
                match_features["season_progress"] = progress
                match_features["is_season_start"] = float(progress <= 0.25)
                match_features["is_season_mid"] = float(0.25 < progress <= 0.50)
                match_features["is_season_late"] = float(0.50 < progress <= 0.75)
                match_features["is_season_end"] = float(progress > 0.75)

            # Rest days
            if self.include_rest_days:
                home_rest = self._get_rest_days_from_history(
                    home_history,
                    match_date,
                )
                away_rest = self._get_rest_days_from_history(
                    away_history,
                    match_date,
                )
                match_features["home_rest_days"] = home_rest
                match_features["away_rest_days"] = away_rest
                match_features["rest_days_diff"] = home_rest - away_rest

            # Fixture density
            if self.include_fixture_density:
                match_features["home_matches_last_14_days"] = (
                    self._count_recent_matches_from_history(
                        home_history,
                        match_date,
                        days=14,
                    )
                )
                match_features["away_matches_last_14_days"] = (
                    self._count_recent_matches_from_history(
                        away_history,
                        match_date,
                        days=14,
                    )
                )

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _get_rest_days(
        self,
        repo: MatchRepository,
        team_id: int,
        match_date: datetime,
    ) -> float:
        """Get days since last match."""
        last_matches = repo.get_team_matches(
            team_id=team_id,
            before_date=match_date,
            limit=1,
        )
        return self._get_rest_days_from_history(last_matches, match_date)

    def _get_rest_days_from_history(
        self,
        matches: list[Any],
        match_date: datetime,
    ) -> float:
        """Get days since last match from an already-fetched history."""
        if not matches:
            return 7.0  # Default: week of rest

        last_match = matches[0]
        rest_days = (match_date - last_match.match_date).days
        return float(min(rest_days, 14))  # Cap at 2 weeks

    def _count_recent_matches(
        self,
        repo: MatchRepository,
        team_id: int,
        match_date: datetime,
        days: int = 14,
    ) -> int:
        """Count matches in recent period."""
        matches = repo.get_team_matches(
            team_id=team_id,
            before_date=match_date,
            limit=20,
        )
        return self._count_recent_matches_from_history(matches, match_date, days)

    @staticmethod
    def _count_recent_matches_from_history(
        matches: list[Any],
        match_date: datetime,
        days: int = 14,
    ) -> int:
        """Count matches in the recent period from an already-fetched history."""
        period_start = match_date - timedelta(days=days)
        recent_count = sum(
            1 for match in matches if period_start <= match.match_date < match_date
        )
        return min(
            recent_count,
            5,
        )

    @staticmethod
    def _get_cached_history(
        repo: MatchRepository,
        cache: dict[tuple[int, datetime], list[Any]],
        team_id: int,
        match_date: datetime,
        limit: int,
    ) -> list[Any]:
        """Fetch a team's recent history once for each team/date pair."""
        cache_key = (team_id, match_date.to_pydatetime())
        if cache_key not in cache:
            cache[cache_key] = repo.get_team_matches(
                team_id=team_id,
                before_date=match_date,
                limit=limit,
            )
        return cache[cache_key]
