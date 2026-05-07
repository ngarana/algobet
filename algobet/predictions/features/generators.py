"""Feature generators for match prediction.

This module provides feature generators that transform raw match data into
ML-ready features. Each generator produces a specific category of features
following the scikit-learn transformer pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from algobet.predictions.data.queries import MatchRepository


@dataclass
class FeatureSchema:
    """Schema definition for a set of features."""

    version: str
    features: dict[str, type]
    description: str | None = None

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Validate that dataframe contains expected features.

        Args:
            df: DataFrame to validate

        Returns:
            List of missing feature names
        """
        missing = []
        for name, _dtype in self.features.items():
            if name not in df.columns:
                missing.append(name)
        return missing

    def get_feature_names(self) -> list[str]:
        """Return list of feature names."""
        return list(self.features.keys())


class FeatureGenerator(ABC):
    """Abstract base class for feature generators.

    Feature generators transform raw match data into feature vectors.
    Each generator produces a specific category of features.
    """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names this generator produces."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return generator name for identification."""

    @abstractmethod
    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate features for matches.

        Args:
            matches: DataFrame with match records, must include:
                - id: match identifier
                - home_team_id: home team ID
                - away_team_id: away team ID
                - match_date: match datetime
                - home_score, away_score: scores (may be None for upcoming)
                - odds_home, odds_draw, odds_away: betting odds (may be None)
            repository: MatchRepository for historical queries

        Returns:
            DataFrame indexed by match_id with generated features
        """

    def get_schema(self) -> FeatureSchema:
        """Return the feature schema for this generator."""
        return FeatureSchema(
            version="v1.0",
            features=dict.fromkeys(self.feature_names, float),
        )


class TeamFormGenerator(FeatureGenerator):
    """Generate team form features based on recent match history.

    Produces features capturing team performance over various time windows,
    including points, goals, win rates, and venue-specific form.
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        include_venue_specific: bool = True,
    ) -> None:
        """Initialize form generator.

        Args:
            window_sizes: List of match window sizes (default: [3, 5, 10])
            include_venue_specific: Whether to include home/away specific form
        """
        self.window_sizes = window_sizes or [3, 5, 10]
        self.include_venue_specific = include_venue_specific

    @property
    def name(self) -> str:
        return "team_form"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for w in self.window_sizes:
            # Home team form
            names.extend(
                [
                    f"home_points_last_{w}",
                    f"home_win_rate_{w}",
                    f"home_goals_for_avg_{w}",
                    f"home_goals_against_avg_{w}",
                    f"home_goal_diff_avg_{w}",
                    # Away team form
                    f"away_points_last_{w}",
                    f"away_win_rate_{w}",
                    f"away_goals_for_avg_{w}",
                    f"away_goals_against_avg_{w}",
                    f"away_goal_diff_avg_{w}",
                ]
            )

        if self.include_venue_specific:
            for w in self.window_sizes:
                names.extend(
                    [
                        f"home_home_form_{w}",
                        f"away_away_form_{w}",
                    ]
                )

        # Form momentum features
        names.extend(
            [
                "home_form_trend",
                "away_form_trend",
                "form_diff",
            ]
        )

        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate form features for each match.

        Args:
            matches: Match data
            repository: Repository for historical queries

        Returns:
            DataFrame with form features indexed by match id
        """
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            match_features: dict[str, Any] = {"match_id": match_id}

            for w in self.window_sizes:
                # Home team form
                home_form = self._calculate_form(
                    repository, home_team_id, match_date, w
                )
                match_features[f"home_points_last_{w}"] = home_form["avg_points"]
                match_features[f"home_win_rate_{w}"] = home_form["win_rate"]
                match_features[f"home_goals_for_avg_{w}"] = home_form["avg_goals_for"]
                match_features[f"home_goals_against_avg_{w}"] = home_form[
                    "avg_goals_against"
                ]
                match_features[f"home_goal_diff_avg_{w}"] = (
                    home_form["avg_goals_for"] - home_form["avg_goals_against"]
                )

                # Away team form
                away_form = self._calculate_form(
                    repository, away_team_id, match_date, w
                )
                match_features[f"away_points_last_{w}"] = away_form["avg_points"]
                match_features[f"away_win_rate_{w}"] = away_form["win_rate"]
                match_features[f"away_goals_for_avg_{w}"] = away_form["avg_goals_for"]
                match_features[f"away_goals_against_avg_{w}"] = away_form[
                    "avg_goals_against"
                ]
                match_features[f"away_goal_diff_avg_{w}"] = (
                    away_form["avg_goals_for"] - away_form["avg_goals_against"]
                )

                if self.include_venue_specific:
                    # Home team's form at home
                    home_home = self._calculate_venue_form(
                        repository, home_team_id, match_date, w, is_home=True
                    )
                    match_features[f"home_home_form_{w}"] = home_home

                    # Away team's form away
                    away_away = self._calculate_venue_form(
                        repository, away_team_id, match_date, w, is_home=False
                    )
                    match_features[f"away_away_form_{w}"] = away_away

            # Form trend (last 3 vs matches 4-6)
            match_features["home_form_trend"] = self._calculate_trend(
                repository, home_team_id, match_date
            )
            match_features["away_form_trend"] = self._calculate_trend(
                repository, away_team_id, match_date
            )
            match_features["form_diff"] = match_features.get(
                "home_points_last_5", 0
            ) - match_features.get("away_points_last_5", 0)

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _calculate_form(
        self,
        repo: MatchRepository,
        team_id: int,
        reference_date: datetime,
        n_matches: int,
    ) -> dict[str, float]:
        """Calculate form metrics for a team."""
        matches = repo.get_team_matches(
            team_id=team_id,
            before_date=reference_date,
            limit=n_matches,
        )

        if not matches:
            return {
                "avg_points": 0.0,
                "win_rate": 0.0,
                "avg_goals_for": 0.0,
                "avg_goals_against": 0.0,
            }

        total_points = 0
        wins = 0
        goals_for = 0
        goals_against = 0

        for match in matches:
            # Determine if team is home or away
            is_home = match.home_team_id == team_id

            if is_home:
                gf = match.home_score or 0
                ga = match.away_score or 0
            else:
                gf = match.away_score or 0
                ga = match.home_score or 0

            goals_for += gf
            goals_against += ga

            if gf > ga:
                total_points += 3
                wins += 1
            elif gf == ga:
                total_points += 1

        n = len(matches)
        return {
            "avg_points": total_points / n,
            "win_rate": wins / n,
            "avg_goals_for": goals_for / n,
            "avg_goals_against": goals_against / n,
        }

    def _calculate_venue_form(
        self,
        repo: MatchRepository,
        team_id: int,
        reference_date: datetime,
        n_matches: int,
        is_home: bool,
    ) -> float:
        """Calculate venue-specific form (home or away)."""
        matches = repo.get_team_matches(
            team_id=team_id,
            before_date=reference_date,
            limit=n_matches,
            home_only=is_home,
            away_only=not is_home,
        )

        if not matches:
            return 0.0

        total_points = 0
        for match in matches:
            if match.home_team_id == team_id:
                gf = match.home_score or 0
                ga = match.away_score or 0
            else:
                gf = match.away_score or 0
                ga = match.home_score or 0

            if gf > ga:
                total_points += 3
            elif gf == ga:
                total_points += 1

        return total_points / len(matches)

    def _calculate_trend(
        self,
        repo: MatchRepository,
        team_id: int,
        reference_date: datetime,
    ) -> float:
        """Calculate form trend (recent vs earlier)."""
        recent = repo.get_team_matches(
            team_id=team_id,
            before_date=reference_date,
            limit=3,
        )

        # Get matches 4-6 (if available)
        earlier = repo.get_team_matches(
            team_id=team_id,
            before_date=reference_date,
            limit=6,
        )

        if len(earlier) < 4:
            return 0.0

        def avg_points(match_list: list) -> float:
            if not match_list:
                return 0.0
            total = 0
            for match in match_list:
                if match.home_team_id == team_id:
                    gf = match.home_score or 0
                    ga = match.away_score or 0
                else:
                    gf = match.away_score or 0
                    ga = match.home_score or 0
                if gf > ga:
                    total += 3
                elif gf == ga:
                    total += 1
            return total / len(match_list)

        recent_avg = avg_points(recent[:3])
        earlier_avg = avg_points(earlier[3:6])

        return recent_avg - earlier_avg


class HeadToHeadGenerator(FeatureGenerator):
    """Generate head-to-head features between teams.

    Produces features capturing historical matchup statistics between
    the home and away teams.
    """

    def __init__(
        self,
        max_h2h_matches: int = 5,
        max_years_back: int = 3,
    ) -> None:
        """Initialize H2H generator.

        Args:
            max_h2h_matches: Maximum H2H matches to consider
            max_years_back: How many years back to look for H2H
        """
        self.max_h2h_matches = max_h2h_matches
        self.max_years_back = max_years_back

    @property
    def name(self) -> str:
        return "head_to_head"

    @property
    def feature_names(self) -> list[str]:
        return [
            "h2h_matches_count",
            "h2h_home_wins",
            "h2h_draws",
            "h2h_away_wins",
            "h2h_home_win_rate",
            "h2h_avg_total_goals",
            "h2h_home_avg_goals",
            "h2h_away_avg_goals",
            "h2h_recent_home_form",
        ]

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate H2H features for each match.

        Args:
            matches: Match data
            repository: Repository for historical queries

        Returns:
            DataFrame with H2H features indexed by match id
        """
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            h2h_matches = repository.get_h2h_matches(
                team1_id=home_team_id,
                team2_id=away_team_id,
                limit=self.max_h2h_matches,
                before_date=match_date,
            )

            match_features = self._calculate_h2h_stats(
                h2h_matches, home_team_id, away_team_id
            )
            match_features["match_id"] = match_id
            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _calculate_h2h_stats(
        self,
        h2h_matches: list,
        home_team_id: int,
        away_team_id: int,
    ) -> dict[str, float]:
        """Calculate H2H statistics from match list."""
        if not h2h_matches:
            return {
                "h2h_matches_count": 0,
                "h2h_home_wins": 0,
                "h2h_draws": 0,
                "h2h_away_wins": 0,
                "h2h_home_win_rate": 0.0,
                "h2h_avg_total_goals": 0.0,
                "h2h_home_avg_goals": 0.0,
                "h2h_away_avg_goals": 0.0,
                "h2h_recent_home_form": 0.0,
            }

        home_wins = 0
        draws = 0
        away_wins = 0
        total_goals = 0
        home_goals = 0
        away_goals = 0

        for match in h2h_matches:
            # Identify which team is home in this historical match
            if match.home_team_id == home_team_id:
                h_score = match.home_score or 0
                a_score = match.away_score or 0
            else:
                # Home team in current match was away in historical
                h_score = match.away_score or 0
                a_score = match.home_score or 0

            home_goals += h_score
            away_goals += a_score
            total_goals += h_score + a_score

            if h_score > a_score:
                home_wins += 1
            elif h_score < a_score:
                away_wins += 1
            else:
                draws += 1

        n = len(h2h_matches)

        # Recent form: points from last 3 H2H matches
        recent_points = 0
        for match in h2h_matches[:3]:
            if match.home_team_id == home_team_id:
                h_score = match.home_score or 0
                a_score = match.away_score or 0
            else:
                h_score = match.away_score or 0
                a_score = match.home_score or 0

            if h_score > a_score:
                recent_points += 3
            elif h_score == a_score:
                recent_points += 1

        return {
            "h2h_matches_count": n,
            "h2h_home_wins": home_wins,
            "h2h_draws": draws,
            "h2h_away_wins": away_wins,
            "h2h_home_win_rate": home_wins / n if n > 0 else 0.0,
            "h2h_avg_total_goals": total_goals / n,
            "h2h_home_avg_goals": home_goals / n,
            "h2h_away_avg_goals": away_goals / n,
            "h2h_recent_home_form": recent_points / min(3, n) if n > 0 else 0.0,
        }


class OddsFeatureGenerator(FeatureGenerator):
    """Generate features from betting odds data.

    Converts odds to implied probabilities and calculates market-derived features.
    Handles missing odds gracefully with imputation.
    """

    def __init__(
        self,
        default_margin: float = 0.05,
        impute_missing: bool = True,
    ) -> None:
        """Initialize odds generator.

        Args:
            default_margin: Default bookmaker margin when odds unavailable
            impute_missing: Whether to impute missing odds with market averages
        """
        self.default_margin = default_margin
        self.impute_missing = impute_missing

    @property
    def name(self) -> str:
        return "odds"

    @property
    def feature_names(self) -> list[str]:
        return [
            "implied_prob_home",
            "implied_prob_draw",
            "implied_prob_away",
            "bookmaker_margin",
            "odds_home_away_ratio",
            "favorite_outcome",
            "favorite_implied_prob",
            "odds_quality_score",
        ]

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository | None = None
    ) -> pd.DataFrame:
        """Generate odds-based features for each match.

        Args:
            matches: Match data with odds columns
            repository: Not used for odds features

        Returns:
            DataFrame with odds features indexed by match id
        """
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]

            odds_home = match.get("odds_home")
            odds_draw = match.get("odds_draw")
            odds_away = match.get("odds_away")
            num_bookmakers = match.get("num_bookmakers", 0)

            # Handle missing odds
            if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
                # Use default probabilities (home advantage)
                match_features = self._default_features()
            else:
                match_features = self._calculate_odds_features(
                    float(odds_home),
                    float(odds_draw),
                    float(odds_away),
                )

            match_features["match_id"] = match_id
            match_features["odds_quality_score"] = min(1.0, (num_bookmakers or 1) / 5.0)

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _calculate_odds_features(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
    ) -> dict[str, float]:
        """Calculate features from odds."""
        # Implied probabilities
        raw_home = 1 / odds_home
        raw_draw = 1 / odds_draw
        raw_away = 1 / odds_away

        # Bookmaker margin
        margin = raw_home + raw_draw + raw_away - 1

        # Normalized probabilities
        total = raw_home + raw_draw + raw_away
        prob_home = raw_home / total
        prob_draw = raw_draw / total
        prob_away = raw_away / total

        # Identify favorite
        probs = [prob_home, prob_draw, prob_away]
        favorite_idx = np.argmax(probs)

        return {
            "implied_prob_home": prob_home,
            "implied_prob_draw": prob_draw,
            "implied_prob_away": prob_away,
            "bookmaker_margin": margin,
            "odds_home_away_ratio": odds_home / odds_away if odds_away > 0 else 1.0,
            "favorite_outcome": float(favorite_idx),
            "favorite_implied_prob": probs[favorite_idx],
        }

    def _default_features(self) -> dict[str, float]:
        """Return default features when odds unavailable.

        Uses neutral estimates reflecting the average distribution across
        major leagues rather than baking in a strong home-win bias.
        """
        prob_home = 0.40
        prob_draw = 0.30
        prob_away = 0.30
        margin = self.default_margin
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total

        probs = [prob_home, prob_draw, prob_away]
        favorite_idx = int(np.argmax(probs))

        return {
            "implied_prob_home": prob_home,
            "implied_prob_draw": prob_draw,
            "implied_prob_away": prob_away,
            "bookmaker_margin": margin,
            "odds_home_away_ratio": 1.0,
            "favorite_outcome": float(favorite_idx),
            "favorite_implied_prob": probs[favorite_idx],
        }


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

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

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
                home_rest = self._get_rest_days(repository, home_team_id, match_date)
                away_rest = self._get_rest_days(repository, away_team_id, match_date)
                match_features["home_rest_days"] = home_rest
                match_features["away_rest_days"] = away_rest
                match_features["rest_days_diff"] = home_rest - away_rest

            # Fixture density
            if self.include_fixture_density:
                match_features["home_matches_last_14_days"] = (
                    self._count_recent_matches(
                        repository, home_team_id, match_date, days=14
                    )
                )
                match_features["away_matches_last_14_days"] = (
                    self._count_recent_matches(
                        repository, away_team_id, match_date, days=14
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

        if not last_matches:
            return 7.0  # Default: week of rest

        last_match = last_matches[0]
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
        # Get count from repository
        count = repo.get_match_count(team_id=team_id, before_date=match_date)

        # This is approximate - would need a more specific query
        # For now, estimate based on typical season density
        return min(count, 5)  # Cap at reasonable maximum


class StandingsFeatureGenerator(FeatureGenerator):
    """Generate league table/standings features for each team at match time.

    Produces features capturing each team's position, points, goal difference,
    and categorical indicators (relegation zone, Euro spots, league leader)
    as they stood at the time of the match.
    """

    def __init__(
        self,
        relegation_threshold: int = 3,
        euro_spot_start: int = 4,
        euro_spot_end: int = 7,
    ) -> None:
        """Initialize standings generator.

        Args:
            relegation_threshold: Number of bottom teams considered "relegation zone"
            euro_spot_start: Position where Euro qualification spots start
            euro_spot_end: Position where Euro qualification spots end
        """
        self.relegation_threshold = relegation_threshold
        self.euro_spot_start = euro_spot_start
        self.euro_spot_end = euro_spot_end

    @property
    def name(self) -> str:
        return "standings"

    @property
    def feature_names(self) -> list[str]:
        return [
            "home_league_position",
            "away_league_position",
            "home_points_total",
            "away_points_total",
            "home_matches_played",
            "away_matches_played",
            "home_goals_for_season",
            "away_goals_for_season",
            "home_goals_against_season",
            "away_goals_against_season",
            "home_goal_diff_season",
            "away_goal_diff_season",
            "home_points_per_game",
            "away_points_per_game",
            "home_win_rate_season",
            "away_win_rate_season",
            "position_diff",
            "points_diff",
            "home_in_relegation",
            "away_in_relegation",
            "home_in_euro_spot",
            "away_in_euro_spot",
            "home_is_leader",
            "away_is_leader",
            "home_position_normalized",
            "away_position_normalized",
        ]

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate standings features for each match.

        Args:
            matches: Match data with tournament_id and season_id
            repository: Repository with preloaded standings

        Returns:
            DataFrame with standings features indexed by match id
        """
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])
            tournament_id = match.get("tournament_id")
            season_id = match.get("season_id")

            match_features: dict[str, Any] = {"match_id": match_id}

            home_standings = None
            away_standings = None

            has_season = (
                tournament_id is not None
                and season_id is not None
                and not pd.isna(tournament_id)
                and not pd.isna(season_id)
            )
            if has_season:
                home_standings = repository.get_team_standings(
                    team_id=home_team_id,
                    tournament_id=int(tournament_id),
                    season_id=int(season_id),
                    before_date=match_date,
                )
                away_standings = repository.get_team_standings(
                    team_id=away_team_id,
                    tournament_id=int(tournament_id),
                    season_id=int(season_id),
                    before_date=match_date,
                )

            # Home team standings
            match_features.update(
                self._standings_to_features(home_standings, prefix="home")
            )
            # Away team standings
            match_features.update(
                self._standings_to_features(away_standings, prefix="away")
            )

            # Differential features
            if home_standings and away_standings:
                match_features["position_diff"] = float(
                    home_standings.position - away_standings.position
                )
                match_features["points_diff"] = float(
                    home_standings.points - away_standings.points
                )
            else:
                match_features["position_diff"] = 0.0
                match_features["points_diff"] = 0.0

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _standings_to_features(
        self,
        standings: object | None,
        prefix: str,
    ) -> dict[str, float]:
        """Convert a TeamStandings object to a feature dictionary.

        Args:
            standings: TeamStandings or None
            prefix: Feature name prefix (home/away)

        Returns:
            Dictionary of feature name to float value
        """
        if standings is None:
            return {
                f"{prefix}_league_position": 0.0,
                f"{prefix}_points_total": 0.0,
                f"{prefix}_matches_played": 0.0,
                f"{prefix}_goals_for_season": 0.0,
                f"{prefix}_goals_against_season": 0.0,
                f"{prefix}_goal_diff_season": 0.0,
                f"{prefix}_points_per_game": 0.0,
                f"{prefix}_win_rate_season": 0.0,
                f"{prefix}_in_relegation": 0.0,
                f"{prefix}_in_euro_spot": 0.0,
                f"{prefix}_is_leader": 0.0,
                f"{prefix}_position_normalized": 0.5,
            }

        total_teams = max(standings.total_teams, 1)
        n_played = max(standings.matches_played, 1)

        in_relegation = (
            1.0 if standings.position > total_teams - self.relegation_threshold else 0.0
        )
        in_euro = (
            1.0
            if self.euro_spot_start <= standings.position <= self.euro_spot_end
            else 0.0
        )
        is_leader = 1.0 if standings.position == 1 else 0.0

        return {
            f"{prefix}_league_position": float(standings.position),
            f"{prefix}_points_total": float(standings.points),
            f"{prefix}_matches_played": float(standings.matches_played),
            f"{prefix}_goals_for_season": float(standings.goals_for),
            f"{prefix}_goals_against_season": float(standings.goals_against),
            f"{prefix}_goal_diff_season": float(standings.goal_diff),
            f"{prefix}_points_per_game": standings.points_per_game,
            f"{prefix}_win_rate_season": standings.wins / n_played,
            f"{prefix}_in_relegation": in_relegation,
            f"{prefix}_in_euro_spot": in_euro,
            f"{prefix}_is_leader": is_leader,
            f"{prefix}_position_normalized": (
                (standings.position - 1) / max(total_teams - 1, 1)
            ),
        }


class EnrichedStatsFeatureGenerator(FeatureGenerator):
    """Generate rolling team features from enriched match and player statistics."""

    _MATCH_STAT_FIELDS: tuple[tuple[str, str], ...] = (
        ("xg_for", "xg"),
        ("xg_against", "xg"),
        ("npxg_for", "npxg"),
        ("npxg_against", "npxg"),
        ("shots_for", "shots"),
        ("shots_against", "shots"),
        ("shots_on_target_for", "shots_on_target"),
        ("shots_on_target_against", "shots_on_target"),
        ("corners_for", "corners"),
        ("corners_against", "corners"),
        ("ppda_for", "ppda"),
        ("ppda_against", "ppda"),
        ("deep_completions_for", "deep_completions"),
        ("deep_completions_against", "deep_completions"),
    )
    _PLAYER_STAT_FIELDS: tuple[tuple[str, str], ...] = (
        ("player_goals", "goals"),
        ("player_assists", "assists"),
        ("player_shots", "shots"),
        ("player_shots_on_target", "shots_on_target"),
        ("player_minutes", "minutes_played"),
    )

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        include_diffs: bool = False,
    ) -> None:
        self.window_sizes = window_sizes or [3, 5]
        self.include_diffs = include_diffs

    @property
    def name(self) -> str:
        return "enriched_stats"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for prefix in ("home", "away"):
            for w in self.window_sizes:
                for feature_name, _ in self._MATCH_STAT_FIELDS:
                    names.append(f"{prefix}_{feature_name}_avg_{w}")
                for feature_name, _ in self._PLAYER_STAT_FIELDS:
                    names.append(f"{prefix}_{feature_name}_avg_{w}")
                names.append(f"{prefix}_enriched_match_coverage_{w}")
                names.append(f"{prefix}_player_stats_coverage_{w}")
        if self.include_diffs:
            for w in self.window_sizes:
                for feature_name, _ in self._MATCH_STAT_FIELDS:
                    names.append(f"{feature_name}_diff_avg_{w}")
                for feature_name, _ in self._PLAYER_STAT_FIELDS:
                    names.append(f"{feature_name}_diff_avg_{w}")
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate enriched rolling features for each fixture."""
        max_window = max(self.window_sizes, default=0)
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            match_features: dict[str, Any] = {"match_id": match_id}
            home_feats = self._build_team_features(
                repository=repository,
                team_id=home_team_id,
                match_date=match_date,
                prefix="home",
                max_window=max_window,
            )
            away_feats = self._build_team_features(
                repository=repository,
                team_id=away_team_id,
                match_date=match_date,
                prefix="away",
                max_window=max_window,
            )
            match_features.update(home_feats)
            match_features.update(away_feats)

            if self.include_diffs:
                diff_feats = self._compute_diffs(home_feats, away_feats)
                match_features.update(diff_feats)

            features.append(match_features)

        return pd.DataFrame(features).set_index("match_id")

    def _compute_diffs(
        self,
        home_feats: dict[str, float],
        away_feats: dict[str, float],
    ) -> dict[str, float]:
        """Compute differential features (home - away) for each stat."""
        diffs: dict[str, float] = {}
        for w in self.window_sizes:
            suffix = f"_avg_{w}"
            for feature_name, _ in self._MATCH_STAT_FIELDS:
                home_key = f"home_{feature_name}{suffix}"
                away_key = f"away_{feature_name}{suffix}"
                diffs[f"{feature_name}_diff{suffix}"] = home_feats.get(
                    home_key, 0.0
                ) - away_feats.get(away_key, 0.0)
            for feature_name, _ in self._PLAYER_STAT_FIELDS:
                home_key = f"home_{feature_name}{suffix}"
                away_key = f"away_{feature_name}{suffix}"
                diffs[f"{feature_name}_diff{suffix}"] = home_feats.get(
                    home_key, 0.0
                ) - away_feats.get(away_key, 0.0)
        return diffs

    def _build_team_features(
        self,
        repository: MatchRepository,
        team_id: int,
        match_date: datetime,
        prefix: str,
        max_window: int,
    ) -> dict[str, float]:
        history = repository.get_team_matches(
            team_id=team_id,
            before_date=match_date,
            limit=max_window,
        )
        team_features: dict[str, float] = {}
        for window_size in self.window_sizes:
            recent_matches = history[:window_size]
            team_features.update(
                self._summarize_window(
                    team_id=team_id,
                    matches=recent_matches,
                    prefix=prefix,
                    window_size=window_size,
                )
            )
        return team_features

    def _summarize_window(
        self,
        team_id: int,
        matches: list[Any],
        prefix: str,
        window_size: int,
    ) -> dict[str, float]:
        match_rows = []
        player_rows = []

        for match in matches:
            match_stats = self._extract_team_match_stats(team_id=team_id, match=match)
            if match_stats is not None:
                match_rows.append(match_stats)

            player_stats = self._extract_team_player_rollup(
                team_id=team_id,
                match=match,
            )
            if player_stats is not None:
                player_rows.append(player_stats)

        denominator = float(len(matches))
        coverage = len(match_rows) / denominator if denominator else 0.0
        player_coverage = len(player_rows) / denominator if denominator else 0.0

        features: dict[str, float] = {}
        for feature_name, _ in self._MATCH_STAT_FIELDS:
            features[f"{prefix}_{feature_name}_avg_{window_size}"] = self._mean(
                rows=match_rows,
                key=feature_name,
            )
        for feature_name, _ in self._PLAYER_STAT_FIELDS:
            features[f"{prefix}_{feature_name}_avg_{window_size}"] = self._mean(
                rows=player_rows,
                key=feature_name,
            )
        features[f"{prefix}_enriched_match_coverage_{window_size}"] = coverage
        features[f"{prefix}_player_stats_coverage_{window_size}"] = player_coverage
        return features

    def _extract_team_match_stats(
        self,
        team_id: int,
        match: Any,
    ) -> dict[str, float] | None:
        statistics = getattr(match, "statistics", None)
        if statistics is None:
            return None

        is_home = getattr(match, "home_team_id", None) == team_id
        team_prefix = "home" if is_home else "away"
        opp_prefix = "away" if is_home else "home"

        raw_stats = {
            "xg_for": getattr(statistics, f"{team_prefix}_xg", None),
            "xg_against": getattr(statistics, f"{opp_prefix}_xg", None),
            "npxg_for": getattr(statistics, f"{team_prefix}_npxg", None),
            "npxg_against": getattr(statistics, f"{opp_prefix}_npxg", None),
            "shots_for": getattr(statistics, f"{team_prefix}_shots", None),
            "shots_against": getattr(statistics, f"{opp_prefix}_shots", None),
            "shots_on_target_for": getattr(
                statistics, f"{team_prefix}_shots_on_target", None
            ),
            "shots_on_target_against": getattr(
                statistics, f"{opp_prefix}_shots_on_target", None
            ),
            "corners_for": getattr(statistics, f"{team_prefix}_corners", None),
            "corners_against": getattr(statistics, f"{opp_prefix}_corners", None),
            "ppda_for": getattr(statistics, f"{team_prefix}_ppda", None),
            "ppda_against": getattr(statistics, f"{opp_prefix}_ppda", None),
            "deep_completions_for": getattr(
                statistics, f"{team_prefix}_deep_completions", None
            ),
            "deep_completions_against": getattr(
                statistics, f"{opp_prefix}_deep_completions", None
            ),
        }
        if all(value is None for value in raw_stats.values()):
            return None

        return {
            feature_name: float(value) if value is not None else 0.0
            for feature_name, value in raw_stats.items()
        }

    def _extract_team_player_rollup(
        self,
        team_id: int,
        match: Any,
    ) -> dict[str, float] | None:
        player_stats = [
            player
            for player in getattr(match, "player_stats", []) or []
            if getattr(player, "team_id", None) == team_id
        ]
        if not player_stats:
            return None

        return {
            "player_goals": self._sum_player_attr(player_stats, "goals"),
            "player_assists": self._sum_player_attr(player_stats, "assists"),
            "player_shots": self._sum_player_attr(player_stats, "shots"),
            "player_shots_on_target": self._sum_player_attr(
                player_stats,
                "shots_on_target",
            ),
            "player_minutes": self._sum_player_attr(player_stats, "minutes_played"),
        }

    @staticmethod
    def _sum_player_attr(players: list[Any], attr_name: str) -> float:
        return float(
            sum(float(getattr(player, attr_name, 0) or 0.0) for player in players)
        )

    @staticmethod
    def _mean(rows: list[dict[str, float]], key: str) -> float:
        if not rows:
            return 0.0
        return float(np.mean([row[key] for row in rows]))


class OddsResidualFeatureGenerator(FeatureGenerator):
    """Generate features that compare team form to market-implied expectations.

    These residual features represent how much better or worse each team is
    performing relative to what the betting market predicts -- the "surprise"
    signal.  This is orthogonal to raw odds and raw form features: instead of
    encoding home advantage three times (odds + venue form + naming), the
    model receives the market expectation once (via odds features) and a
    deviation signal (via these residuals).

    Expected points from odds:
        home_expected_pts = 3 * implied_prob_home + 1 * implied_prob_draw
        away_expected_pts = 3 * implied_prob_away + 1 * implied_prob_draw

    Surprise = actual PPG - expected_pts_per_game
    A positive surprise means the team is outperforming the market; negative
    means underperforming.
    """

    def __init__(
        self,
        form_windows: list[int] | None = None,
    ) -> None:
        self.form_windows = form_windows or [5, 10]

    @property
    def name(self) -> str:
        return "odds_residual"

    @property
    def feature_names(self) -> list[str]:
        names = [
            "home_form_surprise",
            "away_form_surprise",
            "home_venue_form_surprise",
            "away_venue_form_surprise",
            "form_surprise_diff",
            "venue_surprise_diff",
            "home_advantage_net",
        ]
        for w in self.form_windows:
            names.extend(
                [
                    f"home_form_surprise_{w}",
                    f"away_form_surprise_{w}",
                ]
            )
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        max_window = max(self.form_windows, default=10)
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            odds_home = match.get("odds_home")
            odds_draw = match.get("odds_draw")
            odds_away = match.get("odds_away")

            implied = self._implied_probabilities(odds_home, odds_draw, odds_away)

            home_form = self._ppg(repository, home_team_id, match_date, max_window)
            away_form = self._ppg(repository, away_team_id, match_date, max_window)
            home_venue_form = self._venue_ppg(
                repository, home_team_id, match_date, max_window, is_home=True
            )
            away_venue_form = self._venue_ppg(
                repository, away_team_id, match_date, max_window, is_home=False
            )

            home_exp_pts = 3 * implied["home"] + implied["draw"]
            away_exp_pts = 3 * implied["away"] + implied["draw"]

            match_feats: dict[str, Any] = {
                "match_id": match_id,
                "home_form_surprise": home_form - home_exp_pts,
                "away_form_surprise": away_form - away_exp_pts,
                "home_venue_form_surprise": home_venue_form - home_exp_pts,
                "away_venue_form_surprise": away_venue_form - away_exp_pts,
                "form_surprise_diff": (home_form - home_exp_pts)
                - (away_form - away_exp_pts),
                "venue_surprise_diff": (home_venue_form - home_exp_pts)
                - (away_venue_form - away_exp_pts),
                "home_advantage_net": implied["home"] - implied["away"],
            }

            for w in self.form_windows:
                home_w = self._ppg(repository, home_team_id, match_date, w)
                away_w = self._ppg(repository, away_team_id, match_date, w)
                match_feats[f"home_form_surprise_{w}"] = home_w - home_exp_pts
                match_feats[f"away_form_surprise_{w}"] = away_w - away_exp_pts

            features.append(match_feats)

        return pd.DataFrame(features).set_index("match_id")

    def _implied_probabilities(
        self,
        odds_home: Any,
        odds_draw: Any,
        odds_away: Any,
    ) -> dict[str, float]:
        if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
            return {"home": 0.40, "draw": 0.30, "away": 0.30}
        try:
            h = 1.0 / float(odds_home)
            d = 1.0 / float(odds_draw)
            a = 1.0 / float(odds_away)
            total = h + d + a
            return {"home": h / total, "draw": d / total, "away": a / total}
        except (ZeroDivisionError, ValueError, TypeError):
            return {"home": 0.40, "draw": 0.30, "away": 0.30}

    @staticmethod
    def _ppg(
        repo: MatchRepository,
        team_id: int,
        before_date: datetime,
        limit: int,
    ) -> float:
        matches = repo.get_team_matches(
            team_id=team_id, before_date=before_date, limit=limit
        )
        if not matches:
            return 0.0
        points = 0
        for match in matches:
            is_home = match.home_team_id == team_id
            gf = match.home_score if is_home else match.away_score
            ga = match.away_score if is_home else match.home_score
            gf = gf or 0
            ga = ga or 0
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
        return points / len(matches)

    @staticmethod
    def _venue_ppg(
        repo: MatchRepository,
        team_id: int,
        before_date: datetime,
        limit: int,
        is_home: bool,
    ) -> float:
        matches = repo.get_team_matches(
            team_id=team_id,
            before_date=before_date,
            limit=limit,
            home_only=is_home,
            away_only=not is_home,
        )
        if not matches:
            return 0.0
        points = 0
        for match in matches:
            home = match.home_team_id == team_id
            gf = match.home_score if home else match.away_score
            ga = match.away_score if home else match.home_score
            gf = gf or 0
            ga = ga or 0
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
        return points / len(matches)


class CompositeFeatureGenerator(FeatureGenerator):
    """Combines multiple feature generators into a single generator.

    Allows composition of feature generators to create comprehensive
    feature sets.
    """

    def __init__(self, generators: list[FeatureGenerator]) -> None:
        """Initialize composite generator.

        Args:
            generators: List of feature generators to combine
        """
        self.generators = generators

    @property
    def name(self) -> str:
        return "composite"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for gen in self.generators:
            names.extend(gen.feature_names)
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate all features from combined generators.

        Args:
            matches: Match data
            repository: Repository for historical queries

        Returns:
            DataFrame with all combined features indexed by match id
        """
        feature_dfs = []

        for gen in self.generators:
            df = gen.generate(matches, repository)
            feature_dfs.append(df)

        # Merge all feature DataFrames
        if not feature_dfs:
            return pd.DataFrame(index=matches["id"])

        result = feature_dfs[0]
        for df in feature_dfs[1:]:
            result = result.join(df, how="outer")

        return result

    def get_schema(self) -> FeatureSchema:
        """Return combined schema from all generators."""
        all_features: dict[str, type] = {}
        for gen in self.generators:
            schema = gen.get_schema()
            all_features.update(schema.features)

        return FeatureSchema(
            version="v1.0",
            features=all_features,
        )


def create_default_generators() -> CompositeFeatureGenerator:
    """Create the default set of feature generators.

    Returns:
        CompositeFeatureGenerator with all standard generators
    """
    return CompositeFeatureGenerator(
        generators=[
            TeamFormGenerator(
                window_sizes=[3, 5, 10],
                include_venue_specific=True,
            ),
            HeadToHeadGenerator(
                max_h2h_matches=5,
                max_years_back=3,
            ),
            OddsFeatureGenerator(
                impute_missing=True,
            ),
            OddsResidualFeatureGenerator(
                form_windows=[5, 10],
            ),
            TemporalFeatureGenerator(
                include_rest_days=True,
                include_fixture_density=True,
                include_season_period=True,
            ),
            StandingsFeatureGenerator(
                relegation_threshold=3,
                euro_spot_start=4,
                euro_spot_end=7,
            ),
        ]
    )
