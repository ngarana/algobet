"""Feature generators for match prediction.

This module provides feature generators that transform raw match data into
ML-ready features. Each generator produces a specific category of features
following the scikit-learn transformer pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
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
        for name, dtype in self.features.items():
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
            features={name: float for name in self.feature_names},
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
            match_features["form_diff"] = (
                match_features.get("home_points_last_5", 0)
                - match_features.get("away_points_last_5", 0)
            )

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
            match_features["odds_quality_score"] = min(
                1.0, (num_bookmakers or 1) / 5.0
            )

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
        outcomes = [0, 1, 2]  # 0=home, 1=draw, 2=away
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
        """Return default features when odds unavailable."""
        return {
            "implied_prob_home": 0.45,
            "implied_prob_draw": 0.28,
            "implied_prob_away": 0.27,
            "bookmaker_margin": self.default_margin,
            "odds_home_away_ratio": 1.0,
            "favorite_outcome": 0.0,  # Home
            "favorite_implied_prob": 0.45,
        }


class TemporalFeatureGenerator(FeatureGenerator):
    """Generate temporal features from match dates.

    Produces features related to timing, rest days, fixture congestion,
    and calendar effects.
    """

    def __init__(
        self,
        include_rest_days: bool = True,
        include_fixture_density: bool = True,
    ) -> None:
        """Initialize temporal generator.

        Args:
            include_rest_days: Calculate rest days for each team
            include_fixture_density: Calculate fixture congestion
        """
        self.include_rest_days = include_rest_days
        self.include_fixture_density = include_fixture_density

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
            season_start = datetime(match_date.year - 1 if match_date.month < 8 else match_date.year, 8, 1)
            if match_date.month < 8:
                season_start = datetime(match_date.year - 1, 8, 1)
            match_features["days_from_season_start"] = (match_date - season_start).days

            # Rest days
            if self.include_rest_days:
                home_rest = self._get_rest_days(
                    repository, home_team_id, match_date
                )
                away_rest = self._get_rest_days(
                    repository, away_team_id, match_date
                )
                match_features["home_rest_days"] = home_rest
                match_features["away_rest_days"] = away_rest
                match_features["rest_days_diff"] = home_rest - away_rest

            # Fixture density
            if self.include_fixture_density:
                match_features["home_matches_last_14_days"] = self._count_recent_matches(
                    repository, home_team_id, match_date, days=14
                )
                match_features["away_matches_last_14_days"] = self._count_recent_matches(
                    repository, away_team_id, match_date, days=14
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
        cutoff = match_date - timedelta(days=days)

        # Get count from repository
        count = repo.get_match_count(team_id=team_id, before_date=match_date)

        # This is approximate - would need a more specific query
        # For now, estimate based on typical season density
        return min(count, 5)  # Cap at reasonable maximum


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
            TemporalFeatureGenerator(
                include_rest_days=True,
                include_fixture_density=True,
            ),
        ]
    )
