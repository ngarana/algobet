"""Enriched match and player statistics feature generator."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class EnrichedStatsFeatureGenerator(FeatureGenerator):
    """Generate rolling team features from enriched match and player statistics."""

    # Available from Understat for 2014/15 onwards (~65-98% coverage per season)
    _UNDERSTAT_FIELDS: tuple[tuple[str, str], ...] = (
        ("xg_for", "xg"),
        ("xg_against", "xg"),
        ("ppda_for", "ppda"),
        ("ppda_against", "ppda"),
        ("deep_completions_for", "deep_completions"),
        ("deep_completions_against", "deep_completions"),
    )
    # Only available from OddsPortal scrape
    _BASIC_STAT_FIELDS: tuple[tuple[str, str], ...] = (("shots_for", "shots"),)
    _MATCH_STAT_FIELDS = _UNDERSTAT_FIELDS + _BASIC_STAT_FIELDS
    _PLAYER_STAT_FIELDS: tuple[tuple[str, str], ...] = (
        ("player_goals", "goals"),
        ("player_assists", "assists"),
        ("player_shots", "shots"),
        ("player_shots_on_target", "shots_on_target"),
        ("player_minutes", "minutes_played"),
        ("player_saves", "saves"),
        ("player_goals_conceded", "goals_conceded"),
        ("player_fouls_committed", "fouls_committed"),
        ("player_fouls_suffered", "fouls_suffered"),
        ("player_yellow_cards", "yellow_cards"),
        ("player_red_cards", "red_cards"),
        ("player_offsides", "offsides"),
        ("starter_minutes", "minutes_played"),
        ("starter_count", "is_starter"),
    )

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        include_diffs: bool = False,
        include_basic_stats: bool = False,
        include_player_stats: bool = True,
    ) -> None:
        self.window_sizes = window_sizes or [3, 5]
        self.include_diffs = False  # Pruned as per report
        self.include_basic_stats = include_basic_stats
        self.include_player_stats = include_player_stats

    @property
    def _active_match_stat_fields(self) -> tuple[tuple[str, str], ...]:
        if self.include_basic_stats:
            return self._MATCH_STAT_FIELDS
        return self._UNDERSTAT_FIELDS

    @property
    def name(self) -> str:
        return "enriched_stats"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for prefix in ("home", "away"):
            for w in self.window_sizes:
                for feature_name, _ in self._active_match_stat_fields:
                    names.append(f"{prefix}_{feature_name}_avg_{w}")
                if self.include_player_stats:
                    for feature_name, _ in self._PLAYER_STAT_FIELDS:
                        names.append(f"{prefix}_{feature_name}_avg_{w}")
                names.append(f"{prefix}_enriched_match_coverage_{w}")
                if self.include_player_stats:
                    names.append(f"{prefix}_player_stats_coverage_{w}")
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
            for feature_name, _ in self._active_match_stat_fields:
                home_key = f"home_{feature_name}{suffix}"
                away_key = f"away_{feature_name}{suffix}"
                diffs[f"{feature_name}_diff{suffix}"] = home_feats.get(
                    home_key, 0.0
                ) - away_feats.get(away_key, 0.0)
            if self.include_player_stats:
                for feature_name, _ in self._PLAYER_STAT_FIELDS:
                    home_key = f"home_{feature_name}{suffix}"
                    away_key = f"away_{feature_name}{suffix}"
                    diffs[f"{feature_name}_diff{suffix}"] = home_feats.get(
                        home_key, 0.0
                    ) - away_feats.get(away_key, 0.0)
            # Derived diffs
            if self.include_basic_stats:
                for derived in (
                    "shot_quality_for",
                    "shot_quality_against",
                    "shots_on_target_rate_for",
                    "shots_on_target_rate_against",
                ):
                    home_key = f"home_{derived}{suffix}"
                    away_key = f"away_{derived}{suffix}"
                    diffs[f"{derived}_diff{suffix}"] = home_feats.get(
                        home_key, 0.0
                    ) - away_feats.get(away_key, 0.0)
            for derived in ("xg_conversion_for", "xg_conversion_against"):
                home_key = f"home_{derived}{suffix}"
                away_key = f"away_{derived}{suffix}"
                diffs[f"{derived}_diff{suffix}"] = home_feats.get(
                    home_key, 0.0
                ) - away_feats.get(away_key, 0.0)
            # PPDA and deep completions diffs (from match stats)
            diffs[f"ppda_diff{suffix}"] = (
                home_feats.get(f"home_ppda_for{suffix}", 0.0)
                - home_feats.get(f"home_ppda_against{suffix}", 0.0)
                - (
                    away_feats.get(f"away_ppda_for{suffix}", 0.0)
                    - away_feats.get(f"away_ppda_against{suffix}", 0.0)
                )
            )
            diffs[f"deep_completions_diff{suffix}"] = (
                home_feats.get(f"home_deep_completions_for{suffix}", 0.0)
                - home_feats.get(f"home_deep_completions_against{suffix}", 0.0)
                - (
                    away_feats.get(f"away_deep_completions_for{suffix}", 0.0)
                    - away_feats.get(f"away_deep_completions_against{suffix}", 0.0)
                )
            )
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
        for feature_name, _ in self._active_match_stat_fields:
            features[f"{prefix}_{feature_name}_avg_{window_size}"] = self._mean(
                rows=match_rows,
                key=feature_name,
            )
        if self.include_player_stats:
            for feature_name, _ in self._PLAYER_STAT_FIELDS:
                features[f"{prefix}_{feature_name}_avg_{window_size}"] = self._mean(
                    rows=player_rows,
                    key=feature_name,
                )
        features[f"{prefix}_enriched_match_coverage_{window_size}"] = coverage
        if self.include_player_stats:
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
            feature_name: float(value) if value is not None else float("nan")
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
            "player_saves": self._sum_player_attr(player_stats, "saves"),
            "player_goals_conceded": self._sum_player_attr(
                player_stats, "goals_conceded"
            ),
            "player_fouls_committed": self._sum_player_attr(
                player_stats, "fouls_committed"
            ),
            "player_fouls_suffered": self._sum_player_attr(
                player_stats, "fouls_suffered"
            ),
            "player_yellow_cards": self._sum_player_attr(player_stats, "yellow_cards"),
            "player_red_cards": self._sum_player_attr(player_stats, "red_cards"),
            "player_offsides": self._sum_player_attr(player_stats, "offsides"),
            "starter_minutes": self._sum_starter_minutes(player_stats),
            "starter_count": self._count_starters(player_stats),
        }

    @staticmethod
    def _sum_player_attr(players: list[Any], attr_name: str) -> float:
        values = [
            float(value)
            for player in players
            if (value := getattr(player, attr_name, None)) is not None
        ]
        if not values:
            return float("nan")
        return float(sum(values))

    @staticmethod
    def _sum_starter_minutes(players: list[Any]) -> float:
        return float(
            sum(
                float(getattr(player, "minutes_played", 0) or 0.0)
                for player in players
                if bool(getattr(player, "is_starter", False))
            )
        )

    @staticmethod
    def _count_starters(players: list[Any]) -> float:
        return float(
            sum(1 for player in players if bool(getattr(player, "is_starter", False)))
        )

    @staticmethod
    def _mean(rows: list[dict[str, float]], key: str) -> float:
        if not rows:
            return float("nan")
        vals = [row[key] for row in rows if not np.isnan(row[key])]
        return float(np.mean(vals)) if vals else float("nan")
