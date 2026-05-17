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
        ("npxg_for", "npxg"),
        ("npxg_against", "npxg"),
        ("ppda_for", "ppda"),
        ("ppda_against", "ppda"),
        ("deep_completions_for", "deep_completions"),
        ("deep_completions_against", "deep_completions"),
    )
    # Fields that get extra trend (slope) + volatility (std) statistics computed
    # on top of the rolling mean. Targets the variance-compression diagnosis:
    # the regressors need more spread-encoding signal than a single mean provides.
    _TREND_VOL_FIELDS: tuple[str, ...] = (
        "xg_for",
        "xg_against",
        "npxg_for",
        "npxg_against",
    )
    # Derived per-match samples computed from raw stats (extracted in
    # _extract_team_match_stats and rolled up the same way as raw fields).
    _DERIVED_PER_MATCH_FIELDS: tuple[str, ...] = (
        "finishing_rate_for",  # goals_for / max(xg_for, 0.1)
        "finishing_rate_against",  # goals_against / max(xg_against, 0.1)
        "xg_for_adj",  # xg_for - opponent rolling xg_against
        "xg_against_adj",  # xg_against - opponent rolling xg_for
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
        active_match_field_names = {name for name, _ in self._active_match_stat_fields}
        for prefix in ("home", "away"):
            for w in self.window_sizes:
                for feature_name, _ in self._active_match_stat_fields:
                    names.append(f"{prefix}_{feature_name}_avg_{w}")
                # Trend (slope) + volatility (std) for fields where dispersion
                # carries predictive signal (xG/npxG for and against).
                for feature_name in self._TREND_VOL_FIELDS:
                    if feature_name not in active_match_field_names:
                        continue
                    names.append(f"{prefix}_{feature_name}_slope_{w}")
                    names.append(f"{prefix}_{feature_name}_std_{w}")
                # Derived per-match samples (finishing rate, opponent-adjusted xG).
                for derived in self._DERIVED_PER_MATCH_FIELDS:
                    names.append(f"{prefix}_{derived}_avg_{w}")
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
                    repository=repository,
                )
            )
        return team_features

    def _summarize_window(
        self,
        team_id: int,
        matches: list[Any],
        prefix: str,
        window_size: int,
        repository: MatchRepository,
    ) -> dict[str, float]:
        match_rows: list[dict[str, Any]] = []
        player_rows = []

        for match in matches:
            match_stats = self._extract_team_match_stats(team_id=team_id, match=match)
            if match_stats is not None:
                # Derived per-sample features that need extra lookups.
                self._attach_derived_sample_fields(
                    sample=match_stats,
                    repository=repository,
                )
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
        active_match_field_names = {name for name, _ in self._active_match_stat_fields}
        for feature_name, _ in self._active_match_stat_fields:
            features[f"{prefix}_{feature_name}_avg_{window_size}"] = self._mean(
                rows=match_rows,
                key=feature_name,
            )
        # Trend (slope) + volatility (std) for high-signal dispersion fields.
        for feature_name in self._TREND_VOL_FIELDS:
            if feature_name not in active_match_field_names:
                continue
            features[f"{prefix}_{feature_name}_slope_{window_size}"] = self._slope(
                rows=match_rows,
                key=feature_name,
            )
            features[f"{prefix}_{feature_name}_std_{window_size}"] = self._std(
                rows=match_rows,
                key=feature_name,
            )
        # Derived per-match aggregates (finishing rate, opponent-adjusted xG).
        for derived in self._DERIVED_PER_MATCH_FIELDS:
            features[f"{prefix}_{derived}_avg_{window_size}"] = self._mean(
                rows=match_rows,
                key=derived,
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

    # Tunables for derived-sample features.
    _XG_FLOOR = 0.1  # Prevents division blow-up in finishing rate.
    _OPP_BASELINE_WINDOW = 5  # Window size for opponent's xg_against / xg_for baseline.

    def _attach_derived_sample_fields(
        self,
        sample: dict[str, Any],
        repository: MatchRepository,
    ) -> None:
        """Compute per-sample derived features (finishing rate, opp-adjusted xG).

        Mutates `sample` in place to add keys named in `_DERIVED_PER_MATCH_FIELDS`.
        Uses NaN for any sample where the required raw inputs are missing —
        downstream mean aggregation already ignores NaN.
        """
        xg_for = sample.get("xg_for", float("nan"))
        xg_against = sample.get("xg_against", float("nan"))
        goals_for = sample.get("_goals_for", float("nan"))
        goals_against = sample.get("_goals_against", float("nan"))

        # Finishing rate: how clinical was the team relative to chance quality?
        sample["finishing_rate_for"] = (
            goals_for / max(xg_for, self._XG_FLOOR)
            if not (np.isnan(xg_for) or np.isnan(goals_for))
            else float("nan")
        )
        sample["finishing_rate_against"] = (
            goals_against / max(xg_against, self._XG_FLOOR)
            if not (np.isnan(xg_against) or np.isnan(goals_against))
            else float("nan")
        )

        # Opponent-adjusted xG: remove strength-of-schedule confound.
        opponent_id = sample.get("_opponent_id")
        match_date = sample.get("_match_date")
        if opponent_id is not None and match_date is not None:
            opp_xg_against_avg, opp_xg_for_avg = self._opponent_baseline(
                repository=repository,
                opponent_id=opponent_id,
                before_date=match_date,
            )
            sample["xg_for_adj"] = (
                xg_for - opp_xg_against_avg
                if not (np.isnan(xg_for) or np.isnan(opp_xg_against_avg))
                else float("nan")
            )
            sample["xg_against_adj"] = (
                xg_against - opp_xg_for_avg
                if not (np.isnan(xg_against) or np.isnan(opp_xg_for_avg))
                else float("nan")
            )
        else:
            sample["xg_for_adj"] = float("nan")
            sample["xg_against_adj"] = float("nan")

    def _opponent_baseline(
        self,
        repository: MatchRepository,
        opponent_id: int,
        before_date: datetime,
    ) -> tuple[float, float]:
        """Return opponent's rolling (xg_against_avg, xg_for_avg) before a date."""
        opp_history = repository.get_team_matches(
            team_id=opponent_id,
            before_date=before_date,
            limit=self._OPP_BASELINE_WINDOW,
        )
        xg_against_vals: list[float] = []
        xg_for_vals: list[float] = []
        for opp_match in opp_history:
            opp_stats = self._extract_team_match_stats(
                team_id=opponent_id,
                match=opp_match,
            )
            if opp_stats is None:
                continue
            xg_against = opp_stats.get("xg_against", float("nan"))
            xg_for = opp_stats.get("xg_for", float("nan"))
            if not np.isnan(xg_against):
                xg_against_vals.append(xg_against)
            if not np.isnan(xg_for):
                xg_for_vals.append(xg_for)
        xg_against_avg = (
            float(np.mean(xg_against_vals)) if xg_against_vals else float("nan")
        )
        xg_for_avg = float(np.mean(xg_for_vals)) if xg_for_vals else float("nan")
        return xg_against_avg, xg_for_avg

    def _extract_team_match_stats(
        self,
        team_id: int,
        match: Any,
    ) -> dict[str, Any] | None:
        statistics = getattr(match, "statistics", None)
        if statistics is None:
            return None

        is_home = getattr(match, "home_team_id", None) == team_id
        team_prefix = "home" if is_home else "away"
        opp_prefix = "away" if is_home else "home"
        opponent_id = (
            getattr(match, "away_team_id", None)
            if is_home
            else getattr(match, "home_team_id", None)
        )
        team_score = getattr(match, f"{team_prefix}_score", None)
        opp_score = getattr(match, f"{opp_prefix}_score", None)

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

        sample: dict[str, Any] = {
            feature_name: float(value) if value is not None else float("nan")
            for feature_name, value in raw_stats.items()
        }
        # Metadata needed for derived features computed in the rollup step.
        sample["_opponent_id"] = int(opponent_id) if opponent_id is not None else None
        sample["_match_date"] = getattr(match, "match_date", None)
        sample["_goals_for"] = (
            float(team_score) if team_score is not None else float("nan")
        )
        sample["_goals_against"] = (
            float(opp_score) if opp_score is not None else float("nan")
        )
        return sample

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
    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return float("nan")
        vals = [row[key] for row in rows if key in row and not np.isnan(row[key])]
        return float(np.mean(vals)) if vals else float("nan")

    @staticmethod
    def _std(rows: list[dict[str, Any]], key: str) -> float:
        """Population std of a series; NaN if fewer than 2 valid samples."""
        if not rows:
            return float("nan")
        vals = [row[key] for row in rows if key in row and not np.isnan(row[key])]
        if len(vals) < 2:
            return float("nan")
        return float(np.std(vals, ddof=0))

    @staticmethod
    def _slope(rows: list[dict[str, Any]], key: str) -> float:
        """Linear-fit slope across rows. Rows are ordered most-recent first;
        we reverse so positive slope = improving (more recent values larger).
        NaN if fewer than 2 valid samples."""
        if not rows:
            return float("nan")
        vals = [row[key] for row in rows if key in row and not np.isnan(row[key])]
        if len(vals) < 2:
            return float("nan")
        ordered = list(reversed(vals))  # chronological: oldest → newest
        x = np.arange(len(ordered), dtype=np.float64)
        y = np.asarray(ordered, dtype=np.float64)
        # np.polyfit(x, y, 1) returns [slope, intercept].
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
