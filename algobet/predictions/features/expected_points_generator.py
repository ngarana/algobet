"""Expected Points (xPts) feature generator from xG data.

Converts per-match expected goals (xG) into Expected Points (xPts) using
a Poisson-based match outcome model, then produces features that are
*complementary* to the raw ``team_form`` features already in the pipeline.

**Collinearity note**: ``xpts_avg_N`` (average expected points over N games)
is highly collinear with ``points_last_N`` from team_form because xPts
tracks actual points closely.  This generator deliberately emits only
the *unique* signals:

- ``xpts_diff_N`` – the schedule-neutral strength differential (home xPts
  minus away xPts).  Unlike ``form_diff`` which uses raw points, this
  accounts for opponent strength via xG.
- ``points_vs_xpts_N`` – over/under-performance indicator (actual points
  minus expected points).  A positive value means the team is winning more
  than their underlying xG deserves (lucky); negative means underperforming.
- ``xpts_coverage_N`` – fraction of recent matches where xG data was
  available.  Useful for the model to know how reliable the xPts signals
  are.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


def _compute_xpts(
    home_xg: float,
    away_xg: float,
    max_goals: int = 10,
) -> tuple[float, float]:
    """Compute expected points for home and away teams from xG.

    Returns:
        (home_xpts, away_xpts)
    """
    if np.isnan(home_xg) or np.isnan(away_xg):
        return (float("nan"), float("nan"))

    home_probs = poisson.pmf(np.arange(max_goals + 1), home_xg)
    away_probs = poisson.pmf(np.arange(max_goals + 1), away_xg)

    score_matrix = np.outer(home_probs, away_probs)

    p_home_win = float(np.sum(np.tril(score_matrix, -1)))
    p_draw = float(np.sum(np.diag(score_matrix)))

    home_xpts = 3.0 * p_home_win + 1.0 * p_draw
    away_xpts = 3.0 * (1.0 - p_home_win - p_draw) + 1.0 * p_draw
    return (home_xpts, away_xpts)


class ExpectedPointsGenerator(FeatureGenerator):
    """Generate Expected Points (xPts) features from xG data.

    Produces only complementary (non-collinear with team_form) features:
    the xPts differential, over/under-performance indicator, and xG
    data coverage flag.
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        max_goals: int = 10,
    ) -> None:
        """Initialize xPts generator.

        Args:
            window_sizes: Window sizes for rolling averages.
                Default [3, 5, 10].
            max_goals: Maximum goals in Poisson xPts computation.
        """
        self.window_sizes = window_sizes or [3, 5, 10]
        self.max_goals = max_goals

    @property
    def name(self) -> str:
        return "expected_points"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for w in self.window_sizes:
            names.extend(
                [
                    f"xpts_diff_{w}",
                    f"home_points_vs_xpts_{w}",
                    f"away_points_vs_xpts_{w}",
                    f"home_xpts_coverage_{w}",
                    f"away_xpts_coverage_{w}",
                ]
            )
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate xPts features for each match.

        Args:
            matches: DataFrame with match records
            repository: Repository for historical data queries

        Returns:
            DataFrame indexed by match_id with xPts features
        """
        max_window = max(self.window_sizes, default=10)
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            match_features: dict[str, Any] = {"match_id": match_id}

            home_history = repository.get_team_matches(
                team_id=home_team_id,
                before_date=match_date,
                limit=max_window,
            )
            away_history = repository.get_team_matches(
                team_id=away_team_id,
                before_date=match_date,
                limit=max_window,
            )

            for w in self.window_sizes:
                home_summary = self._summarize_xpts(home_history[:w], home_team_id)
                away_summary = self._summarize_xpts(away_history[:w], away_team_id)

                # xPts differential: schedule-neutral strength gap
                home_avg = home_summary["xpts_avg"]
                away_avg = away_summary["xpts_avg"]
                if not np.isnan(home_avg) and not np.isnan(away_avg):
                    match_features[f"xpts_diff_{w}"] = home_avg - away_avg
                else:
                    match_features[f"xpts_diff_{w}"] = float("nan")

                # Over/under-performance: actual points minus xPts
                match_features[f"home_points_vs_xpts_{w}"] = home_summary[
                    "points_vs_xpts"
                ]
                match_features[f"away_points_vs_xpts_{w}"] = away_summary[
                    "points_vs_xpts"
                ]

                # xG data coverage (how reliable the xPts signal is)
                match_features[f"home_xpts_coverage_{w}"] = home_summary["coverage"]
                match_features[f"away_xpts_coverage_{w}"] = away_summary["coverage"]

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _summarize_xpts(self, matches: list[Any], team_id: int) -> dict[str, float]:
        """Compute xPts statistics over a window of recent matches.

        Args:
            matches: List of recent match objects (most recent first).
            team_id: The team whose perspective to use.

        Returns:
            Dictionary with xpts_avg, points_vs_xpts, and coverage.
        """
        if not matches:
            nan = float("nan")
            return {"xpts_avg": nan, "points_vs_xpts": nan, "coverage": 0.0}

        xpts_list: list[float] = []
        points_vs_xpts_list: list[float] = []

        for match in matches:
            is_home = match.home_team_id == team_id
            stats = getattr(match, "statistics", None)

            if stats is None:
                continue

            home_xg = getattr(stats, "home_xg", None)
            away_xg = getattr(stats, "away_xg", None)

            if home_xg is None or away_xg is None:
                continue

            try:
                home_xg_f = float(home_xg)
                away_xg_f = float(away_xg)
            except (TypeError, ValueError):
                continue

            home_xpts, away_xpts = _compute_xpts(home_xg_f, away_xg_f, self.max_goals)

            if np.isnan(home_xpts):
                continue

            # Actual points earned
            home_score = match.home_score or 0
            away_score = match.away_score or 0

            if is_home:
                if home_score > away_score:
                    actual_pts = 3.0
                elif home_score == away_score:
                    actual_pts = 1.0
                else:
                    actual_pts = 0.0
                team_xpts = home_xpts
            else:
                if away_score > home_score:
                    actual_pts = 3.0
                elif away_score == home_score:
                    actual_pts = 1.0
                else:
                    actual_pts = 0.0
                team_xpts = away_xpts

            xpts_list.append(team_xpts)
            points_vs_xpts_list.append(actual_pts - team_xpts)

        coverage = len(xpts_list) / len(matches) if matches else 0.0

        if xpts_list:
            return {
                "xpts_avg": float(np.mean(xpts_list)),
                "points_vs_xpts": float(np.mean(points_vs_xpts_list)),
                "coverage": coverage,
            }

        nan = float("nan")
        return {"xpts_avg": nan, "points_vs_xpts": nan, "coverage": coverage}
