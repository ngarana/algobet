"""Draw-signal composite feature generator.

Produces interaction and transformation features specifically engineered
to capture draw conditions. These features compress draw-relevant signals
from multiple groups into a small set of orthogonal, match-level features.
"""

from typing import Any

import numpy as np
import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class DrawSignalFeatureGenerator(FeatureGenerator):
    """Generate draw-signal composite features.

    These features are intentionally non-collinear with raw team-form
    features because they use interactions (products), absolute differences,
    and nonlinear transforms that capture *closeness* rather than raw
    magnitude.
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        xg_window_sizes: list[int] | None = None,
    ) -> None:
        """Initialize draw-signal generator.

        Args:
            window_sizes: Match window sizes for most features (default: [3, 5, 10])
            xg_window_sizes: Window sizes for xG-based features (default: [3, 5])
        """
        self.window_sizes = window_sizes or [3, 5, 10]
        self.xg_window_sizes = xg_window_sizes or [3, 5]

    @property
    def name(self) -> str:
        return "draw_signals"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for w in self.window_sizes:
            names.extend(
                [
                    f"defensive_balance_{w}",
                    f"low_scoring_probability_{w}",
                    f"clean_sheet_interaction_{w}",
                    f"goal_convergence_{w}",
                    f"volatility_sum_{w}",
                ]
            )
        for w in self.xg_window_sizes:
            names.extend(
                [
                    f"xg_parity_{w}",
                    # xG attack-defense clash
                    f"bilateral_xg_neutralization_{w}",
                ]
            )
        names.extend(
            [
                "strength_parity",
                "h2h_draw_boost",
            ]
        )
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate draw-signal features for each match.

        Args:
            matches: Match data
            repository: Repository for historical queries

        Returns:
            DataFrame with draw-signal features indexed by match id
        """
        features = []
        max_window = max(self.window_sizes, default=0)

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            match_features: dict[str, Any] = {"match_id": match_id}

            # Fetch team histories
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

            # Fetch H2H history
            h2h_matches = repository.get_h2h_matches(
                team1_id=home_team_id,
                team2_id=away_team_id,
                limit=5,
                before_date=match_date,
            )

            # Windowed features
            for w in self.window_sizes:
                home_stats = self._summarize_team_history(
                    home_history[:w], home_team_id
                )
                away_stats = self._summarize_team_history(
                    away_history[:w], away_team_id
                )
                match_features.update(
                    self._compute_window_features(home_stats, away_stats, w)
                )

            # xG windowed features
            for w in self.xg_window_sizes:
                home_xg_stats = self._summarize_xg_history(
                    home_history[:w], home_team_id
                )
                away_xg_stats = self._summarize_xg_history(
                    away_history[:w], away_team_id
                )
                match_features[f"xg_parity_{w}"] = self._compute_xg_parity(
                    home_xg_stats, away_xg_stats
                )
                match_features[f"bilateral_xg_neutralization_{w}"] = (
                    self._compute_bilateral_xg_neutralization(
                        home_xg_stats, away_xg_stats
                    )
                )

            # Global features
            home_form_5 = self._summarize_team_history(home_history[:5], home_team_id)
            away_form_5 = self._summarize_team_history(away_history[:5], away_team_id)
            match_features["strength_parity"] = abs(
                home_form_5.get("avg_points", float("nan"))
                - away_form_5.get("avg_points", float("nan"))
            )

            h2h_stats = self._summarize_h2h(h2h_matches, home_team_id, away_team_id)
            n_h2h = h2h_stats["matches_count"]
            draw_rate = h2h_stats["draw_rate"] if n_h2h > 0 else 0.0
            match_features["h2h_draw_boost"] = draw_rate * min(n_h2h / 5.0, 1.0)

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _summarize_team_history(
        self,
        matches: list[Any],
        team_id: int,
    ) -> dict[str, float]:
        """Summarize team history into base stats for draw-signal features."""
        if not matches:
            nan = float("nan")
            return {
                "avg_points": nan,
                "draw_rate": nan,
                "avg_goals_for": nan,
                "avg_goals_against": nan,
                "clean_sheet_rate": nan,
                "failed_to_score_rate": nan,
                "low_scoring_rate": nan,
                "goal_variance": nan,
            }

        total_points = 0
        draws = 0
        goals_for = 0
        goals_against = 0
        clean_sheets = 0
        failed_to_score = 0
        low_scoring = 0
        goals_for_list = []

        for match in matches:
            gf, ga = self._goals_for_against(match, team_id)
            goals_for += gf
            goals_against += ga
            goals_for_list.append(gf)

            if gf > ga:
                total_points += 3
            elif gf == ga:
                total_points += 1
                draws += 1
            else:
                pass  # loss

            if ga == 0:
                clean_sheets += 1
            if gf == 0:
                failed_to_score += 1
            if gf + ga < 2.5:
                low_scoring += 1

        n = len(matches)
        goal_variance = (
            float(np.var(goals_for_list)) if len(goals_for_list) > 1 else 0.0
        )

        return {
            "avg_points": total_points / n,
            "draw_rate": draws / n,
            "avg_goals_for": goals_for / n,
            "avg_goals_against": goals_against / n,
            "clean_sheet_rate": clean_sheets / n,
            "failed_to_score_rate": failed_to_score / n,
            "low_scoring_rate": low_scoring / n,
            "goal_variance": goal_variance,
        }

    def _summarize_xg_history(
        self,
        matches: list[Any],
        team_id: int,
    ) -> dict[str, float]:
        """Summarize xG history for a team."""
        xg_for_list = []
        xg_against_list = []

        for match in matches:
            statistics = getattr(match, "statistics", None)
            if statistics is None:
                continue

            is_home = getattr(match, "home_team_id", None) == team_id
            if is_home:
                xg_for = getattr(statistics, "home_xg", None)
                xg_against = getattr(statistics, "away_xg", None)
            else:
                xg_for = getattr(statistics, "away_xg", None)
                xg_against = getattr(statistics, "home_xg", None)

            if xg_for is not None:
                xg_for_list.append(float(xg_for))
            if xg_against is not None:
                xg_against_list.append(float(xg_against))

        if not xg_for_list or not xg_against_list:
            return {
                "avg_xg_for": float("nan"),
                "avg_xg_against": float("nan"),
            }

        return {
            "avg_xg_for": float(np.mean(xg_for_list)),
            "avg_xg_against": float(np.mean(xg_against_list)),
        }

    def _compute_window_features(
        self,
        home_stats: dict[str, float],
        away_stats: dict[str, float],
        window_size: int,
    ) -> dict[str, float]:
        """Compute draw-signal interaction features for a window."""
        defensive_balance = abs(
            home_stats["avg_goals_against"] - away_stats["avg_goals_against"]
        )
        low_scoring_prob = (
            home_stats["low_scoring_rate"] * away_stats["low_scoring_rate"]
        )
        clean_sheet_interaction = (
            home_stats["clean_sheet_rate"] * away_stats["failed_to_score_rate"]
            + away_stats["clean_sheet_rate"] * home_stats["failed_to_score_rate"]
        )

        gf_diff = abs(home_stats["avg_goals_for"] - away_stats["avg_goals_for"])
        goal_convergence = 1.0 / (1.0 + gf_diff)

        volatility_sum = home_stats["goal_variance"] + away_stats["goal_variance"]

        return {
            f"defensive_balance_{window_size}": defensive_balance,
            f"low_scoring_probability_{window_size}": low_scoring_prob,
            f"clean_sheet_interaction_{window_size}": clean_sheet_interaction,
            f"goal_convergence_{window_size}": goal_convergence,
            f"volatility_sum_{window_size}": volatility_sum,
        }

    def _compute_xg_parity(
        self,
        home_xg_stats: dict[str, float],
        away_xg_stats: dict[str, float],
    ) -> float:
        """Compute xG parity feature: 1 / (1 + abs(home_xg_for - away_xg_against))."""
        home_xg_for = home_xg_stats.get("avg_xg_for", float("nan"))
        away_xg_against = away_xg_stats.get("avg_xg_against", float("nan"))
        if np.isnan(home_xg_for) or np.isnan(away_xg_against):
            return float("nan")
        return 1.0 / (1.0 + abs(home_xg_for - away_xg_against))

    def _compute_bilateral_xg_neutralization(
        self,
        home_xg_stats: dict[str, float],
        away_xg_stats: dict[str, float],
    ) -> float:
        """1 / (1 + |home_xg_for - away_xg_against| + |away_xg_for - home_xg_against|).

        High value (→ 1.0) means both attacks are simultaneously neutralized by
        the opposing defense in xG terms — the strongest xG-based draw signal.
        """
        h_for = home_xg_stats.get("avg_xg_for", float("nan"))
        h_against = home_xg_stats.get("avg_xg_against", float("nan"))
        a_for = away_xg_stats.get("avg_xg_for", float("nan"))
        a_against = away_xg_stats.get("avg_xg_against", float("nan"))
        if any(np.isnan(v) for v in (h_for, h_against, a_for, a_against)):
            return float("nan")
        home_clash = abs(h_for - a_against)
        away_clash = abs(a_for - h_against)
        return 1.0 / (1.0 + home_clash + away_clash)

    def _summarize_h2h(
        self,
        h2h_matches: list[Any],
        home_team_id: int,
        away_team_id: int,
    ) -> dict[str, float]:
        """Summarize H2H stats for draw boost."""
        if not h2h_matches:
            return {"matches_count": 0, "draw_rate": 0.0}

        draws = 0
        for match in h2h_matches:
            if match.home_team_id == home_team_id:
                h_score = match.home_score or 0
                a_score = match.away_score or 0
            else:
                h_score = match.away_score or 0
                a_score = match.home_score or 0

            if h_score == a_score:
                draws += 1

        n = len(h2h_matches)
        return {"matches_count": n, "draw_rate": draws / n}

    @staticmethod
    def _goals_for_against(match: Any, team_id: int) -> tuple[int, int]:
        """Return goals for/against from the team's perspective."""
        if match.home_team_id == team_id:
            return match.home_score or 0, match.away_score or 0
        return match.away_score or 0, match.home_score or 0
