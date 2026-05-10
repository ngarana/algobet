"""Team-form feature generator."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


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
            names.extend(
                [
                    f"home_points_last_{w}",
                    f"home_win_rate_{w}",
                    f"home_goals_for_avg_{w}",
                    f"home_goals_against_avg_{w}",
                    f"home_goal_diff_avg_{w}",
                    f"away_points_last_{w}",
                    f"away_win_rate_{w}",
                    f"away_goals_for_avg_{w}",
                    f"away_goals_against_avg_{w}",
                    f"away_goal_diff_avg_{w}",
                ]
            )

        for w in self.window_sizes:
            names.extend(
                [
                    f"home_draw_rate_{w}",
                    f"away_draw_rate_{w}",
                    f"home_loss_rate_{w}",
                    f"away_loss_rate_{w}",
                    f"home_clean_sheet_rate_{w}",
                    f"away_clean_sheet_rate_{w}",
                    f"home_failed_to_score_rate_{w}",
                    f"away_failed_to_score_rate_{w}",
                    f"home_btts_rate_{w}",
                    f"away_btts_rate_{w}",
                    f"home_low_scoring_rate_{w}",
                    f"away_low_scoring_rate_{w}",
                    f"home_goal_variance_{w}",
                    f"away_goal_variance_{w}",
                    f"home_points_volatility_{w}",
                    f"away_points_volatility_{w}",
                ]
            )

        if self.include_venue_specific:
            for w in self.window_sizes:
                names.extend(
                    [
                        f"home_home_form_{w}",
                        f"away_away_form_{w}",
                        f"home_home_draw_rate_{w}",
                        f"away_away_draw_rate_{w}",
                        f"away_away_win_rate_{w}",
                        f"away_away_clean_sheet_rate_{w}",
                    ]
                )

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
        features = []
        max_window = max(self.window_sizes, default=0)
        trend_window = max(max_window, 6)

        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            match_features: dict[str, Any] = {"match_id": match_id}
            home_history = repository.get_team_matches(
                team_id=home_team_id,
                before_date=match_date,
                limit=trend_window,
            )
            away_history = repository.get_team_matches(
                team_id=away_team_id,
                before_date=match_date,
                limit=trend_window,
            )
            home_venue_history = (
                repository.get_team_matches(
                    team_id=home_team_id,
                    before_date=match_date,
                    limit=max_window,
                    home_only=True,
                )
                if self.include_venue_specific
                else []
            )
            away_venue_history = (
                repository.get_team_matches(
                    team_id=away_team_id,
                    before_date=match_date,
                    limit=max_window,
                    away_only=True,
                )
                if self.include_venue_specific
                else []
            )

            for w in self.window_sizes:
                home_form = self._summarize_extended_form(
                    home_history[:w], home_team_id
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
                match_features[f"home_draw_rate_{w}"] = home_form["draw_rate"]
                match_features[f"home_loss_rate_{w}"] = home_form["loss_rate"]
                match_features[f"home_clean_sheet_rate_{w}"] = home_form[
                    "clean_sheet_rate"
                ]
                match_features[f"home_failed_to_score_rate_{w}"] = home_form[
                    "failed_to_score_rate"
                ]
                match_features[f"home_btts_rate_{w}"] = home_form["btts_rate"]
                match_features[f"home_low_scoring_rate_{w}"] = home_form[
                    "low_scoring_rate"
                ]
                match_features[f"home_goal_variance_{w}"] = home_form["goal_variance"]
                match_features[f"home_points_volatility_{w}"] = home_form[
                    "points_volatility"
                ]

                away_form = self._summarize_extended_form(
                    away_history[:w], away_team_id
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
                match_features[f"away_draw_rate_{w}"] = away_form["draw_rate"]
                match_features[f"away_loss_rate_{w}"] = away_form["loss_rate"]
                match_features[f"away_clean_sheet_rate_{w}"] = away_form[
                    "clean_sheet_rate"
                ]
                match_features[f"away_failed_to_score_rate_{w}"] = away_form[
                    "failed_to_score_rate"
                ]
                match_features[f"away_btts_rate_{w}"] = away_form["btts_rate"]
                match_features[f"away_low_scoring_rate_{w}"] = away_form[
                    "low_scoring_rate"
                ]
                match_features[f"away_goal_variance_{w}"] = away_form["goal_variance"]
                match_features[f"away_points_volatility_{w}"] = away_form[
                    "points_volatility"
                ]

                if self.include_venue_specific:
                    home_home_stats = self._summarize_venue_form(
                        home_venue_history[:w], home_team_id
                    )
                    match_features[f"home_home_form_{w}"] = home_home_stats[
                        "avg_points"
                    ]
                    match_features[f"home_home_draw_rate_{w}"] = home_home_stats[
                        "draw_rate"
                    ]

                    away_away_stats = self._summarize_venue_form(
                        away_venue_history[:w], away_team_id
                    )
                    match_features[f"away_away_form_{w}"] = away_away_stats[
                        "avg_points"
                    ]
                    match_features[f"away_away_draw_rate_{w}"] = away_away_stats[
                        "draw_rate"
                    ]
                    match_features[f"away_away_win_rate_{w}"] = away_away_stats[
                        "win_rate"
                    ]
                    match_features[f"away_away_clean_sheet_rate_{w}"] = away_away_stats[
                        "clean_sheet_rate"
                    ]

            match_features["home_form_trend"] = self._calculate_trend_from_history(
                home_history,
                home_team_id,
            )
            match_features["away_form_trend"] = self._calculate_trend_from_history(
                away_history,
                away_team_id,
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
        return self._summarize_form(matches, team_id)

    def _summarize_extended_form(
        self,
        matches: list[Any],
        team_id: int,
    ) -> dict[str, float]:
        if not matches:
            return {
                "avg_points": 0.0,
                "win_rate": 0.0,
                "avg_goals_for": 0.0,
                "avg_goals_against": 0.0,
                "draw_rate": 0.0,
                "loss_rate": 0.0,
                "clean_sheet_rate": 0.0,
                "failed_to_score_rate": 0.0,
                "btts_rate": 0.0,
                "low_scoring_rate": 0.0,
                "goal_variance": 0.0,
                "points_volatility": 0.0,
            }

        total_points = 0
        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0
        clean_sheets = 0
        failed_to_score = 0
        btts_count = 0
        low_scoring = 0
        points_list = []
        goals_for_list = []

        for match in matches:
            gf, ga = self._goals_for_against(match, team_id)
            goals_for += gf
            goals_against += ga
            goals_for_list.append(gf)

            if gf > ga:
                total_points += 3
                wins += 1
                points_list.append(3)
            elif gf == ga:
                total_points += 1
                draws += 1
                points_list.append(1)
            else:
                losses += 1
                points_list.append(0)

            if ga == 0:
                clean_sheets += 1
            if gf == 0:
                failed_to_score += 1
            if gf > 0 and ga > 0:
                btts_count += 1
            if gf + ga < 2.5:
                low_scoring += 1

        n = len(matches)
        goal_variance = (
            float(np.var(goals_for_list)) if len(goals_for_list) > 1 else 0.0
        )
        points_volatility = float(np.std(points_list)) if len(points_list) > 1 else 0.0

        return {
            "avg_points": total_points / n,
            "win_rate": wins / n,
            "avg_goals_for": goals_for / n,
            "avg_goals_against": goals_against / n,
            "draw_rate": draws / n,
            "loss_rate": losses / n,
            "clean_sheet_rate": clean_sheets / n,
            "failed_to_score_rate": failed_to_score / n,
            "btts_rate": btts_count / n,
            "low_scoring_rate": low_scoring / n,
            "goal_variance": goal_variance,
            "points_volatility": points_volatility,
        }

    def _summarize_venue_form(
        self,
        matches: list[Any],
        team_id: int,
    ) -> dict[str, float]:
        if not matches:
            return {
                "avg_points": 0.0,
                "win_rate": 0.0,
                "draw_rate": 0.0,
                "clean_sheet_rate": 0.0,
            }
        total_points = 0
        wins = 0
        draws = 0
        clean_sheets = 0
        for match in matches:
            gf, ga = self._goals_for_against(match, team_id)
            if gf > ga:
                total_points += 3
                wins += 1
            elif gf == ga:
                total_points += 1
                draws += 1
            if ga == 0:
                clean_sheets += 1
        n = len(matches)
        return {
            "avg_points": total_points / n,
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "clean_sheet_rate": clean_sheets / n,
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
        return self._average_points(matches, team_id)

    def _average_points(self, matches: list[Any], team_id: int) -> float:
        """Calculate average points from an already-fetched match list."""
        if not matches:
            return 0.0

        total_points = 0
        for match in matches:
            gf, ga = self._goals_for_against(match, team_id)

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
        matches = repo.get_team_matches(
            team_id=team_id,
            before_date=reference_date,
            limit=6,
        )

        return self._calculate_trend_from_history(matches, team_id)

    def _calculate_trend_from_history(
        self,
        matches: list[Any],
        team_id: int,
    ) -> float:
        """Calculate form trend from an already-fetched recent history."""
        if len(matches) < 4:
            return 0.0

        recent_avg = self._average_points(matches[:3], team_id)
        earlier_avg = self._average_points(matches[3:6], team_id)

        return recent_avg - earlier_avg

    @staticmethod
    def _goals_for_against(match: Any, team_id: int) -> tuple[int, int]:
        """Return goals for/against from the team's perspective."""
        if match.home_team_id == team_id:
            return match.home_score or 0, match.away_score or 0
        return match.away_score or 0, match.home_score or 0
