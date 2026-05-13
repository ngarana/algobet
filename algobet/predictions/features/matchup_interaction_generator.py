"""Matchup interaction feature generator.

Produces cross-team interaction features that capture how two teams'
styles and strengths relate to each other. Unlike univariate form
features, these measure the specific matchup — a genuine predictor
for draws (which happen when styles neutralize each other).
"""

from typing import Any

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class MatchupInteractionGenerator(FeatureGenerator):
    """Generate cross-team matchup interaction features.

    These features capture pairwise relationships between the two teams'
    recent form, addressing the core issue that univariate team summaries
    cannot distinguish draws from narrow home/away wins.
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
    ) -> None:
        """Initialize matchup interaction generator.

        Args:
            window_sizes: Rolling windows to use (default: [5, 10])
        """
        self.window_sizes = window_sizes or [5, 10]

    @property
    def name(self) -> str:
        return "matchup_interaction"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for w in self.window_sizes:
            names.extend(
                [
                    # Strength parity: how close are the teams in quality
                    f"strength_parity_{w}",
                    # Combined defensive strength: both teams defend well → low scoring
                    f"combined_defensive_strength_{w}",
                    # Offensive balance: both teams attack similarly
                    f"offensive_balance_{w}",
                    # Total goal parity: how close are total goal rates
                    f"total_goal_parity_{w}",
                    # Low-scoring matchup: both score few → 0-0 draw
                    f"low_scoring_matchup_{w}",
                    # Win rate gap: decisive strength difference
                    f"win_rate_gap_{w}",
                    # Attack-defense clash: home attack vs away defense (and vice versa)
                    # Draws are likely when BOTH clashes are simultaneously tight
                    f"bilateral_neutralization_{w}",
                    f"home_attack_nullified_{w}",
                    f"away_attack_nullified_{w}",
                ]
            )
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate matchup interaction features.

        Args:
            matches: Match data
            repository: Repository for historical queries

        Returns:
            DataFrame with interaction features indexed by match id
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
                home_stats = self._compute_team_stats(home_history[:w], home_team_id)
                away_stats = self._compute_team_stats(away_history[:w], away_team_id)

                if home_stats is None or away_stats is None:
                    for feat in [
                        f"strength_parity_{w}",
                        f"combined_defensive_strength_{w}",
                        f"offensive_balance_{w}",
                        f"total_goal_parity_{w}",
                        f"low_scoring_matchup_{w}",
                        f"win_rate_gap_{w}",
                        f"bilateral_neutralization_{w}",
                        f"home_attack_nullified_{w}",
                        f"away_attack_nullified_{w}",
                    ]:
                        match_features[feat] = float("nan")
                    continue

                # Strength parity: 1/(1 + |points_diff|) → 1.0 = equal
                match_features[f"strength_parity_{w}"] = 1.0 / (
                    1.0 + abs(home_stats["ppg"] - away_stats["ppg"])
                )

                # Combined defensive strength: 1.0 / (1.0 + sum of goals against)
                ha_def = home_stats["goals_against_avg"]
                aa_def = away_stats["goals_against_avg"]
                match_features[f"combined_defensive_strength_{w}"] = 1.0 / (
                    1.0 + ha_def + aa_def
                )

                # Offensive balance: similarity of attacking output
                ha_off = home_stats["goals_for_avg"]
                aa_off = away_stats["goals_for_avg"]
                match_features[f"offensive_balance_{w}"] = 1.0 / (
                    1.0 + abs(ha_off - aa_off)
                )

                # Total goal parity: how close are total goal rates
                home_total_goals = (
                    home_stats["goals_for_avg"] + home_stats["goals_against_avg"]
                )
                away_total_goals = (
                    away_stats["goals_for_avg"] + away_stats["goals_against_avg"]
                )
                match_features[f"total_goal_parity_{w}"] = 1.0 / (
                    1.0 + abs(home_total_goals - away_total_goals)
                )

                # Low scoring matchup: both teams in low-scoring matches
                match_features[f"low_scoring_matchup_{w}"] = (
                    home_stats["low_scoring_rate"] * away_stats["low_scoring_rate"]
                )

                # Win rate gap: absolute difference (high → decisive, low → draw)
                match_features[f"win_rate_gap_{w}"] = abs(
                    home_stats["win_rate"] - away_stats["win_rate"]
                )

                # Attack-defense clash: how well does each team's attack match
                # the opposing defense? Draws happen when BOTH clashes are tight
                # (home attack ≈ away defense AND away attack ≈ home defense).
                home_clash = abs(ha_off - aa_def)
                away_clash = abs(aa_off - ha_def)
                match_features[f"bilateral_neutralization_{w}"] = 1.0 / (
                    1.0 + home_clash + away_clash
                )
                match_features[f"home_attack_nullified_{w}"] = 1.0 / (1.0 + home_clash)
                match_features[f"away_attack_nullified_{w}"] = 1.0 / (1.0 + away_clash)

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    @staticmethod
    def _compute_team_stats(
        matches: list[Any], team_id: int
    ) -> dict[str, float] | None:
        """Compute summary statistics for a team's recent matches.

        Args:
            matches: Recent match history
            team_id: Team to compute stats for

        Returns:
            Dict of computed stats, or None if insufficient data
        """
        if not matches:
            return None

        wins = 0
        draws = 0
        goals_for = 0
        goals_against = 0
        low_scoring = 0
        n = len(matches)

        for m in matches:
            is_home = m.home_team_id == team_id
            gf = m.home_score if is_home else m.away_score
            ga = m.away_score if is_home else m.home_score

            if gf is None or ga is None:
                continue

            goals_for += gf
            goals_against += ga

            if gf > ga:
                wins += 1
            elif gf == ga:
                draws += 1

            if gf + ga <= 1:
                low_scoring += 1

        return {
            "ppg": (wins * 3 + draws) / n,
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "goals_for_avg": goals_for / n,
            "goals_against_avg": goals_against / n,
            "low_scoring_rate": low_scoring / n,
        }
