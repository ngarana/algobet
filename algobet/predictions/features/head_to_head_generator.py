"""Head-to-head feature generator."""

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


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
            "h2h_home_win_rate",
            "h2h_avg_total_goals",
            "h2h_home_avg_goals",
            "h2h_away_avg_goals",
            "h2h_recent_home_form",
            "h2h_draw_rate",
            "h2h_away_win_rate",
            "h2h_low_scoring_rate",
            "h2h_btts_rate",
            "h2h_goal_diff_avg_from_home_perspective",
            "h2h_recency_weighted_home_points",
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
        if not h2h_matches:
            nan = float("nan")
            return {
                "h2h_matches_count": 0,
                "h2h_home_win_rate": nan,
                "h2h_avg_total_goals": nan,
                "h2h_home_avg_goals": nan,
                "h2h_away_avg_goals": nan,
                "h2h_recent_home_form": nan,
                "h2h_draw_rate": nan,
                "h2h_away_win_rate": nan,
                "h2h_low_scoring_rate": nan,
                "h2h_btts_rate": nan,
                "h2h_goal_diff_avg_from_home_perspective": nan,
                "h2h_recency_weighted_home_points": nan,
            }

        home_wins = 0
        draws = 0
        away_wins = 0
        total_goals = 0
        home_goals = 0
        away_goals = 0
        low_scoring = 0
        btts_count = 0

        for match in h2h_matches:
            if match.home_team_id == home_team_id:
                h_score = match.home_score or 0
                a_score = match.away_score or 0
            else:
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

            if h_score + a_score < 2.5:
                low_scoring += 1
            if h_score > 0 and a_score > 0:
                btts_count += 1

        n = len(h2h_matches)

        recent_points = 0
        weight_total = 0.0
        for idx, match in enumerate(h2h_matches[:5]):
            weight = 1.0 / (idx + 1)
            if match.home_team_id == home_team_id:
                h_score = match.home_score or 0
                a_score = match.away_score or 0
            else:
                h_score = match.away_score or 0
                a_score = match.home_score or 0
            if h_score > a_score:
                recent_points += 3 * weight
            elif h_score == a_score:
                recent_points += 1 * weight
            weight_total += weight

        return {
            "h2h_matches_count": n,
            "h2h_home_win_rate": home_wins / n,
            "h2h_avg_total_goals": total_goals / n,
            "h2h_home_avg_goals": home_goals / n,
            "h2h_away_avg_goals": away_goals / n,
            "h2h_recent_home_form": recent_points / min(3, n) if n > 0 else 0.0,
            "h2h_draw_rate": draws / n,
            "h2h_away_win_rate": away_wins / n,
            "h2h_low_scoring_rate": low_scoring / n,
            "h2h_btts_rate": btts_count / n,
            "h2h_goal_diff_avg_from_home_perspective": (home_goals - away_goals) / n,
            "h2h_recency_weighted_home_points": (
                recent_points / weight_total if weight_total > 0 else 0.0
            ),
        }
