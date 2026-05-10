"""Standings feature generator."""

from typing import Any

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


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
            "home_draw_rate_season",
            "away_draw_rate_season",
            "home_loss_rate_season",
            "away_loss_rate_season",
            "draw_rate_diff_season",
            "loss_rate_diff_season",
            "points_per_game_diff",
            "home_top_six",
            "away_top_six",
            "home_bottom_six",
            "away_bottom_six",
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
