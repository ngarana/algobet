"""Elo Rating feature generator for schedule-adjusted team strength.

Computes Elo ratings incrementally across historical matches, producing
features that capture each team's true strength accounting for the
quality of their opponents.

**Collinearity note**: Raw Elo ratings and win rates are highly collinear
with team_form's ``points_last_N`` and ``win_rate_N``.  This generator
deliberately emits only the *complementary* signals that team_form cannot
provide:

- ``elo_diff`` – the schedule-adjusted strength differential (team_form
  computes ``form_diff`` from raw points, which ignores opponent quality).
- ``elo_expected_home`` – the probability of a home win derived from both
  teams' ratings *and* a home-advantage offset; this interaction term is
  unique to Elo.
- ``elo_change_avg_N`` – rolling *schedule-adjusted* momentum (average Elo
  change per match).  Unlike ``form_trend`` which just compares recent vs
  earlier raw points, this accounts for opponent strength.
"""

from typing import Any

import numpy as np
import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class EloRatingGenerator(FeatureGenerator):
    """Generate schedule-adjusted Elo rating features.

    Produces only the complementary (non-collinear with team_form) Elo
    signals: the Elo differential, expected home-win probability, and
    rolling average Elo change as a momentum indicator.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        home_advantage: float = 65.0,
        initial_rating: float = 1500.0,
        window_sizes: list[int] | None = None,
    ) -> None:
        """Initialize Elo rating generator.

        Args:
            k_factor: Elo K-factor controlling rating sensitivity per match.
            home_advantage: Elo points added to home team rating when
                computing expected score.
            initial_rating: Starting Elo rating for teams with no history.
            window_sizes: Window sizes for rolling momentum features.
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.window_sizes = window_sizes or [5, 10, 20]

    @property
    def name(self) -> str:
        return "elo_rating"

    @property
    def feature_names(self) -> list[str]:
        names = [
            "elo_diff",
            "elo_expected_home",
        ]
        for w in self.window_sizes:
            names.extend(
                [
                    f"home_elo_change_avg_{w}",
                    f"away_elo_change_avg_{w}",
                ]
            )
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        """Generate Elo rating features for each match.

        Builds a complete Elo history from all available matches in the
        repository, then looks up pre-match ratings for each input match.

        Args:
            matches: DataFrame with match records
            repository: Repository for historical data queries

        Returns:
            DataFrame indexed by match_id with Elo features
        """
        all_team_ids = set()
        for _, match in matches.iterrows():
            all_team_ids.add(int(match["home_team_id"]))
            all_team_ids.add(int(match["away_team_id"]))

        seen_ids: set[int] = set()
        all_matches: list[Any] = []
        for team_id in all_team_ids:
            team_matches = repository.get_team_matches(
                team_id=team_id,
                limit=10000,
            )
            for m in team_matches:
                if m.id not in seen_ids:
                    all_matches.append(m)
                    seen_ids.add(m.id)

        # Sort chronologically for incremental Elo computation.
        all_matches.sort(key=lambda m: m.match_date)

        # Compute Elo ratings incrementally.
        # elo_history: team_id -> list of (date, pre_match_rating, change)
        elo_ratings: dict[int, float] = {}
        elo_history: dict[int, list[tuple[Any, float, float]]] = {}

        for m in all_matches:
            h_id = m.home_team_id
            a_id = m.away_team_id

            h_rating = elo_ratings.get(h_id, self.initial_rating)
            a_rating = elo_ratings.get(a_id, self.initial_rating)

            h_eff = h_rating + self.home_advantage
            e_h = 1.0 / (1.0 + 10.0 ** ((a_rating - h_eff) / 400.0))
            e_a = 1.0 - e_h

            h_score = m.home_score or 0
            a_score = m.away_score or 0

            if h_score > a_score:
                actual_h, actual_a = 1.0, 0.0
            elif h_score == a_score:
                actual_h, actual_a = 0.5, 0.5
            else:
                actual_h, actual_a = 0.0, 1.0

            h_change = self.k_factor * (actual_h - e_h)
            a_change = self.k_factor * (actual_a - e_a)

            elo_history.setdefault(h_id, []).append((m.match_date, h_rating, h_change))
            elo_history.setdefault(a_id, []).append((m.match_date, a_rating, a_change))

            elo_ratings[h_id] = h_rating + h_change
            elo_ratings[a_id] = a_rating + a_change

        # Build features for each input match
        features = []
        for _, match in matches.iterrows():
            match_id = match["id"]
            match_date = pd.to_datetime(match["match_date"])
            home_team_id = int(match["home_team_id"])
            away_team_id = int(match["away_team_id"])

            match_features: dict[str, Any] = {"match_id": match_id}

            home_elo = self._rating_before(
                elo_history.get(home_team_id, []),
                elo_ratings,
                home_team_id,
                match_date,
            )
            away_elo = self._rating_before(
                elo_history.get(away_team_id, []),
                elo_ratings,
                away_team_id,
                match_date,
            )

            # Only emit complementary features (not raw ratings which
            # are collinear with team_form's points_last / win_rate).
            match_features["elo_diff"] = home_elo - away_elo
            h_eff = home_elo + self.home_advantage
            match_features["elo_expected_home"] = 1.0 / (
                1.0 + 10.0 ** ((away_elo - h_eff) / 400.0)
            )

            # Momentum: average Elo change per match (schedule-adjusted)
            home_before = [
                (d, r, c)
                for d, r, c in elo_history.get(home_team_id, [])
                if d < match_date
            ]
            away_before = [
                (d, r, c)
                for d, r, c in elo_history.get(away_team_id, [])
                if d < match_date
            ]

            for w in self.window_sizes:
                recent_home = home_before[-w:] if home_before else []
                recent_away = away_before[-w:] if away_before else []

                if recent_home:
                    changes = [c for _, _, c in recent_home]
                    match_features[f"home_elo_change_avg_{w}"] = float(np.mean(changes))
                else:
                    match_features[f"home_elo_change_avg_{w}"] = float("nan")

                if recent_away:
                    changes = [c for _, _, c in recent_away]
                    match_features[f"away_elo_change_avg_{w}"] = float(np.mean(changes))
                else:
                    match_features[f"away_elo_change_avg_{w}"] = float("nan")

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _rating_before(
        self,
        history: list[tuple[Any, float, float]],
        current_ratings: dict[int, float],
        team_id: int,
        match_date: pd.Timestamp,
    ) -> float:
        """Get the Elo rating for a team just before a given date."""
        if not history:
            return current_ratings.get(team_id, self.initial_rating)

        latest_rating = self.initial_rating
        for d, r, c in history:
            if d < match_date:
                latest_rating = r + c
            else:
                break

        if latest_rating == self.initial_rating and not any(
            d < match_date for d, _, _ in history
        ):
            return current_ratings.get(team_id, self.initial_rating)

        return latest_rating
