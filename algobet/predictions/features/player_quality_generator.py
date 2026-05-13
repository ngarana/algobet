"""Player quality feature generator based on lineup stability."""

from datetime import datetime

import numpy as np
import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class PlayerQualityGenerator(FeatureGenerator):
    """Generate squad stability features from historical lineup data.

    Captures lineup continuity and rotation patterns — information that is
    orthogonal to form stats, xG, and standings. A team with a stable XI
    (high overlap between recent lineups) is likely to have better on-field
    coordination and indicates squad health. High starting-pool size signals
    heavy rotation or injury-driven disruption.

    Features are computed solely from player_match_stats rows PRIOR to the
    prediction date, so they are safe to use for live pre-match inference.
    """

    def __init__(self, window_sizes: list[int] | None = None) -> None:
        self.window_sizes = window_sizes or [3, 5]

    @property
    def name(self) -> str:
        return "player_quality"

    @property
    def feature_names(self) -> list[str]:
        names = []
        for w in self.window_sizes:
            names.extend(
                [
                    f"home_xi_stability_{w}",
                    f"away_xi_stability_{w}",
                    f"xi_stability_diff_{w}",
                ]
            )
        max_w = max(self.window_sizes)
        names.extend(
            [
                f"home_starting_pool_{max_w}",
                f"away_starting_pool_{max_w}",
            ]
        )
        return names

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository
    ) -> pd.DataFrame:
        max_w = max(self.window_sizes)
        records = []

        for _, row in matches.iterrows():
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            match_date: datetime = row["match_date"]

            home_lineups = self._get_lineup_history(
                repository, home_id, match_date, max_w
            )
            away_lineups = self._get_lineup_history(
                repository, away_id, match_date, max_w
            )

            feat: dict[str, float] = {"id": row["id"]}

            for w in self.window_sizes:
                h_stab = self._xi_stability(home_lineups[:w])
                a_stab = self._xi_stability(away_lineups[:w])
                feat[f"home_xi_stability_{w}"] = h_stab
                feat[f"away_xi_stability_{w}"] = a_stab
                if not np.isnan(h_stab) and not np.isnan(a_stab):
                    feat[f"xi_stability_diff_{w}"] = h_stab - a_stab
                else:
                    feat[f"xi_stability_diff_{w}"] = float("nan")

            feat[f"home_starting_pool_{max_w}"] = self._starting_pool(home_lineups)
            feat[f"away_starting_pool_{max_w}"] = self._starting_pool(away_lineups)

            records.append(feat)

        return pd.DataFrame(records).set_index("id")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_lineup_history(
        repository: MatchRepository,
        team_id: int,
        before_date: datetime,
        n: int,
    ) -> list[set[str]]:
        """Return the last *n* starting XIs for a team before *before_date*.

        Each element is a set of player names who were recorded as starters
        (`is_starter=True`) for that team in that match. Empty sets (no
        player-stats data) are excluded so they don't dilute stability scores.

        Most recent lineup is first.
        """
        matches = repository.get_team_matches(team_id, before_date=before_date, limit=n)
        lineups: list[set[str]] = []
        for match in matches:
            starters: set[str] = {
                ps.player_name
                for ps in (match.player_stats or [])
                if ps.team_id == team_id and ps.is_starter
            }
            if starters:
                lineups.append(starters)
        return lineups

    @staticmethod
    def _xi_stability(lineups: list[set[str]]) -> float:
        """Average pairwise overlap between consecutive starting XIs.

        Returns a value in [0, 1]: 1.0 means the exact same players started
        in every adjacent pair; 0.0 means completely different lineups.
        Returns NaN when fewer than two lineups with data are available.
        """
        if len(lineups) < 2:
            return float("nan")
        overlaps: list[float] = []
        for i in range(len(lineups) - 1):
            a, b = lineups[i], lineups[i + 1]
            if not a or not b:
                continue
            # Jaccard-style overlap normalised by the larger XI size
            overlaps.append(len(a & b) / max(len(a), len(b)))
        return float(np.mean(overlaps)) if overlaps else float("nan")

    @staticmethod
    def _starting_pool(lineups: list[set[str]]) -> float:
        """Ratio of distinct starters across all lineups to a standard XI (11).

        A ratio near 1.0 means nearly the same 11 players started every game.
        Values above 1.0 (e.g. 1.5) indicate heavy rotation (15 distinct
        starters over the window), which can signal injury problems or
        deliberate squad rotation.
        """
        if not lineups:
            return float("nan")
        all_starters: set[str] = set()
        for xi in lineups:
            all_starters |= xi
        return len(all_starters) / 11.0
