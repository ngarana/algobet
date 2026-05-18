"""Market-mediation feature generator.

The generator intentionally uses only opening/current market fields.  Closing
odds are labels for CLV training and evaluation, never inference features.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class MarketMediationFeatureGenerator(FeatureGenerator):
    """Generate market features for selective market mediation."""

    @property
    def name(self) -> str:
        return "market_mediation"

    @property
    def feature_names(self) -> list[str]:
        return [
            "mediation_opening_implied_prob_home",
            "mediation_opening_implied_prob_draw",
            "mediation_opening_implied_prob_away",
            "mediation_opening_overround",
            "mediation_opening_entropy",
            "mediation_opening_favorite_outcome",
            "mediation_opening_favorite_prob",
            "mediation_opening_home_away_ratio",
            "mediation_avg_implied_prob_home",
            "mediation_avg_implied_prob_draw",
            "mediation_avg_implied_prob_away",
            "mediation_max_implied_prob_home",
            "mediation_max_implied_prob_draw",
            "mediation_max_implied_prob_away",
            "mediation_bookmaker_disagreement_home",
            "mediation_bookmaker_disagreement_draw",
            "mediation_bookmaker_disagreement_away",
            "mediation_asian_handicap_line",
            "mediation_asian_handicap_home_odds",
            "mediation_asian_handicap_away_odds",
            "mediation_over_under_line",
            "mediation_over_25_odds",
            "mediation_under_25_odds",
            "mediation_over_under_implied_total",
            "mediation_market_data_quality",
        ]

    def generate(
        self,
        matches: pd.DataFrame,
        repository: MatchRepository | None = None,
    ) -> pd.DataFrame:
        rows = []
        for _, match in matches.iterrows():
            features = self._calculate_features(match)
            features["match_id"] = match["id"]
            rows.append(features)
        return pd.DataFrame(rows).set_index("match_id")

    def _calculate_features(self, match: pd.Series) -> dict[str, float]:
        opening = self._first_odds_matrix(
            match,
            (
                ("opening_odds_home", "odds_home"),
                ("opening_odds_draw", "odds_draw"),
                ("opening_odds_away", "odds_away"),
            ),
        )
        avg = self._first_odds_matrix(
            match,
            (
                ("opening_avg_home_odds", "avg_home_odds"),
                ("opening_avg_draw_odds", "avg_draw_odds"),
                ("opening_avg_away_odds", "avg_away_odds"),
            ),
        )
        max_odds = self._first_odds_matrix(
            match,
            (
                ("opening_max_home_odds", "max_home_odds"),
                ("opening_max_draw_odds", "max_draw_odds"),
                ("opening_max_away_odds", "max_away_odds"),
            ),
        )

        opening_probs, opening_overround = self._implied_probs(opening)
        avg_probs, _avg_overround = self._implied_probs(avg)
        max_probs, _max_overround = self._implied_probs(max_odds)
        favorite = max(range(3), key=lambda idx: opening_probs[idx])
        disagreement = [
            self._disagreement(avg[idx], max_odds[idx]) for idx in range(3)
        ]

        ah_line = self._safe_float(match.get("opening_asian_handicap_line"))
        if ah_line is None:
            ah_line = self._safe_float(match.get("odds_asian_handicap_line"))
        ah_home = self._safe_float(match.get("opening_asian_handicap"))
        if ah_home is None:
            ah_home = self._safe_float(match.get("odds_asian_handicap"))
        ah_away = self._safe_float(match.get("opening_asian_handicap_away"))

        ou_line = self._safe_float(match.get("opening_over_under_line"))
        if ou_line is None:
            ou_line = self._safe_float(match.get("odds_over_under_line"))
        over_25 = self._safe_float(match.get("opening_over_under_25"))
        if over_25 is None:
            over_25 = self._safe_float(match.get("odds_over_under_25"))
        under_25 = self._safe_float(match.get("opening_over_under_25_under"))
        if under_25 is None and over_25:
            remaining = 1.05 - (1.0 / over_25)
            under_25 = (1.0 / remaining) if remaining > 0 else None

        ou_total = self._ou_implied_total(over_25, under_25, ou_line or 2.5)
        quality = self._quality_score(opening, avg, max_odds, over_25, ah_line)

        return {
            "mediation_opening_implied_prob_home": opening_probs[0],
            "mediation_opening_implied_prob_draw": opening_probs[1],
            "mediation_opening_implied_prob_away": opening_probs[2],
            "mediation_opening_overround": opening_overround,
            "mediation_opening_entropy": self._entropy(opening_probs),
            "mediation_opening_favorite_outcome": float(favorite),
            "mediation_opening_favorite_prob": opening_probs[favorite],
            "mediation_opening_home_away_ratio": (
                opening[0] / opening[2] if opening[0] > 0 and opening[2] > 0 else 1.0
            ),
            "mediation_avg_implied_prob_home": avg_probs[0],
            "mediation_avg_implied_prob_draw": avg_probs[1],
            "mediation_avg_implied_prob_away": avg_probs[2],
            "mediation_max_implied_prob_home": max_probs[0],
            "mediation_max_implied_prob_draw": max_probs[1],
            "mediation_max_implied_prob_away": max_probs[2],
            "mediation_bookmaker_disagreement_home": disagreement[0],
            "mediation_bookmaker_disagreement_draw": disagreement[1],
            "mediation_bookmaker_disagreement_away": disagreement[2],
            "mediation_asian_handicap_line": ah_line or 0.0,
            "mediation_asian_handicap_home_odds": ah_home or 0.0,
            "mediation_asian_handicap_away_odds": ah_away or 0.0,
            "mediation_over_under_line": ou_line or 2.5,
            "mediation_over_25_odds": over_25 or 0.0,
            "mediation_under_25_odds": under_25 or 0.0,
            "mediation_over_under_implied_total": ou_total,
            "mediation_market_data_quality": quality,
        }

    def _first_odds_matrix(
        self,
        match: pd.Series,
        columns: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    ) -> tuple[float, float, float]:
        values = []
        for candidates in columns:
            value = None
            for column in candidates:
                value = self._safe_float(match.get(column))
                if value is not None:
                    break
            values.append(value if value is not None else 0.0)
        return values[0], values[1], values[2]

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(numeric) or numeric <= 0:
            return None
        return numeric

    @staticmethod
    def _implied_probs(odds: tuple[float, float, float]) -> tuple[list[float], float]:
        if not all(value > 0 for value in odds):
            return [0.40, 0.30, 0.30], 0.0
        raw = [1.0 / value for value in odds]
        total = sum(raw)
        return [value / total for value in raw], total - 1.0

    @staticmethod
    def _entropy(probs: list[float]) -> float:
        return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))

    @staticmethod
    def _disagreement(avg_odds: float, max_odds: float) -> float:
        if avg_odds <= 0 or max_odds <= 0:
            return 0.0
        return float((max_odds - avg_odds) / avg_odds)

    @staticmethod
    def _ou_implied_total(
        over_25: float | None,
        under_25: float | None,
        line: float,
    ) -> float:
        if not over_25 or not under_25:
            return float(line)
        raw_over = 1.0 / over_25
        raw_under = 1.0 / under_25
        total = raw_over + raw_under
        if total <= 0:
            return float(line)
        over_prob = raw_over / total
        return float(line - 0.5 + over_prob)

    @staticmethod
    def _quality_score(
        opening: tuple[float, float, float],
        avg: tuple[float, float, float],
        max_odds: tuple[float, float, float],
        over_25: float | None,
        ah_line: float | None,
    ) -> float:
        present = 0
        present += int(all(value > 0 for value in opening))
        present += int(all(value > 0 for value in avg))
        present += int(all(value > 0 for value in max_odds))
        present += int(over_25 is not None)
        present += int(ah_line is not None)
        return present / 5.0
