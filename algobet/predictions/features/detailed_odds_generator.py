"""Detailed odds feature generator from Football-Data.co.uk data.

Generates features from multi-bookmaker odds, Asian handicap, and
over/under markets. These markets contain independent information
about expected goal totals and spreads that 1X2 odds alone cannot capture.
"""

import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class DetailedOddsFeatureGenerator(FeatureGenerator):
    """Generate features from detailed betting odds data.

    Extracts signals from:
    - Multi-bookmaker consensus (avg/max odds across bookmakers)
    - Asian handicap (implied goal spread)
    - Over/Under 2.5 (implied total goals)
    - Bookmaker disagreement (max - avg as uncertainty signal)
    """

    def __init__(
        self,
        default_margin: float = 0.05,
        impute_missing: bool = True,
    ) -> None:
        """Initialize detailed odds generator.

        Args:
            default_margin: Default bookmaker margin when odds unavailable
            impute_missing: Whether to impute missing odds with neutral values
        """
        self.default_margin = default_margin
        self.impute_missing = impute_missing

    @property
    def name(self) -> str:
        return "detailed_odds"

    @property
    def feature_names(self) -> list[str]:
        return [
            # Multi-bookmaker consensus features
            "avg_implied_prob_home",
            "avg_implied_prob_draw",
            "avg_implied_prob_away",
            "avg_bookmaker_margin",
            # Max odds features (best available price)
            "max_implied_prob_home",
            "max_implied_prob_draw",
            "max_implied_prob_away",
            # Bookmaker disagreement (uncertainty signal)
            "odds_disagreement_home",
            "odds_disagreement_draw",
            "odds_disagreement_away",
            # Asian handicap features
            "asian_handicap_line",
            "asian_handicap_implied_margin",
            # Over/Under features
            "over_under_25_over_odds",
            "over_under_25_under_odds",
            "over_under_implied_total",
            "over_under_implied_margin",
            # Cross-market features
            "ah_vs_1x2_spread_diff",
            "ou_vs_1x2_total_diff",
        ]

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository | None = None
    ) -> pd.DataFrame:
        """Generate detailed odds features for each match.

        Args:
            matches: Match data with detailed odds columns
            repository: Not used for detailed odds features

        Returns:
            DataFrame with detailed odds features indexed by match id
        """
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]

            match_features = self._calculate_features(match)
            match_features["match_id"] = match_id

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _calculate_features(self, match: pd.Series) -> dict[str, float]:
        """Calculate all detailed odds features for a single match."""
        # Extract odds values
        avg_home = self._safe_float(match.get("avg_home_odds"))
        avg_draw = self._safe_float(match.get("avg_draw_odds"))
        avg_away = self._safe_float(match.get("avg_away_odds"))
        max_home = self._safe_float(match.get("max_home_odds"))
        max_draw = self._safe_float(match.get("max_draw_odds"))
        max_away = self._safe_float(match.get("max_away_odds"))

        # AH line can be negative (handicap), so use separate extraction
        ah_line_raw = match.get("odds_asian_handicap_line")
        ah_line = self._safe_float_or_zero(ah_line_raw)

        ou_over = self._safe_float(match.get("odds_over_under_25"))
        # We don't store ou_under in the DB currently.
        # We can approximate it assuming a 5% bookmaker margin: 1/over + 1/under = 1.05
        ou_under = None
        if ou_over and (1.05 - 1.0 / ou_over) > 0:
            ou_under = 1.0 / (1.05 - 1.0 / ou_over)

        # Also get 1X2 odds for cross-market comparison
        odds_home = self._safe_float(match.get("odds_home"))
        odds_draw = self._safe_float(match.get("odds_draw"))
        odds_away = self._safe_float(match.get("odds_away"))

        # Calculate consensus features
        if avg_home and avg_draw and avg_away:
            avg_probs = self._implied_probs(avg_home, avg_draw, avg_away)
            avg_margin = sum([1 / avg_home, 1 / avg_draw, 1 / avg_away]) - 1.0
        else:
            avg_probs = self._default_probs()
            avg_margin = self.default_margin

        # Calculate max odds features
        if max_home and max_draw and max_away:
            max_probs = self._implied_probs(max_home, max_draw, max_away)
        else:
            max_probs = self._default_probs()

        # Bookmaker disagreement (max - avg as % of avg)
        disagreement_home = self._disagreement(avg_home, max_home)
        disagreement_draw = self._disagreement(avg_draw, max_draw)
        disagreement_away = self._disagreement(avg_away, max_away)

        # Asian handicap features
        # We do not store the away AH odds, so we assume a standard 5% margin
        ah_implied_margin = 0.05

        # Over/Under features
        ou_implied_total = 2.5  # default
        ou_implied_margin = 0.0
        if ou_over and ou_under:
            ou_implied_margin = (1 / ou_over + 1 / ou_under) - 1.0
            # Implied total goals from O/U odds
            over_prob = (1 / ou_over) / (1 / ou_over + 1 / ou_under)
            # Simple mapping: higher over prob -> higher expected total
            ou_implied_total = 2.0 + over_prob * 1.5  # range ~2.0 to ~3.5

        # Cross-market features
        ah_vs_1x2_spread_diff = 0.0
        if ah_line and odds_home and odds_away:
            # Compare AH line to 1X2 implied spread
            home_prob = 1 / odds_home
            away_prob = 1 / odds_away
            total = home_prob + away_prob
            implied_spread = (home_prob - away_prob) / total
            ah_vs_1x2_spread_diff = ah_line - (implied_spread * 3.0)

        ou_vs_1x2_total_diff = 0.0
        if odds_home and odds_draw and odds_away:
            # 1X2 implied total goals (simplified)
            draw_prob = 1 / odds_draw
            # Higher draw prob -> lower expected total
            implied_total_1x2 = 3.0 - draw_prob * 2.0
            ou_vs_1x2_total_diff = ou_implied_total - implied_total_1x2

        return {
            # Consensus features
            "avg_implied_prob_home": avg_probs[0],
            "avg_implied_prob_draw": avg_probs[1],
            "avg_implied_prob_away": avg_probs[2],
            "avg_bookmaker_margin": avg_margin,
            # Max odds features
            "max_implied_prob_home": max_probs[0],
            "max_implied_prob_draw": max_probs[1],
            "max_implied_prob_away": max_probs[2],
            # Disagreement features
            "odds_disagreement_home": disagreement_home,
            "odds_disagreement_draw": disagreement_draw,
            "odds_disagreement_away": disagreement_away,
            # Asian handicap features
            "asian_handicap_line": ah_line if ah_line else 0.0,
            "asian_handicap_implied_margin": ah_implied_margin,
            # Over/Under features
            "over_under_25_over_odds": ou_over if ou_over else 0.0,
            "over_under_25_under_odds": ou_under if ou_under else 0.0,
            "over_under_implied_total": ou_implied_total,
            "over_under_implied_margin": ou_implied_margin,
            # Cross-market features
            "ah_vs_1x2_spread_diff": ah_vs_1x2_spread_diff,
            "ou_vs_1x2_total_diff": ou_vs_1x2_total_diff,
        }

    def _implied_probs(
        self, odds_home: float, odds_draw: float, odds_away: float
    ) -> tuple[float, float, float]:
        """Calculate normalized implied probabilities from odds."""
        raw_home = 1 / odds_home
        raw_draw = 1 / odds_draw
        raw_away = 1 / odds_away
        total = raw_home + raw_draw + raw_away
        return raw_home / total, raw_draw / total, raw_away / total

    def _default_probs(self) -> tuple[float, float, float]:
        """Return neutral default probabilities."""
        return (0.40, 0.30, 0.30)

    def _disagreement(self, avg_odds: float | None, max_odds: float | None) -> float:
        """Calculate bookmaker disagreement as (max - avg) / avg."""
        if not avg_odds or not max_odds or avg_odds == 0:
            return 0.0
        return (max_odds - avg_odds) / avg_odds

    def _safe_float(self, val: object) -> float | None:
        """Safely convert a value to float, returning None on failure."""
        if val is None:
            return None
        try:
            v = float(val)
            if pd.isna(v) or v <= 0:
                return None
            return v
        except (ValueError, TypeError):
            return None

    def _safe_float_or_zero(self, val: object) -> float:
        """Safely convert a value to float, returning 0.0 on failure.

        Used for values that can be negative (e.g., AH line).
        """
        if val is None:
            return 0.0
        try:
            v = float(val)
            return 0.0 if pd.isna(v) else v
        except (ValueError, TypeError):
            return 0.0
