"""Odds-derived feature generator."""

import numpy as np
import pandas as pd

from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.base import FeatureGenerator


class OddsFeatureGenerator(FeatureGenerator):
    """Generate features from betting odds data.

    Converts odds to implied probabilities and calculates market-derived features.
    Handles missing odds gracefully with imputation.
    """

    def __init__(
        self,
        default_margin: float = 0.05,
        impute_missing: bool = True,
    ) -> None:
        """Initialize odds generator.

        Args:
            default_margin: Default bookmaker margin when odds unavailable
            impute_missing: Whether to impute missing odds with market averages
        """
        self.default_margin = default_margin
        self.impute_missing = impute_missing

    @property
    def name(self) -> str:
        return "odds"

    @property
    def feature_names(self) -> list[str]:
        return [
            "implied_prob_home",
            "implied_prob_draw",
            "implied_prob_away",
            "bookmaker_margin",
            "odds_home_away_ratio",
            "favorite_outcome",
            "favorite_implied_prob",
            "odds_quality_score",
        ]

    def generate(
        self, matches: pd.DataFrame, repository: MatchRepository | None = None
    ) -> pd.DataFrame:
        """Generate odds-based features for each match.

        Args:
            matches: Match data with odds columns
            repository: Not used for odds features

        Returns:
            DataFrame with odds features indexed by match id
        """
        features = []

        for _, match in matches.iterrows():
            match_id = match["id"]

            odds_home = match.get("odds_home")
            odds_draw = match.get("odds_draw")
            odds_away = match.get("odds_away")
            num_bookmakers = match.get("num_bookmakers", 0)

            # Handle missing odds
            if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
                # Use default probabilities (home advantage)
                match_features = self._default_features()
            else:
                match_features = self._calculate_odds_features(
                    float(odds_home),
                    float(odds_draw),
                    float(odds_away),
                )

            match_features["match_id"] = match_id
            match_features["odds_quality_score"] = min(1.0, (num_bookmakers or 1) / 5.0)

            features.append(match_features)

        df = pd.DataFrame(features)
        return df.set_index("match_id")

    def _calculate_odds_features(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
    ) -> dict[str, float]:
        """Calculate features from odds."""
        # Implied probabilities
        raw_home = 1 / odds_home
        raw_draw = 1 / odds_draw
        raw_away = 1 / odds_away

        # Bookmaker margin
        margin = raw_home + raw_draw + raw_away - 1

        # Normalized probabilities
        total = raw_home + raw_draw + raw_away
        prob_home = raw_home / total
        prob_draw = raw_draw / total
        prob_away = raw_away / total

        # Identify favorite
        probs = [prob_home, prob_draw, prob_away]
        favorite_idx = np.argmax(probs)

        return {
            "implied_prob_home": prob_home,
            "implied_prob_draw": prob_draw,
            "implied_prob_away": prob_away,
            "bookmaker_margin": margin,
            "odds_home_away_ratio": odds_home / odds_away if odds_away > 0 else 1.0,
            "favorite_outcome": float(favorite_idx),
            "favorite_implied_prob": probs[favorite_idx],
        }

    def _default_features(self) -> dict[str, float]:
        """Return default features when odds unavailable.

        Uses neutral estimates reflecting the average distribution across
        major leagues rather than baking in a strong home-win bias.
        """
        prob_home = 0.40
        prob_draw = 0.30
        prob_away = 0.30
        margin = self.default_margin
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total

        probs = [prob_home, prob_draw, prob_away]
        favorite_idx = int(np.argmax(probs))

        return {
            "implied_prob_home": prob_home,
            "implied_prob_draw": prob_draw,
            "implied_prob_away": prob_away,
            "bookmaker_margin": margin,
            "odds_home_away_ratio": 1.0,
            "favorite_outcome": float(favorite_idx),
            "favorite_implied_prob": probs[favorite_idx],
        }
