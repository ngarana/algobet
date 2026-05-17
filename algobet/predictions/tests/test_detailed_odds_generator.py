"""Tests for DetailedOddsFeatureGenerator."""

import pandas as pd

from algobet.predictions.features.detailed_odds_generator import (
    DetailedOddsFeatureGenerator,
)


class TestDetailedOddsFeatureGenerator:
    def test_feature_names_count(self) -> None:
        gen = DetailedOddsFeatureGenerator()
        assert len(gen.feature_names) == 18
        assert gen.name == "detailed_odds"

    def test_required_features_present(self) -> None:
        gen = DetailedOddsFeatureGenerator()
        assert "avg_implied_prob_home" in gen.feature_names
        assert "avg_implied_prob_draw" in gen.feature_names
        assert "avg_implied_prob_away" in gen.feature_names
        assert "asian_handicap_line" in gen.feature_names
        assert "over_under_25_over_odds" in gen.feature_names
        assert "over_under_implied_total" in gen.feature_names
        assert "ah_vs_1x2_spread_diff" in gen.feature_names
        assert "ou_vs_1x2_total_diff" in gen.feature_names

    def test_generate_with_all_odds(self) -> None:
        gen = DetailedOddsFeatureGenerator()

        matches = pd.DataFrame(
            [
                {
                    "id": 1,
                    "odds_home": 2.0,
                    "odds_draw": 3.5,
                    "odds_away": 3.8,
                    "avg_home_odds": 2.05,
                    "avg_draw_odds": 3.45,
                    "avg_away_odds": 3.75,
                    "max_home_odds": 2.15,
                    "max_draw_odds": 3.60,
                    "max_away_odds": 3.90,
                    "odds_asian_handicap_line": -0.5,
                    "odds_asian_handicap": 1.90,
                    "odds_over_under_25": 1.85,
                    "odds_over_under_line": 2.5,
                }
            ]
        )

        result = gen.generate(matches)
        assert len(result) == 1
        assert result.index[0] == 1

        row = result.iloc[0]
        # Implied probabilities should sum to ~1
        total_prob = (
            row["avg_implied_prob_home"]
            + row["avg_implied_prob_draw"]
            + row["avg_implied_prob_away"]
        )
        assert abs(total_prob - 1.0) < 0.01

        # Disagreement should be positive (max > avg)
        assert row["odds_disagreement_home"] > 0
        assert row["odds_disagreement_draw"] > 0
        assert row["odds_disagreement_away"] > 0

        # AH line should be present
        assert row["asian_handicap_line"] == -0.5

    def test_generate_with_missing_odds(self) -> None:
        gen = DetailedOddsFeatureGenerator()

        matches = pd.DataFrame(
            [
                {
                    "id": 2,
                    "odds_home": 2.0,
                    "odds_draw": 3.5,
                    "odds_away": 3.8,
                    "avg_home_odds": None,
                    "avg_draw_odds": None,
                    "avg_away_odds": None,
                    "max_home_odds": None,
                    "max_draw_odds": None,
                    "max_away_odds": None,
                    "odds_asian_handicap_line": None,
                    "odds_asian_handicap": None,
                    "odds_over_under_25": None,
                    "odds_over_under_line": None,
                }
            ]
        )

        result = gen.generate(matches)
        assert len(result) == 1

        row = result.iloc[0]
        # Should use default probabilities
        assert abs(row["avg_implied_prob_home"] - 0.40) < 0.01
        assert abs(row["avg_implied_prob_draw"] - 0.30) < 0.01
        assert abs(row["avg_implied_prob_away"] - 0.30) < 0.01

    def test_generate_with_partial_odds(self) -> None:
        gen = DetailedOddsFeatureGenerator()

        matches = pd.DataFrame(
            [
                {
                    "id": 3,
                    "odds_home": 2.0,
                    "odds_draw": 3.5,
                    "odds_away": 3.8,
                    "avg_home_odds": 2.05,
                    "avg_draw_odds": 3.45,
                    "avg_away_odds": None,
                    "max_home_odds": 2.15,
                    "max_draw_odds": None,
                    "max_away_odds": None,
                    "odds_asian_handicap_line": 0.0,
                    "odds_asian_handicap": 1.95,
                    "odds_over_under_25": 1.90,
                    "odds_over_under_line": 2.5,
                }
            ]
        )

        result = gen.generate(matches)
        assert len(result) == 1
        # Should not raise, should handle partial data gracefully

    def test_disagreement_calculation(self) -> None:
        gen = DetailedOddsFeatureGenerator()

        matches = pd.DataFrame(
            [
                {
                    "id": 4,
                    "odds_home": 2.0,
                    "odds_draw": 3.5,
                    "odds_away": 3.8,
                    "avg_home_odds": 2.00,
                    "avg_draw_odds": 3.50,
                    "avg_away_odds": 3.80,
                    "max_home_odds": 2.20,
                    "max_draw_odds": 3.85,
                    "max_away_odds": 4.18,
                    "odds_asian_handicap_line": None,
                    "odds_asian_handicap": None,
                    "odds_over_under_25": None,
                    "odds_over_under_line": None,
                }
            ]
        )

        result = gen.generate(matches)
        row = result.iloc[0]

        # Disagreement = (max - avg) / avg
        expected_home_disagreement = (2.20 - 2.00) / 2.00
        assert abs(row["odds_disagreement_home"] - expected_home_disagreement) < 1e-6

    def test_cross_market_features(self) -> None:
        gen = DetailedOddsFeatureGenerator()

        matches = pd.DataFrame(
            [
                {
                    "id": 5,
                    "odds_home": 1.8,
                    "odds_draw": 3.6,
                    "odds_away": 4.5,
                    "avg_home_odds": 1.85,
                    "avg_draw_odds": 3.55,
                    "avg_away_odds": 4.40,
                    "max_home_odds": 1.90,
                    "max_draw_odds": 3.70,
                    "max_away_odds": 4.60,
                    "odds_asian_handicap_line": -1.0,
                    "odds_asian_handicap": 1.85,
                    "odds_over_under_25": 1.75,
                    "odds_over_under_line": 2.5,
                }
            ]
        )

        result = gen.generate(matches)
        row = result.iloc[0]

        # Cross-market features should be computed
        assert "ah_vs_1x2_spread_diff" in row.index
        assert "ou_vs_1x2_total_diff" in row.index
