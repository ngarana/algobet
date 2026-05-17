"""Tests for EloRatingGenerator and ExpectedPointsGenerator."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from algobet.predictions.features.elo_rating_generator import EloRatingGenerator
from algobet.predictions.features.expected_points_generator import (
    ExpectedPointsGenerator,
    _compute_xpts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeStatistics:
    """Mimic MatchStatistics for test matches."""

    home_xg: float | None = None
    away_xg: float | None = None
    home_npxg: float | None = None
    away_npxg: float | None = None
    home_ppda: float | None = None
    away_ppda: float | None = None
    home_deep_completions: int | None = None
    away_deep_completions: int | None = None


@dataclass
class FakeMatch:
    """Minimal match object that satisfies feature generator attribute access."""

    id: int
    home_team_id: int
    away_team_id: int
    match_date: datetime
    home_score: int | None
    away_score: int | None
    status: str = "FINISHED"
    tournament_id: int | None = None
    season_id: int | None = None
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None
    num_bookmakers: int | None = None
    statistics: Any = None
    player_stats: list[Any] | None = None


class FakeMatchRepository:
    """In-memory MatchRepository backed by a list of FakeMatch objects."""

    def __init__(self, matches: list[FakeMatch] | None = None) -> None:
        self._matches: list[FakeMatch] = matches or []
        self._team_cache: dict[int, list[FakeMatch]] = {}

    def build_cache(self) -> None:
        """Build per-team match cache sorted by date descending."""
        by_team: dict[int, list[FakeMatch]] = {}
        for m in self._matches:
            by_team.setdefault(m.home_team_id, []).append(m)
            by_team.setdefault(m.away_team_id, []).append(m)
        for tid in by_team:
            by_team[tid].sort(key=lambda m: m.match_date, reverse=True)
        self._team_cache = by_team

    def get_team_matches(
        self,
        team_id: int,
        before_date: datetime | None = None,
        limit: int = 10,
        home_only: bool = False,
        away_only: bool = False,
    ) -> list[FakeMatch]:
        if team_id not in self._team_cache:
            return []
        candidates = self._team_cache[team_id]
        if before_date is not None:
            before_ts = pd.Timestamp(before_date)
            candidates = [
                m for m in candidates if pd.Timestamp(m.match_date) < before_ts
            ]
        if home_only:
            candidates = [m for m in candidates if m.home_team_id == team_id]
        if away_only:
            candidates = [m for m in candidates if m.away_team_id == team_id]
        return candidates[:limit]

    def get_h2h_matches(
        self,
        team1_id: int,
        team2_id: int,
        limit: int = 5,
        before_date: datetime | None = None,
    ) -> list[FakeMatch]:
        results = [
            m
            for m in self._matches
            if {m.home_team_id, m.away_team_id} == {team1_id, team2_id}
        ]
        if before_date is not None:
            results = [m for m in results if m.match_date < before_date]
        results.sort(key=lambda m: m.match_date, reverse=True)
        return results[:limit]

    def get_team_standings(self, *args: Any, **kwargs: Any) -> None:
        return None

    def preload_team_matches(self, *args: Any, **kwargs: Any) -> None:
        self.build_cache()

    def preload_h2h_matches(self, *args: Any, **kwargs: Any) -> None:
        pass

    def preload_season_standings(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_historical_matches(self, **kwargs: Any) -> list[FakeMatch]:
        return list(self._matches)

    def clear_cache(self) -> None:
        self._team_cache.clear()


_BASE_DATE = datetime(2024, 1, 1)


def _make_matches_for_elo() -> list[FakeMatch]:
    """Build a sequence of 10 matches between teams 1-4."""
    matches = []
    results = [
        (1, 2, 3, 1),
        (3, 4, 2, 0),
        (1, 3, 2, 2),
        (2, 4, 1, 0),
        (4, 1, 0, 2),
        (2, 3, 1, 1),
        (1, 4, 4, 0),
        (3, 2, 2, 1),
        (4, 2, 1, 1),
        (3, 1, 0, 1),
    ]
    for i, (home, away, hg, ag) in enumerate(results):
        matches.append(
            FakeMatch(
                id=100 + i,
                home_team_id=home,
                away_team_id=away,
                match_date=_BASE_DATE + timedelta(days=i),
                home_score=hg,
                away_score=ag,
                tournament_id=1,
                season_id=1,
            )
        )
    return matches


def _make_matches_for_xpts() -> list[FakeMatch]:
    """Build matches with xG data attached for xPts testing."""
    matches = []
    match_data = [
        (1, 2, 2, 1, 1.8, 0.7),
        (3, 4, 1, 0, 1.2, 0.5),
        (1, 3, 1, 1, 1.0, 1.2),
        (2, 4, 3, 2, 2.5, 1.5),
        (4, 1, 0, 2, 0.4, 2.1),
    ]
    for i, (home, away, hg, ag, hxg, axg) in enumerate(match_data):
        matches.append(
            FakeMatch(
                id=200 + i,
                home_team_id=home,
                away_team_id=away,
                match_date=_BASE_DATE + timedelta(days=i),
                home_score=hg,
                away_score=ag,
                tournament_id=1,
                season_id=1,
                statistics=FakeStatistics(home_xg=hxg, away_xg=axg),
            )
        )
    # Prediction target match
    matches.append(
        FakeMatch(
            id=999,
            home_team_id=1,
            away_team_id=2,
            match_date=_BASE_DATE + timedelta(days=10),
            home_score=None,
            away_score=None,
            tournament_id=1,
            season_id=1,
        )
    )
    return matches


# ---------------------------------------------------------------------------
# Poisson xPts unit tests
# ---------------------------------------------------------------------------


class TestPoissonMatchProbs:
    """Test Poisson-derived xPts mathematics via _compute_xpts."""

    def test_home_advantage_reflected_in_xpts(self) -> None:
        """With equal xG, home and away xPts should be nearly equal."""
        home_xpts, away_xpts = _compute_xpts(1.3, 1.3)
        assert abs(home_xpts - away_xpts) < 1e-5  # symmetric input

    def test_dominant_home(self) -> None:
        home_xpts, away_xpts = _compute_xpts(3.0, 0.5)
        assert home_xpts > 2.0  # Strong home should get > 2 xPts
        assert away_xpts < 0.5  # Weak away should get < 0.5 xPts

    def test_nan_handling(self) -> None:
        home_xpts, away_xpts = _compute_xpts(float("nan"), 1.0)
        assert np.isnan(home_xpts)
        assert np.isnan(away_xpts)

    def test_zero_xg_all_draw(self) -> None:
        """With both teams at xG=0, xPts should reflect pure draw probability."""
        home_xpts, away_xpts = _compute_xpts(0.0, 0.0)
        # Both teams get same xPts (all draws)
        assert abs(home_xpts - away_xpts) < 1e-10
        # With 0-0 xG, P(draw) ≈ 1, so xPts ≈ 1
        assert home_xpts > 0.9


class TestComputeXpts:
    def test_typical_match(self) -> None:
        home_xpts, away_xpts = _compute_xpts(1.5, 0.9)
        assert home_xpts > away_xpts
        assert 0 < home_xpts < 3
        assert 0 < away_xpts < 3

    def test_balanced_match(self) -> None:
        home_xpts, away_xpts = _compute_xpts(1.3, 1.3)
        assert abs(home_xpts - away_xpts) < 1e-5

    def test_xpts_range(self) -> None:
        for hxg, axg in [(0.3, 0.3), (2.5, 0.4), (0.5, 2.8), (1.5, 1.5)]:
            home_xpts, away_xpts = _compute_xpts(hxg, axg)
            assert 0 <= home_xpts <= 3
            assert 0 <= away_xpts <= 3


# ---------------------------------------------------------------------------
# EloRatingGenerator tests
# ---------------------------------------------------------------------------


class TestEloRatingGenerator:
    def test_feature_names_count(self) -> None:
        gen = EloRatingGenerator()
        assert len(gen.feature_names) == 8
        assert gen.name == "elo_rating"

    def test_custom_window_sizes(self) -> None:
        gen = EloRatingGenerator(window_sizes=[5, 10])
        # 2 base features + 2 per window = 2 + 2*2 = 6
        assert len(gen.feature_names) == 6
        assert "home_elo_change_avg_5" in gen.feature_names
        assert "home_elo_change_avg_10" in gen.feature_names

    def test_no_raw_elo_rating_features(self) -> None:
        """Slim generator should NOT emit home_elo_rating or away_elo_rating."""
        gen = EloRatingGenerator()
        assert "home_elo_rating" not in gen.feature_names
        assert "away_elo_rating" not in gen.feature_names
        assert "home_elo_win_rate_5" not in gen.feature_names
        assert "away_elo_win_rate_5" not in gen.feature_names

    def test_required_features_present(self) -> None:
        gen = EloRatingGenerator()
        assert "elo_diff" in gen.feature_names
        assert "elo_expected_home" in gen.feature_names
        assert "home_elo_change_avg_5" in gen.feature_names

    def test_generate_elo_features(self) -> None:
        matches = _make_matches_for_elo()
        repo = FakeMatchRepository(matches)
        repo.build_cache()
        gen = EloRatingGenerator(k_factor=32.0, home_advantage=65.0)

        target_df = pd.DataFrame(
            [
                {
                    "id": 999,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "match_date": _BASE_DATE + timedelta(days=20),
                    "home_score": None,
                    "away_score": None,
                    "tournament_id": 1,
                    "season_id": 1,
                    "status": "SCHEDULED",
                }
            ]
        )

        result = gen.generate(target_df, repo)  # type: ignore[arg-type]
        assert len(result) == 1
        assert result.index[0] == 999

        # Team 1 should have higher Elo diff (won most matches)
        assert result.iloc[0]["elo_diff"] > 0

        # Elo expected home should be > 0.5 (stronger team + home advantage)
        assert result.iloc[0]["elo_expected_home"] > 0.5

    def test_generate_with_no_history(self) -> None:
        gen = EloRatingGenerator(initial_rating=1500.0)
        target_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "home_team_id": 100,
                    "away_team_id": 200,
                    "match_date": _BASE_DATE,
                    "home_score": None,
                    "away_score": None,
                    "tournament_id": 1,
                    "season_id": 1,
                    "status": "SCHEDULED",
                }
            ]
        )
        repo = FakeMatchRepository([])
        repo.build_cache()

        result = gen.generate(target_df, repo)  # type: ignore[arg-type]
        assert len(result) == 1
        # Elo diff should be 0 (equal unknown teams)
        assert abs(result.iloc[0]["elo_diff"]) < 1e-6

        import math

        assert math.isnan(result.iloc[0]["home_elo_change_avg_5"])

    def test_elo_ratings_update_after_match(self) -> None:
        gen = EloRatingGenerator(k_factor=32.0)

        matches = [
            FakeMatch(
                id=1,
                home_team_id=1,
                away_team_id=2,
                match_date=_BASE_DATE,
                home_score=2,
                away_score=0,
            ),
        ]
        repo = FakeMatchRepository(matches)
        repo.build_cache()

        target_df = pd.DataFrame(
            [
                {
                    "id": 999,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "match_date": _BASE_DATE + timedelta(days=1),
                    "home_score": None,
                    "away_score": None,
                    "tournament_id": 1,
                    "season_id": 1,
                    "status": "SCHEDULED",
                }
            ]
        )

        result = gen.generate(target_df, repo)  # type: ignore[arg-type]
        # Team 1 won -> positive Elo change avg; Team 2 lost -> negative
        home_change = result.iloc[0]["home_elo_change_avg_5"]
        away_change = result.iloc[0]["away_elo_change_avg_5"]
        assert home_change > 0
        assert away_change < 0


# ---------------------------------------------------------------------------
# ExpectedPointsGenerator tests
# ---------------------------------------------------------------------------


class TestExpectedPointsGenerator:
    def test_feature_names_count(self) -> None:
        gen = ExpectedPointsGenerator()
        assert len(gen.feature_names) == 15
        assert gen.name == "expected_points"

    def test_custom_window_sizes(self) -> None:
        gen = ExpectedPointsGenerator(window_sizes=[5])
        # 5 features per window: xpts_diff, home/away points_vs_xpts, home/away coverage
        assert len(gen.feature_names) == 5

    def test_no_xpts_avg_features(self) -> None:
        """Slim generator should NOT emit avg features collinear with team_form."""
        gen = ExpectedPointsGenerator()
        for name in gen.feature_names:
            assert "xpts_avg" not in name

    def test_required_features_present(self) -> None:
        gen = ExpectedPointsGenerator()
        assert "xpts_diff_3" in gen.feature_names
        assert "home_points_vs_xpts_3" in gen.feature_names
        assert "away_points_vs_xpts_3" in gen.feature_names
        assert "home_xpts_coverage_3" in gen.feature_names

    def test_generate_xpts_features(self) -> None:
        matches = _make_matches_for_xpts()
        repo = FakeMatchRepository(matches)
        repo.build_cache()
        gen = ExpectedPointsGenerator(window_sizes=[3, 5])

        target_df = pd.DataFrame([matches[-1]])
        result = gen.generate(target_df, repo)  # type: ignore[arg-type]
        assert len(result) == 1
        assert result.index[0] == 999

        row = result.iloc[0]
        # xpts_diff should exist
        assert not np.isnan(row["xpts_diff_3"])

    def test_generate_with_no_xg_data(self) -> None:
        matches = []
        for i in range(5):
            matches.append(
                FakeMatch(
                    id=i + 1,
                    home_team_id=1,
                    away_team_id=2,
                    match_date=_BASE_DATE + timedelta(days=i),
                    home_score=1,
                    away_score=0,
                    statistics=None,
                )
            )
        target = FakeMatch(
            id=999,
            home_team_id=1,
            away_team_id=2,
            match_date=_BASE_DATE + timedelta(days=10),
            home_score=None,
            away_score=None,
            tournament_id=1,
            season_id=1,
        )
        matches.append(target)

        repo = FakeMatchRepository(matches)
        repo.build_cache()
        gen = ExpectedPointsGenerator(window_sizes=[3])

        target_df = pd.DataFrame([target])
        result = gen.generate(target_df, repo)  # type: ignore[arg-type]
        row = result.iloc[0]
        assert row["home_xpts_coverage_3"] == 0.0
        assert np.isnan(row["xpts_diff_3"])

    def test_points_vs_xpts_positive_for_lucky_team(self) -> None:
        matches = [
            FakeMatch(
                id=1,
                home_team_id=1,
                away_team_id=2,
                match_date=_BASE_DATE,
                home_score=2,
                away_score=0,
                statistics=FakeStatistics(home_xg=0.5, away_xg=0.8),
            ),
        ]
        target = FakeMatch(
            id=999,
            home_team_id=1,
            away_team_id=3,
            match_date=_BASE_DATE + timedelta(days=1),
            home_score=None,
            away_score=None,
            tournament_id=1,
            season_id=1,
        )
        matches.append(target)

        repo = FakeMatchRepository(matches)
        repo.build_cache()
        gen = ExpectedPointsGenerator(window_sizes=[3])

        target_df = pd.DataFrame([target])
        result = gen.generate(target_df, repo)  # type: ignore[arg-type]
        row = result.iloc[0]
        assert row["home_points_vs_xpts_3"] > 0


# ---------------------------------------------------------------------------
# Composite generator integration tests
# ---------------------------------------------------------------------------


class TestCompositeIntegration:
    def test_elo_and_xpts_via_composite(self) -> None:
        from algobet.predictions.features.composite import create_generators_by_names

        gen = create_generators_by_names(["elo_rating", "expected_points"])
        names = gen.feature_names
        assert len(names) == 23  # 8 elo + 15 xpts
        assert "elo_diff" in names
        assert "elo_expected_home" in names
        assert "xpts_diff_3" in names
        assert "home_points_vs_xpts_3" in names

    def test_all_groups_including_new(self) -> None:
        from algobet.predictions.features.composite import create_generators_by_names

        all_groups = [
            "team_form",
            "head_to_head",
            "temporal",
            "standings",
            "enriched_stats",
            "elo_rating",
            "expected_points",
        ]
        gen = create_generators_by_names(all_groups)
        assert len(gen.feature_names) > 0

    def test_team_form_plus_elo_plus_xpts_count(self) -> None:
        from algobet.predictions.features.composite import create_generators_by_names

        gen = create_generators_by_names(["team_form", "elo_rating", "expected_points"])
        # 27 team_form + 8 elo + 15 xpts = 50
        assert len(gen.feature_names) == 50


class TestFeatureClassification:
    def test_elo_features_classify_as_elo(self) -> None:
        from algobet.predictions.training.feature_selection import classify_feature

        gen = EloRatingGenerator()
        for name in gen.feature_names:
            family = classify_feature(name)
            assert family == "elo", (
                f"Feature '{name}' classified as '{family}', expected 'elo'"
            )

    def test_xpts_features_classify_as_xpts(self) -> None:
        from algobet.predictions.training.feature_selection import classify_feature

        gen = ExpectedPointsGenerator()
        for name in gen.feature_names:
            family = classify_feature(name)
            assert family == "xpts", (
                f"Feature '{name}' classified as '{family}', expected 'xpts'"
            )

    def test_elo_features_group_by_generator(self) -> None:
        from algobet.predictions.evaluation.ablation import group_features_by_generator
        from algobet.predictions.features.elo_rating_generator import EloRatingGenerator
        from algobet.predictions.features.expected_points_generator import (
            ExpectedPointsGenerator,
        )

        elo_gen = EloRatingGenerator()
        xpts_gen = ExpectedPointsGenerator()
        all_features = elo_gen.feature_names + xpts_gen.feature_names

        groups = group_features_by_generator(
            all_features, ["elo_rating", "expected_points"]
        )

        for f in elo_gen.feature_names:
            assert f in groups.get("elo_rating", []), f"'{f}' not in elo_rating group"

        for f in xpts_gen.feature_names:
            assert f in groups.get("expected_points", []), (
                f"'{f}' not in expected_points group"
            )
