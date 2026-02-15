"""Unit tests for FormCalculator."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from algobet.predictions.features.form_features import FormCalculator


def _make_match(
    home_team_id: int,
    away_team_id: int,
    home_score: int,
    away_score: int,
    match_date: datetime | None = None,
) -> MagicMock:
    """Create a mock Match object."""
    match = MagicMock()
    match.home_team_id = home_team_id
    match.away_team_id = away_team_id
    match.home_score = home_score
    match.away_score = away_score
    match.match_date = match_date or datetime(2025, 1, 1)
    return match


class TestFormCalculator:
    """Tests for FormCalculator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.repo = MagicMock()
        self.calc = FormCalculator(self.repo)
        self.team_id = 1
        self.ref_date = datetime(2025, 6, 1)

    def test_calculate_recent_form_all_wins(self) -> None:
        """Average points should be 3.0 when team wins every match."""
        wins = [
            _make_match(self.team_id, 2, 2, 0),  # Home win
            _make_match(self.team_id, 3, 1, 0),  # Home win
            _make_match(4, self.team_id, 0, 1),  # Away win
        ]
        self.repo.get_team_matches.return_value = wins
        result = self.calc.calculate_recent_form(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )
        assert result == 3.0

    def test_calculate_recent_form_all_losses(self) -> None:
        """Average points should be 0.0 when team loses every match."""
        losses = [
            _make_match(self.team_id, 2, 0, 2),  # Home loss
            _make_match(3, self.team_id, 3, 0),  # Away loss
        ]
        self.repo.get_team_matches.return_value = losses
        result = self.calc.calculate_recent_form(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )
        assert result == 0.0

    def test_calculate_recent_form_draws(self) -> None:
        """Average points should be 1.0 when team draws every match."""
        draws = [
            _make_match(self.team_id, 2, 1, 1),  # Home draw
            _make_match(3, self.team_id, 0, 0),  # Away draw
        ]
        self.repo.get_team_matches.return_value = draws
        result = self.calc.calculate_recent_form(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )
        assert result == 1.0

    def test_calculate_recent_form_mixed(self) -> None:
        """Average points for W/D/L = (3+1+0)/3 = 1.333..."""
        mixed = [
            _make_match(self.team_id, 2, 3, 1),  # Home win (3 pts)
            _make_match(self.team_id, 3, 1, 1),  # Home draw (1 pt)
            _make_match(4, self.team_id, 2, 0),  # Away loss (0 pts)
        ]
        self.repo.get_team_matches.return_value = mixed
        result = self.calc.calculate_recent_form(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )
        assert result == pytest.approx(4.0 / 3.0)

    def test_calculate_recent_form_empty_history(self) -> None:
        """Returns 0.0 when no matches found."""
        self.repo.get_team_matches.return_value = []
        result = self.calc.calculate_recent_form(
            team_id=self.team_id, reference_date=self.ref_date
        )
        assert result == 0.0

    def test_calculate_goals_scored_home(self) -> None:
        """Goals scored as home team."""
        matches = [
            _make_match(self.team_id, 2, 3, 1),  # Scored 3
            _make_match(self.team_id, 3, 1, 0),  # Scored 1
        ]
        self.repo.get_team_matches.return_value = matches
        result = self.calc.calculate_goals_scored(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )
        assert result == 2.0  # (3 + 1) / 2

    def test_calculate_goals_scored_away(self) -> None:
        """Goals scored as away team."""
        matches = [
            _make_match(2, self.team_id, 0, 2),  # Scored 2
            _make_match(3, self.team_id, 1, 4),  # Scored 4
        ]
        self.repo.get_team_matches.return_value = matches
        result = self.calc.calculate_goals_scored(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )
        assert result == 3.0  # (2 + 4) / 2

    def test_calculate_goals_scored_empty(self) -> None:
        """Returns 0.0 when no matches found."""
        self.repo.get_team_matches.return_value = []
        result = self.calc.calculate_goals_scored(
            team_id=self.team_id, reference_date=self.ref_date
        )
        assert result == 0.0

    def test_calculate_goals_conceded_home(self) -> None:
        """Goals conceded as home team = opponent's score."""
        matches = [
            _make_match(self.team_id, 2, 3, 1),  # Conceded 1
            _make_match(self.team_id, 3, 0, 2),  # Conceded 2
        ]
        self.repo.get_team_matches.return_value = matches
        result = self.calc.calculate_goals_conceded(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )
        assert result == 1.5  # (1 + 2) / 2

    def test_calculate_goals_conceded_away(self) -> None:
        """Goals conceded as away team = home team's score."""
        matches = [
            _make_match(2, self.team_id, 3, 0),  # Conceded 3
            _make_match(3, self.team_id, 1, 2),  # Conceded 1
        ]
        self.repo.get_team_matches.return_value = matches
        result = self.calc.calculate_goals_conceded(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )
        assert result == 2.0  # (3 + 1) / 2

    def test_get_form_breakdown(self) -> None:
        """Full form breakdown with all metrics."""
        matches = [
            _make_match(self.team_id, 2, 2, 0),  # Win, GF 2, GA 0
            _make_match(self.team_id, 3, 1, 1),  # Draw, GF 1, GA 1
            _make_match(4, self.team_id, 3, 0),  # Loss, GF 0, GA 3
        ]
        self.repo.get_team_matches.return_value = matches
        breakdown = self.calc.get_form_breakdown(
            team_id=self.team_id, reference_date=self.ref_date, n_matches=5
        )

        assert breakdown["avg_points"] == pytest.approx(4.0 / 3.0)
        assert breakdown["win_rate"] == pytest.approx(1.0 / 3.0)
        assert breakdown["draw_rate"] == pytest.approx(1.0 / 3.0)
        assert breakdown["loss_rate"] == pytest.approx(1.0 / 3.0)
        assert breakdown["avg_goals_for"] == pytest.approx(3.0 / 3.0)
        assert breakdown["avg_goals_against"] == pytest.approx(4.0 / 3.0)

    def test_get_form_breakdown_empty(self) -> None:
        """Empty match history returns zeroed breakdown."""
        self.repo.get_team_matches.return_value = []
        breakdown = self.calc.get_form_breakdown(
            team_id=self.team_id, reference_date=self.ref_date
        )

        assert all(v == 0.0 for v in breakdown.values())

    def test_repo_called_with_correct_params(self) -> None:
        """Verify the repository is called with correct parameters."""
        self.repo.get_team_matches.return_value = []
        self.calc.calculate_recent_form(
            team_id=42, reference_date=self.ref_date, n_matches=10
        )
        self.repo.get_team_matches.assert_called_once_with(
            team_id=42, before_date=self.ref_date, limit=10
        )
