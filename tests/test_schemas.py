"""Tests for Pydantic schemas."""

from datetime import datetime

import pytest

from algobet.api.schemas import (
    FormBreakdown,
    MatchFilters,
    MatchResponse,
    MatchStatus,
    ModelVersionResponse,
    PredictedOutcome,
    PredictionResponse,
    SeasonResponse,
    TeamResponse,
    TournamentResponse,
    ValueBetResponse,
)


class TestTournamentSchema:
    """Tests for TournamentResponse schema."""

    def test_valid_tournament(self) -> None:
        """Test creating a valid tournament response."""
        tournament = TournamentResponse(
            id=1,
            name="Premier League",
            country="England",
            url_slug="premier-league",
        )
        assert tournament.id == 1
        assert tournament.name == "Premier League"
        assert tournament.country == "England"
        assert tournament.url_slug == "premier-league"


class TestSeasonSchema:
    """Tests for SeasonResponse schema."""

    def test_valid_season(self) -> None:
        """Test creating a valid season response."""
        season = SeasonResponse(
            id=1,
            tournament_id=1,
            name="2023/2024",
            start_year=2023,
            end_year=2024,
            url_suffix="2023-2024",
        )
        assert season.id == 1
        assert season.tournament_id == 1
        assert season.name == "2023/2024"
        assert season.start_year == 2023
        assert season.end_year == 2024

    def test_season_without_url_suffix(self) -> None:
        """Test creating a season without URL suffix."""
        season = SeasonResponse(
            id=1,
            tournament_id=1,
            name="2024/2025",
            start_year=2024,
            end_year=2025,
            url_suffix=None,
        )
        assert season.url_suffix is None


class TestTeamSchema:
    """Tests for TeamResponse schema."""

    def test_valid_team(self) -> None:
        """Test creating a valid team response."""
        team = TeamResponse(id=1, name="Manchester United")
        assert team.id == 1
        assert team.name == "Manchester United"


class TestFormBreakdownSchema:
    """Tests for FormBreakdown schema."""

    def test_valid_form_breakdown(self) -> None:
        """Test creating a valid form breakdown."""
        form = FormBreakdown(
            avg_points=2.1,
            win_rate=0.7,
            draw_rate=0.2,
            loss_rate=0.1,
            avg_goals_for=2.3,
            avg_goals_against=0.9,
        )
        assert form.avg_points == 2.1
        assert form.win_rate + form.draw_rate + form.loss_rate == pytest.approx(1.0)


class TestMatchSchema:
    """Tests for MatchResponse schema."""

    def test_valid_scheduled_match(self) -> None:
        """Test creating a valid scheduled match."""
        match = MatchResponse(
            id=1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime.now(),
            home_score=None,
            away_score=None,
            status=MatchStatus.SCHEDULED,
            odds_home=2.10,
            odds_draw=3.40,
            odds_away=3.20,
            num_bookmakers=12,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            result=None,
        )
        assert match.status == MatchStatus.SCHEDULED
        assert match.result is None

    def test_valid_finished_match(self) -> None:
        """Test creating a valid finished match."""
        match = MatchResponse(
            id=1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime.now(),
            home_score=2,
            away_score=1,
            status=MatchStatus.FINISHED,
            odds_home=None,
            odds_draw=None,
            odds_away=None,
            num_bookmakers=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            result=PredictedOutcome.HOME,
        )
        assert match.status == MatchStatus.FINISHED
        assert match.home_score == 2
        assert match.away_score == 1
        assert match.result == PredictedOutcome.HOME

    def test_invalid_status_raises_error(self) -> None:
        """Test that invalid status raises validation error."""
        with pytest.raises(ValueError, match="Invalid status"):
            MatchResponse(
                id=1,
                tournament_id=1,
                season_id=1,
                home_team_id=1,
                away_team_id=2,
                match_date=datetime.now(),
                home_score=None,
                away_score=None,
                status="INVALID_STATUS",
                odds_home=None,
                odds_draw=None,
                odds_away=None,
                num_bookmakers=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                result=None,
            )

    def test_invalid_limit_raises_error(self) -> None:
        """Test that invalid limit raises validation error."""
        with pytest.raises(ValueError, match="limit must be between"):
            MatchFilters(limit=150)


class TestMatchFiltersSchema:
    """Tests for MatchFilters schema."""

    def test_default_filters(self) -> None:
        """Test default filter values."""
        filters = MatchFilters()
        assert filters.limit == 50
        assert filters.offset == 0
        assert filters.status is None

    def test_filters_with_status(self) -> None:
        """Test filters with status specified."""
        filters = MatchFilters(status=MatchStatus.SCHEDULED, limit=20)
        assert filters.status == MatchStatus.SCHEDULED
        assert filters.limit == 20


class TestPredictionSchema:
    """Tests for PredictionResponse schema."""

    def test_valid_prediction(self) -> None:
        """Test creating a valid prediction."""
        prediction = PredictionResponse(
            id=1,
            match_id=1,
            model_version_id=1,
            prob_home=0.45,
            prob_draw=0.30,
            prob_away=0.25,
            predicted_outcome=PredictedOutcome.HOME,
            confidence=0.65,
            predicted_at=datetime.now(),
            actual_roi=0.10,
            max_probability=0.45,
        )
        assert (
            prediction.prob_home + prediction.prob_draw + prediction.prob_away
            == pytest.approx(1.0)
        )
        assert prediction.predicted_outcome == PredictedOutcome.HOME
        assert prediction.max_probability == 0.45

    def test_invalid_probability_raises_error(self) -> None:
        """Test that invalid probability raises validation error."""
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            PredictionResponse(
                id=1,
                match_id=1,
                model_version_id=1,
                prob_home=1.5,  # Invalid
                prob_draw=0.30,
                prob_away=0.25,
                predicted_outcome=PredictedOutcome.HOME,
                confidence=0.65,
                predicted_at=datetime.now(),
                actual_roi=None,
                max_probability=1.5,
            )

    def test_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence raises validation error."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            PredictionResponse(
                id=1,
                match_id=1,
                model_version_id=1,
                prob_home=0.45,
                prob_draw=0.30,
                prob_away=0.25,
                predicted_outcome=PredictedOutcome.HOME,
                confidence=1.5,  # Invalid
                predicted_at=datetime.now(),
                actual_roi=None,
                max_probability=0.45,
            )


class TestModelVersionSchema:
    """Tests for ModelVersionResponse schema."""

    def test_valid_model_version(self) -> None:
        """Test creating a valid model version."""
        model = ModelVersionResponse(
            id=1,
            name="XGBoost Predictor",
            version="1.0.0",
            algorithm="xgboost",
            accuracy=0.72,
            file_path="/models/xgboost_v1.pkl",
            is_active=True,
            created_at=datetime.now(),
            metrics={"accuracy": 0.72, "precision": 0.68, "recall": 0.65},
            hyperparameters={"n_estimators": 100, "max_depth": 6},
            feature_schema_version="1.0",
            description="XGBoost model for match outcome prediction",
        )
        assert model.name == "XGBoost Predictor"
        assert model.algorithm == "xgboost"
        assert model.is_active is True
        assert model.accuracy == 0.72

    def test_invalid_accuracy_raises_error(self) -> None:
        """Test that invalid accuracy raises validation error."""
        with pytest.raises(ValueError, match="Accuracy must be between 0 and 1"):
            ModelVersionResponse(
                id=1,
                name="Test Model",
                version="1.0.0",
                algorithm="xgboost",
                accuracy=1.5,  # Invalid
                file_path="/models/test.pkl",
                is_active=False,
                created_at=datetime.now(),
                metrics=None,
                hyperparameters=None,
                feature_schema_version=None,
                description=None,
            )


class TestValueBetSchema:
    """Tests for ValueBetResponse schema."""

    def test_valid_value_bet(self) -> None:
        """Test creating a valid value bet response."""
        match = MatchResponse(
            id=1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime.now(),
            home_score=None,
            away_score=None,
            status=MatchStatus.SCHEDULED,
            odds_home=2.50,
            odds_draw=3.40,
            odds_away=2.80,
            num_bookmakers=15,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            result=None,
        )
        value_bet = ValueBetResponse(
            match=match,
            prediction_id=1,
            predicted_outcome="H",
            predicted_probability=0.50,
            market_odds=2.50,
            expected_value=0.25,
            kelly_fraction=0.10,
            confidence=0.65,
        )
        assert value_bet.prediction_id == 1
        assert value_bet.predicted_outcome == "H"
        assert value_bet.predicted_probability == 0.50
        assert value_bet.expected_value == 0.25
        assert value_bet.match.id == 1

    def test_invalid_predicted_outcome_raises_error(self) -> None:
        """Test that invalid predicted outcome raises validation error."""
        match = MatchResponse(
            id=1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime.now(),
            home_score=None,
            away_score=None,
            status=MatchStatus.SCHEDULED,
            odds_home=2.50,
            odds_draw=3.40,
            odds_away=2.80,
            num_bookmakers=15,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            result=None,
        )
        with pytest.raises(ValueError, match="Invalid predicted outcome"):
            ValueBetResponse(
                match=match,
                prediction_id=1,
                predicted_outcome="X",  # Invalid
                predicted_probability=0.50,
                market_odds=2.50,
                expected_value=0.25,
                kelly_fraction=0.10,
                confidence=0.65,
            )

    def test_invalid_predicted_probability_raises_error(self) -> None:
        """Test that invalid predicted probability raises validation error."""
        match = MatchResponse(
            id=1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime.now(),
            home_score=None,
            away_score=None,
            status=MatchStatus.SCHEDULED,
            odds_home=2.50,
            odds_draw=3.40,
            odds_away=2.80,
            num_bookmakers=15,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            result=None,
        )
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            ValueBetResponse(
                match=match,
                prediction_id=1,
                predicted_outcome="H",
                predicted_probability=1.5,  # Invalid
                market_odds=2.50,
                expected_value=0.25,
                kelly_fraction=0.10,
                confidence=0.65,
            )

    def test_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence raises validation error."""
        match = MatchResponse(
            id=1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime.now(),
            home_score=None,
            away_score=None,
            status=MatchStatus.SCHEDULED,
            odds_home=2.50,
            odds_draw=3.40,
            odds_away=2.80,
            num_bookmakers=15,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            result=None,
        )
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            ValueBetResponse(
                match=match,
                prediction_id=1,
                predicted_outcome="H",
                predicted_probability=0.50,
                market_odds=2.50,
                expected_value=0.25,
                kelly_fraction=0.10,
                confidence=1.5,  # Invalid
            )

    def test_invalid_expected_value_raises_error(self) -> None:
        """Test that unreasonable expected value raises validation error."""
        match = MatchResponse(
            id=1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime.now(),
            home_score=None,
            away_score=None,
            status=MatchStatus.SCHEDULED,
            odds_home=2.50,
            odds_draw=3.40,
            odds_away=2.80,
            num_bookmakers=15,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            result=None,
        )
        with pytest.raises(ValueError, match="Expected value seems unreasonable"):
            ValueBetResponse(
                match=match,
                prediction_id=1,
                predicted_outcome="H",
                predicted_probability=0.50,
                market_odds=2.50,
                expected_value=150,  # Unreasonable
                kelly_fraction=0.10,
                confidence=0.65,
            )
