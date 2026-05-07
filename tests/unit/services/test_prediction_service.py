"""Unit tests for PredictionService."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from algobet.predictions.training.pipeline import MODEL_FEATURE_SCHEMA_VERSION
from algobet.services.prediction_service import PredictionResult, PredictionService


def _make_match(
    match_id: int = 1,
    home_team_id: int = 10,
    away_team_id: int = 20,
    match_date: datetime | None = None,
) -> MagicMock:
    """Create a mock Match object."""
    match = MagicMock()
    match.id = match_id
    match.home_team_id = home_team_id
    match.away_team_id = away_team_id
    match.match_date = match_date or datetime(2025, 6, 15)
    match.home_team.name = "Team A"
    match.away_team.name = "Team B"
    match.home_score = None
    match.away_score = None
    match.status = "SCHEDULED"
    match.tournament_id = 1
    match.season_id = 1
    match.odds_home = 1.5
    match.odds_draw = 3.5
    match.odds_away = 6.0
    match.num_bookmakers = 10
    return match


def _make_model_record(
    version: str = "xgboost_20260507_010203",
    feature_pipeline_path: str = "/tmp/feature-pipeline",
    feature_schema_version: str = MODEL_FEATURE_SCHEMA_VERSION,
) -> MagicMock:
    """Create a mock model version record."""
    model_record = MagicMock()
    model_record.version = version
    model_record.is_active = True
    model_record.feature_schema_version = feature_schema_version
    model_record.hyperparameters = {
        "feature_pipeline_path": feature_pipeline_path,
    }
    return model_record


class TestPredictionServiceInit:
    """Tests for PredictionService initialization."""

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_init(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """PredictionService initializes all components."""
        session = MagicMock()
        service = PredictionService(session)

        mock_repo_cls.assert_called_once_with(session)
        assert service.pipelines_path == Path("data/pipelines")
        assert service._feature_pipeline is None
        assert service._feature_pipeline_path is None


class TestLoadModel:
    """Tests for model loading and pipeline validation."""

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_rejects_legacy_feature_schema(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """Prediction generation should reject pre-odds-free models."""
        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = (
            _make_model_record(feature_schema_version="v1.0")
        )

        service = PredictionService(session)

        with pytest.raises(ValueError, match="Retrain it under v2.0_odds_free"):
            service.load_model("xgboost_legacy")

    @patch("algobet.services.prediction_service.FeaturePipeline.load")
    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_loads_saved_feature_pipeline_for_model(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_pipeline_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Prediction generation should use the pipeline saved with the model."""
        pipeline_dir = tmp_path / "feature_pipeline"
        pipeline_dir.mkdir()
        (pipeline_dir / "config.json").write_text("{}", encoding="utf-8")

        fitted_pipeline = MagicMock()
        fitted_pipeline.is_fitted = True
        fitted_pipeline.feature_names = ["home_points_last_5"]
        mock_pipeline_load.return_value = fitted_pipeline

        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = (
            _make_model_record(feature_pipeline_path=str(pipeline_dir))
        )

        registry = mock_registry_cls.return_value
        registry.load_model.return_value = MagicMock()

        service = PredictionService(session)
        model, version = service.load_model("xgboost_20260507_010203")

        assert model is registry.load_model.return_value
        assert version == "xgboost_20260507_010203"
        assert service._feature_pipeline is fitted_pipeline
        assert service._feature_pipeline_path == pipeline_dir.resolve()
        mock_pipeline_load.assert_called_once_with(pipeline_dir.resolve())


class TestGenerateFeatures:
    """Tests for generate_features (FormCalculator-based)."""

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_returns_expected_keys(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """generate_features returns dict with 6 keys."""
        calc_instance = mock_calc_cls.return_value
        calc_instance.calculate_recent_form.return_value = 2.0
        calc_instance.calculate_goals_scored.return_value = 1.5
        calc_instance.calculate_goals_conceded.return_value = 0.8

        service = PredictionService(MagicMock())
        match = _make_match()
        features = service.generate_features(match)

        expected_keys = {
            "home_form",
            "away_form",
            "home_goals_scored",
            "away_goals_scored",
            "home_goals_conceded",
            "away_goals_conceded",
        }
        assert set(features.keys()) == expected_keys

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_calls_calculator_with_correct_team_ids(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """generate_features passes correct team IDs to calculator."""
        calc_instance = mock_calc_cls.return_value
        calc_instance.calculate_recent_form.return_value = 0.0
        calc_instance.calculate_goals_scored.return_value = 0.0
        calc_instance.calculate_goals_conceded.return_value = 0.0

        service = PredictionService(MagicMock())
        match = _make_match(home_team_id=10, away_team_id=20)
        service.generate_features(match)

        # Check first call was for home team, second for away
        form_calls = calc_instance.calculate_recent_form.call_args_list
        assert form_calls[0].kwargs["team_id"] == 10
        assert form_calls[1].kwargs["team_id"] == 20


class TestGetPrediction:
    """Tests for get_prediction method."""

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_home_win_prediction(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """Model predicting high home probability returns HOME."""
        service = PredictionService(MagicMock())
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.7, 0.2, 0.1]])

        outcome, confidence, probs = service.get_prediction(
            model, {"home_form": 2.0, "away_form": 1.0}
        )

        assert outcome == "HOME"
        assert confidence == pytest.approx(0.7)
        assert probs["home"] == pytest.approx(0.7)
        assert probs["draw"] == pytest.approx(0.2)
        assert probs["away"] == pytest.approx(0.1)

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_draw_prediction(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """Model predicting high draw probability returns DRAW."""
        service = PredictionService(MagicMock())
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])

        outcome, confidence, probs = service.get_prediction(
            model, {"home_form": 1.0, "away_form": 1.0}
        )

        assert outcome == "DRAW"
        assert confidence == pytest.approx(0.6)

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_away_win_prediction(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """Model predicting high away probability returns AWAY."""
        service = PredictionService(MagicMock())
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])

        outcome, confidence, probs = service.get_prediction(
            model, {"home_form": 0.5, "away_form": 2.5}
        )

        assert outcome == "AWAY"
        assert confidence == pytest.approx(0.8)

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_accepts_numpy_array(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """get_prediction accepts numpy array features (from FeaturePipeline)."""
        service = PredictionService(MagicMock())
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.5, 0.3, 0.2]])

        features = np.array([[1.0, 2.0, 3.0, 4.0]])
        outcome, confidence, probs = service.get_prediction(model, features)

        assert outcome == "HOME"
        assert confidence == pytest.approx(0.5)
        model.predict_proba.assert_called_once()

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_fallback_to_predict(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """Falls back to model.predict when predict_proba is unavailable."""
        service = PredictionService(MagicMock())
        model = MagicMock()
        model.predict_proba.side_effect = AttributeError
        model.predict.return_value = np.array([[0.3, 0.4, 0.3]])

        outcome, confidence, probs = service.get_prediction(model, {"f1": 1.0})

        assert outcome == "DRAW"
        model.predict.assert_called_once()


class TestPredictMatch:
    """Tests for predict_match (end-to-end with mocks)."""

    @patch("algobet.services.prediction_service.ModelRegistry")
    @patch("algobet.services.prediction_service.MatchRepository")
    @patch("algobet.services.prediction_service.FormCalculator")
    def test_returns_prediction_result(
        self,
        mock_calc_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
    ) -> None:
        """predict_match returns a PredictionResult with correct fields."""
        # Set up mocks
        calc = mock_calc_cls.return_value
        calc.calculate_recent_form.return_value = 2.0
        calc.calculate_goals_scored.return_value = 1.5
        calc.calculate_goals_conceded.return_value = 0.8

        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = (
            _make_model_record()
        )
        registry = mock_registry_cls.return_value
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.6, 0.25, 0.15]])
        registry.load_model.return_value = model

        service = PredictionService(session)
        service._feature_pipeline = MagicMock(is_fitted=True)
        service._feature_pipeline.transform.return_value = np.array(
            [[1.0, 2.0, 3.0]],
            dtype=np.float64,
        )
        service._feature_pipeline_path = Path("/tmp/feature-pipeline")
        match = _make_match(match_id=42)
        result = service.predict_match(match)

        assert isinstance(result, PredictionResult)
        assert result.match_id == 42
        assert result.predicted_outcome == "HOME"
        assert result.confidence == pytest.approx(0.6)
        assert result.prob_home == pytest.approx(0.6)
        assert result.prob_draw == pytest.approx(0.25)
        assert result.prob_away == pytest.approx(0.15)
        assert result.home_team == "Team A"
        assert result.away_team == "Team B"
