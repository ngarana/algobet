"""Integration tests for prediction router contracts used by the frontend."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from algobet.models import Match, ModelVersion, Prediction, Season, Team, Tournament
from algobet.predictions.training.pipeline import MODEL_FEATURE_SCHEMA_VERSION


@pytest.fixture
def scheduled_prediction_data(test_session: Session) -> dict[str, int]:
    """Create a scheduled match, model, and prediction for API contract tests."""
    tournament = Tournament(
        id=101,
        name="Premier League",
        country="England",
        url_slug="premier-league",
    )
    season = Season(
        id=101,
        tournament_id=101,
        name="2025/26",
        start_year=2025,
        end_year=2026,
    )
    home_team = Team(id=101, name="Arsenal")
    away_team = Team(id=102, name="Chelsea")
    model = ModelVersion(
        id=101,
        name="Frontend Model",
        version="v1.0.0",
        algorithm="xgboost",
        accuracy=0.6,
        file_path="data/models/xgboost/v1.0.0/model.pkl",
        is_active=True,
        metrics={"test_accuracy": 0.6},
    )
    match = Match(
        id=101,
        tournament_id=101,
        season_id=101,
        home_team_id=101,
        away_team_id=102,
        match_date=datetime.now(timezone.utc) + timedelta(days=2),
        status="SCHEDULED",
        odds_home=2.0,
        odds_draw=3.4,
        odds_away=3.8,
    )
    prediction = Prediction(
        id=101,
        match_id=101,
        model_version_id=101,
        prob_home=0.52,
        prob_draw=0.24,
        prob_away=0.24,
        predicted_outcome="H",
        confidence=0.52,
        predicted_at=datetime.now(timezone.utc),
    )

    test_session.add_all(
        [tournament, season, home_team, away_team, model, match, prediction]
    )
    test_session.commit()

    return {
        "model_id": model.id,
        "match_id": match.id,
        "prediction_id": prediction.id,
    }


class TestPredictionRouterContracts:
    """Prediction router behavior used by the frontend pages."""

    def test_upcoming_predictions_include_match_and_model(
        self,
        test_client: TestClient,
        scheduled_prediction_data: dict[str, int],
    ) -> None:
        """Upcoming predictions should include match summary and model metadata."""
        response = test_client.get(
            f"/api/v1/predictions/upcoming?days=7&model_version_id={scheduled_prediction_data['model_id']}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["items"]

        item = data["items"][0]
        assert item["match"]["home_team_name"] == "Arsenal"
        assert item["match"]["away_team_name"] == "Chelsea"
        assert item["model_version"]["version"] == "v1.0.0"

    def test_generate_predictions_uses_selected_model(
        self,
        test_client: TestClient,
        test_session: Session,
        tmp_path: Path,
    ) -> None:
        """Prediction generation should persist results for the selected model."""
        tournament = Tournament(
            id=201,
            name="Serie A",
            country="Italy",
            url_slug="serie-a",
        )
        season = Season(
            id=201,
            tournament_id=201,
            name="2025/26",
            start_year=2025,
            end_year=2026,
        )
        home_team = Team(id=201, name="Inter")
        away_team = Team(id=202, name="Milan")
        model = ModelVersion(
            id=201,
            name="Selected Model",
            version="xgboost_20260507_010203",
            algorithm="xgboost",
            accuracy=0.62,
            file_path="data/models/xgboost/xgboost_20260507_010203/model.pkl",
            is_active=False,
            metrics={"test_accuracy": 0.62},
            hyperparameters={"feature_pipeline_path": ""},
            feature_schema_version=MODEL_FEATURE_SCHEMA_VERSION,
        )
        match = Match(
            id=201,
            tournament_id=201,
            season_id=201,
            home_team_id=201,
            away_team_id=202,
            match_date=datetime.now(timezone.utc) + timedelta(days=1),
            status="SCHEDULED",
            odds_home=2.1,
            odds_draw=3.2,
            odds_away=3.5,
        )

        test_session.add_all([tournament, season, home_team, away_team, model, match])
        test_session.commit()

        pipeline_dir = tmp_path / "feature-pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        (pipeline_dir / "config.json").write_text("{}", encoding="utf-8")
        model.hyperparameters = {"feature_pipeline_path": str(pipeline_dir)}
        test_session.commit()

        loaded_pipeline = MagicMock()
        loaded_pipeline.is_fitted = True
        loaded_pipeline.feature_names = ["home_points_last_5"]
        loaded_pipeline.transform.return_value = np.array(
            [[1.0, 2.0, 3.0]],
            dtype=np.float64,
        )
        loaded_model = MagicMock()
        loaded_model.predict_proba.return_value = np.array([[0.67, 0.2, 0.13]])

        with (
            patch(
                "algobet.services.prediction_service.ModelRegistry.load_model",
                return_value=loaded_model,
            ),
            patch(
                "algobet.services.prediction_service.FeaturePipeline.load",
                return_value=loaded_pipeline,
            ),
        ):
            response = test_client.post(
                "/api/v1/predictions/generate",
                json={"model_version": "xgboost_20260507_010203", "days_ahead": 7},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["generated"] == 1
        assert data["model_version"] == "xgboost_20260507_010203"
        assert data["existing_predictions_skipped"] == 0

    def test_generate_predictions_rejects_legacy_model_schema(
        self,
        test_client: TestClient,
        test_session: Session,
    ) -> None:
        """Prediction generation should require retraining for legacy schemas."""
        model = ModelVersion(
            id=301,
            name="Legacy Model",
            version="xgboost_legacy",
            algorithm="xgboost",
            accuracy=0.58,
            file_path="data/models/xgboost/xgboost_legacy/model.pkl",
            is_active=False,
            metrics={"test_accuracy": 0.58},
            feature_schema_version="v1.0",
        )
        test_session.add(model)
        test_session.commit()

        response = test_client.post(
            "/api/v1/predictions/generate",
            json={"model_version": "xgboost_legacy", "days_ahead": 7},
        )

        assert response.status_code == 400
        assert "Retrain it under v2.0_odds_free" in response.json()["detail"]
