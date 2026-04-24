"""Integration tests for prediction router contracts used by the frontend."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from algobet.models import Match, ModelVersion, Prediction, Season, Team, Tournament


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
            version="v2.0.0",
            algorithm="xgboost",
            accuracy=0.62,
            file_path="data/models/xgboost/v2.0.0/model.pkl",
            is_active=False,
            metrics={"test_accuracy": 0.62},
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

        with (
            patch(
                "algobet.api.routers.predictions.PredictionService.load_model",
                return_value=(MagicMock(), "v2.0.0"),
            ),
            patch(
                "algobet.api.routers.predictions.PredictionService.generate_features_v2",
                return_value=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            ),
            patch(
                "algobet.api.routers.predictions.PredictionService.get_prediction",
                return_value=(
                    "HOME",
                    0.67,
                    {"home": 0.67, "draw": 0.2, "away": 0.13},
                ),
            ),
        ):
            response = test_client.post(
                "/api/v1/predictions/generate",
                json={"model_version": "v2.0.0", "days_ahead": 7},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["generated"] == 1
        assert data["model_version"] == "v2.0.0"
        assert data["existing_predictions_skipped"] == 0
