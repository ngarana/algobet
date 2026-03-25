"""Integration tests for ML operations API endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from algobet.models import Match, ModelVersion, Season, Team, Tournament


@pytest.fixture
def sample_matches(test_session: Session) -> list[Match]:
    """Create sample matches with results for testing."""
    tournament = Tournament(
        id=1, name="Test League", country="Test Country", url_slug="test-league"
    )
    test_session.add(tournament)

    season = Season(
        id=1,
        tournament_id=1,
        name="2024",
        start_year=2024,
        end_year=2025,
    )
    test_session.add(season)

    home_team = Team(id=1, name="Home United")
    away_team = Team(id=2, name="Away City")
    test_session.add_all([home_team, away_team])

    matches = []
    from datetime import datetime, timedelta

    for i in range(150):
        match = Match(
            id=i + 1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime(2024, 1, 1) + timedelta(days=i),
            home_score=i % 3,
            away_score=(i + 1) % 3,
            status="FINISHED",
            odds_home=2.0 + (i % 5) * 0.1,
            odds_draw=3.3,
            odds_away=2.8 - (i % 5) * 0.1,
        )
        matches.append(match)
        test_session.add(match)

    test_session.commit()
    return matches


@pytest.fixture
def sample_model(test_session: Session) -> ModelVersion:
    """Create a sample model version for testing."""
    model = ModelVersion(
        id=1,
        name="Test Model",
        version="v1.0.0",
        algorithm="xgboost",
        accuracy=0.55,
        file_path="test_model.pkl",
        is_active=True,
        metrics={"accuracy": 0.55},
    )
    test_session.add(model)
    test_session.commit()
    return model


class TestBacktestEndpoint:
    """Tests for POST /api/v1/ml/backtest endpoint."""

    def test_backtest_requires_model(
        self, test_client: TestClient, sample_matches: list[Match]
    ) -> None:
        """Backtest should fail when no active model exists."""
        response = test_client.post(
            "/api/v1/ml/backtest",
            json={"min_matches": 100},
        )

        assert response.status_code == 404

    def test_backtest_insufficient_matches(
        self, test_client: TestClient, sample_model: ModelVersion
    ) -> None:
        """Backtest should fail with insufficient matches."""
        with patch(
            "algobet.api.routers.ml_operations.ModelRegistry"
        ) as mock_registry_cls:
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=10
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/backtest",
                json={"min_matches": 1000},
            )

        assert response.status_code == 400
        assert "Insufficient matches" in response.json()["detail"]

    def test_backtest_success(
        self,
        test_client: TestClient,
        sample_matches: list[Match],
        sample_model: ModelVersion,
    ) -> None:
        """Backtest should return metrics on success."""
        n_test_samples = 105

        with patch(
            "algobet.api.routers.ml_operations.ModelRegistry"
        ) as mock_registry_cls:
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=n_test_samples
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/backtest",
                json={"min_matches": 100},
            )

        assert response.status_code == 200
        data = response.json()

        assert "model_version" in data
        assert "num_samples" in data
        assert "classification" in data
        assert "expected_calibration_error" in data

    def test_backtest_with_date_range(
        self,
        test_client: TestClient,
        sample_matches: list[Match],
        sample_model: ModelVersion,
    ) -> None:
        """Backtest should respect date range parameters."""
        n_test_samples = 70

        with patch(
            "algobet.api.routers.ml_operations.ModelRegistry"
        ) as mock_registry_cls:
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=n_test_samples
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/backtest",
                json={
                    "min_matches": 50,
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-30",
                },
            )

        assert response.status_code == 200

    def test_backtest_classification_metrics(
        self,
        test_client: TestClient,
        sample_matches: list[Match],
        sample_model: ModelVersion,
    ) -> None:
        """Backtest should return all classification metrics."""
        n_test_samples = 105

        with patch(
            "algobet.api.routers.ml_operations.ModelRegistry"
        ) as mock_registry_cls:
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=n_test_samples
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/backtest",
                json={"min_matches": 100},
            )

        assert response.status_code == 200
        data = response.json()
        classification = data["classification"]

        assert "accuracy" in classification
        assert "log_loss" in classification
        assert "brier_score" in classification
        assert "f1_macro" in classification
        assert "per_class_f1" in classification
        assert "confusion_matrix" in classification
        assert "top_2_accuracy" in classification
        assert "cohen_kappa" in classification

        assert 0.0 <= classification["accuracy"] <= 1.0
        assert classification["log_loss"] > 0.0

    def test_backtest_betting_metrics(
        self,
        test_client: TestClient,
        sample_matches: list[Match],
        sample_model: ModelVersion,
    ) -> None:
        """Backtest should return betting simulation metrics."""
        n_test_samples = 105

        with patch(
            "algobet.api.routers.ml_operations.ModelRegistry"
        ) as mock_registry_cls:
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=n_test_samples
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/backtest",
                json={"min_matches": 100},
            )

        assert response.status_code == 200
        data = response.json()

        if data["betting"] is not None:
            betting = data["betting"]
            assert "total_bets" in betting
            assert "win_rate" in betting
            assert "roi_percent" in betting
            assert "sharpe_ratio" in betting
            assert "max_drawdown" in betting


class TestCalibrateEndpoint:
    """Tests for POST /api/v1/ml/calibrate endpoint."""

    def test_calibrate_requires_model(self, test_client: TestClient) -> None:
        """Calibrate should fail when no active model exists."""
        response = test_client.post(
            "/api/v1/ml/calibrate",
            json={"method": "isotonic"},
        )

        assert response.status_code == 404

    def test_calibrate_insufficient_matches(
        self, test_client: TestClient, sample_model: ModelVersion
    ) -> None:
        """Calibrate should fail with insufficient historical matches."""
        with patch(
            "algobet.api.routers.ml_operations.ModelRegistry"
        ) as mock_registry_cls:
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=50
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0"),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/calibrate",
                json={"method": "isotonic"},
            )

        assert response.status_code == 400
        assert "Insufficient historical matches" in response.json()["detail"]

    def test_calibrate_success(
        self,
        test_client: TestClient,
        sample_matches: list[Match],
        sample_model: ModelVersion,
    ) -> None:
        """Calibrate should return before/after metrics on success."""
        n_samples = 400

        with patch(
            "algobet.api.routers.ml_operations.ModelRegistry"
        ) as mock_registry_cls:
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=n_samples
            ).astype(np.float64)

            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0"),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry.save_model.return_value = "v1.0.1"
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/calibrate",
                json={"method": "isotonic", "activate": False},
            )

        assert response.status_code == 200
        data = response.json()

        assert "base_model_version" in data
        assert "calibrated_model_version" in data
        assert "before_metrics" in data
        assert "after_metrics" in data
        assert "improvement" in data


class TestBacktestHistoryEndpoint:
    """Tests for GET /api/v1/ml/backtest/history endpoint."""

    def test_get_history_empty(self, test_client: TestClient) -> None:
        """History should return empty list initially."""
        response = test_client.get("/api/v1/ml/backtest/history")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_get_history_pagination(
        self,
        test_client: TestClient,
        sample_matches: list[Match],
        sample_model: ModelVersion,
    ) -> None:
        """History should support pagination."""
        n_test_samples = 105

        with patch(
            "algobet.api.routers.ml_operations.ModelRegistry"
        ) as mock_registry_cls:
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=n_test_samples
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            test_client.post(
                "/api/v1/ml/backtest",
                json={"min_matches": 100},
            )

        response = test_client.get("/api/v1/ml/backtest/history?limit=5&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data


class TestBacktestValidation:
    """Tests for backtest request validation."""

    def test_min_matches_bounds(self, test_client: TestClient) -> None:
        """min_matches should have valid bounds."""
        response = test_client.post(
            "/api/v1/ml/backtest",
            json={"min_matches": 5},
        )
        assert response.status_code == 422

        response = test_client.post(
            "/api/v1/ml/backtest",
            json={"min_matches": 15000},
        )
        assert response.status_code == 422

    def test_min_edge_bounds(self, test_client: TestClient) -> None:
        """min_edge should be between 0 and 1."""
        response = test_client.post(
            "/api/v1/ml/backtest",
            json={"min_edge": -0.1},
        )
        assert response.status_code == 422

        response = test_client.post(
            "/api/v1/ml/backtest",
            json={"min_edge": 1.5},
        )
        assert response.status_code == 422

    def test_valid_min_edge(self, test_client: TestClient) -> None:
        """Valid min_edge values should be accepted."""
        response = test_client.post(
            "/api/v1/ml/backtest",
            json={"min_matches": 100, "min_edge": 0.05},
        )
        assert response.status_code in [404, 400, 200]


class TestCalibrateValidation:
    """Tests for calibrate request validation."""

    def test_method_validation(self, test_client: TestClient) -> None:
        """Method should be isotonic or sigmoid."""
        response = test_client.post(
            "/api/v1/ml/calibrate",
            json={"method": "invalid"},
        )
        assert response.status_code == 422

    def test_validation_split_bounds(self, test_client: TestClient) -> None:
        """validation_split should be between 0.1 and 0.5."""
        response = test_client.post(
            "/api/v1/ml/calibrate",
            json={"validation_split": 0.05},
        )
        assert response.status_code == 422

        response = test_client.post(
            "/api/v1/ml/calibrate",
            json={"validation_split": 0.6},
        )
        assert response.status_code == 422
