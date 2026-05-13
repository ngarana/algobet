"""Tests for the ablation / permutation importance API endpoint."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from algobet.api.schemas.ablation import (
    AblationFamilyResult,
    AblationStudyResponse,
    PermutationFamilyResultSchema,
    PermutationImportanceResponse,
)
from algobet.models import Match, ModelVersion, Season, Team, Tournament
from algobet.services.ml_ops import MLOperationsOrchestrator


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
    for i in range(150):
        home_score = i % 3
        away_score = home_score if i % 5 == 0 else (i + 1) % 3
        match = Match(
            id=i + 1,
            tournament_id=1,
            season_id=1,
            home_team_id=1,
            away_team_id=2,
            match_date=datetime(2024, 1, 1) + timedelta(days=i),
            home_score=home_score,
            away_score=away_score,
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
        version="xgboost_20240101_000000",
        algorithm="xgboost",
        accuracy=0.55,
        file_path="data/models/xgboost/xgboost_20240101_000000/model.pkl",
        is_active=True,
        metrics={"accuracy": 0.55},
    )
    test_session.add(model)
    test_session.commit()
    return model


def _make_permutation_response():
    """Build a sample PermutationImportanceResponse."""
    return PermutationImportanceResponse(
        method="permutation",
        model_version="xgboost_20240101_000000",
        num_samples=45,
        n_repeats=10,
        baseline_log_loss=1.05,
        baseline_accuracy=0.52,
        families=[
            PermutationFamilyResultSchema(
                family="form",
                features_in_family=["home_points_last_5", "away_points_last_5"],
                features_found=["home_points_last_5", "away_points_last_5"],
                baseline_log_loss=1.05,
                permuted_log_loss=1.08,
                log_loss_increase=0.03,
                baseline_accuracy=0.52,
                permuted_accuracy=0.49,
                accuracy_decrease=0.03,
                importance_score=0.6,
                importance_rank=1,
            ),
            PermutationFamilyResultSchema(
                family="temporal",
                features_in_family=["day_of_week", "month"],
                features_found=["day_of_week", "month"],
                baseline_log_loss=1.05,
                permuted_log_loss=1.06,
                log_loss_increase=0.01,
                baseline_accuracy=0.52,
                permuted_accuracy=0.51,
                accuracy_decrease=0.01,
                importance_score=0.2,
                importance_rank=2,
            ),
        ],
        raw_feature_importance={"home_points_last_5": 0.15},
    )


def _make_ablation_response():
    """Build a sample AblationStudyResponse."""
    return AblationStudyResponse(
        method="ablation",
        baseline_model_version="xgboost_20240101_000000",
        baseline_num_features=120,
        baseline_train_metrics={"accuracy": 0.70, "log_loss": 0.85},
        baseline_val_metrics={"accuracy": 0.55, "log_loss": 1.05},
        baseline_test_metrics={"accuracy": 0.54, "log_loss": 1.08},
        families=[
            AblationFamilyResult(
                family="team_form",
                features_excluded=["home_points_last_5", "away_points_last_5"],
                num_features_used=110,
                model_version="xgboost_20240102_000000",
                train_metrics={"accuracy": 0.67, "log_loss": 0.90},
                val_metrics={"accuracy": 0.53, "log_loss": 1.10},
                test_metrics={"accuracy": 0.52, "log_loss": 1.12},
                log_loss_delta=0.04,
                accuracy_delta=-0.02,
            ),
            AblationFamilyResult(
                family="temporal",
                features_excluded=["day_of_week", "month"],
                num_features_used=118,
                model_version="xgboost_20240103_000000",
                train_metrics={"accuracy": 0.69, "log_loss": 0.87},
                val_metrics={"accuracy": 0.54, "log_loss": 1.06},
                test_metrics={"accuracy": 0.53, "log_loss": 1.09},
                log_loss_delta=0.01,
                accuracy_delta=-0.01,
            ),
        ],
    )


class TestAblationPermutationEndpoint:
    """Tests for POST /api/v1/ml/ablation with method=permutation."""

    def test_permutation_success(self, test_client: TestClient) -> None:
        """Permutation should return per-family importance scores."""
        mock_orchestrator = MagicMock(spec=MLOperationsOrchestrator)
        mock_orchestrator.run_ablation.return_value = _make_permutation_response()

        from algobet.api.main import app
        from algobet.api.routers.ml_operations import _ml_ops

        app.dependency_overrides[_ml_ops] = lambda: mock_orchestrator
        try:
            response = test_client.post(
                "/api/v1/ml/ablation",
                json={
                    "method": "permutation",
                    "model_version": "xgboost_20240101_000000",
                    "n_repeats": 5,
                    "min_matches": 50,
                },
            )
        finally:
            app.dependency_overrides.pop(_ml_ops, None)

        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "permutation"
        assert "families" in data
        assert len(data["families"]) == 2
        assert data["families"][0]["family"] == "form"
        assert data["families"][0]["importance_score"] == pytest.approx(0.6)
        assert data["baseline_log_loss"] > 0

    def test_permutation_group_by_generator(self, test_client: TestClient) -> None:
        """Permutation with group_by=generator groups by feature generator."""
        resp = _make_permutation_response()
        resp = resp.model_copy(
            update={
                "families": [
                    PermutationFamilyResultSchema(
                        family="team_form",
                        features_in_family=["home_points_last_5"],
                        features_found=["home_points_last_5"],
                        baseline_log_loss=1.05,
                        permuted_log_loss=1.10,
                        log_loss_increase=0.05,
                        baseline_accuracy=0.52,
                        permuted_accuracy=0.48,
                        accuracy_decrease=0.04,
                        importance_score=0.7,
                        importance_rank=1,
                    ),
                ],
            }
        )

        mock_orchestrator = MagicMock(spec=MLOperationsOrchestrator)
        mock_orchestrator.run_ablation.return_value = resp

        from algobet.api.main import app
        from algobet.api.routers.ml_operations import _ml_ops

        app.dependency_overrides[_ml_ops] = lambda: mock_orchestrator
        try:
            response = test_client.post(
                "/api/v1/ml/ablation",
                json={
                    "method": "permutation",
                    "group_by": "generator",
                    "min_matches": 50,
                },
            )
        finally:
            app.dependency_overrides.pop(_ml_ops, None)

        assert response.status_code == 200
        data = response.json()
        assert data["families"][0]["family"] == "team_form"


class TestAblationLeaveOneOutEndpoint:
    """Tests for POST /api/v1/ml/ablation with method=ablation."""

    def test_ablation_success(self, test_client: TestClient) -> None:
        """Ablation should return per-group retraining results."""
        mock_orchestrator = MagicMock(spec=MLOperationsOrchestrator)
        mock_orchestrator.run_ablation.return_value = _make_ablation_response()

        from algobet.api.main import app
        from algobet.api.routers.ml_operations import _ml_ops

        app.dependency_overrides[_ml_ops] = lambda: mock_orchestrator
        try:
            response = test_client.post(
                "/api/v1/ml/ablation",
                json={
                    "method": "ablation",
                    "feature_families": ["team_form", "temporal"],
                    "min_matches": 50,
                },
            )
        finally:
            app.dependency_overrides.pop(_ml_ops, None)

        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "ablation"
        assert "families" in data
        assert len(data["families"]) == 2
        assert data["families"][0]["family"] == "team_form"
        assert data["families"][0]["log_loss_delta"] > 0


class TestAblationValidation:
    """Tests for ablation request validation."""

    def test_method_validation(self, test_client: TestClient) -> None:
        """Method must be 'permutation' or 'ablation'."""
        response = test_client.post(
            "/api/v1/ml/ablation",
            json={"method": "invalid"},
        )
        assert response.status_code == 422

    def test_n_repeats_bounds(self, test_client: TestClient) -> None:
        """n_repeats must be between 1 and 100."""
        response = test_client.post(
            "/api/v1/ml/ablation",
            json={"method": "permutation", "n_repeats": 0},
        )
        assert response.status_code == 422

        response = test_client.post(
            "/api/v1/ml/ablation",
            json={"method": "permutation", "n_repeats": 101},
        )
        assert response.status_code == 422

    def test_min_matches_bounds(self, test_client: TestClient) -> None:
        """min_matches must be between 10 and 10000."""
        response = test_client.post(
            "/api/v1/ml/ablation",
            json={"method": "permutation", "min_matches": 5},
        )
        assert response.status_code == 422

    def test_group_by_validation(self, test_client: TestClient) -> None:
        """group_by must be 'family' or 'generator'."""
        response = test_client.post(
            "/api/v1/ml/ablation",
            json={"method": "permutation", "group_by": "invalid"},
        )
        assert response.status_code == 422

    def test_valid_permutation_request(self, test_client: TestClient) -> None:
        """Valid permutation request should pass schema validation."""
        response = test_client.post(
            "/api/v1/ml/ablation",
            json={
                "method": "permutation",
                "n_repeats": 10,
                "group_by": "family",
                "min_matches": 100,
            },
        )
        # May be 404/400/500 (no model/data), but not 422 (validation)
        assert response.status_code in [200, 400, 404, 500]

    def test_ablation_model_config_defaults(self, test_client: TestClient) -> None:
        """Ablation request should accept nested model config."""
        mock_orchestrator = MagicMock(spec=MLOperationsOrchestrator)
        mock_orchestrator.run_ablation.return_value = _make_ablation_response()

        from algobet.api.main import app
        from algobet.api.routers.ml_operations import _ml_ops

        app.dependency_overrides[_ml_ops] = lambda: mock_orchestrator
        try:
            response = test_client.post(
                "/api/v1/ml/ablation",
                json={
                    "method": "ablation",
                    "min_matches": 100,
                    "ablation_model_config": {
                        "model_type": "xgboost",
                        "calibrate_probabilities": True,
                    },
                },
            )
        finally:
            app.dependency_overrides.pop(_ml_ops, None)

        assert response.status_code == 200
