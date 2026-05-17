"""Integration tests for ML operations API endpoints."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from algobet.models import Match, ModelVersion, Season, Team, Tournament
from algobet.predictions.training.pipeline import MODEL_FEATURE_SCHEMA_VERSION


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


def _fitted_feature_pipeline() -> MagicMock:
    """Create a fitted feature pipeline mock for API runner tests."""
    pipeline = MagicMock()
    pipeline.is_fitted = True
    pipeline.feature_names = ["f1", "f2", "f3"]
    pipeline.transform.side_effect = lambda df, repo: np.ones(
        (len(df), 3), dtype=np.float64
    )
    return pipeline


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
        with (
            patch(
                "algobet.services.ml_ops.backtest_runner.ModelRegistry"
            ) as mock_registry_cls,
            patch(
                "algobet.services.ml_ops.backtest_runner."
                "BacktestRunner._load_saved_feature_pipeline",
                return_value=_fitted_feature_pipeline(),
            ),
        ):
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=10
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1, model_id=1),
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
        with (
            patch(
                "algobet.services.ml_ops.backtest_runner.ModelRegistry"
            ) as mock_registry_cls,
            patch(
                "algobet.services.ml_ops.backtest_runner."
                "BacktestRunner._load_saved_feature_pipeline",
                return_value=_fitted_feature_pipeline(),
            ),
        ):
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = lambda X: np.random.dirichlet(
                [1, 1, 1], size=len(X)
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1, model_id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/backtest",
                json={
                    "min_matches": 100,
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-30",
                },
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
        with (
            patch(
                "algobet.services.ml_ops.backtest_runner.ModelRegistry"
            ) as mock_registry_cls,
            patch(
                "algobet.services.ml_ops.backtest_runner."
                "BacktestRunner._load_saved_feature_pipeline",
                return_value=_fitted_feature_pipeline(),
            ),
        ):
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = lambda X: np.random.dirichlet(
                [1, 1, 1], size=len(X)
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1, model_id=1),
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
        with (
            patch(
                "algobet.services.ml_ops.backtest_runner.ModelRegistry"
            ) as mock_registry_cls,
            patch(
                "algobet.services.ml_ops.backtest_runner."
                "BacktestRunner._load_saved_feature_pipeline",
                return_value=_fitted_feature_pipeline(),
            ),
        ):
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = lambda X: np.random.dirichlet(
                [1, 1, 1], size=len(X)
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1, model_id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/backtest",
                json={
                    "min_matches": 100,
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-30",
                },
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
        with (
            patch(
                "algobet.services.ml_ops.backtest_runner.ModelRegistry"
            ) as mock_registry_cls,
            patch(
                "algobet.services.ml_ops.backtest_runner."
                "BacktestRunner._load_saved_feature_pipeline",
                return_value=_fitted_feature_pipeline(),
            ),
        ):
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = lambda X: np.random.dirichlet(
                [1, 1, 1], size=len(X)
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1, model_id=1),
            )
            mock_registry.list_models.return_value = [MagicMock(version="v1.0.0")]
            mock_registry_cls.return_value = mock_registry

            response = test_client.post(
                "/api/v1/ml/backtest",
                json={
                    "min_matches": 100,
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-30",
                },
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


class TestTrainEndpoint:
    """Tests for POST /api/v1/ml/train endpoint."""

    def test_train_success(
        self,
        test_client: TestClient,
        test_session: Session,
    ) -> None:
        """Training should return the new model summary."""
        trained_model = ModelVersion(
            name="Frontend Model",
            version="xgboost_20260424_000000",
            algorithm="xgboost",
            accuracy=0.61,
            file_path="data/models/xgboost/xgboost_20260424_000000/model.pkl",
            is_active=False,
            metrics={"test_accuracy": 0.61},
        )
        test_session.add(trained_model)
        test_session.commit()

        with (
            patch(
                "algobet.services.ml_ops.training_runner.TrainingPipeline"
            ) as mock_pipeline_cls,
            patch(
                "algobet.services.ml_ops.training_runner.ModelRegistry"
            ) as mock_registry_cls,
        ):
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = MagicMock(
                model_version="xgboost_20260424_000000",
                model_type="xgboost",
                feature_schema_version=MODEL_FEATURE_SCHEMA_VERSION,
                num_features=42,
                trained_at=datetime(2026, 4, 24, 0, 0, 0),
                training_duration_seconds=12.5,
                train_metrics={"accuracy": 0.7},
                val_metrics={"accuracy": 0.63},
                test_metrics={"accuracy": 0.61},
                feature_importance={"home_form": 0.21},
                ensemble_weights=None,
                ensemble_validation_metrics=None,
            )
            mock_pipeline_cls.return_value = mock_pipeline
            mock_registry_cls.return_value = MagicMock()

            response = test_client.post(
                "/api/v1/ml/train",
                json={
                    "model_type": "xgboost",
                    "tune_hyperparameters": False,
                    "activate": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "xgboost_20260424_000000"
        assert data["model_type"] == "xgboost"
        assert data["is_active"] is True
        assert data["feature_schema_version"] == MODEL_FEATURE_SCHEMA_VERSION
        assert data["num_features"] == 42
        assert "test_metrics" in data
        train_config = mock_pipeline_cls.call_args.kwargs["config"]
        assert train_config.feature_schema_version == MODEL_FEATURE_SCHEMA_VERSION
        assert train_config.calibration_method == "temperature"
        assert train_config.outcome_balance is False
        assert not hasattr(train_config, "require_odds")
        assert not hasattr(train_config, "odds_blend")

    def test_train_accepts_epl_feature_selection_request(
        self,
        test_client: TestClient,
    ) -> None:
        """The recommended odds-free EPL request should map into TrainingConfig."""
        with (
            patch(
                "algobet.services.ml_ops.training_runner.TrainingPipeline"
            ) as mock_pipeline_cls,
            patch(
                "algobet.services.ml_ops.training_runner.ModelRegistry"
            ) as mock_registry_cls,
        ):
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = MagicMock(
                model_version="xgboost_20260507_192247",
                model_type="xgboost",
                feature_schema_version=MODEL_FEATURE_SCHEMA_VERSION,
                num_features=30,
                trained_at=datetime(2026, 5, 7, 19, 22, 47),
                training_duration_seconds=18.0,
                train_metrics={"accuracy": 0.7},
                val_metrics={"accuracy": 0.63},
                test_metrics={
                    "accuracy": 0.61,
                    "market_log_loss": 1.02,
                    "market_model_probability_mae": 0.12,
                },
                feature_importance={"home_points_last_5": 0.21},
                ensemble_weights=None,
                ensemble_validation_metrics=None,
            )
            mock_pipeline_cls.return_value = mock_pipeline
            mock_registry_cls.return_value = MagicMock()

            response = test_client.post(
                "/api/v1/ml/train",
                json={
                    "model_type": "xgboost",
                    "description": "EPL odds-free calibrated probability model",
                    "tournament_ids": [359],
                    "feature_groups": [
                        "team_form",
                        "head_to_head",
                        "temporal",
                        "standings",
                        "enriched_stats",
                    ],
                    "feature_selection": True,
                    "feature_selection_threshold": 0.005,
                    "min_samples_per_feature": 40,
                    "min_matches": 150,
                    "outcome_balance": False,
                    "tune_hyperparameters": False,
                    "calibrate_probabilities": True,
                    "calibration_method": "sigmoid",
                    "hyperparameters": {
                        "max_depth": 3,
                        "learning_rate": 0.03,
                        "n_estimators": 1200,
                    },
                    "tags": {"model_scope": "epl", "odds_policy": "pure_model"},
                },
            )

        assert response.status_code == 200
        train_config = mock_pipeline_cls.call_args.kwargs["config"]
        assert train_config.tournament_ids == [359]
        assert train_config.feature_groups == [
            "team_form",
            "head_to_head",
            "temporal",
            "standings",
            "enriched_stats",
        ]
        assert train_config.feature_selection is True
        assert train_config.feature_selection_threshold == pytest.approx(0.005)
        assert train_config.min_samples_per_feature == 40
        assert train_config.outcome_balance is False
        assert train_config.calibration_method == "sigmoid"

    def test_train_model_type_validation(self, test_client: TestClient) -> None:
        """Training should validate model type values."""
        response = test_client.post(
            "/api/v1/ml/train",
            json={"model_type": "invalid-model"},
        )

        assert response.status_code == 422

    def test_train_accepts_explicit_odds_feature_group(
        self, test_client: TestClient
    ) -> None:
        """Training should pass explicit odds groups through to TrainingConfig."""
        with (
            patch(
                "algobet.services.ml_ops.training_runner.TrainingPipeline"
            ) as mock_pipeline_cls,
            patch(
                "algobet.services.ml_ops.training_runner.ModelRegistry"
            ) as mock_registry_cls,
        ):
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = MagicMock(
                model_version="xgboost_20260513_010203",
                model_type="xgboost",
                feature_schema_version=MODEL_FEATURE_SCHEMA_VERSION,
                num_features=12,
                trained_at=datetime(2026, 5, 13, 1, 2, 3),
                training_duration_seconds=5.0,
                train_metrics={"accuracy": 0.7},
                val_metrics={"accuracy": 0.63},
                test_metrics={"accuracy": 0.61},
                feature_importance={"implied_prob_home": 0.3},
                ensemble_weights=None,
                ensemble_validation_metrics=None,
            )
            mock_pipeline_cls.return_value = mock_pipeline
            mock_registry_cls.return_value = MagicMock()

            response = test_client.post(
                "/api/v1/ml/train",
                json={"model_type": "xgboost", "feature_groups": ["odds"]},
            )

        assert response.status_code == 200
        train_config = mock_pipeline_cls.call_args.kwargs["config"]
        assert train_config.feature_groups == ["odds"]


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
        with (
            patch(
                "algobet.services.ml_ops.calibration_runner.ModelRegistry"
            ) as mock_registry_cls,
            patch(
                "algobet.services.ml_ops.calibration_runner."
                "CalibrationRunner._load_saved_feature_pipeline",
                return_value=_fitted_feature_pipeline(),
            ),
        ):
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
        with (
            patch(
                "algobet.services.ml_ops.calibration_runner.ModelRegistry"
            ) as mock_registry_cls,
            patch(
                "algobet.services.ml_ops.calibration_runner."
                "CalibrationRunner._load_saved_feature_pipeline",
                return_value=_fitted_feature_pipeline(),
            ),
        ):
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = lambda X: np.random.dirichlet(
                [1, 1, 1], size=len(X)
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

        with (
            patch(
                "algobet.services.ml_ops.backtest_runner.ModelRegistry"
            ) as mock_registry_cls,
            patch(
                "algobet.services.ml_ops.backtest_runner."
                "BacktestRunner._load_saved_feature_pipeline",
                return_value=_fitted_feature_pipeline(),
            ),
        ):
            mock_registry = MagicMock()
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], size=n_test_samples
            ).astype(np.float64)
            mock_registry.load_model.return_value = mock_model
            mock_registry.get_active_model.return_value = (
                mock_model,
                MagicMock(version="v1.0.0", id=1, model_id=1),
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
        """Method should validate against supported calibration methods."""
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
