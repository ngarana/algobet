"""Unit tests for TrainingPipeline."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from algobet.predictions.data.queries import MatchRepository, TeamStandings
from algobet.predictions.training.pipeline import (
    MODEL_FEATURE_SCHEMA_VERSION,
    TrainingConfig,
    TrainingPipeline,
)


class TestTrainingConfig:
    """Tests for TrainingConfig defaults."""

    def test_default_values(self) -> None:
        """TrainingConfig should have sensible defaults."""
        config = TrainingConfig()
        assert config.model_type == "xgboost"
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.tune_hyperparameters is False
        assert config.calibrate_probabilities is True
        assert config.calibration_method == "sigmoid"
        assert config.outcome_balance is False
        assert config.feature_schema_version == MODEL_FEATURE_SCHEMA_VERSION
        assert config.use_ensemble is False
        assert config.random_seed == 42
        assert not hasattr(config, "require_odds")
        assert not hasattr(config, "odds_blend")

    def test_custom_values(self) -> None:
        """TrainingConfig accepts custom values."""
        config = TrainingConfig(
            model_type="lightgbm",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            tune_hyperparameters=True,
            tuning_trials=100,
            description="Test model",
        )
        assert config.model_type == "lightgbm"
        assert config.train_ratio == 0.8
        assert config.tune_hyperparameters is True
        assert config.tuning_trials == 100
        assert config.description == "Test model"

    def test_default_xgboost_probability_hyperparameters(self) -> None:
        """XGBoost defaults should be conservative for odds-free probabilities."""
        from algobet.predictions.training.classifiers import XGBoostPredictor

        defaults = XGBoostPredictor().default_hyperparameters

        assert defaults["max_depth"] == 3
        assert defaults["learning_rate"] == pytest.approx(0.03)
        assert defaults["n_estimators"] == 1200
        assert defaults["reg_lambda"] == pytest.approx(10.0)

    def test_ratios_sum_to_one(self) -> None:
        """Default split ratios should sum to 1.0."""
        config = TrainingConfig()
        total = config.train_ratio + config.val_ratio + config.test_ratio
        assert total == pytest.approx(1.0)

    def test_ensemble_types(self) -> None:
        """Ensemble types default to xgboost and lightgbm."""
        config = TrainingConfig(use_ensemble=True)
        assert "xgboost" in config.ensemble_types
        assert "lightgbm" in config.ensemble_types


class TestTrainingPipelineInit:
    """Tests for TrainingPipeline initialization."""

    @patch("algobet.predictions.training.pipeline.FeaturePipeline")
    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_init_creates_components(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """TrainingPipeline initializes all required components."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        session = MagicMock()
        config = TrainingConfig()
        models_path = Path("test/models")

        pipeline = TrainingPipeline(
            config=config,
            session=session,
            models_path=models_path,
        )

        assert pipeline.config is config
        mock_repo_cls.assert_called_once_with(session)
        mock_registry_cls.assert_called_once_with(
            storage_path=models_path, session=session
        )

    @patch("algobet.predictions.training.pipeline.FeaturePipeline")
    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_init_uses_custom_feature_pipeline(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """TrainingPipeline uses custom FeaturePipeline when provided."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        custom_pipeline = MagicMock()
        pipeline = TrainingPipeline(
            config=TrainingConfig(),
            session=MagicMock(),
            feature_pipeline=custom_pipeline,
        )
        assert pipeline.feature_pipeline is custom_pipeline


class TestTrainingPipelineEvaluate:
    """Tests for TrainingPipeline._evaluate method."""

    @patch("algobet.predictions.training.pipeline.FeaturePipeline")
    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_evaluate_returns_expected_keys(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """_evaluate should return accuracy, log_loss, precision, recall, f1."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(
            config=TrainingConfig(),
            session=MagicMock(),
        )

        # Create mock predictor
        predictor = MagicMock()
        n_samples = 50
        # Simulate probabilities for 3 classes
        probas = np.random.dirichlet([1, 1, 1], size=n_samples).astype(np.float64)
        predictor.predict_proba.return_value = probas

        X = np.random.randn(n_samples, 10).astype(np.float64)
        y = np.random.randint(0, 3, size=n_samples).astype(np.int64)

        metrics = pipeline._evaluate(predictor, X, y)

        assert "accuracy" in metrics
        assert "log_loss" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["log_loss"] > 0.0

    @patch("algobet.predictions.training.pipeline.FeaturePipeline")
    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_evaluate_with_calibrator(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """_evaluate uses calibrator when available."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(
            config=TrainingConfig(),
            session=MagicMock(),
        )

        # Set up calibrator
        calibrator = MagicMock()
        calibrated_probas = np.random.dirichlet([1, 1, 1], size=30).astype(np.float64)
        calibrator.calibrate.return_value = calibrated_probas
        pipeline._calibrator = calibrator

        predictor = MagicMock()
        raw_probas = np.random.dirichlet([1, 1, 1], size=30).astype(np.float64)
        predictor.predict_proba.return_value = raw_probas

        X = np.random.randn(30, 5).astype(np.float64)
        y = np.random.randint(0, 3, size=30).astype(np.int64)

        pipeline._evaluate(predictor, X, y)

        calibrator.calibrate.assert_called_once()

    @patch("algobet.predictions.training.pipeline.FeaturePipeline")
    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_evaluate_adds_market_diagnostics_without_training_on_odds(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """Evaluation may compare with odds, but only as post-hoc diagnostics."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(
            config=TrainingConfig(),
            session=MagicMock(),
        )

        predictor = MagicMock()
        predictor.predict_proba.return_value = np.array(
            [
                [0.20, 0.20, 0.60],
                [0.55, 0.25, 0.20],
                [0.25, 0.50, 0.25],
            ],
            dtype=np.float64,
        )

        X = np.ones((3, 2), dtype=np.float64)
        y = np.array([2, 0, 1], dtype=np.int64)
        matches_df = pd.DataFrame(
            {
                "odds_home": [5.5, 2.0, 3.2],
                "odds_draw": [4.2, 3.4, 2.8],
                "odds_away": [1.55, 4.0, 3.1],
            }
        )

        metrics = pipeline._evaluate(predictor, X, y, matches_df)

        assert metrics["market_samples"] == pytest.approx(3.0)
        assert metrics["market_log_loss"] > 0.0
        assert 0.0 <= metrics["market_favorite_accuracy"] <= 1.0
        assert 0.0 <= metrics["market_favorite_agreement"] <= 1.0
        assert "market_model_probability_mae" in metrics


class TestTrainingPipelineFeatureSelection:
    """Tests for the two-pass feature selection helper."""

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_choose_selected_features_respects_threshold_and_sample_cap(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Selection should use normalized gain and min-samples caps."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        feature_pipeline = MagicMock()
        pipeline = TrainingPipeline(
            config=TrainingConfig(
                feature_selection_threshold=0.1,
                min_samples_per_feature=2,
            ),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )

        selected = pipeline._choose_selected_feature_names(
            feature_names=["f1", "f2", "f3", "f4"],
            feature_importance={"f1": 8.0, "f2": 1.0, "f3": 0.5, "f4": 0.0},
            n_samples=4,
        )

        assert selected == ["f1", "f2"]

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_apply_feature_selection_reduces_arrays_and_records_subset(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """The probe pass should reduce the feature matrix before final training."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2", "f3"]
        feature_pipeline.set_selected_features = MagicMock()
        pipeline = TrainingPipeline(
            config=TrainingConfig(
                feature_selection=True,
                feature_selection_threshold=0.2,
            ),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )

        probe = MagicMock()
        probe.feature_importance = {"f1": 9.0, "f2": 1.0, "f3": 0.0}
        pipeline._train_model = MagicMock(return_value=probe)

        # Set the dataframes so the production refit path is exercised
        pipeline._train_df = pd.DataFrame({"a": [1, 2, 3, 4]})
        pipeline._val_df = pd.DataFrame({"a": [5, 6]})
        pipeline._test_df = pd.DataFrame({"a": [7, 8]})

        # Simulate fit_transform/transform returning reduced matrices
        feature_pipeline.fit_transform.return_value = np.array(
            [[1], [2], [3], [4]], dtype=np.float64
        )
        feature_pipeline.transform.side_effect = [
            np.array([[5], [6]], dtype=np.float64),
            np.array([[7], [8]], dtype=np.float64),
        ]

        X_train = np.arange(12, dtype=np.float64).reshape(4, 3)
        X_val = np.arange(6, dtype=np.float64).reshape(2, 3)
        X_test = np.arange(6, dtype=np.float64).reshape(2, 3)

        selected_train, selected_val, selected_test = pipeline._apply_feature_selection(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=np.array([0, 1, 2, 0], dtype=np.int64),
            y_val=np.array([1, 2], dtype=np.int64),
            class_weights=None,
            hyperparameters={},
        )

        assert pipeline._selected_feature_names == ["f1"]
        feature_pipeline.set_selected_features.assert_called_once_with(["f1"])
        # In the production path the shapes come from the refit pipeline
        assert selected_train.shape == (4, 1)
        assert selected_val.shape == (2, 1)
        assert selected_test.shape == (2, 1)
        feature_pipeline.fit_transform.assert_called_once()

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_apply_feature_selection_reuses_cached_raw_features(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Feature selection should refit from cached raw frames, not regenerate."""
        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2", "f3"]
        feature_pipeline.set_selected_features = MagicMock()
        feature_pipeline.fit_transform_raw_features.return_value = np.array(
            [[1], [2], [3], [4]], dtype=np.float64
        )
        feature_pipeline.transform_raw_features.side_effect = [
            np.array([[5], [6]], dtype=np.float64),
            np.array([[7], [8]], dtype=np.float64),
        ]

        pipeline = TrainingPipeline(
            config=TrainingConfig(
                feature_selection=True,
                feature_selection_threshold=0.2,
            ),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )
        pipeline._train_raw_features = pd.DataFrame(
            {"f1": [1, 2, 3, 4], "f2": [0, 0, 0, 0], "f3": [9, 9, 9, 9]}
        )
        pipeline._val_raw_features = pd.DataFrame(
            {"f1": [5, 6], "f2": [0, 0], "f3": [9, 9]}
        )
        pipeline._test_raw_features = pd.DataFrame(
            {"f1": [7, 8], "f2": [0, 0], "f3": [9, 9]}
        )

        probe = MagicMock()
        probe.feature_importance = {"f1": 9.0, "f2": 1.0, "f3": 0.0}
        pipeline._train_model = MagicMock(return_value=probe)

        y_train = np.array([0, 1, 2, 0], dtype=np.int64)
        selected_train, selected_val, selected_test = pipeline._apply_feature_selection(
            X_train=np.arange(12, dtype=np.float64).reshape(4, 3),
            X_val=np.arange(6, dtype=np.float64).reshape(2, 3),
            X_test=np.arange(6, dtype=np.float64).reshape(2, 3),
            y_train=y_train,
            y_val=np.array([1, 2], dtype=np.int64),
            class_weights=None,
            hyperparameters={},
        )

        feature_pipeline.set_selected_features.assert_called_once_with(["f1"])
        feature_pipeline.fit_transform_raw_features.assert_called_once_with(
            pipeline._train_raw_features,
            y_train,
        )
        assert feature_pipeline.transform_raw_features.call_count == 2
        feature_pipeline.fit_transform.assert_not_called()
        assert selected_train.shape == (4, 1)
        assert selected_val.shape == (2, 1)
        assert selected_test.shape == (2, 1)


class TestTrainingPipelineOddsFree:
    """Tests for odds-free training behavior."""

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_init_rejects_unsupported_feature_groups(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Odds-derived feature groups should not be accepted."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        with pytest.raises(ValueError, match="Unsupported feature groups: odds"):
            TrainingPipeline(
                config=TrainingConfig(feature_groups=["odds"]),
                session=MagicMock(),
            )

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_init_accepts_enriched_stats_feature_group(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Enriched stats should remain available as a non-odds feature group."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(
            config=TrainingConfig(feature_groups=["enriched_stats"]),
            session=MagicMock(),
        )

        assert "home_xg_for_avg_3" in pipeline.feature_pipeline.feature_names

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_run_saves_feature_pipeline_beside_model(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Training should persist the odds-free pipeline with the model."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2"]
        pipeline = TrainingPipeline(
            config=TrainingConfig(calibrate_probabilities=False),
            session=MagicMock(),
            models_path=Path("test/models"),
            feature_pipeline=feature_pipeline,
        )

        mock_registry = mock_registry_cls.return_value
        mock_registry.save_model.return_value = "xgboost_20260507_010203"

        pipeline._prepare_data = MagicMock(
            return_value=(
                np.ones((2, 2), dtype=np.float64),
                np.ones((2, 2), dtype=np.float64),
                np.ones((2, 2), dtype=np.float64),
                np.array([0, 1], dtype=np.int64),
                np.array([1, 2], dtype=np.int64),
                np.array([2, 0], dtype=np.int64),
            )
        )
        predictor = MagicMock()
        predictor.feature_importance = {"f1": 0.6}
        pipeline._train_model = MagicMock(return_value=predictor)
        pipeline._evaluate = MagicMock(
            side_effect=[
                {"accuracy": 0.7},
                {"accuracy": 0.6},
                {"accuracy": 0.5},
            ]
        )

        result = pipeline.run()
        expected_path = (
            Path("test/models")
            / "xgboost"
            / "xgboost_20260507_010203"
            / "feature_pipeline"
        )

        feature_pipeline.save.assert_called_once_with(expected_path)
        mock_registry.update_model_hyperparameters.assert_called_once()
        assert mock_registry.update_model_hyperparameters.call_args.kwargs[
            "hyperparameters"
        ]["feature_pipeline_path"] == str(expected_path)
        assert pipeline._train_model.call_args.args[5] is None
        assert result.feature_schema_version == MODEL_FEATURE_SCHEMA_VERSION
        assert result.model_path == Path("test/models/xgboost/xgboost_20260507_010203")

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_run_applies_class_weights_only_when_enabled(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Class weights should be opt-in for calibrated probability training."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2"]
        pipeline = TrainingPipeline(
            config=TrainingConfig(
                calibrate_probabilities=False,
                outcome_balance=True,
            ),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )

        mock_registry = mock_registry_cls.return_value
        mock_registry.save_model.return_value = "xgboost_20260507_010203"

        pipeline._prepare_data = MagicMock(
            return_value=(
                np.ones((5, 2), dtype=np.float64),
                np.ones((2, 2), dtype=np.float64),
                np.ones((2, 2), dtype=np.float64),
                np.array([0, 0, 0, 1, 2], dtype=np.int64),
                np.array([1, 2], dtype=np.int64),
                np.array([2, 0], dtype=np.int64),
            )
        )
        predictor = MagicMock()
        predictor.feature_importance = {"f1": 0.6}
        pipeline._train_model = MagicMock(return_value=predictor)
        pipeline._evaluate = MagicMock(
            side_effect=[
                {"accuracy": 0.7},
                {"accuracy": 0.6},
                {"accuracy": 0.5},
            ]
        )

        pipeline.run()

        class_weights = pipeline._train_model.call_args.args[5]
        assert class_weights is not None
        assert set(class_weights) == {0, 1, 2}

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_run_recovers_collapsed_xgboost_model(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Training should not save an XGBoost model that predicts one class."""
        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2"]
        pipeline = TrainingPipeline(
            config=TrainingConfig(
                calibrate_probabilities=False,
                outcome_balance=False,
            ),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )
        pipeline._prepare_data = MagicMock(
            return_value=(
                np.ones((6, 2), dtype=np.float64),
                np.ones((3, 2), dtype=np.float64),
                np.ones((3, 2), dtype=np.float64),
                np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
                np.array([0, 1, 2], dtype=np.int64),
                np.array([0, 1, 2], dtype=np.int64),
            )
        )

        collapsed = MagicMock()
        collapsed.predict_proba.return_value = np.array(
            [[0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.2]],
            dtype=np.float64,
        )
        recovered = MagicMock()
        recovered.predict_proba.return_value = np.array(
            [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.2, 0.3, 0.5]],
            dtype=np.float64,
        )
        recovered.feature_importance = {"f1": 0.5, "f2": 0.5}
        recovered.effective_hyperparameters = {}
        pipeline._train_model = MagicMock(side_effect=[collapsed, recovered])
        pipeline._evaluate = MagicMock(
            side_effect=[
                {"accuracy": 0.7},
                {"accuracy": 0.6},
                {"accuracy": 0.5},
            ]
        )
        mock_registry = mock_registry_cls.return_value
        mock_registry.save_model.return_value = "xgboost_20260508_200000"

        pipeline.run()

        assert pipeline._train_model.call_count == 2
        class_weights = pipeline._train_model.call_args.args[5]
        assert class_weights is not None
        saved_hyperparameters = mock_registry.save_model.call_args.kwargs[
            "hyperparameters"
        ]
        assert saved_hyperparameters["outcome_balance"] is True
        assert saved_hyperparameters["collapse_recovery"]["triggered"] is True
        assert (
            saved_hyperparameters["collapse_recovery"]["strategy"]
            == "class_weighted_full_features"
        )

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_run_rejects_collapsed_model_when_recovery_disabled(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Degenerate models should fail before they are saved."""
        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2"]
        pipeline = TrainingPipeline(
            config=TrainingConfig(
                calibrate_probabilities=False,
                collapse_recovery=False,
            ),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )
        pipeline._prepare_data = MagicMock(
            return_value=(
                np.ones((6, 2), dtype=np.float64),
                np.ones((3, 2), dtype=np.float64),
                np.ones((3, 2), dtype=np.float64),
                np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
                np.array([0, 1, 2], dtype=np.int64),
                np.array([0, 1, 2], dtype=np.int64),
            )
        )
        collapsed = MagicMock()
        collapsed.predict_proba.return_value = np.array(
            [[0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.2]],
            dtype=np.float64,
        )
        pipeline._train_model = MagicMock(return_value=collapsed)

        with pytest.raises(ValueError, match="degenerate model"):
            pipeline.run()

        mock_registry_cls.return_value.save_model.assert_not_called()

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_apply_feature_selection_rejects_odds_features(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Odds-derived features must never survive feature selection."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "implied_prob_home", "f3"]
        feature_pipeline.set_selected_features = MagicMock()
        pipeline = TrainingPipeline(
            config=TrainingConfig(
                feature_selection=True,
                feature_selection_threshold=0.2,
            ),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )

        probe = MagicMock()
        probe.feature_importance = {"f1": 1.0, "implied_prob_home": 9.0, "f3": 0.0}
        pipeline._train_model = MagicMock(return_value=probe)

        with pytest.raises(ValueError, match="Odds-derived features detected"):
            pipeline._apply_feature_selection(
                X_train=np.ones((4, 3), dtype=np.float64),
                X_val=np.ones((2, 3), dtype=np.float64),
                X_test=np.ones((2, 3), dtype=np.float64),
                y_train=np.array([0, 1, 2, 0], dtype=np.int64),
                y_val=np.array([1, 2], dtype=np.int64),
                class_weights=None,
                hyperparameters={},
            )


class TestMatchRepositoryStandings:
    """Tests for time-aware standings lookup."""

    def _standing(self, as_of: datetime, points: int) -> TeamStandings:
        return TeamStandings(
            team_id=1,
            as_of=as_of,
            matches_played=points // 3,
            points=points,
            goals_for=points,
            goals_against=0,
            goal_diff=points,
            wins=points // 3,
            draws=0,
            losses=0,
            position=1,
            total_teams=2,
            points_per_game=float(points),
        )

    def test_get_team_standings_uses_latest_snapshot_before_match_date(self) -> None:
        """Standings features should not see final-season snapshots."""
        repo = MatchRepository(MagicMock())
        repo._standings_cache[(10, 20)] = {
            1: [
                self._standing(datetime(2026, 1, 1), 3),
                self._standing(datetime(2026, 2, 1), 6),
                self._standing(datetime(2026, 3, 1), 9),
            ]
        }

        standing = repo.get_team_standings(
            team_id=1,
            tournament_id=10,
            season_id=20,
            before_date=datetime(2026, 2, 15),
        )

        assert standing is not None
        assert standing.points == 6
        assert (
            repo.get_team_standings(
                team_id=1,
                tournament_id=10,
                season_id=20,
                before_date=datetime(2025, 12, 31),
            )
            is None
        )

    def test_compute_standings_history_uses_full_season_team_count(self) -> None:
        """Early-season snapshots must reflect the true league size."""
        from types import SimpleNamespace

        repo = MatchRepository(MagicMock())
        # Only two teams have played so far, but the league has 4 teams total
        matches = [
            SimpleNamespace(
                match_date=datetime(2026, 1, 1),
                home_team_id=1,
                away_team_id=2,
                home_score=1,
                away_score=0,
            ),
        ]
        repo._compute_standings_history(10, 20, matches)

        history = repo._standings_cache[(10, 20)]
        assert history[1][0].total_teams == 2  # Only 2 teams appeared in matches


class TestTrainModel:
    """Tests for the train_model convenience function."""

    def test_train_model_creates_config(self) -> None:
        """train_model creates correct TrainingConfig."""
        config = TrainingConfig(
            model_type="lightgbm",
            tune_hyperparameters=True,
            description="Test description",
        )
        assert config.model_type == "lightgbm"
        assert config.tune_hyperparameters is True
        assert config.description == "Test description"
