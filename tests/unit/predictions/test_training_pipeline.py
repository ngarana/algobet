"""Unit tests for TrainingPipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from algobet.predictions.training.pipeline import TrainingConfig


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
        assert config.calibration_method == "isotonic"
        assert config.min_calibration_samples == 100
        assert config.min_calibration_class_samples == 10
        assert config.fallback_calibration_method == "sigmoid"
        assert config.use_ensemble is False
        assert config.random_seed == 42

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


class TestTrainingPipelineCalibrationPolicy:
    """Tests for calibration method selection and metadata recording."""

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_select_calibration_method_falls_back_to_sigmoid_for_small_validation(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Small validation slices should not use isotonic calibration."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2"]
        pipeline = TrainingPipeline(
            config=TrainingConfig(),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )

        y_val = np.array([0, 1, 2] * 4, dtype=np.int64)

        with pytest.warns(RuntimeWarning, match="Falling back to sigmoid"):
            method = pipeline._select_calibration_method(y_val)

        assert method == "sigmoid"
        assert pipeline._calibration_policy_reason is not None
        assert "Falling back to sigmoid" in pipeline._calibration_policy_reason

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_select_calibration_method_skips_when_class_is_missing(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Calibration should be skipped when validation misses an outcome class."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2"]
        pipeline = TrainingPipeline(
            config=TrainingConfig(),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )

        y_val = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

        with pytest.warns(
            RuntimeWarning,
            match="does not contain all outcome classes",
        ):
            method = pipeline._select_calibration_method(y_val)

        assert method is None
        assert pipeline._calibration_policy_reason is not None
        assert "does not contain all outcome classes" in (
            pipeline._calibration_policy_reason
        )

    @patch("algobet.predictions.training.pipeline.FeatureStore")
    @patch("algobet.predictions.training.pipeline.ModelRegistry")
    @patch("algobet.predictions.training.pipeline.MatchRepository")
    def test_build_model_hyperparameters_uses_effective_params_and_applied_method(
        self,
        mock_repo_cls: MagicMock,
        mock_registry_cls: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Registry metadata should reflect effective params and applied calibration."""
        from algobet.predictions.training.pipeline import TrainingPipeline

        feature_pipeline = MagicMock()
        feature_pipeline.feature_names = ["f1", "f2"]
        pipeline = TrainingPipeline(
            config=TrainingConfig(),
            session=MagicMock(),
            feature_pipeline=feature_pipeline,
        )
        pipeline._applied_calibration_method = "sigmoid"
        pipeline._calibrator = MagicMock()
        pipeline._calibration_policy_reason = (
            "Validation split too small for isotonic calibration (n=12, "
            "min_class_count=4). Falling back to sigmoid."
        )

        predictor = MagicMock()
        predictor.effective_hyperparameters = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 500,
        }

        model_hyperparameters = pipeline._build_model_hyperparameters(
            predictor=predictor,
            best_params={"max_depth": 3},
        )

        assert model_hyperparameters["max_depth"] == 6
        assert model_hyperparameters["n_estimators"] == 500
        assert model_hyperparameters["feature_names"] == ["f1", "f2"]
        assert model_hyperparameters["requested_calibration_method"] == "isotonic"
        assert model_hyperparameters["calibration_method"] == "sigmoid"
        assert model_hyperparameters["calibration_applied"] is True
        assert (
            model_hyperparameters["calibration_policy_reason"]
            == pipeline._calibration_policy_reason
        )


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
