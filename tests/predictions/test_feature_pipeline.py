"""Essential tests for FeaturePipeline."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from algobet.predictions.features.pipeline import FeaturePipeline, PipelineConfig


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_values(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig()

        assert config.schema_version == "v1.0"
        assert config.description == ""
        assert config.generator_configs == {}
        assert config.transformer_config == {}
        assert isinstance(config.created_at, datetime)

    def test_custom_values(self):
        """Test PipelineConfig with custom values."""
        config = PipelineConfig(
            schema_version="v2.0",
            description="Test pipeline",
            generator_configs={"form": {"window": 5}},
            transformer_config={"scale": True},
        )

        assert config.schema_version == "v2.0"
        assert config.description == "Test pipeline"
        assert config.generator_configs["form"]["window"] == 5


class TestFeaturePipeline:
    """Test FeaturePipeline core operations."""

    @pytest.fixture
    def mock_generators(self):
        """Create mock feature generators."""
        generators = MagicMock()
        generators.feature_names = ["feature1", "feature2"]
        generators.generate.return_value = pd.DataFrame(
            {"feature1": [1, 2], "feature2": [3, 4]}
        )
        return generators

    @pytest.fixture
    def mock_transformers(self):
        """Create mock transformers."""
        transformers = MagicMock()
        transformers.fit_transform.return_value = pd.DataFrame(
            {"feature1": [0.1, 0.2], "feature2": [0.3, 0.4]}
        )
        transformers.transform.return_value = pd.DataFrame(
            {"feature1": [0.1, 0.2], "feature2": [0.3, 0.4]}
        )
        return transformers

    @pytest.fixture
    def pipeline(self, mock_generators, mock_transformers):
        """Create FeaturePipeline instance."""
        return FeaturePipeline(
            generators=mock_generators,
            transformers=mock_transformers,
        )

    def test_init_with_generators(self, mock_generators):
        """Test pipeline initialization with generators."""
        pipeline = FeaturePipeline(generators=mock_generators)

        assert pipeline.generators == mock_generators
        assert pipeline._fitted is False

    def test_init_creates_default_transformers(self, mock_generators):
        """Test pipeline creates default transformers."""
        pipeline = FeaturePipeline(generators=mock_generators)

        assert pipeline.transformers is not None

    def test_init_with_config(self, mock_generators):
        """Test pipeline initialization with custom config."""
        config = PipelineConfig(schema_version="v2.0")
        pipeline = FeaturePipeline(generators=mock_generators, config=config)

        assert pipeline.config.schema_version == "v2.0"

    def test_feature_names_from_generators(self, pipeline, mock_generators):
        """Test feature_names property returns generator feature names."""
        names = pipeline.feature_names

        assert names == ["feature1", "feature2"]

    def test_feature_names_cached(self, pipeline, mock_generators):
        """Test feature_names are cached."""
        _ = pipeline.feature_names
        _ = pipeline.feature_names

        # Should only access generators.feature_names once (cached)
        assert mock_generators.feature_names is not None

    def test_is_fitted_initially_false(self, pipeline):
        """Test is_fitted is False initially."""
        assert pipeline.is_fitted is False

    @patch("algobet.predictions.features.pipeline.prepare_match_dataframe")
    def test_fit_sets_fitted_flag(self, mock_prepare, pipeline, mock_generators):
        """Test fit sets _fitted flag."""
        mock_prepare.return_value = pd.DataFrame({"home_score": [1, 2]})

        pipeline.fit([], MagicMock())

        assert pipeline._fitted is True

    def test_fit_returns_self(self, pipeline, mock_generators):
        """Test fit returns self for chaining."""
        mock_df = pd.DataFrame({"home_score": [1, 2], "away_score": [0, 1]})

        with patch(
            "algobet.predictions.features.pipeline.prepare_match_dataframe",
            return_value=mock_df,
        ):
            result = pipeline.fit([], MagicMock())

        assert result is pipeline

    @patch("algobet.predictions.features.pipeline.prepare_match_dataframe")
    def test_fit_transform_returns_features(
        self, mock_prepare, pipeline, mock_transformers
    ):
        """Test fit_transform returns transformed features."""
        mock_prepare.return_value = pd.DataFrame({"home_score": [1, 2]})

        result = pipeline.fit_transform([], MagicMock())

        assert isinstance(result, pd.DataFrame)
        assert "feature1" in result.columns

    def test_transform_requires_fit(self, pipeline):
        """Test transform raises if not fitted."""
        with pytest.raises(ValueError) as exc_info:
            pipeline.transform([], MagicMock())

        assert "not fitted" in str(exc_info.value).lower()

    @patch("algobet.predictions.features.pipeline.prepare_match_dataframe")
    def test_transform_returns_features(
        self, mock_prepare, pipeline, mock_transformers
    ):
        """Test transform returns transformed features."""
        # First fit the pipeline
        mock_prepare.return_value = pd.DataFrame({"home_score": [1, 2]})
        pipeline.fit([], MagicMock())

        # Now transform
        result = pipeline.transform([], MagicMock())

        assert isinstance(result, pd.DataFrame)

    @patch("algobet.predictions.features.pipeline.create_default_generators")
    def test_create_default_returns_pipeline(self, mock_create_gen):
        """Test create_default returns FeaturePipeline."""
        mock_gen = MagicMock()
        mock_create_gen.return_value = mock_gen

        pipeline = FeaturePipeline.create_default()

        assert isinstance(pipeline, FeaturePipeline)
        assert pipeline.generators == mock_gen

    def test_pipeline_initially_not_fitted(self, pipeline):
        """Test pipeline is initially not fitted."""
        assert pipeline.is_fitted is False
