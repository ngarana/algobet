"""Essential tests for FeaturePipeline."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from algobet.predictions.features.generators import (
    EnrichedStatsFeatureGenerator,
    FeatureSchema,
    HeadToHeadGenerator,
)
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

    def test_fit_transform_generates_raw_features_once(self, pipeline, mock_generators):
        """fit_transform should not regenerate expensive raw features."""
        matches = pd.DataFrame(
            [
                {"id": 1, "home_team_id": 10, "away_team_id": 20},
                {"id": 2, "home_team_id": 30, "away_team_id": 40},
            ]
        )

        pipeline.fit_transform(matches, MagicMock())

        assert mock_generators.generate.call_count == 1
        assert pipeline.last_raw_features is not None

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

    def test_create_default_excludes_odds_features(self):
        """Default training features should be free of implied-odds signals."""
        pipeline = FeaturePipeline.create_default()

        forbidden_terms = (
            "odds",
            "implied_prob",
            "bookmaker",
            "favorite",
            "market_",
        )
        assert pipeline.feature_names
        assert all(
            not any(term in feature_name for term in forbidden_terms)
            for feature_name in pipeline.feature_names
        )
        assert "h2h_goal_diff_avg_from_home_perspective" in pipeline.feature_names
        assert "h2h_goal_diff_avg" not in pipeline.feature_names
        assert "home_starter_minutes_avg_3" in pipeline.feature_names
        assert "away_starter_count_avg_5" in pipeline.feature_names

    def test_set_selected_features_filters_names_and_schema(
        self, pipeline, mock_generators
    ):
        """Selected feature subsets should also constrain the feature schema."""
        mock_generators.get_schema.return_value = FeatureSchema(
            version="v1.0",
            features={"feature1": float, "feature2": float},
        )

        pipeline.set_selected_features(["feature1"])

        assert pipeline.feature_names == ["feature1"]
        assert list(pipeline.get_schema().features) == ["feature1"]

    def test_clear_selected_features_restores_full_feature_set(
        self, pipeline, mock_generators
    ):
        """The training recovery path can restore pruned features."""
        pipeline.set_selected_features(["feature1"])

        pipeline.clear_selected_features()

        assert pipeline.selected_feature_names is None
        assert pipeline.feature_names == ["feature1", "feature2"]

    def test_save_load_preserves_selected_feature_subset(self, tmp_path):
        """Saved pipelines should reload the same selected feature shape."""
        pipeline = FeaturePipeline.create_default()
        selected_features = ["home_points_last_3", "away_points_last_3"]
        pipeline.set_selected_features(selected_features)

        pipeline_path = tmp_path / "pipeline"
        pipeline.save(pipeline_path)

        loaded = FeaturePipeline.load(pipeline_path)

        assert loaded.selected_feature_names == selected_features
        assert loaded.feature_names == selected_features

    def test_load_raises_when_saved_selected_features_are_unavailable(self, tmp_path):
        """Loading should fail loudly if generator code drift removed saved features."""
        pipeline = FeaturePipeline.create_default()
        pipeline.set_selected_features(["home_points_last_3", "away_points_last_3"])

        pipeline_path = tmp_path / "pipeline"
        pipeline.save(pipeline_path)

        # Corrupt the saved config to reference a feature that no longer exists
        with open(pipeline_path / "config.json") as f:
            config_data = json.load(f)
        config_data["selected_feature_names"] = ["home_points_last_3", "ghost_feature"]
        with open(pipeline_path / "config.json", "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="cannot be restored"):
            FeaturePipeline.load(pipeline_path)


class TestEnrichedStatsFeatureGenerator:
    """Tests for enriched stats feature generation."""

    def test_generate_uses_understat_and_player_rollups(self) -> None:
        """Historical enriched stats should feed rolling team features."""
        generator = EnrichedStatsFeatureGenerator(window_sizes=[2])
        repository = MagicMock()

        team_one_matches = [
            SimpleNamespace(
                home_team_id=1,
                away_team_id=3,
                statistics=SimpleNamespace(
                    home_xg=1.6,
                    away_xg=0.8,
                    home_npxg=1.4,
                    away_npxg=0.7,
                    home_shots=14,
                    away_shots=9,
                    home_shots_on_target=6,
                    away_shots_on_target=3,
                    home_corners=7,
                    away_corners=4,
                    home_ppda=9.5,
                    away_ppda=12.0,
                    home_deep_completions=11,
                    away_deep_completions=6,
                ),
                player_stats=[
                    SimpleNamespace(
                        team_id=1,
                        goals=2,
                        assists=1,
                        shots=5,
                        shots_on_target=3,
                        minutes_played=90,
                        is_starter=True,
                    ),
                    SimpleNamespace(
                        team_id=1,
                        goals=0,
                        assists=1,
                        shots=2,
                        shots_on_target=1,
                        minutes_played=85,
                        is_starter=False,
                    ),
                ],
            ),
            SimpleNamespace(
                home_team_id=4,
                away_team_id=1,
                statistics=SimpleNamespace(
                    home_xg=0.7,
                    away_xg=1.2,
                    home_npxg=0.6,
                    away_npxg=1.1,
                    home_shots=8,
                    away_shots=12,
                    home_shots_on_target=2,
                    away_shots_on_target=5,
                    home_corners=3,
                    away_corners=6,
                    home_ppda=13.0,
                    away_ppda=8.0,
                    home_deep_completions=5,
                    away_deep_completions=10,
                ),
                player_stats=[
                    SimpleNamespace(
                        team_id=1,
                        goals=1,
                        assists=0,
                        shots=4,
                        shots_on_target=2,
                        minutes_played=88,
                        is_starter=True,
                    )
                ],
            ),
        ]
        team_two_matches = [
            SimpleNamespace(
                home_team_id=2,
                away_team_id=5,
                statistics=SimpleNamespace(
                    home_xg=1.0,
                    away_xg=0.9,
                    home_npxg=0.8,
                    away_npxg=0.7,
                    home_shots=11,
                    away_shots=10,
                    home_shots_on_target=4,
                    away_shots_on_target=4,
                    home_corners=5,
                    away_corners=5,
                    home_ppda=10.0,
                    away_ppda=11.5,
                    home_deep_completions=8,
                    away_deep_completions=7,
                ),
                player_stats=[
                    SimpleNamespace(
                        team_id=2,
                        goals=1,
                        assists=1,
                        shots=3,
                        shots_on_target=2,
                        minutes_played=90,
                        is_starter=True,
                    )
                ],
            )
        ]

        def get_team_matches(team_id: int, **_: object) -> list[object]:
            if team_id == 1:
                return team_one_matches
            if team_id == 2:
                return team_two_matches
            return []

        repository.get_team_matches.side_effect = get_team_matches

        matches = pd.DataFrame(
            [
                {
                    "id": 99,
                    "match_date": datetime(2026, 5, 4, 19, 0, 0),
                    "home_team_id": 1,
                    "away_team_id": 2,
                }
            ]
        )

        result = generator.generate(matches, repository)

        assert result.loc[99, "home_xg_for_avg_2"] == pytest.approx(1.4)
        assert result.loc[99, "home_xg_against_avg_2"] == pytest.approx(0.75)
        assert result.loc[99, "home_player_shots_avg_2"] == pytest.approx(5.5)
        assert result.loc[99, "home_player_minutes_avg_2"] == pytest.approx(131.5)
        assert result.loc[99, "home_starter_minutes_avg_2"] == pytest.approx(89.0)
        assert result.loc[99, "home_starter_count_avg_2"] == pytest.approx(1.0)
        assert result.loc[99, "home_enriched_match_coverage_2"] == pytest.approx(1.0)
        assert result.loc[99, "away_xg_for_avg_2"] == pytest.approx(1.0)
        assert result.loc[99, "away_starter_minutes_avg_2"] == pytest.approx(90.0)
        assert result.loc[99, "away_starter_count_avg_2"] == pytest.approx(1.0)
        assert result.loc[99, "away_player_stats_coverage_2"] == pytest.approx(1.0)


class TestHeadToHeadGenerator:
    """Tests for head-to-head feature generation."""

    def test_goal_diff_uses_home_perspective_schema_name(self) -> None:
        """H2H goal difference should be keyed to the current home team."""
        generator = HeadToHeadGenerator(max_h2h_matches=5)
        repository = MagicMock()
        repository.get_h2h_matches.return_value = [
            SimpleNamespace(
                home_team_id=1,
                away_team_id=2,
                home_score=2,
                away_score=1,
            ),
            SimpleNamespace(
                home_team_id=2,
                away_team_id=1,
                home_score=3,
                away_score=1,
            ),
        ]
        matches = pd.DataFrame(
            [
                {
                    "id": 101,
                    "match_date": datetime(2026, 5, 4, 19, 0, 0),
                    "home_team_id": 1,
                    "away_team_id": 2,
                }
            ]
        )

        result = generator.generate(matches, repository)

        assert "h2h_goal_diff_avg" not in generator.feature_names
        assert "h2h_goal_diff_avg" not in result.columns
        assert result.loc[
            101, "h2h_goal_diff_avg_from_home_perspective"
        ] == pytest.approx(-0.5)
