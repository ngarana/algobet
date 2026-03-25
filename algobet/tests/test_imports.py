"""Test suite for feature-root architecture imports.

Tests that imports work correctly across the new architecture.
"""

import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base


class TestFeatureImports:
    """Test imports from feature packages."""

    def test_import_teams_models(self):
        """Test importing from teams.models."""
        from algobet.teams.models import Team, Tournament, Season

        assert Team is not None
        assert Tournament is not None
        assert Season is not None

    def test_import_matches_models(self):
        """Test importing from matches.models."""
        from algobet.matches.models import Match, MatchStatistics

        assert Match is not None
        assert MatchStatistics is not None

    def test_import_predictions_models(self):
        """Test importing from predictions.models."""
        from algobet.predictions.models import (
            Prediction,
            ModelVersion,
            ModelFeature,
            BacktestHistory,
        )

        assert Prediction is not None
        assert ModelVersion is not None
        assert ModelFeature is not None
        assert BacktestHistory is not None

    def test_import_scheduling_models(self):
        """Test importing from scheduling.models."""
        from algobet.scheduling.models import ScheduledTask, TaskExecution

        assert ScheduledTask is not None
        assert TaskExecution is not None

    def test_import_scraping_models(self):
        """Test importing from scraping.models."""
        from algobet.scraping.models import ScrapingJob, ScrapingLog, ScrapedOdds

        assert ScrapingJob is not None
        assert ScrapingLog is not None
        assert ScrapedOdds is not None


class TestInfrastructureImports:
    """Test imports from infrastructure package."""

    def test_import_infrastructure_base(self):
        """Test importing Base from infrastructure."""
        from algobet.infrastructure.models import Base

        assert Base is not None

    def test_import_infrastructure_config(self):
        """Test importing config from infrastructure."""
        from algobet.infrastructure import get_config, AlgobetConfig

        assert get_config is not None
        assert AlgobetConfig is not None

    def test_import_infrastructure_exceptions(self):
        """Test importing exceptions from infrastructure."""
        from algobet.infrastructure import AlgoBetError, DatabaseError

        assert AlgoBetError is not None
        assert DatabaseError is not None


class TestModelRelationships:
    """Test that model relationships work correctly."""

    def test_team_match_relationship(self):
        """Test that Team-Match relationship is set up."""
        from algobet.teams.models import Team
        from algobet.matches.models import Match

        # Check that Team has match relationships
        assert hasattr(Team, "home_matches")
        assert hasattr(Team, "away_matches")

        # Check that Match has team relationships
        assert hasattr(Match, "home_team")
        assert hasattr(Match, "away_team")

    def test_match_prediction_relationship(self):
        """Test that Match-Prediction relationship is set up."""
        from algobet.matches.models import Match
        from algobet.predictions.models import Prediction

        # Check that Match has predictions
        assert hasattr(Match, "predictions")

        # Check that Prediction has match
        assert hasattr(Prediction, "match")

    def test_prediction_model_version_relationship(self):
        """Test that Prediction-ModelVersion relationship is set up."""
        from algobet.predictions.models import Prediction, ModelVersion

        # Check relationships
        assert hasattr(Prediction, "model_version")
        assert hasattr(ModelVersion, "predictions")


class TestPackageStructure:
    """Test that packages are properly structured."""

    def test_all_features_have_init(self):
        """Test that all feature packages have __init__.py."""
        from pathlib import Path

        algobet_path = Path(__file__).parent.parent
        features = ["teams", "matches", "predictions", "scheduling", "scraping"]

        for feature in features:
            init_file = algobet_path / feature / "__init__.py"
            assert init_file.exists(), f"{feature} missing __init__.py"

    def test_infrastructure_has_init(self):
        """Test that infrastructure has __init__.py."""
        from pathlib import Path

        algobet_path = Path(__file__).parent.parent
        init_file = algobet_path / "infrastructure" / "__init__.py"

        assert init_file.exists(), "infrastructure missing __init__.py"


class TestNoBackwardCompatibility:
    """Test that old import paths no longer work."""

    def test_old_models_import_fails(self):
        """Test that importing from old models.py raises ImportError."""
        with pytest.raises(ImportError):
            from algobet.models import Match

    def test_old_database_import_fails(self):
        """Test that importing from old database.py raises ImportError."""
        with pytest.raises(ImportError):
            from algobet.database import get_session

    def test_old_config_import_fails(self):
        """Test that importing from old config.py raises ImportError."""
        with pytest.raises(ImportError):
            from algobet.config import get_config

    def test_old_exceptions_import_fails(self):
        """Test that importing from old exceptions.py raises ImportError."""
        with pytest.raises(ImportError):
            from algobet.exceptions import AlgoBetError


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
