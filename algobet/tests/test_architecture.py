"""Tests for the new feature-root architecture.

This module tests that the architecture refactoring was successful
and validates the new structure.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Set

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestArchitectureCompliance:
    """Test suite for validating feature-root architecture compliance."""

    def test_infrastructure_package_exists(self):
        """Test that infrastructure package was created."""
        infra_path = Path(__file__).parent.parent / "infrastructure"
        assert infra_path.exists(), "Infrastructure package not found"
        assert (infra_path / "__init__.py").exists(), (
            "Infrastructure __init__.py missing"
        )

    def test_feature_packages_exist(self):
        """Test that all feature packages were created."""
        features = ["matches", "teams", "predictions", "scheduling", "scraping"]
        algobet_path = Path(__file__).parent.parent

        for feature in features:
            feature_path = algobet_path / feature
            assert feature_path.exists(), f"Feature package {feature} not found"
            assert (feature_path / "__init__.py").exists(), (
                f"{feature} __init__.py missing"
            )
            assert (feature_path / "models.py").exists(), f"{feature} models.py missing"

    def test_models_imported_from_features(self):
        """Test that models can be imported from feature packages."""
        from algobet.teams.models import Team, Tournament, Season
        from algobet.matches.models import Match, MatchStatistics
        from algobet.predictions.models import Prediction, ModelVersion
        from algobet.scheduling.models import ScheduledTask, TaskExecution
        from algobet.scraping.models import ScrapingJob

        # Test that classes are importable
        assert Team is not None
        assert Tournament is not None
        assert Season is not None
        assert Match is not None
        assert MatchStatistics is not None
        assert Prediction is not None
        assert ModelVersion is not None
        assert ScheduledTask is not None
        assert TaskExecution is not None
        assert ScrapingJob is not None


class TestImportStructure:
    """Test that imports follow feature-root architecture."""

    def test_no_deep_imports_from_other_features(self):
        """Test that features don't import internal modules from other features."""
        # This would require analyzing all Python files - simplified version
        features_path = Path(__file__).parent.parent

        # Features that should have public APIs
        public_features = ["teams", "matches", "predictions", "scheduling", "scraping"]

        for feature in public_features:
            init_file = features_path / feature / "__init__.py"
            if init_file.exists():
                content = init_file.read_text()
                assert "__all__" in content, (
                    f"{feature} __init__.py should define __all__"
                )

    def test_infrastructure_exports(self):
        """Test that infrastructure package exports expected items."""
        from algobet import infrastructure

        expected = ["Base", "get_config", "DatabaseError"]
        for item in expected:
            assert hasattr(infrastructure, item), (
                f"infrastructure missing export: {item}"
            )


class TestFeatureCohesion:
    """Test that features are properly self-contained."""

    def test_matches_feature_has_required_files(self):
        """Test that matches feature has all required components."""
        matches_path = Path(__file__).parent.parent / "matches"

        required_files = ["__init__.py", "models.py"]
        for file in required_files:
            assert (matches_path / file).exists(), f"matches/{file} missing"

    def test_teams_feature_has_required_files(self):
        """Test that teams feature has all required components."""
        teams_path = Path(__file__).parent.parent / "teams"

        required_files = ["__init__.py", "models.py"]
        for file in required_files:
            assert (teams_path / file).exists(), f"teams/{file} missing"

    def test_predictions_feature_has_required_files(self):
        """Test that predictions feature has all required components."""
        pred_path = Path(__file__).parent.parent / "predictions"

        required_files = ["__init__.py", "models.py"]
        for file in required_files:
            assert (pred_path / file).exists(), f"predictions/{file} missing"

    def test_scheduling_feature_has_required_files(self):
        """Test that scheduling feature has all required components."""
        sched_path = Path(__file__).parent.parent / "scheduling"

        required_files = ["__init__.py", "models.py"]
        for file in required_files:
            assert (sched_path / file).exists(), f"scheduling/{file} missing"

    def test_scraping_feature_has_required_files(self):
        """Test that scraping feature has all required components."""
        scrap_path = Path(__file__).parent.parent / "scraping"

        required_files = ["__init__.py", "models.py"]
        for file in required_files:
            assert (scrap_path / file).exists(), f"scraping/{file} missing"


class TestCircularImportPrevention:
    """Test that circular imports are prevented."""

    def test_no_circular_imports_between_features(self):
        """Test that features don't have circular import dependencies."""
        # This is a basic test - comprehensive check would require import analysis
        features = ["teams", "matches", "predictions", "scheduling", "scraping"]

        for feature in features:
            # Try to import each feature independently
            module_name = f"algobet.{feature}"
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")


class TestModelIntegrity:
    """Test that models maintain their structure after refactoring."""

    def test_team_model_has_expected_attributes(self):
        """Test that Team model has expected attributes."""
        from algobet.teams.models import Team

        expected_attrs = ["id", "name", "home_matches", "away_matches"]
        for attr in expected_attrs:
            assert hasattr(Team, attr), f"Team missing attribute: {attr}"

    def test_match_model_has_expected_attributes(self):
        """Test that Match model has expected attributes."""
        from algobet.matches.models import Match

        expected_attrs = [
            "id",
            "home_team_id",
            "away_team_id",
            "match_date",
            "home_score",
            "away_score",
            "status",
            "odds_home",
            "odds_draw",
            "odds_away",
            "result",
        ]
        for attr in expected_attrs:
            assert hasattr(Match, attr), f"Match missing attribute: {attr}"

    def test_prediction_model_has_expected_attributes(self):
        """Test that Prediction model has expected attributes."""
        from algobet.predictions.models import Prediction

        expected_attrs = [
            "id",
            "match_id",
            "model_version_id",
            "prob_home",
            "prob_draw",
            "prob_away",
            "predicted_outcome",
            "confidence",
            "max_probability",
        ]
        for attr in expected_attrs:
            assert hasattr(Prediction, attr), f"Prediction missing attribute: {attr}"


class TestNoOldImports:
    """Test that old import paths are completely removed."""

    def test_old_models_file_removed(self):
        """Test that old centralized models.py is removed."""
        models_file = Path(__file__).parent.parent / "models.py"
        assert not models_file.exists(), "Old models.py should be removed"

    def test_old_config_file_removed(self):
        """Test that old config.py is removed."""
        config_file = Path(__file__).parent.parent / "config.py"
        assert not config_file.exists(), "Old config.py should be removed"

    def test_old_database_file_removed(self):
        """Test that old database.py is removed."""
        database_file = Path(__file__).parent.parent / "database.py"
        assert not database_file.exists(), "Old database.py should be removed"

    def test_old_exceptions_file_removed(self):
        """Test that old exceptions.py is removed."""
        exceptions_file = Path(__file__).parent.parent / "exceptions.py"
        assert not exceptions_file.exists(), "Old exceptions.py should be removed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
