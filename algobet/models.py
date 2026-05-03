"""Central re-export of all SQLAlchemy models for backward compatibility.

This module provides a single import point for all models in the AlgoBet system.
For new code, consider importing directly from the feature-specific modules
(e.g., algobet.matches.models, algobet.predictions.models) to reduce coupling.
"""

# Base class
from algobet.infrastructure.models import Base, MetadataMixin, TimestampMixin

# Match models
from algobet.matches.models import Match, MatchStatistics

# Prediction models
from algobet.predictions.models import (
    BacktestHistory,
    ModelFeature,
    ModelVersion,
    Prediction,
)

# Scheduling models
from algobet.scheduling.models import ScheduledTask, TaskExecution

# Scraping models
from algobet.scraping.models import (
    ScrapedOdds,
    ScrapingJob,
    ScrapingLog,
    ScrapingSource,
)

# Team/Tournament models
from algobet.teams.models import Season, Team, Tournament

__all__ = [
    # Base
    "Base",
    "TimestampMixin",
    "MetadataMixin",
    # Matches
    "Match",
    "MatchStatistics",
    # Teams/Tournaments
    "Team",
    "Tournament",
    "Season",
    # Predictions
    "Prediction",
    "ModelVersion",
    "ModelFeature",
    "BacktestHistory",
    # Scraping
    "ScrapedOdds",
    "ScrapingJob",
    "ScrapingLog",
    "ScrapingSource",
    # Scheduling
    "ScheduledTask",
    "TaskExecution",
]
