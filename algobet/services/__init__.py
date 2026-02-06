"""Service layer for AlgoBet business logic."""

from __future__ import annotations

from algobet.services.base import BaseService
from algobet.services.prediction_service import PredictionResult, PredictionService
from algobet.services.scheduler_service import SchedulerService, TaskDefinition
from algobet.services.scraping_service import (
    JobStatus,
    ScrapingJob,
    ScrapingProgress,
    ScrapingService,
)

__all__ = [
    "BaseService",
    "PredictionService",
    "PredictionResult",
    "ScrapingService",
    "ScrapingJob",
    "ScrapingProgress",
    "JobStatus",
    "SchedulerService",
    "TaskDefinition",
]
