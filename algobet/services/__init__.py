"""Service layer for AlgoBet business logic."""

from __future__ import annotations

from algobet.services.analysis_service import AnalysisService
from algobet.services.async_base import AsyncBaseService
from algobet.services.async_database_service import AsyncDatabaseService
from algobet.services.async_model_management_service import AsyncModelManagementService
from algobet.services.async_query_service import AsyncQueryService
from algobet.services.base import BaseService
from algobet.services.database_service import DatabaseService
from algobet.services.model_management_service import ModelManagementService
from algobet.services.prediction_service import PredictionResult, PredictionService
from algobet.services.query_service import QueryService
from algobet.services.scheduler_service import SchedulerService, TaskDefinition
from algobet.services.scraping_service import (
    JobStatus,
    ScrapingJob,
    ScrapingProgress,
    ScrapingService,
)

__all__ = [
    # Sync services
    "AnalysisService",
    "BaseService",
    "DatabaseService",
    "ModelManagementService",
    "PredictionService",
    "PredictionResult",
    "QueryService",
    "ScrapingService",
    "ScrapingJob",
    "ScrapingProgress",
    "JobStatus",
    "SchedulerService",
    "TaskDefinition",
    # Async services
    "AsyncBaseService",
    "AsyncDatabaseService",
    "AsyncModelManagementService",
    "AsyncQueryService",
]
