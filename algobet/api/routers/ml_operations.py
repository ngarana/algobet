"""API router for ML operations (train, backtest, calibrate, ablation)."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.schemas.ablation import (
    AblationRequest,
    AblationStudyResponse,
    PermutationImportanceResponse,
)
from algobet.api.schemas.ml_operations import (
    BacktestHistoryListResponse,
    BacktestRequest,
    BacktestResultResponse,
    CalibrateRequest,
    CalibrateResultResponse,
    TrainModelRequest,
    TrainModelResponse,
)
from algobet.services.ml_ops import MLOperationsOrchestrator

router = APIRouter()


def _ml_ops() -> MLOperationsOrchestrator:
    return MLOperationsOrchestrator()


@router.post("/train", response_model=TrainModelResponse)
def run_training(
    request: TrainModelRequest,
    db: Session = Depends(get_db),
    orchestrator: MLOperationsOrchestrator = Depends(_ml_ops),
) -> TrainModelResponse:
    """Train a prediction model and optionally activate it."""
    return orchestrator.run_training(request, db)


@router.post("/backtest", response_model=BacktestResultResponse)
def run_backtest(
    request: BacktestRequest,
    db: Session = Depends(get_db),
    orchestrator: MLOperationsOrchestrator = Depends(_ml_ops),
) -> BacktestResultResponse:
    """Run a historical backtest on model predictions."""
    return orchestrator.run_backtest(request, db)


@router.post("/calibrate", response_model=CalibrateResultResponse)
def run_calibrate(
    request: CalibrateRequest,
    db: Session = Depends(get_db),
    orchestrator: MLOperationsOrchestrator = Depends(_ml_ops),
) -> CalibrateResultResponse:
    """Calibrate model probabilities."""
    return orchestrator.run_calibrate(request, db)


@router.get("/backtest/history", response_model=BacktestHistoryListResponse)
def get_backtest_history(
    model_version_id: int | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    orchestrator: MLOperationsOrchestrator = Depends(_ml_ops),
) -> BacktestHistoryListResponse:
    """Get backtest history for analysis and comparison."""
    return orchestrator.get_backtest_history(model_version_id, limit, offset, db)


@router.get("/backtest/{backtest_id}", response_model=BacktestResultResponse)
def get_backtest_detail(
    backtest_id: int,
    db: Session = Depends(get_db),
    orchestrator: MLOperationsOrchestrator = Depends(_ml_ops),
) -> BacktestResultResponse:
    """Get detailed backtest result by ID."""
    return orchestrator.get_backtest_detail(backtest_id, db)


@router.post(
    "/ablation",
    response_model=PermutationImportanceResponse | AblationStudyResponse,
)
def run_ablation(
    request: AblationRequest,
    db: Session = Depends(get_db),
    orchestrator: MLOperationsOrchestrator = Depends(_ml_ops),
) -> PermutationImportanceResponse | AblationStudyResponse:
    """Run feature ablation or permutation importance analysis.

    - **permutation**: Shuffles feature columns per family on a trained model
      and measures the performance drop. Fast, no retraining.
    - **ablation**: Retrains the model excluding each feature group
      (leave-one-out) and compares metrics. Slow.
    """
    return orchestrator.run_ablation(request, db)
