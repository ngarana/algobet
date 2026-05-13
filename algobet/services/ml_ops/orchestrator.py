"""Facade for ML operation use cases."""

from sqlalchemy.orm import Session

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
from algobet.services.ml_ops.ablation_runner import AblationRunner
from algobet.services.ml_ops.backtest_runner import BacktestRunner
from algobet.services.ml_ops.calibration_runner import CalibrationRunner
from algobet.services.ml_ops.history_reader import BacktestHistoryReader
from algobet.services.ml_ops.training_runner import TrainingRunner


class MLOperationsOrchestrator:
    """Thin facade that coordinates ML operation collaborators."""

    def __init__(
        self,
        training_runner: TrainingRunner | None = None,
        backtest_runner: BacktestRunner | None = None,
        calibration_runner: CalibrationRunner | None = None,
        history_reader: BacktestHistoryReader | None = None,
        ablation_runner: AblationRunner | None = None,
    ) -> None:
        self.training_runner = training_runner or TrainingRunner()
        self.backtest_runner = backtest_runner or BacktestRunner()
        self.calibration_runner = calibration_runner or CalibrationRunner()
        self.history_reader = history_reader or BacktestHistoryReader()
        self.ablation_runner = ablation_runner or AblationRunner()

    def run_training(
        self,
        request: TrainModelRequest,
        db: Session,
    ) -> TrainModelResponse:
        return self.training_runner.run_training(request, db)

    def run_backtest(
        self,
        request: BacktestRequest,
        db: Session,
    ) -> BacktestResultResponse:
        return self.backtest_runner.run_backtest(request, db)

    def run_calibrate(
        self,
        request: CalibrateRequest,
        db: Session,
    ) -> CalibrateResultResponse:
        return self.calibration_runner.run_calibrate(request, db)

    def get_backtest_history(
        self,
        model_version_id: int | None,
        limit: int,
        offset: int,
        db: Session,
    ) -> BacktestHistoryListResponse:
        return self.history_reader.get_backtest_history(
            model_version_id,
            limit,
            offset,
            db,
        )

    def get_backtest_detail(
        self,
        backtest_id: int,
        db: Session,
    ) -> BacktestResultResponse:
        return self.history_reader.get_backtest_detail(backtest_id, db)

    def run_ablation(
        self,
        request: AblationRequest,
        db: Session,
    ) -> PermutationImportanceResponse | AblationStudyResponse:
        return self.ablation_runner.run(request, db)
