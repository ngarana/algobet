"""Analysis service facade - coordinates backtest, value bets, and calibration.

This module provides a unified interface for analysis operations by delegating
to specialized services: BacktestService, ValueBetService, and CalibrationService.
"""

from pathlib import Path

from sqlalchemy.orm import Session

from algobet.logging_config import get_logger
from algobet.services.base import BaseService
from algobet.services.dto import (
    BacktestRequest,
    BacktestResponse,
    CalibrateRequest,
    CalibrateResponse,
    ValueBetsRequest,
    ValueBetsResponse,
)


class AnalysisService(BaseService[Session]):
    """Facade service for analysis operations.

    Delegates to specialized services:
    - BacktestService: Historical model evaluation
    - ValueBetService: Betting opportunity detection
    - CalibrationService: Probability calibration
    """

    def __init__(
        self, session: Session, models_path: Path = Path("data/models")
    ) -> None:
        """Initialize the analysis facade.

        Args:
            session: SQLAlchemy database session
            models_path: Path to model storage directory
        """
        super().__init__(session)
        self.logger = get_logger("services.analysis")
        self.models_path = models_path

        from algobet.services.backtest_service import BacktestService
        from algobet.services.calibration_service import CalibrationService
        from algobet.services.value_bet_service import ValueBetService

        self._backtest_service = BacktestService(session, models_path)
        self._value_bet_service = ValueBetService(session, models_path)
        self._calibration_service = CalibrationService(session, models_path)

    def run_backtest(self, request: BacktestRequest) -> BacktestResponse:
        """Run a backtest on historical data."""
        return self._backtest_service.run(request)

    def find_value_bets(self, request: ValueBetsRequest) -> ValueBetsResponse:
        """Find value bets based on model predictions."""
        return self._value_bet_service.run(request)

    def calibrate_model(self, request: CalibrateRequest) -> CalibrateResponse:
        """Calibrate model probabilities."""
        return self._calibration_service.run(request)
