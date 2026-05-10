"""Analysis runner compatibility shim - deprecated.

This module provides backwards-compatible wrapper classes that delegate
to the new dedicated services. New code should use the services directly.
"""

from typing import Any

from algobet.services.backtest_service import BacktestService
from algobet.services.calibration_service import CalibrationService
from algobet.services.dto import (
    BacktestRequest,
    BacktestResponse,
    CalibrateRequest,
    CalibrateResponse,
    ValueBetsRequest,
    ValueBetsResponse,
)
from algobet.services.value_bet_service import ValueBetService


class AnalysisBacktestRunner:
    """Deprecated: Use BacktestService directly."""

    def __init__(self, implementation: Any) -> None:
        self._service = BacktestService(
            implementation.session, implementation.models_path
        )

    def run(self, request: BacktestRequest) -> BacktestResponse:
        return self._service.run(request)


class ValueBetFinder:
    """Deprecated: Use ValueBetService directly."""

    def __init__(self, implementation: Any) -> None:
        self._service = ValueBetService(
            implementation.session, implementation.models_path
        )

    def run(self, request: ValueBetsRequest) -> ValueBetsResponse:
        return self._service.run(request)


class ModelCalibrator:
    """Deprecated: Use CalibrationService directly."""

    def __init__(self, implementation: Any) -> None:
        self._service = CalibrationService(
            implementation.session, implementation.models_path
        )

    def run(self, request: CalibrateRequest) -> CalibrateResponse:
        return self._service.run(request)
