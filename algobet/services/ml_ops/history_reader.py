"""Backtest history read use cases for ML operations."""

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

from algobet.api.schemas.ml_operations import (
    BacktestHistoryItem,
    BacktestHistoryListResponse,
    BacktestResultResponse,
    BettingMetricsResponse,
    ClassificationMetricsResponse,
)
from algobet.models import BacktestHistory


class BacktestHistoryReader:
    """Run this ML operation use case."""

    def get_backtest_history(
        self,
        model_version_id: int | None,
        limit: int,
        offset: int,
        db: Session,
    ) -> BacktestHistoryListResponse:
        """Get backtest history for analysis and comparison.

        Returns a paginated list of historical backtest results, optionally
        filtered by model version.

        Args:
            model_version_id: Optional filter by model version
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination
            db: Database session

        Returns:
            BacktestHistoryListResponse with list of backtest results
        """
        query = db.query(BacktestHistory).options(
            joinedload(BacktestHistory.model_version)
        )

        if model_version_id:
            query = query.filter(BacktestHistory.model_version_id == model_version_id)

        total = query.count()

        history = (
            query.order_by(BacktestHistory.evaluated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = []
        for h in history:
            items.append(
                BacktestHistoryItem(
                    id=h.id,
                    model_version_id=h.model_version_id,
                    model_name=h.model_version.name if h.model_version else None,
                    model_version=h.model_version.version if h.model_version else None,
                    num_samples=h.num_samples,
                    date_range_start=h.date_range_start,
                    date_range_end=h.date_range_end,
                    accuracy=h.accuracy,
                    log_loss=h.log_loss,
                    f1_macro=h.f1_macro,
                    roi_percent=h.roi_percent,
                    win_rate=h.win_rate,
                    evaluated_at=h.evaluated_at.isoformat(),
                )
            )

        return BacktestHistoryListResponse(items=items, total=total)

    def get_backtest_detail(
        self, backtest_id: int, db: Session
    ) -> BacktestResultResponse:
        """Get detailed backtest result by ID.

        Returns the complete backtest metrics for a specific backtest run.

        Args:
            backtest_id: Backtest history record ID
            db: Database session

        Returns:
            BacktestResultResponse with full metrics

        Raises:
            HTTPException: If backtest not found
        """
        history = (
            db.query(BacktestHistory)
            .options(joinedload(BacktestHistory.model_version))
            .filter(BacktestHistory.id == backtest_id)
            .first()
        )

        if not history:
            raise HTTPException(status_code=404, detail="Backtest not found")

        full_metrics = history.full_metrics or {}
        classification = full_metrics.get("classification", {})
        betting = full_metrics.get("betting")

        return BacktestResultResponse(
            model_version=history.model_version.version
            if history.model_version
            else "unknown",
            evaluated_at=history.evaluated_at.isoformat(),
            num_samples=history.num_samples,
            date_range=(history.date_range_start, history.date_range_end or "")
            if history.date_range_start
            else None,
            classification=ClassificationMetricsResponse(
                accuracy=history.accuracy,
                log_loss=history.log_loss,
                brier_score=history.brier_score,
                precision_macro=history.precision_macro,
                recall_macro=history.recall_macro,
                f1_macro=history.f1_macro,
                precision_weighted=history.f1_weighted,
                recall_weighted=history.recall_macro,
                f1_weighted=history.f1_weighted,
                per_class_precision=classification.get(
                    "per_class_precision", {"H": 0.0, "D": 0.0, "A": 0.0}
                ),
                per_class_recall=classification.get(
                    "per_class_recall", {"H": 0.0, "D": 0.0, "A": 0.0}
                ),
                per_class_f1={
                    "H": history.f1_home,
                    "D": history.f1_draw,
                    "A": history.f1_away,
                },
                confusion_matrix=classification.get(
                    "confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
                top_2_accuracy=history.top_2_accuracy,
                cohen_kappa=history.cohen_kappa,
            ),
            betting=(
                BettingMetricsResponse(
                    total_bets=betting.get("total_bets", 0),
                    winning_bets=int(
                        betting.get("total_bets", 0) * betting.get("win_rate", 0)
                    ),
                    losing_bets=int(
                        betting.get("total_bets", 0) * (1 - betting.get("win_rate", 0))
                    ),
                    total_stake=(
                        history.profit_loss / (history.roi_percent / 100)
                        if history.profit_loss
                        and history.roi_percent
                        and history.roi_percent != 0
                        else 0
                    ),
                    total_return=(history.profit_loss or 0)
                    + (
                        history.profit_loss / (history.roi_percent / 100)
                        if history.profit_loss
                        and history.roi_percent
                        and history.roi_percent != 0
                        else 0
                    ),
                    profit_loss=history.profit_loss or 0,
                    roi_percent=history.roi_percent or 0,
                    yield_percent=history.roi_percent or 0,
                    sharpe_ratio=history.sharpe_ratio or 0,
                    max_drawdown=history.max_drawdown or 0,
                    win_rate=history.win_rate or 0,
                    average_winning_odds=0,
                    average_losing_odds=0,
                    average_kelly_fraction=0,
                    optimal_kelly_fraction=0.25,
                )
                if betting
                else None
            ),
            expected_calibration_error=history.expected_calibration_error,
            maximum_calibration_error=history.maximum_calibration_error,
            outcome_accuracy=full_metrics.get(
                "outcome_accuracy", {"H": 0.0, "D": 0.0, "A": 0.0}
            ),
        )
