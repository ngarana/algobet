"""API router for ML operations (backtest, calibrate)."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

from algobet.api.dependencies import get_db
from algobet.models import Match
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.evaluation import evaluate_predictions
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.calibration import (
    ProbabilityCalibrator,
    calculate_calibration_metrics,
)

router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================


class BacktestRequest(BaseModel):
    """Request schema for backtest operation."""

    model_version: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_matches: int = Field(default=100, ge=10, le=10000)
    min_edge: float = Field(default=0.0, ge=0.0, le=1.0)


class ClassificationMetricsResponse(BaseModel):
    """Classification metrics response."""

    accuracy: float
    log_loss: float
    brier_score: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    top_2_accuracy: float
    cohen_kappa: float


class BettingMetricsResponse(BaseModel):
    """Betting metrics response."""

    total_bets: int
    winning_bets: int
    losing_bets: int
    total_stake: float
    total_return: float
    profit_loss: float
    roi_percent: float
    yield_percent: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    average_winning_odds: float
    average_losing_odds: float
    average_kelly_fraction: float
    optimal_kelly_fraction: float


class BacktestResultResponse(BaseModel):
    """Response schema for backtest results."""

    model_version: str
    evaluated_at: str
    num_samples: int
    date_range: tuple[str, str] | None
    classification: ClassificationMetricsResponse
    betting: BettingMetricsResponse | None = None
    expected_calibration_error: float
    maximum_calibration_error: float
    outcome_accuracy: dict[str, float]


class CalibrateRequest(BaseModel):
    """Request schema for calibrate operation."""

    model_version: str | None = None
    method: str = Field(default="isotonic", pattern="^(isotonic|sigmoid)$")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    activate: bool = True


class CalibrationMetricsResponse(BaseModel):
    """Calibration metrics response."""

    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    log_loss: float


class CalibrateResultResponse(BaseModel):
    """Response schema for calibrate results."""

    base_model_version: str
    calibrated_model_version: str
    method: str
    samples_used: int
    before_metrics: CalibrationMetricsResponse
    after_metrics: CalibrationMetricsResponse
    improvement: dict[str, float]
    is_active: bool


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/backtest", response_model=BacktestResultResponse)
def run_backtest(
    request: BacktestRequest,
    db: Session = Depends(get_db),
) -> BacktestResultResponse:
    """Run a historical backtest on model predictions.

    Evaluates model performance on historical match data with
    comprehensive classification and betting metrics.

    Args:
        request: Backtest parameters

    Returns:
        BacktestResultResponse with evaluation metrics

    Raises:
        HTTPException: If model not found or insufficient data
    """
    # Get model
    registry = ModelRegistry(storage_path=Path("data/models"), session=db)

    try:
        if request.model_version:
            model = registry.load_model(request.model_version)
            model_meta = next(
                (
                    m
                    for m in registry.list_models()
                    if m.version == request.model_version
                ),
                None,
            )
        else:
            model, model_meta = registry.get_active_model()
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {e}",
        ) from None

    # Set default date range
    end_date = request.end_date or datetime.now()
    start_date = request.start_date or (end_date - timedelta(days=365))

    # Get historical matches with results
    matches = (
        db.query(Match)
        .options(joinedload(Match.home_team), joinedload(Match.away_team))
        .filter(
            and_(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
                Match.match_date >= start_date,
                Match.match_date <= end_date,
                Match.odds_home.is_not(None),
                Match.odds_draw.is_not(None),
                Match.odds_away.is_not(None),
            )
        )
        .order_by(Match.match_date)
        .all()
    )

    if len(matches) < request.min_matches:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient matches: {len(matches)} < {request.min_matches}",
        )

    # Prepare data
    matches_df = prepare_match_dataframe(matches)

    # Calculate results
    matches_df["result"] = matches_df.apply(
        lambda m: "H"
        if m["home_score"] > m["away_score"]
        else ("A" if m["home_score"] < m["away_score"] else "D"),
        axis=1,
    )

    # Generate features
    repo = MatchRepository(db)
    feature_pipeline = FeaturePipeline.create_default()

    # Use temporal split - first 30% for training features, rest for backtest
    train_size = int(len(matches) * 0.3)
    train_matches = matches_df.iloc[:train_size]
    test_matches = matches_df.iloc[train_size:]

    feature_pipeline.fit(train_matches, repo)
    X_test = feature_pipeline.transform(test_matches, repo)

    # Get odds for betting simulation
    odds = test_matches[["odds_home", "odds_draw", "odds_away"]].values

    # Get predictions
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    # Encode true labels
    result_map = {"H": 0, "D": 1, "A": 2}
    y_true = test_matches["result"].map(result_map).values

    # Evaluate
    date_range = (
        str(test_matches["match_date"].min().date()),
        str(test_matches["match_date"].max().date()),
    )

    result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        odds=odds,
        model_version=model_meta.version if model_meta else "unknown",
        date_range=date_range,
    )

    # Build response
    return BacktestResultResponse(
        model_version=result.model_version,
        evaluated_at=result.evaluated_at,
        num_samples=result.num_samples,
        date_range=result.date_range,
        classification=ClassificationMetricsResponse(
            accuracy=result.classification.accuracy,
            log_loss=result.classification.log_loss,
            brier_score=result.classification.brier_score,
            precision_macro=result.classification.precision_macro,
            recall_macro=result.classification.recall_macro,
            f1_macro=result.classification.f1_macro,
            precision_weighted=result.classification.precision_weighted,
            recall_weighted=result.classification.recall_weighted,
            f1_weighted=result.classification.f1_weighted,
            per_class_precision=result.classification.per_class_precision,
            per_class_recall=result.classification.per_class_recall,
            per_class_f1=result.classification.per_class_f1,
            confusion_matrix=result.classification.confusion_matrix,
            top_2_accuracy=result.classification.top_2_accuracy,
            cohen_kappa=result.classification.cohen_kappa,
        ),
        betting=(
            BettingMetricsResponse(
                total_bets=result.betting.total_bets,
                winning_bets=result.betting.winning_bets,
                losing_bets=result.betting.losing_bets,
                total_stake=result.betting.total_stake,
                total_return=result.betting.total_return,
                profit_loss=result.betting.profit_loss,
                roi_percent=result.betting.roi_percent,
                yield_percent=result.betting.yield_percent,
                sharpe_ratio=result.betting.sharpe_ratio,
                max_drawdown=result.betting.max_drawdown,
                win_rate=result.betting.win_rate,
                average_winning_odds=result.betting.average_winning_odds,
                average_losing_odds=result.betting.average_losing_odds,
                average_kelly_fraction=result.betting.average_kelly_fraction,
                optimal_kelly_fraction=result.betting.optimal_kelly_fraction,
            )
            if result.betting
            else None
        ),
        expected_calibration_error=result.expected_calibration_error,
        maximum_calibration_error=result.maximum_calibration_error,
        outcome_accuracy=result.outcome_accuracy,
    )


@router.post("/calibrate", response_model=CalibrateResultResponse)
def run_calibrate(
    request: CalibrateRequest,
    db: Session = Depends(get_db),
) -> CalibrateResultResponse:
    """Calibrate model probabilities.

    Fits a calibrator to improve probability estimates for
    better value betting accuracy.

    Args:
        request: Calibration parameters

    Returns:
        CalibrateResultResponse with before/after metrics

    Raises:
        HTTPException: If model not found or insufficient data
    """
    from algobet.predictions.training.calibration import CalibratedPredictor

    # Get model
    registry = ModelRegistry(storage_path=Path("data/models"), session=db)

    try:
        if request.model_version:
            model = registry.load_model(request.model_version)
            model_meta = next(
                (
                    m
                    for m in registry.list_models()
                    if m.version == request.model_version
                ),
                None,
            )
            base_version = request.model_version
        else:
            model, model_meta = registry.get_active_model()
            base_version = model_meta.version if model_meta else "unknown"
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {e}",
        ) from None

    # Get historical matches for calibration
    matches = (
        db.query(Match)
        .filter(
            and_(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                Match.away_score.is_not(None),
            )
        )
        .order_by(Match.match_date.desc())
        .limit(2000)
        .all()
    )

    if len(matches) < 100:
        raise HTTPException(
            status_code=400,
            detail="Insufficient historical matches for calibration (< 100)",
        )

    # Prepare data
    repo = MatchRepository(db)
    feature_pipeline = FeaturePipeline.create_default()

    matches_df = prepare_match_dataframe(matches)
    matches_df["result"] = matches_df.apply(
        lambda m: "H"
        if m["home_score"] > m["away_score"]
        else ("A" if m["home_score"] < m["away_score"] else "D"),
        axis=1,
    )

    # Sort by date for temporal split
    matches_df = matches_df.sort_values("match_date")

    # Split into train/val for calibration
    val_size = int(len(matches_df) * request.validation_split)
    train_df = matches_df.iloc[:-val_size]
    val_df = matches_df.iloc[-val_size:]

    # Fit pipeline on training data
    feature_pipeline.fit(train_df, repo)

    # Generate features
    X_val = feature_pipeline.transform(val_df, repo)

    # Encode targets
    result_map = {"H": 0, "D": 1, "A": 2}
    y_val = val_df["result"].map(result_map).values

    # Get raw predictions
    y_proba = model.predict_proba(X_val)

    # Calculate before metrics
    before_metrics = calculate_calibration_metrics(y_val, y_proba)

    # Fit calibrator
    calibrator = ProbabilityCalibrator(method=request.method, n_classes=3)
    calibrator.fit(y_proba, y_val)

    # Calculate after metrics
    calibrated_proba = calibrator.calibrate(y_proba)
    after_metrics = calculate_calibration_metrics(y_val, calibrated_proba)

    # Calculate improvement
    improvement = {
        "ece_improvement": before_metrics["expected_calibration_error"]
        - after_metrics["expected_calibration_error"],
        "brier_improvement": before_metrics["brier_score"]
        - after_metrics["brier_score"],
        "log_loss_improvement": before_metrics["log_loss"] - after_metrics["log_loss"],
    }

    # Save calibrated model
    calibrated_model = CalibratedPredictor(predictor=model, calibrator=calibrator)

    metrics: dict[str, float] = {
        "calibration_ece": float(after_metrics["expected_calibration_error"]),
        "calibration_brier": float(after_metrics["brier_score"]),
    }

    new_version = registry.save_model(
        model=calibrated_model,
        name="match_predictor_calibrated",
        metrics=metrics,
        model_type="calibrated",
        description=f"Calibrated version of {base_version} using {request.method}",
    )

    # Activate if requested
    is_active = False
    if request.activate:
        registry.activate_model(new_version)
        is_active = True

    return CalibrateResultResponse(
        base_model_version=base_version,
        calibrated_model_version=new_version,
        method=request.method,
        samples_used=len(val_df),
        before_metrics=CalibrationMetricsResponse(
            expected_calibration_error=before_metrics["expected_calibration_error"],
            maximum_calibration_error=before_metrics["maximum_calibration_error"],
            brier_score=before_metrics["brier_score"],
            log_loss=before_metrics["log_loss"],
        ),
        after_metrics=CalibrationMetricsResponse(
            expected_calibration_error=after_metrics["expected_calibration_error"],
            maximum_calibration_error=after_metrics["maximum_calibration_error"],
            brier_score=after_metrics["brier_score"],
            log_loss=after_metrics["log_loss"],
        ),
        improvement=improvement,
        is_active=is_active,
    )
