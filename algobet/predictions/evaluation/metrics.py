"""Evaluation metrics for match prediction models.

Provides comprehensive metrics for evaluating prediction quality,
including classification metrics, betting-specific metrics, and
calibration measures.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    """Standard classification metrics for match prediction."""

    # Overall metrics
    accuracy: float
    log_loss: float
    brier_score: float

    # Macro-averaged metrics (treat all classes equally)
    precision_macro: float
    recall_macro: float
    f1_macro: float

    # Weighted metrics (weighted by class frequency)
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float

    # Per-class metrics
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]

    # Confusion matrix
    confusion_matrix: list[list[int]]

    # Additional metrics
    top_2_accuracy: float = 0.0
    cohen_kappa: float = 0.0


@dataclass
class BettingMetrics:
    """Betting-specific metrics for prediction evaluation."""

    # Basic betting stats
    total_bets: int
    winning_bets: int
    losing_bets: int

    # Profit/Loss
    total_stake: float
    total_return: float
    profit_loss: float

    # ROI metrics
    roi_percent: float
    yield_percent: float

    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # Average odds
    average_winning_odds: float
    average_losing_odds: float

    # Kelly criterion
    average_kelly_fraction: float
    optimal_kelly_fraction: float


@dataclass
class EvaluationResult:
    """Complete evaluation result for a prediction model."""

    # Model info
    model_version: str
    evaluated_at: str

    # Data info
    num_samples: int
    date_range: tuple[str, str] | None

    # Metrics
    classification: ClassificationMetrics
    betting: BettingMetrics | None = None

    # Calibration
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0

    # Per-outcome breakdown
    outcome_accuracy: dict[str, float] = field(default_factory=dict)


# Outcome labels
OUTCOME_LABELS = {0: "H", 1: "D", 2: "A"}
OUTCOME_NAMES = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def calculate_classification_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    y_proba: NDArray[np.float64],
) -> ClassificationMetrics:
    """Calculate all classification metrics.

    Args:
        y_true: True labels (0=H, 1=D, 2=A)
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        ClassificationMetrics with all calculated metrics
    """
    n_classes = 3

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    log_loss_val = log_loss(y_true, y_proba)

    # Brier score (average over all classes)
    brier = 0.0
    for cls in range(n_classes):
        y_binary = (y_true == cls).astype(float)
        brier += np.mean((y_proba[:, cls] - y_binary) ** 2)
    brier /= n_classes

    # Precision, Recall, F1
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    for cls in range(n_classes):
        label = OUTCOME_LABELS[cls]
        per_class_precision[label] = float(precisions[cls])
        per_class_recall[label] = float(recalls[cls])
        per_class_f1[label] = float(f1s[cls])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()

    # Top-2 accuracy (is correct answer in top 2 predictions?)
    top_2_correct = 0
    for i in range(len(y_true)):
        top_2 = np.argsort(y_proba[i])[-2:]
        if y_true[i] in top_2:
            top_2_correct += 1
    top_2_accuracy = top_2_correct / len(y_true)

    # Cohen's Kappa
    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(y_true, y_pred)

    return ClassificationMetrics(
        accuracy=accuracy,
        log_loss=log_loss_val,
        brier_score=brier,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=cm_list,
        top_2_accuracy=top_2_accuracy,
        cohen_kappa=kappa,
    )


def calculate_betting_metrics(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    odds: NDArray[np.float64],
    stake: float = 1.0,
    min_edge: float = 0.0,
    kelly_fraction: float = 0.25,
) -> BettingMetrics:
    """Calculate betting-specific metrics.

    Simulates betting using the predictions and calculates ROI,
    yield, and other betting metrics.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        odds: Odds matrix (n_samples, 3) for [H, D, A]
        stake: Fixed stake per bet
        min_edge: Minimum edge threshold to place bet
        kelly_fraction: Fraction of Kelly criterion to use

    Returns:
        BettingMetrics with simulated betting results
    """
    n_samples = len(y_true)

    # Track betting results
    total_stake = 0.0
    total_return = 0.0
    winning_bets = 0
    losing_bets = 0
    winning_odds = []
    losing_odds = []
    kelly_fractions = []

    # Track equity curve for drawdown
    equity = [0.0]

    for i in range(n_samples):
        true_outcome = y_true[i]
        probas = y_proba[i]
        match_odds = odds[i]

        # Find value bets (where predicted prob > implied prob)
        for cls in range(3):
            implied_prob = 1.0 / match_odds[cls]
            edge = probas[cls] - implied_prob

            if edge > min_edge:
                # Calculate Kelly stake
                b = match_odds[cls] - 1
                p = probas[cls]
                q = 1 - p
                kelly = (b * p - q) / b if b > 0 else 0
                kelly = max(0, kelly) * kelly_fraction
                kelly_fractions.append(kelly)

                bet_stake = stake * kelly if kelly > 0 else stake
                total_stake += bet_stake

                if cls == true_outcome:
                    # Won
                    winnings = bet_stake * match_odds[cls]
                    total_return += winnings
                    winning_bets += 1
                    winning_odds.append(match_odds[cls])
                    equity.append(equity[-1] + (winnings - bet_stake))
                else:
                    # Lost
                    losing_bets += 1
                    losing_odds.append(match_odds[cls])
                    equity.append(equity[-1] - bet_stake)

    # Calculate metrics
    total_bets = winning_bets + losing_bets
    profit_loss = total_return - total_stake

    roi = (profit_loss / total_stake * 100) if total_stake > 0 else 0.0
    yield_pct = (profit_loss / total_stake * 100) if total_stake > 0 else 0.0

    # Win rate
    win_rate = winning_bets / total_bets if total_bets > 0 else 0.0

    # Sharpe ratio (simplified)
    if len(equity) > 1:
        returns = np.diff(equity)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10))
    else:
        sharpe = 0.0

    # Maximum drawdown
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / (peak + 1e-10) if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Average odds
    avg_winning_odds = np.mean(winning_odds) if winning_odds else 0.0
    avg_losing_odds = np.mean(losing_odds) if losing_odds else 0.0

    # Optimal Kelly fraction (maximize growth)
    avg_kelly = np.mean(kelly_fractions) if kelly_fractions else 0.0
    optimal_kelly = avg_kelly * 4 if avg_kelly > 0 else 0.25  # Rough estimate

    return BettingMetrics(
        total_bets=total_bets,
        winning_bets=winning_bets,
        losing_bets=losing_bets,
        total_stake=total_stake,
        total_return=total_return,
        profit_loss=profit_loss,
        roi_percent=roi,
        yield_percent=yield_pct,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        average_winning_odds=avg_winning_odds,
        average_losing_odds=avg_losing_odds,
        average_kelly_fraction=avg_kelly,
        optimal_kelly_fraction=optimal_kelly,
    )


def calculate_outcome_accuracy(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> dict[str, float]:
    """Calculate accuracy for each outcome type.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary mapping outcome to accuracy
    """
    accuracy = {}

    for cls in range(3):
        label = OUTCOME_LABELS[cls]
        mask = y_pred == cls
        if mask.sum() > 0:
            correct = (y_true[mask] == cls).sum()
            accuracy[label] = correct / mask.sum()
        else:
            accuracy[label] = 0.0

    return accuracy


def evaluate_predictions(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    odds: NDArray[np.float64] | None = None,
    model_version: str = "unknown",
    date_range: tuple[str, str] | None = None,
) -> EvaluationResult:
    """Complete evaluation of predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        odds: Optional odds for betting simulation
        model_version: Model version identifier
        date_range: Optional (start, end) date range

    Returns:
        EvaluationResult with all metrics
    """
    from datetime import datetime

    # Classification metrics
    class_metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

    # Betting metrics (if odds provided)
    betting_metrics = None
    if odds is not None:
        betting_metrics = calculate_betting_metrics(y_true, y_proba, odds)

    # Calibration metrics
    from algobet.predictions.training.calibration import calculate_calibration_metrics

    cal_metrics = calculate_calibration_metrics(y_true, y_proba)

    # Per-outcome accuracy
    outcome_acc = calculate_outcome_accuracy(y_true, y_pred)

    return EvaluationResult(
        model_version=model_version,
        evaluated_at=datetime.now().isoformat(),
        num_samples=len(y_true),
        date_range=date_range,
        classification=class_metrics,
        betting=betting_metrics,
        expected_calibration_error=cal_metrics["expected_calibration_error"],
        maximum_calibration_error=cal_metrics["maximum_calibration_error"],
        outcome_accuracy=outcome_acc,
    )


def compare_models(
    results: list[EvaluationResult],
) -> dict[str, Any]:
    """Compare multiple model evaluation results.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Dictionary with comparison data
    """
    comparison = {
        "models": [],
        "accuracy": [],
        "log_loss": [],
        "f1_macro": [],
        "roi_percent": [],
    }

    for result in results:
        comparison["models"].append(result.model_version)
        comparison["accuracy"].append(result.classification.accuracy)
        comparison["log_loss"].append(result.classification.log_loss)
        comparison["f1_macro"].append(result.classification.f1_macro)
        comparison["roi_percent"].append(
            result.betting.roi_percent if result.betting else 0.0
        )

    # Find best model for each metric
    comparison["best_accuracy"] = comparison["models"][
        np.argmax(comparison["accuracy"])
    ]
    comparison["best_log_loss"] = comparison["models"][
        np.argmin(comparison["log_loss"])
    ]
    comparison["best_f1"] = comparison["models"][np.argmax(comparison["f1_macro"])]
    comparison["best_roi"] = comparison["models"][np.argmax(comparison["roi_percent"])]

    return comparison
