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

    # Draw-specific metrics
    draw_recall: float = 0.0
    draw_precision: float = 0.0
    draw_f1: float = 0.0
    draw_ece: float = 0.0
    draw_predicted_share: float = 0.0

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

    # Closing Line Value (CLV) metrics
    mean_clv: float = 0.0
    clv_hit_rate: float = 0.0
    clv_weighted_roi: float = 0.0


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

    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
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

    # Draw-specific metrics
    draw_recall = float(recalls[1])
    draw_precision = float(precisions[1])
    draw_f1 = float(f1s[1])
    draw_predicted_share = float((y_pred == 1).mean())

    # Draw ECE
    draw_probas = y_proba[:, 1]
    draw_true = (y_true == 1).astype(int)
    draw_ece = 0.0
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (draw_probas > bin_boundaries[i]) & (
            draw_probas <= bin_boundaries[i + 1]
        )
        if in_bin.sum() > 0:
            confidence = draw_probas[in_bin].mean()
            accuracy_in_bin = draw_true[in_bin].mean()
            draw_ece += abs(confidence - accuracy_in_bin) * in_bin.mean()

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
        draw_recall=draw_recall,
        draw_precision=draw_precision,
        draw_f1=draw_f1,
        draw_ece=draw_ece,
        draw_predicted_share=draw_predicted_share,
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
    closing_odds: NDArray[np.float64] | None = None,
    use_model_clv: bool = False,
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
        closing_odds: Optional closing odds matrix (n_samples, 3) for CLV.
            When provided (and use_model_clv is False), classical
            dual-snapshot CLV = taken/closing - 1 is computed.
        use_model_clv: When True, compute model-CLV instead of classical
            CLV. In this mode `odds` is treated as the closing market
            price, the model's own probabilities (re-scaled to match
            the closing overround) act as the synthetic "taken" odds,
            and CLV becomes closing_odds * model_prob * overround - 1.
            Positive across many bets indicates the closing market
            under-priced the outcomes the model picked. Use this when
            only one odds snapshot per match exists.

    Returns:
        BettingMetrics with simulated betting results including CLV
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

        # Place at most one bet per match: the outcome with the single
        # highest positive edge.  Betting on multiple outcomes per match
        # only makes sense when the model is perfectly calibrated; with
        # any calibration error it triggers false value across every
        # outcome and destroys ROI through over-betting.
        edges = []
        for cls in range(3):
            implied_prob = 1.0 / match_odds[cls]
            edge = probas[cls] - implied_prob
            edges.append((edge, cls))

        best_edge, best_cls = max(edges, key=lambda x: x[0])

        if best_edge > min_edge:
            # Calculate Kelly stake
            b = match_odds[best_cls] - 1
            p = probas[best_cls]
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0
            kelly = max(0, kelly) * kelly_fraction
            kelly_fractions.append(kelly)

            bet_stake = stake * kelly if kelly > 0 else stake
            total_stake += bet_stake

            if best_cls == true_outcome:
                winnings = bet_stake * match_odds[best_cls]
                total_return += winnings
                winning_bets += 1
                winning_odds.append(match_odds[best_cls])
                equity.append(equity[-1] + (winnings - bet_stake))
            else:
                losing_bets += 1
                losing_odds.append(match_odds[best_cls])
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

    # Closing Line Value (CLV) metrics
    # Classical CLV needs two snapshots (taken vs closing). With only one
    # snapshot, callers can opt into model-CLV: treat `odds` as closing and
    # use the model's overround-adjusted implied price as synthetic "taken".
    closing = closing_odds if closing_odds is not None else odds
    clv_values = []
    clv_weighted_returns = []
    clv_total_stake = 0.0

    for i in range(n_samples):
        probas = y_proba[i]
        match_odds = odds[i]
        close = closing[i]

        # Determine the best-edge bet (same logic as above)
        edges = []
        for cls in range(3):
            implied_prob = 1.0 / match_odds[cls]
            edge = probas[cls] - implied_prob
            edges.append((edge, cls))

        best_edge, best_cls = max(edges, key=lambda x: x[0])

        if best_edge > min_edge:
            closing_price = close[best_cls]

            if use_model_clv:
                # Re-scale model probs onto the closing overround so the
                # synthetic taken price lives in the same vig-loaded scale
                # as the closing market; then CLV = closing * adj_prob - 1.
                # Positive => closing market under-priced the model's pick.
                closing_implieds = 1.0 / close
                closing_overround = float(closing_implieds.sum())
                model_prob_adj = probas[best_cls] * closing_overround
                if closing_price > 0 and model_prob_adj > 0:
                    clv = closing_price * model_prob_adj - 1.0
                else:
                    clv = 0.0
            else:
                # Classical CLV: compare the price taken (opening `odds`)
                # against the closing price. Positive means the taken
                # odds were longer than where the market settled.
                taken_odds = match_odds[best_cls]
                if taken_odds > 0 and closing_price > 0:
                    clv = taken_odds / closing_price - 1.0
                else:
                    clv = 0.0
            clv_values.append(clv)

            # Weight CLV by actual bet result
            b = match_odds[best_cls] - 1
            p = probas[best_cls]
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0
            kelly = max(0, kelly) * kelly_fraction
            bet_stake_clv = stake * kelly if kelly > 0 else stake
            clv_total_stake += bet_stake_clv

            clv_weighted_returns.append(clv * bet_stake_clv)

    mean_clv = float(np.mean(clv_values)) if clv_values else 0.0
    clv_hit_rate = (
        float(np.mean([1 if v > 0 else 0 for v in clv_values])) if clv_values else 0.0
    )
    clv_weighted_roi = (
        (sum(clv_weighted_returns) / clv_total_stake * 100)
        if clv_total_stake > 0
        else 0.0
    )

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
        mean_clv=mean_clv,
        clv_hit_rate=clv_hit_rate,
        clv_weighted_roi=clv_weighted_roi,
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
    min_edge: float = 0.0,
    closing_odds: NDArray[np.float64] | None = None,
    use_model_clv: bool = False,
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
        betting_metrics = calculate_betting_metrics(
            y_true,
            y_proba,
            odds,
            min_edge=min_edge,
            closing_odds=closing_odds,
            use_model_clv=use_model_clv,
        )

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
    comparison: dict[str, list[Any]] = {
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
