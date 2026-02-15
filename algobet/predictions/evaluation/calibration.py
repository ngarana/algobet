"""Calibration analysis for match prediction models.

Provides detailed calibration analysis including reliability diagrams,
calibration curves, and statistical tests for probability calibration.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class CalibrationBin:
    """A single bin in a calibration curve."""

    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_actual: float
    count: int
    accuracy: float


@dataclass
class CalibrationCurveResult:
    """Result of calibration curve analysis."""

    outcome: str
    bins: list[CalibrationBin]
    expected_calibration_error: float
    maximum_calibration_error: float
    root_mean_squared_calibration_error: float
    correlation: float  # Correlation between predicted and actual


@dataclass
class CalibrationAnalysisResult:
    """Complete calibration analysis for all outcomes."""

    home_win: CalibrationCurveResult
    draw: CalibrationCurveResult
    away_win: CalibrationCurveResult
    overall_ece: float
    reliability_score: float  # 0-1, higher is better
    calibration_statistics: dict[str, Any] = field(default_factory=dict)


# Outcome labels
OUTCOME_LABELS = {0: "H", 1: "D", 2: "A"}
OUTCOME_NAMES = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def compute_calibration_curve(
    y_true_binary: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> list[CalibrationBin]:
    """Compute calibration curve bins.

    Args:
        y_true_binary: Binary labels (0 or 1)
        y_proba: Predicted probabilities for this class
        n_bins: Number of bins
        strategy: 'uniform' for equal width, 'quantile' for equal count

    Returns:
        List of CalibrationBin objects
    """
    bins = []

    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_edges = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Handle edge case for first bin
        if i == 0:
            mask = (y_proba >= lower) & (y_proba <= upper)
        else:
            mask = (y_proba > lower) & (y_proba <= upper)

        count = mask.sum()

        if count > 0:
            mean_predicted = y_proba[mask].mean()
            mean_actual = y_true_binary[mask].mean()
            accuracy = mean_actual  # Same as mean_actual for binary
        else:
            mean_predicted = (lower + upper) / 2
            mean_actual = 0.0
            accuracy = 0.0

        bins.append(
            CalibrationBin(
                bin_lower=float(lower),
                bin_upper=float(upper),
                mean_predicted=float(mean_predicted),
                mean_actual=float(mean_actual),
                count=int(count),
                accuracy=float(accuracy),
            )
        )

    return bins


def compute_calibration_errors(
    bins: list[CalibrationBin],
) -> tuple[float, float, float]:
    """Compute calibration error metrics from bins.

    Args:
        bins: List of CalibrationBin objects

    Returns:
        Tuple of (ECE, MCE, RMSE)
    """
    total_count = sum(b.count for b in bins)
    if total_count == 0:
        return 0.0, 0.0, 0.0

    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    sse = 0.0  # Sum of squared errors

    for b in bins:
        if b.count > 0:
            error = abs(b.mean_predicted - b.mean_actual)
            ece += error * (b.count / total_count)
            mce = max(mce, error)
            sse += (b.mean_predicted - b.mean_actual) ** 2 * b.count

    rmse = np.sqrt(sse / total_count)

    return ece, mce, rmse


def analyze_calibration(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    n_bins: int = 10,
) -> CalibrationAnalysisResult:
    """Perform complete calibration analysis.

    Args:
        y_true: True labels (0=H, 1=D, 2=A)
        y_proba: Predicted probabilities (n_samples, 3)
        n_bins: Number of bins for calibration curves

    Returns:
        CalibrationAnalysisResult with detailed analysis
    """
    results = {}

    for cls in range(3):
        label = OUTCOME_LABELS[cls]
        y_binary = (y_true == cls).astype(int)
        probas = y_proba[:, cls]

        # Compute calibration curve
        bins = compute_calibration_curve(y_binary, probas, n_bins)

        # Compute errors
        ece, mce, rmse = compute_calibration_errors(bins)

        # Compute correlation
        # Bin the predictions and compute correlation
        bin_means_pred = [b.mean_predicted for b in bins if b.count > 0]
        bin_means_actual = [b.mean_actual for b in bins if b.count > 0]

        if len(bin_means_pred) > 1:
            correlation = float(np.corrcoef(bin_means_pred, bin_means_actual)[0, 1])
        else:
            correlation = 1.0

        results[label] = CalibrationCurveResult(
            outcome=OUTCOME_NAMES[label],
            bins=bins,
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            root_mean_squared_calibration_error=rmse,
            correlation=correlation,
        )

    # Overall metrics
    overall_ece = np.mean([r.expected_calibration_error for r in results.values()])

    # Reliability score (0-1, higher is better)
    reliability = 1.0 - min(overall_ece, 1.0)

    # Statistical tests
    statistics = compute_calibration_statistics(y_true, y_proba)

    return CalibrationAnalysisResult(
        home_win=results["H"],
        draw=results["D"],
        away_win=results["A"],
        overall_ece=overall_ece,
        reliability_score=reliability,
        calibration_statistics=statistics,
    )


def compute_calibration_statistics(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
) -> dict[str, Any]:
    """Compute statistical tests for calibration.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary with statistical test results
    """
    statistics = {}

    for cls in range(3):
        label = OUTCOME_LABELS[cls]
        y_binary = (y_true == cls).astype(float)
        probas = y_proba[:, cls]

        # Hosmer-Lemeshow test (simplified)
        # Groups predictions into deciles and tests goodness of fit
        n_bins = 10
        bin_edges = np.percentile(probas, np.linspace(0, 100, n_bins + 1))

        observed = []
        expected = []

        for i in range(n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == 0:
                mask = (probas >= lower) & (probas <= upper)
            else:
                mask = (probas > lower) & (probas <= upper)

            count = mask.sum()
            if count > 0:
                observed.append(y_binary[mask].sum())
                expected.append(probas[mask].sum())

        if len(observed) > 2:
            observed = np.array(observed)
            expected = np.array(expected)
            # Chi-square goodness of fit
            chi2_stat = np.sum((observed - expected) ** 2 / (expected + 1e-10))
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=len(observed) - 1)
        else:
            chi2_stat = 0.0
            p_value = 1.0

        statistics[f"{label}_hosmer_lemeshow_chi2"] = float(chi2_stat)
        statistics[f"{label}_hosmer_lemeshow_pvalue"] = float(p_value)

        # Spiegelhalter's test statistic
        # Tests if predictions are well calibrated
        y_pred_max = probas
        y_true_float = y_binary

        # Brier score decomposition
        brier = np.mean((y_pred_max - y_true_float) ** 2)
        statistics[f"{label}_brier_score"] = float(brier)

        # Calibration-in-the-large (intercept)
        # Perfect calibration: intercept = 0
        mean_pred = probas.mean()
        mean_actual = y_binary.mean()
        statistics[f"{label}_calibration_intercept"] = float(mean_actual - mean_pred)

    return statistics


def reliability_diagram_data(
    analysis: CalibrationAnalysisResult,
) -> dict[str, Any]:
    """Generate data for plotting reliability diagrams.

    Args:
        analysis: CalibrationAnalysisResult from analyze_calibration

    Returns:
        Dictionary with plotting data
    """
    data = {
        "outcomes": [],
        "perfect_calibration": list(zip([0, 1], [0, 1], strict=False)),
    }

    for label, curve in [
        ("Home Win", analysis.home_win),
        ("Draw", analysis.draw),
        ("Away Win", analysis.away_win),
    ]:
        predicted = [b.mean_predicted for b in curve.bins if b.count > 0]
        actual = [b.mean_actual for b in curve.bins if b.count > 0]
        counts = [b.count for b in curve.bins if b.count > 0]

        data["outcomes"].append(
            {
                "name": label,
                "predicted": predicted,
                "actual": actual,
                "counts": counts,
                "ece": curve.expected_calibration_error,
                "mce": curve.maximum_calibration_error,
            }
        )

    return data


def calibration_summary(analysis: CalibrationAnalysisResult) -> dict[str, str]:
    """Generate human-readable calibration summary.

    Args:
        analysis: CalibrationAnalysisResult

    Returns:
        Dictionary with summary messages
    """
    summary = {}

    # Overall assessment
    if analysis.overall_ece < 0.05:
        summary["overall"] = "Excellent calibration (ECE < 0.05)"
    elif analysis.overall_ece < 0.10:
        summary["overall"] = "Good calibration (ECE < 0.10)"
    elif analysis.overall_ece < 0.15:
        summary["overall"] = "Acceptable calibration (ECE < 0.15)"
    else:
        summary["overall"] = "Poor calibration (ECE >= 0.15) - consider recalibration"

    # Per-outcome
    for label, curve in [
        ("home_win", analysis.home_win),
        ("draw", analysis.draw),
        ("away_win", analysis.away_win),
    ]:
        ece = curve.expected_calibration_error
        if ece < 0.05:
            status = "Excellent"
        elif ece < 0.10:
            status = "Good"
        elif ece < 0.15:
            status = "Acceptable"
        else:
            status = "Poor"

        summary[label] = f"{status} (ECE = {ece:.4f})"

    return summary


def detect_calibration_issues(
    analysis: CalibrationAnalysisResult,
) -> list[dict[str, Any]]:
    """Detect specific calibration issues.

    Args:
        analysis: CalibrationAnalysisResult

    Returns:
        List of detected issues with recommendations
    """
    issues = []

    # Check for overconfidence
    for label, curve in [
        ("Home Win", analysis.home_win),
        ("Draw", analysis.draw),
        ("Away Win", analysis.away_win),
    ]:
        # Check if high predictions are too confident
        high_pred_bins = [
            b for b in curve.bins if b.mean_predicted > 0.7 and b.count > 0
        ]
        if high_pred_bins:
            avg_gap = np.mean(
                [b.mean_predicted - b.mean_actual for b in high_pred_bins]
            )
            if avg_gap > 0.1:
                issues.append(
                    {
                        "type": "overconfidence",
                        "outcome": label,
                        "description": (
                            f"Model is overconfident for {label} at high probabilities"
                        ),
                        "avg_gap": avg_gap,
                        "recommendation": "Consider isotonic regression calibration",
                    }
                )

        # Check for underconfidence
        low_pred_bins = [
            b for b in curve.bins if b.mean_predicted < 0.3 and b.count > 0
        ]
        if low_pred_bins:
            avg_gap = np.mean([b.mean_actual - b.mean_predicted for b in low_pred_bins])
            if avg_gap > 0.1:
                issues.append(
                    {
                        "type": "underconfidence",
                        "outcome": label,
                        "description": (
                            f"Model is underconfident for {label} at low probabilities"
                        ),
                        "avg_gap": avg_gap,
                        "recommendation": "Consider sigmoid calibration",
                    }
                )

    # Check for uncalibrated draw predictions (common issue)
    draw_ece = analysis.draw.expected_calibration_error
    if draw_ece > 0.15:
        issues.append(
            {
                "type": "draw_calibration",
                "outcome": "Draw",
                "description": (
                    "Draw predictions are poorly calibrated (common in football)"
                ),
                "draw_ece": draw_ece,
                "recommendation": "Consider class weighting or specialized draw model",
            }
        )

    return issues
