"""Model evaluation module for match prediction.

Provides comprehensive evaluation including classification metrics,
betting simulation, calibration analysis, and report generation.
"""

from algobet.predictions.evaluation.calibration import (
    CalibrationAnalysisResult,
    CalibrationBin,
    CalibrationCurveResult,
    analyze_calibration,
    calibration_summary,
    compute_calibration_curve,
    detect_calibration_issues,
    reliability_diagram_data,
)
from algobet.predictions.evaluation.metrics import (
    BettingMetrics,
    ClassificationMetrics,
    EvaluationResult,
    calculate_betting_metrics,
    calculate_classification_metrics,
    calculate_outcome_accuracy,
    compare_models,
    evaluate_predictions,
)
from algobet.predictions.evaluation.reports import (
    ReportConfig,
    ReportGenerator,
    generate_evaluation_report,
)

__all__ = [
    # Metrics
    "BettingMetrics",
    "ClassificationMetrics",
    "EvaluationResult",
    "calculate_betting_metrics",
    "calculate_classification_metrics",
    "calculate_outcome_accuracy",
    "compare_models",
    "evaluate_predictions",
    # Calibration
    "CalibrationAnalysisResult",
    "CalibrationBin",
    "CalibrationCurveResult",
    "analyze_calibration",
    "calibration_summary",
    "compute_calibration_curve",
    "detect_calibration_issues",
    "reliability_diagram_data",
    # Reports
    "ReportConfig",
    "ReportGenerator",
    "generate_evaluation_report",
]
