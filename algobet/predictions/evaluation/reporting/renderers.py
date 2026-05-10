"""Renderer collaborators for evaluation reports."""

from typing import Protocol

from algobet.predictions.evaluation.calibration import CalibrationAnalysisResult
from algobet.predictions.evaluation.metrics import EvaluationResult


class ReportRenderingImplementation(Protocol):
    """Private rendering implementation exposed by the report facade."""

    def _generate_markdown_impl(
        self,
        result: EvaluationResult,
        calibration: CalibrationAnalysisResult | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> str: ...

    def _generate_html_impl(
        self,
        result: EvaluationResult,
        calibration: CalibrationAnalysisResult | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> str: ...


class MarkdownReportRenderer:
    """Render markdown reports behind the ReportGenerator facade."""

    def __init__(self, implementation: ReportRenderingImplementation) -> None:
        self.implementation = implementation

    def render(
        self,
        result: EvaluationResult,
        calibration: CalibrationAnalysisResult | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> str:
        return self.implementation._generate_markdown_impl(
            result,
            calibration=calibration,
            feature_importance=feature_importance,
        )


class HtmlReportRenderer:
    """Render HTML reports behind the ReportGenerator facade."""

    def __init__(self, implementation: ReportRenderingImplementation) -> None:
        self.implementation = implementation

    def render(
        self,
        result: EvaluationResult,
        calibration: CalibrationAnalysisResult | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> str:
        return self.implementation._generate_html_impl(
            result,
            calibration=calibration,
            feature_importance=feature_importance,
        )
