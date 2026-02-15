"""Report generation for prediction model evaluation.

Provides comprehensive report generation in HTML and Markdown formats,
including visualizations and detailed analysis.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from algobet.predictions.evaluation.calibration import (
    CalibrationAnalysisResult,
    analyze_calibration,
    calibration_summary,
    detect_calibration_issues,
    reliability_diagram_data,
)
from algobet.predictions.evaluation.metrics import (
    BettingMetrics,
    ClassificationMetrics,
    EvaluationResult,
    OUTCOME_NAMES,
    calculate_classification_metrics,
    compare_models,
    evaluate_predictions,
)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str = "Model Evaluation Report"
    include_calibration: bool = True
    include_betting: bool = True
    include_feature_importance: bool = True
    include_recommendations: bool = True
    output_dir: Path = Path("data/reports")


class ReportGenerator:
    """Generate evaluation reports in multiple formats."""

    def __init__(self, config: ReportConfig | None = None) -> None:
        """Initialize report generator.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()

    def generate_markdown(
        self,
        result: EvaluationResult,
        calibration: CalibrationAnalysisResult | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> str:
        """Generate Markdown report.

        Args:
            result: Evaluation result
            calibration: Optional calibration analysis
            feature_importance: Optional feature importance dict

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append(f"# {self.config.title}")
        lines.append("")
        lines.append(f"**Model Version:** {result.model_version}")
        lines.append(f"**Evaluated At:** {result.evaluated_at}")
        lines.append(f"**Samples:** {result.num_samples:,}")

        if result.date_range:
            lines.append(f"**Date Range:** {result.date_range[0]} to {result.date_range[1]}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Classification Metrics
        lines.append("## Classification Metrics")
        lines.append("")
        lines.append(self._classification_table(result.classification))
        lines.append("")

        # Per-Class Metrics
        lines.append("### Per-Class Metrics")
        lines.append("")
        lines.append(self._per_class_table(result.classification))
        lines.append("")

        # Confusion Matrix
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append(self._confusion_matrix_table(result.classification.confusion_matrix))
        lines.append("")

        # Betting Metrics
        if self.config.include_betting and result.betting:
            lines.append("## Betting Simulation")
            lines.append("")
            lines.append(self._betting_table(result.betting))
            lines.append("")

        # Calibration Analysis
        if self.config.include_calibration and calibration:
            lines.append("## Calibration Analysis")
            lines.append("")
            summary = calibration_summary(calibration)
            lines.append(f"**Overall:** {summary['overall']}")
            lines.append("")
            lines.append("| Outcome | Calibration Quality |")
            lines.append("|---------|---------------------|")
            for key, value in summary.items():
                if key != "overall":
                    outcome_name = key.replace("_", " ").title()
                    lines.append(f"| {outcome_name} | {value} |")
            lines.append("")

            # Calibration issues
            issues = detect_calibration_issues(calibration)
            if issues and self.config.include_recommendations:
                lines.append("### Detected Issues")
                lines.append("")
                for issue in issues:
                    lines.append(f"- **{issue['type'].replace('_', ' ').title()}** ({issue['outcome']}): {issue['description']}")
                    lines.append(f"  - Recommendation: {issue['recommendation']}")
                lines.append("")

        # Feature Importance
        if self.config.include_feature_importance and feature_importance:
            lines.append("## Feature Importance")
            lines.append("")
            lines.append(self._feature_importance_table(feature_importance))
            lines.append("")

        # Recommendations
        if self.config.include_recommendations:
            lines.append("## Recommendations")
            lines.append("")
            lines.extend(self._generate_recommendations(result, calibration))
            lines.append("")

        return "\n".join(lines)

    def generate_html(
        self,
        result: EvaluationResult,
        calibration: CalibrationAnalysisResult | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> str:
        """Generate HTML report.

        Args:
            result: Evaluation result
            calibration: Optional calibration analysis
            feature_importance: Optional feature importance dict

        Returns:
            HTML string
        """
        # Generate markdown first
        md_content = self.generate_markdown(result, calibration, feature_importance)

        # Convert to HTML with styling
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .metric-good {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .metric-warning {{
            color: #FF9800;
            font-weight: bold;
        }}
        .metric-bad {{
            color: #f44336;
            font-weight: bold;
        }}
        .highlight {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
            margin: 15px 0;
        }}
        .warning {{
            background-color: #fff3e0;
            border-left-color: #FF9800;
        }}
        .error {{
            background-color: #ffebee;
            border-left-color: #f44336;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #ddd;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {self._markdown_to_html(md_content)}
    </div>
</body>
</html>"""
        return html

    def save_report(
        self,
        result: EvaluationResult,
        format: str = "markdown",
        calibration: CalibrationAnalysisResult | None = None,
        feature_importance: dict[str, float] | None = None,
        filename: str | None = None,
    ) -> Path:
        """Save report to file.

        Args:
            result: Evaluation result
            format: 'markdown' or 'html'
            calibration: Optional calibration analysis
            feature_importance: Optional feature importance dict
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{result.model_version}_{timestamp}"

        ext = ".md" if format == "markdown" else ".html"
        filepath = self.config.output_dir / f"{filename}{ext}"

        if format == "markdown":
            content = self.generate_markdown(result, calibration, feature_importance)
        else:
            content = self.generate_html(result, calibration, feature_importance)

        filepath.write_text(content)
        return filepath

    def _classification_table(self, metrics: ClassificationMetrics) -> str:
        """Generate classification metrics table."""
        lines = [
            "| Metric | Value | Target | Status |",
            "|--------|-------|--------|--------|",
        ]

        metrics_data = [
            ("Accuracy", metrics.accuracy, 0.53, "higher"),
            ("Log Loss", metrics.log_loss, 0.95, "lower"),
            ("Brier Score", metrics.brier_score, 0.20, "lower"),
            ("F1 (Macro)", metrics.f1_macro, 0.45, "higher"),
            ("F1 (Weighted)", metrics.f1_weighted, 0.50, "higher"),
            ("Top-2 Accuracy", metrics.top_2_accuracy, 0.76, "higher"),
            ("Cohen's Kappa", metrics.cohen_kappa, 0.30, "higher"),
        ]

        for name, value, target, direction in metrics_data:
            if direction == "higher":
                status = self._get_status(value >= target)
            else:
                status = self._get_status(value <= target)

            lines.append(f"| {name} | {value:.4f} | {target:.2f} | {status} |")

        return "\n".join(lines)

    def _per_class_table(self, metrics: ClassificationMetrics) -> str:
        """Generate per-class metrics table."""
        lines = [
            "| Outcome | Precision | Recall | F1-Score |",
            "|---------|-----------|--------|----------|",
        ]

        for outcome in ["H", "D", "A"]:
            name = OUTCOME_NAMES[outcome]
            prec = metrics.per_class_precision.get(outcome, 0)
            rec = metrics.per_class_recall.get(outcome, 0)
            f1 = metrics.per_class_f1.get(outcome, 0)
            lines.append(f"| {name} | {prec:.4f} | {rec:.4f} | {f1:.4f} |")

        return "\n".join(lines)

    def _confusion_matrix_table(self, cm: list[list[int]]) -> str:
        """Generate confusion matrix table."""
        lines = [
            "| | Predicted H | Predicted D | Predicted A |",
            "|---|-------------|-------------|-------------|",
        ]

        for i, row_label in enumerate(["Actual H", "Actual D", "Actual A"]):
            row = cm[i]
            lines.append(f"| {row_label} | {row[0]} | {row[1]} | {row[2]} |")

        return "\n".join(lines)

    def _betting_table(self, metrics: BettingMetrics) -> str:
        """Generate betting metrics table."""
        lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Bets | {metrics.total_bets} |",
            f"| Win Rate | {metrics.win_rate:.1%} |",
            f"| ROI | {metrics.roi_percent:.2f}% |",
            f"| Yield | {metrics.yield_percent:.2f}% |",
            f"| Profit/Loss | ${metrics.profit_loss:.2f} |",
            f"| Max Drawdown | {metrics.max_drawdown:.1%} |",
            f"| Sharpe Ratio | {metrics.sharpe_ratio:.3f} |",
            f"| Avg Winning Odds | {metrics.average_winning_odds:.2f} |",
            f"| Avg Losing Odds | {metrics.average_losing_odds:.2f} |",
        ]

        return "\n".join(lines)

    def _feature_importance_table(self, importance: dict[str, float]) -> str:
        """Generate feature importance table."""
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

        lines = [
            "| Rank | Feature | Importance |",
            "|------|---------|------------|",
        ]

        for rank, (feature, imp) in enumerate(sorted_features, 1):
            lines.append(f"| {rank} | {feature} | {imp:.4f} |")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        result: EvaluationResult,
        calibration: CalibrationAnalysisResult | None,
    ) -> list[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Classification recommendations
        if result.classification.accuracy < 0.50:
            recommendations.append(
                "- Consider adding more features or using ensemble methods to improve accuracy"
            )

        if result.classification.f1_macro < 0.40:
            recommendations.append(
                "- Per-class F1 is low. Consider class weighting or resampling techniques"
            )

        # Draw prediction is typically hardest
        if result.classification.per_class_f1.get("D", 0) < 0.25:
            recommendations.append(
                "- Draw prediction is weak. Consider specialized draw detection features"
            )

        # Calibration recommendations
        if calibration and calibration.overall_ece > 0.10:
            recommendations.append(
                "- Model calibration is poor. Apply isotonic or sigmoid calibration"
            )

        # Betting recommendations
        if result.betting:
            if result.betting.roi_percent < 0:
                recommendations.append(
                    "- Negative ROI indicates predictions are not beating the market. "
                    "Focus on value bet detection rather than outright prediction"
                )
            if result.betting.max_drawdown > 0.3:
                recommendations.append(
                    "- High drawdown detected. Consider reducing stake sizes using Kelly criterion"
                )

        if not recommendations:
            recommendations.append("- Model performance is satisfactory. Continue monitoring in production.")

        return recommendations

    def _get_status(self, condition: bool) -> str:
        """Get status indicator."""
        return "✅" if condition else "⚠️"

    def _markdown_to_html(self, md: str) -> str:
        """Convert markdown to HTML (simple conversion)."""
        html = md

        # Headers
        html = html.replace("# ", "<h1>").replace("\n# ", "</h1>\n<h1>")
        html = html.replace("## ", "<h2>").replace("\n## ", "</h2>\n<h2>")
        html = html.replace("### ", "<h3>").replace("\n### ", "</h3>\n<h3>")

        # Close open headers at end of line
        lines = html.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith("<h1>"):
                if not line.endswith("</h1>"):
                    line = line + "</h1>"
            elif line.startswith("<h2>"):
                if not line.endswith("</h2>"):
                    line = line + "</h2>"
            elif line.startswith("<h3>"):
                if not line.endswith("</h3>"):
                    line = line + "</h3>"
            new_lines.append(line)
        html = "\n".join(new_lines)

        # Bold
        import re

        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

        # Horizontal rules
        html = html.replace("---", "<hr>")

        # Tables (simple conversion)
        lines = html.split("\n")
        in_table = False
        table_lines = []
        result_lines = []

        for line in lines:
            if "|" in line and not line.startswith("<"):
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            else:
                if in_table:
                    result_lines.append(self._convert_table(table_lines))
                    in_table = False
                    table_lines = []
                result_lines.append(line)

        if in_table:
            result_lines.append(self._convert_table(table_lines))

        html = "\n".join(result_lines)

        # Lists
        lines = html.split("\n")
        in_list = False
        result_lines = []

        for line in lines:
            if line.strip().startswith("- "):
                if not in_list:
                    result_lines.append("<ul>")
                    in_list = True
                result_lines.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    result_lines.append("</ul>")
                    in_list = False
                result_lines.append(line)

        if in_list:
            result_lines.append("</ul>")

        return "\n".join(result_lines)

    def _convert_table(self, lines: list[str]) -> str:
        """Convert markdown table to HTML."""
        if len(lines) < 2:
            return "\n".join(lines)

        html_lines = ["<table>"]

        for i, line in enumerate(lines):
            cells = [c.strip() for c in line.split("|") if c.strip()]

            if i == 0:
                html_lines.append("<thead><tr>")
                for cell in cells:
                    html_lines.append(f"<th>{cell}</th>")
                html_lines.append("</tr></thead><tbody>")
            elif i == 1:
                # Skip separator row
                continue
            else:
                html_lines.append("<tr>")
                for cell in cells:
                    html_lines.append(f"<td>{cell}</td>")
                html_lines.append("</tr>")

        html_lines.append("</tbody></table>")
        return "\n".join(html_lines)


def generate_evaluation_report(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    model_version: str,
    odds: NDArray[np.float64] | None = None,
    feature_importance: dict[str, float] | None = None,
    date_range: tuple[str, str] | None = None,
    output_dir: Path | str = "data/reports",
    format: str = "html",
) -> Path:
    """Generate complete evaluation report.

    Convenience function that performs full evaluation and generates report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_version: Model version string
        odds: Optional odds for betting simulation
        feature_importance: Optional feature importance
        date_range: Optional date range
        output_dir: Output directory for reports
        format: 'html' or 'markdown'

    Returns:
        Path to generated report
    """
    # Perform evaluation
    result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        odds=odds,
        model_version=model_version,
        date_range=date_range,
    )

    # Perform calibration analysis
    calibration = analyze_calibration(y_true, y_proba)

    # Generate report
    config = ReportConfig(output_dir=Path(output_dir))
    generator = ReportGenerator(config)

    return generator.save_report(
        result=result,
        format=format,
        calibration=calibration,
        feature_importance=feature_importance,
    )
