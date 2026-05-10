"use client";

import { BarChart3, TrendingUp } from "lucide-react";

import { CalibrationCurve } from "@/components/backtest/calibration-curve";
import { ConfusionMatrixHeatmap } from "@/components/backtest/confusion-matrix-heatmap";
import { EquityCurveChart } from "@/components/backtest/equity-curve-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { BacktestResult } from "@/lib/types/ml-operations";

function ClassificationMetricsCard({
  metrics,
}: {
  metrics: BacktestResult["classification"];
}) {
  const metricRows = [
    {
      label: "Accuracy",
      value: `${(metrics.accuracy * 100).toFixed(1)}%`,
      target: ">=50%",
    },
    { label: "Log Loss", value: metrics.log_loss.toFixed(3), target: "<=0.95" },
    { label: "Brier Score", value: metrics.brier_score.toFixed(3), target: "<=0.20" },
    { label: "F1 (Macro)", value: metrics.f1_macro.toFixed(3), target: ">=0.45" },
    {
      label: "Top-2 Accuracy",
      value: `${(metrics.top_2_accuracy * 100).toFixed(1)}%`,
      target: ">=75%",
    },
    { label: "Cohen's Kappa", value: metrics.cohen_kappa.toFixed(3), target: ">=0.30" },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <BarChart3 className="h-5 w-5" />
          Classification Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="py-2 text-left">Metric</th>
              <th className="py-2 text-right">Value</th>
              <th className="py-2 text-right text-muted-foreground">Target</th>
            </tr>
          </thead>
          <tbody>
            {metricRows.map((row) => (
              <tr key={row.label} className="border-b">
                <td className="py-2">{row.label}</td>
                <td className="text-right font-mono">{row.value}</td>
                <td className="text-right text-muted-foreground">{row.target}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="mt-4">
          <h4 className="mb-2 font-medium">Per-Class F1 Scores</h4>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="rounded bg-muted p-2">
              <div className="text-lg font-bold">
                {(metrics.per_class_f1.H * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">Home</div>
            </div>
            <div className="rounded bg-muted p-2">
              <div className="text-lg font-bold">
                {(metrics.per_class_f1.D * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">Draw</div>
            </div>
            <div className="rounded bg-muted p-2">
              <div className="text-lg font-bold">
                {(metrics.per_class_f1.A * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">Away</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function BettingMetricsCard({
  metrics,
}: {
  metrics: NonNullable<BacktestResult["betting"]>;
}) {
  const isPositiveROI = metrics.roi_percent >= 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <TrendingUp className="h-5 w-5" />
          Betting Simulation
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4 grid grid-cols-2 gap-4">
          <div className="rounded-lg bg-muted p-3 text-center">
            <div className="text-2xl font-bold">{metrics.total_bets}</div>
            <div className="text-xs text-muted-foreground">Total Bets</div>
          </div>
          <div className="rounded-lg bg-muted p-3 text-center">
            <div className="text-2xl font-bold">
              {(metrics.win_rate * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground">Win Rate</div>
          </div>
        </div>

        <div className="mb-4 rounded-lg bg-muted p-4">
          <div className="text-center">
            <div
              className={`text-3xl font-bold ${
                isPositiveROI ? "text-green-600" : "text-red-600"
              }`}
            >
              {metrics.roi_percent >= 0 ? "+" : ""}
              {metrics.roi_percent.toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">Return on Investment</div>
          </div>
        </div>

        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Profit/Loss</span>
            <span
              className={`font-mono ${
                isPositiveROI ? "text-green-600" : "text-red-600"
              }`}
            >
              ${metrics.profit_loss.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Max Drawdown</span>
            <span className="font-mono">
              {(metrics.max_drawdown * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Sharpe Ratio</span>
            <span className="font-mono">{metrics.sharpe_ratio.toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Avg Winning Odds</span>
            <span className="font-mono">{metrics.average_winning_odds.toFixed(2)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function CalibrationCard({ result }: { result: BacktestResult }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Calibration</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg bg-muted p-3 text-center">
            <div className="text-xl font-bold">
              {result.expected_calibration_error.toFixed(4)}
            </div>
            <div className="text-xs text-muted-foreground">
              Expected Calibration Error
            </div>
          </div>
          <div className="rounded-lg bg-muted p-3 text-center">
            <div className="text-xl font-bold">
              {result.maximum_calibration_error.toFixed(4)}
            </div>
            <div className="text-xs text-muted-foreground">
              Maximum Calibration Error
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function BacktestResultsPanel({ result }: { result: BacktestResult }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Backtest Results</h2>
          <p className="text-sm text-muted-foreground">
            {result.num_samples.toLocaleString()} samples
            {result.date_range &&
              ` - ${result.date_range[0]} to ${result.date_range[1]}`}
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-muted-foreground">Model</div>
          <div className="font-medium">{result.model_version}</div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <ClassificationMetricsCard metrics={result.classification} />
        {result.betting && <BettingMetricsCard metrics={result.betting} />}
        <CalibrationCard result={result} />
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <ConfusionMatrixHeatmap
          confusionMatrix={result.classification.confusion_matrix}
          title="Prediction Confusion Matrix"
        />
        <CalibrationCurve
          expectedCalibrationError={result.expected_calibration_error}
          maximumCalibrationError={result.maximum_calibration_error}
          outcomeAccuracy={result.outcome_accuracy}
          title="Probability Calibration"
        />
      </div>

      {result.betting && (
        <EquityCurveChart
          profitLoss={result.betting.profit_loss}
          roiPercent={result.betting.roi_percent}
          maxDrawdown={result.betting.max_drawdown}
          sharpeRatio={result.betting.sharpe_ratio}
          winRate={result.betting.win_rate}
          totalBets={result.betting.total_bets}
          title="Betting Performance Simulation"
        />
      )}
    </div>
  );
}
