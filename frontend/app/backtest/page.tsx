"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, Play, TrendingUp, Target, BarChart3 } from "lucide-react";
import { useBacktest } from "@/lib/queries/use-ml-operations";
import { useActiveModel, useModels } from "@/lib/queries/use-models";
import type { BacktestResult } from "@/lib/types/ml-operations";

function BacktestForm({
  onSubmit,
  isLoading,
}: {
  onSubmit: (data: { startDate: string; endDate: string; minMatches: number }) => void;
  isLoading: boolean;
}) {
  const { data: activeModel } = useActiveModel();
  const { data: modelsData } = useModels();

  // Default dates
  const today = new Date();
  const oneYearAgo = new Date(today);
  oneYearAgo.setFullYear(today.getFullYear() - 1);

  const [startDate, setStartDate] = useState(oneYearAgo.toISOString().split("T")[0]);
  const [endDate, setEndDate] = useState(today.toISOString().split("T")[0]);
  const [minMatches, setMinMatches] = useState(100);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ startDate, endDate, minMatches });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Run Backtest
        </CardTitle>
        <CardDescription>
          Evaluate model performance on historical match data
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
            <Target className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">
              Active Model:{" "}
              {activeModel ? (
                <span className="font-medium">
                  {activeModel.name} ({activeModel.algorithm})
                </span>
              ) : (
                <span className="text-destructive">No active model</span>
              )}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start-date">Start Date</Label>
              <Input
                id="start-date"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="end-date">End Date</Label>
              <Input
                id="end-date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="min-matches">Minimum Matches</Label>
            <Input
              id="min-matches"
              type="number"
              min={10}
              max={10000}
              value={minMatches}
              onChange={(e) => setMinMatches(parseInt(e.target.value) || 100)}
            />
          </div>

          <Button type="submit" className="w-full" disabled={isLoading || !activeModel}>
            {isLoading ? (
              <>
                <span className="mr-2 animate-spin">⏳</span>
                Running Backtest...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Backtest
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function ClassificationMetricsCard({
  metrics,
}: {
  metrics: BacktestResult["classification"];
}) {
  const metricRows = [
    {
      label: "Accuracy",
      value: `${(metrics.accuracy * 100).toFixed(1)}%`,
      target: "≥50%",
    },
    { label: "Log Loss", value: metrics.log_loss.toFixed(3), target: "≤0.95" },
    { label: "Brier Score", value: metrics.brier_score.toFixed(3), target: "≤0.20" },
    { label: "F1 (Macro)", value: metrics.f1_macro.toFixed(3), target: "≥0.45" },
    {
      label: "Top-2 Accuracy",
      value: `${(metrics.top_2_accuracy * 100).toFixed(1)}%`,
      target: "≥75%",
    },
    { label: "Cohen's Kappa", value: metrics.cohen_kappa.toFixed(3), target: "≥0.30" },
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
              className={`text-3xl font-bold ${isPositiveROI ? "text-green-600" : "text-red-600"}`}
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
              className={`font-mono ${isPositiveROI ? "text-green-600" : "text-red-600"}`}
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

function BacktestResults({ result }: { result: BacktestResult }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Backtest Results</h2>
          <p className="text-sm text-muted-foreground">
            {result.num_samples.toLocaleString()} samples
            {result.date_range &&
              ` • ${result.date_range[0]} to ${result.date_range[1]}`}
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
    </div>
  );
}

export default function BacktestPage() {
  const backtestMutation = useBacktest();
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (data: {
    startDate: string;
    endDate: string;
    minMatches: number;
  }) => {
    setError(null);
    setResult(null);

    try {
      const result = await backtestMutation.mutateAsync({
        start_date: data.startDate,
        end_date: data.endDate,
        min_matches: data.minMatches,
      });
      setResult(result);
    } catch (err) {
      console.error("Backtest error:", err);
      setError(
        err instanceof Error ? err.message : "Failed to run backtest. Please try again."
      );
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Backtest</h1>
        <p className="text-muted-foreground">
          Evaluate model performance on historical match data
        </p>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-2 p-4 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <p>{error}</p>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <BacktestForm
            onSubmit={handleSubmit}
            isLoading={backtestMutation.isPending}
          />
        </div>

        <div className="lg:col-span-2">
          {backtestMutation.isPending ? (
            <div className="space-y-4">
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-64 w-full" />
            </div>
          ) : result ? (
            <BacktestResults result={result} />
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <BarChart3 className="mb-4 h-12 w-12" />
                <p className="text-lg font-medium">No results yet</p>
                <p className="text-sm">Run a backtest to see performance metrics</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
