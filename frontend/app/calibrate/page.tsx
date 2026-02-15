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
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertCircle,
  Play,
  Target,
  TrendingDown,
  TrendingUp,
  CheckCircle,
} from "lucide-react";
import { useCalibrate } from "@/lib/queries/use-ml-operations";
import { useActiveModel, useModels } from "@/lib/queries/use-models";
import type { CalibrateResult, CalibrationMetrics } from "@/lib/types/ml-operations";

function CalibrateForm({
  onSubmit,
  isLoading,
}: {
  onSubmit: (data: {
    method: "isotonic" | "sigmoid";
    validationSplit: number;
    activate: boolean;
  }) => void;
  isLoading: boolean;
}) {
  const { data: activeModel } = useActiveModel();

  const [method, setMethod] = useState<"isotonic" | "sigmoid">("isotonic");
  const [validationSplit, setValidationSplit] = useState(0.2);
  const [activate, setActivate] = useState(true);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ method, validationSplit, activate });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Calibrate Model
        </CardTitle>
        <CardDescription>
          Improve probability estimates for better value betting
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
            <Target className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">
              Base Model:{" "}
              {activeModel ? (
                <span className="font-medium">
                  {activeModel.name} ({activeModel.algorithm})
                </span>
              ) : (
                <span className="text-destructive">No active model</span>
              )}
            </span>
          </div>

          <div className="space-y-2">
            <Label htmlFor="method">Calibration Method</Label>
            <Select
              value={method}
              onValueChange={(v) => setMethod(v as "isotonic" | "sigmoid")}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="isotonic">Isotonic Regression</SelectItem>
                <SelectItem value="sigmoid">Sigmoid (Platt Scaling)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {method === "isotonic"
                ? "More flexible, works better with more data"
                : "Simpler, less prone to overfitting"}
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="validation-split">Validation Split</Label>
            <Select
              value={validationSplit.toString()}
              onValueChange={(v) => setValidationSplit(parseFloat(v))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0.1">10%</SelectItem>
                <SelectItem value="0.2">20%</SelectItem>
                <SelectItem value="0.3">30%</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="activate"
              checked={activate}
              onChange={(e) => setActivate(e.target.checked)}
              className="rounded border-gray-300"
            />
            <Label htmlFor="activate" className="text-sm font-normal">
              Activate after calibration
            </Label>
          </div>

          <Button type="submit" className="w-full" disabled={isLoading || !activeModel}>
            {isLoading ? (
              <>
                <span className="mr-2 animate-spin">⏳</span>
                Calibrating...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Calibrate Model
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function MetricsComparisonCard({
  before,
  after,
  improvement,
}: {
  before: CalibrationMetrics;
  after: CalibrationMetrics;
  improvement: CalibrateResult["improvement"];
}) {
  const metrics = [
    {
      label: "Expected Calibration Error",
      before: before.expected_calibration_error,
      after: after.expected_calibration_error,
      improvement: improvement.ece_improvement,
      lowerIsBetter: true,
    },
    {
      label: "Maximum Calibration Error",
      before: before.maximum_calibration_error,
      after: after.maximum_calibration_error,
      improvement: 0, // Not provided in improvement
      lowerIsBetter: true,
    },
    {
      label: "Brier Score",
      before: before.brier_score,
      after: after.brier_score,
      improvement: improvement.brier_improvement,
      lowerIsBetter: true,
    },
    {
      label: "Log Loss",
      before: before.log_loss,
      after: after.log_loss,
      improvement: improvement.log_loss_improvement,
      lowerIsBetter: true,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <TrendingDown className="h-5 w-5" />
          Calibration Improvement
        </CardTitle>
        <CardDescription>Lower values are better for all metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="py-2 text-left">Metric</th>
              <th className="py-2 text-right">Before</th>
              <th className="py-2 text-right">After</th>
              <th className="py-2 text-right">Change</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => {
              const change = metric.after - metric.before;
              const isPositive = metric.lowerIsBetter ? change < 0 : change > 0;

              return (
                <tr key={metric.label} className="border-b">
                  <td className="py-2">{metric.label}</td>
                  <td className="text-right font-mono">{metric.before.toFixed(4)}</td>
                  <td className="text-right font-mono">{metric.after.toFixed(4)}</td>
                  <td
                    className={`text-right font-mono ${isPositive ? "text-green-600" : "text-red-600"}`}
                  >
                    {change >= 0 ? "+" : ""}
                    {change.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function CalibrateResults({ result }: { result: CalibrateResult }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Calibration Complete</h2>
          <p className="text-sm text-muted-foreground">
            {result.samples_used.toLocaleString()} samples used for calibration
          </p>
        </div>
        <div className="flex items-center gap-2">
          {result.is_active && (
            <span className="flex items-center gap-1 text-sm text-green-600">
              <CheckCircle className="h-4 w-4" />
              Active
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Base Model</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="font-mono text-sm">{result.base_model_version}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Calibrated Model</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="font-mono text-sm">{result.calibrated_model_version}</p>
          </CardContent>
        </Card>
      </div>

      <MetricsComparisonCard
        before={result.before_metrics}
        after={result.after_metrics}
        improvement={result.improvement}
      />
    </div>
  );
}

export default function CalibratePage() {
  const calibrateMutation = useCalibrate();
  const [result, setResult] = useState<CalibrateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (data: {
    method: "isotonic" | "sigmoid";
    validationSplit: number;
    activate: boolean;
  }) => {
    setError(null);
    setResult(null);

    try {
      const result = await calibrateMutation.mutateAsync({
        method: data.method,
        validation_split: data.validationSplit,
        activate: data.activate,
      });
      setResult(result);
    } catch (err) {
      console.error("Calibrate error:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to calibrate model. Please try again."
      );
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Calibrate</h1>
        <p className="text-muted-foreground">
          Improve probability estimates for better value betting accuracy
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
          <CalibrateForm
            onSubmit={handleSubmit}
            isLoading={calibrateMutation.isPending}
          />
        </div>

        <div className="lg:col-span-2">
          {calibrateMutation.isPending ? (
            <div className="space-y-4">
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-64 w-full" />
            </div>
          ) : result ? (
            <CalibrateResults result={result} />
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <TrendingUp className="mb-4 h-12 w-12" />
                <p className="text-lg font-medium">No calibration yet</p>
                <p className="text-sm">
                  Run calibration to improve probability estimates
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
