"use client";

import { GitCompare, History } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useBacktestHistory } from "@/lib/queries/use-ml-operations";
import type { BacktestHistoryItem } from "@/lib/types/ml-operations";

interface BacktestParamsCardProps {
  onSelect: (item: BacktestHistoryItem) => void;
  selectedIds: number[];
}

export function BacktestParamsCard({ onSelect, selectedIds }: BacktestParamsCardProps) {
  const { data, isLoading } = useBacktestHistory({ limit: 10 });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Backtest History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {[...Array(3)].map((_, index) => (
              <Skeleton key={index} className="h-12 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data?.items.length) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5" />
          Backtest History
        </CardTitle>
        <CardDescription>Click to compare previous backtest results</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="max-h-64 space-y-2 overflow-y-auto">
          {data.items.map((item) => (
            <button
              key={item.id}
              onClick={() => onSelect(item)}
              className={`w-full rounded-lg border p-3 text-left transition-colors ${
                selectedIds.includes(item.id)
                  ? "border-primary bg-primary/10"
                  : "border-border hover:bg-muted"
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-medium">
                    {item.model_name || item.model_version || "Unknown"}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {new Date(item.evaluated_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="text-right">
                  <p className="font-mono text-sm">
                    {(item.accuracy * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {item.num_samples.toLocaleString()} samples
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function ModelComparisonCard({ items }: { items: BacktestHistoryItem[] }) {
  if (items.length < 2) {
    return null;
  }

  const bestAccuracy = items.reduce((best, item) =>
    item.accuracy > best.accuracy ? item : best
  );
  const bestF1 = items.reduce((best, item) =>
    item.f1_macro > best.f1_macro ? item : best
  );
  const bestROI = items.reduce((best, item) =>
    (item.roi_percent ?? -Infinity) > (best.roi_percent ?? -Infinity) ? item : best
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitCompare className="h-5 w-5" />
          Model Comparison
        </CardTitle>
        <CardDescription>Comparing {items.length} backtest results</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="py-2 text-left">Model</th>
                <th className="py-2 text-right">Accuracy</th>
                <th className="py-2 text-right">F1 Macro</th>
                <th className="py-2 text-right">ROI%</th>
                <th className="py-2 text-right">Win Rate</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id} className="border-b">
                  <td className="py-2">
                    <div>
                      <p className="font-medium">
                        {item.model_name || item.model_version}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {item.date_range_start} to {item.date_range_end}
                      </p>
                    </div>
                  </td>
                  <td className="py-2 text-right font-mono">
                    <span
                      className={
                        bestAccuracy.id === item.id ? "font-bold text-green-600" : ""
                      }
                    >
                      {(item.accuracy * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-2 text-right font-mono">
                    <span
                      className={
                        bestF1.id === item.id ? "font-bold text-green-600" : ""
                      }
                    >
                      {item.f1_macro.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-2 text-right font-mono">
                    <span
                      className={
                        bestROI.id === item.id ? "font-bold text-green-600" : ""
                      }
                    >
                      {item.roi_percent !== null
                        ? `${item.roi_percent.toFixed(1)}%`
                        : "N/A"}
                    </span>
                  </td>
                  <td className="py-2 text-right font-mono">
                    {item.win_rate !== null
                      ? `${(item.win_rate * 100).toFixed(1)}%`
                      : "N/A"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
