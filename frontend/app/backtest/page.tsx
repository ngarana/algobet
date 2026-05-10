"use client";

import { AlertCircle, BarChart3 } from "lucide-react";

import {
  BacktestForm,
  BacktestParamsCard,
  BacktestResultsPanel,
  ModelComparisonCard,
} from "@/components/backtest";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useBacktest } from "@/hooks/useBacktest";

export default function BacktestPage() {
  const { compareItems, error, isPending, result, runBacktest, toggleComparisonItem } =
    useBacktest();

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
        <div className="space-y-6 lg:col-span-1">
          <BacktestForm onSubmit={runBacktest} isLoading={isPending} />
          <BacktestParamsCard
            onSelect={toggleComparisonItem}
            selectedIds={compareItems.map((item) => item.id)}
          />
        </div>

        <div className="space-y-6 lg:col-span-2">
          {isPending ? (
            <div className="space-y-4">
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-64 w-full" />
            </div>
          ) : result ? (
            <BacktestResultsPanel result={result} />
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <BarChart3 className="mb-4 h-12 w-12" />
                <p className="text-lg font-medium">No results yet</p>
                <p className="text-sm">Run a backtest to see performance metrics</p>
              </CardContent>
            </Card>
          )}

          <ModelComparisonCard items={compareItems} />
        </div>
      </div>
    </div>
  );
}
