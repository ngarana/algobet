"use client";

import { useState } from "react";

import { useBacktest as useRunBacktest } from "@/lib/queries/use-ml-operations";
import type { BacktestHistoryItem, BacktestResult } from "@/lib/types/ml-operations";

interface BacktestFormValues {
  startDate: string;
  endDate: string;
  minMatches: number;
}

export function useBacktest() {
  const backtestMutation = useRunBacktest();
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [compareItems, setCompareItems] = useState<BacktestHistoryItem[]>([]);

  const runBacktest = async (data: BacktestFormValues) => {
    setError(null);
    setResult(null);

    try {
      const nextResult = await backtestMutation.mutateAsync({
        start_date: data.startDate,
        end_date: data.endDate,
        min_matches: data.minMatches,
      });
      setResult(nextResult);
    } catch (err) {
      console.error("Backtest error:", err);
      setError(
        err instanceof Error ? err.message : "Failed to run backtest. Please try again."
      );
    }
  };

  const toggleComparisonItem = (item: BacktestHistoryItem) => {
    setCompareItems((previousItems) => {
      const exists = previousItems.find((existingItem) => existingItem.id === item.id);
      if (exists) {
        return previousItems.filter((existingItem) => existingItem.id !== item.id);
      }
      if (previousItems.length >= 5) {
        return [...previousItems.slice(1), item];
      }
      return [...previousItems, item];
    });
  };

  return {
    compareItems,
    error,
    isPending: backtestMutation.isPending,
    result,
    runBacktest,
    toggleComparisonItem,
  };
}
