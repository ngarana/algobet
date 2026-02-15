"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { BarChart3 } from "lucide-react";

interface PerformanceTrendsProps {
  data?: any[];
  isLoading?: boolean;
}

export function PerformanceTrendsChart({
  data,
  isLoading = false,
}: PerformanceTrendsProps) {
  if (isLoading) {
    return (
      <div className="h-64">
        <Skeleton className="h-full w-full" />
      </div>
    );
  }

  return (
    <div className="flex h-64 flex-col">
      <div className="flex flex-1 items-center justify-center text-muted-foreground">
        <div className="text-center">
          <BarChart3 className="mx-auto mb-2 h-12 w-12 opacity-50" />
          <p className="text-sm">Performance chart will be displayed here</p>
          <p className="mt-1 text-xs text-muted-foreground">
            This chart will show your ROI and profit trends over time
          </p>
        </div>
      </div>
      <div className="mt-4 flex justify-center gap-4">
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-green-500"></div>
          <span className="text-xs">Profit</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-blue-500"></div>
          <span className="text-xs">ROI</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-purple-500"></div>
          <span className="text-xs">Predictions</span>
        </div>
      </div>
    </div>
  );
}
