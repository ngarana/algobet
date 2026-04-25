"use client";

import { Target } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { formatDuration } from "./utils";
import type { TrainingResultDisplayProps } from "./types";

export function TrainingResultDisplay({ result }: TrainingResultDisplayProps) {
  const testAccuracy = result.test_metrics.accuracy;

  return (
    <div className="rounded-md border bg-muted/30 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-medium">Latest training run</div>
          <div className="font-mono text-sm text-muted-foreground">
            {result.model_version}
          </div>
        </div>
        {result.is_active && (
          <Badge className="bg-green-600">
            <Target className="mr-1 h-3 w-3" />
            Active
          </Badge>
        )}
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            Test Accuracy
          </div>
          <div className="mt-1 text-xl font-semibold">
            {typeof testAccuracy === "number"
              ? `${(testAccuracy * 100).toFixed(1)}%`
              : "N/A"}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            Features
          </div>
          <div className="mt-1 text-xl font-semibold">{result.num_features}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            Duration
          </div>
          <div className="mt-1 text-xl font-semibold">
            {formatDuration(result.training_duration_seconds)}
          </div>
        </div>
      </div>
    </div>
  );
}
