"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useModelMetrics } from "@/lib/queries/use-models";
import { formatMetricValue } from "./utils";
import type { ModelMetricsPanelProps } from "./types";

export function ModelMetricsPanel({ model, onClose }: ModelMetricsPanelProps) {
  const { data, isLoading } = useModelMetrics(model.id);

  return (
    <Card className="m-4 border-dashed">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-lg">{model.version}</CardTitle>
            <p className="text-sm text-muted-foreground">
              Detailed metrics and training metadata
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {[...Array(5)].map((_, index) => (
              <Skeleton key={index} className="h-10 w-full" />
            ))}
          </div>
        ) : data ? (
          <div className="space-y-6">
            <div className="grid gap-3 md:grid-cols-3">
              <Card>
                <CardContent className="p-4">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">
                    Algorithm
                  </div>
                  <div className="mt-2 text-lg font-semibold">{data.algorithm}</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">
                    Accuracy
                  </div>
                  <div className="mt-2 text-lg font-semibold">
                    {data.accuracy !== null
                      ? `${(data.accuracy * 100).toFixed(1)}%`
                      : "N/A"}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">
                    Feature Schema
                  </div>
                  <div className="mt-2 text-lg font-semibold">
                    {data.feature_schema_version ?? "N/A"}
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="space-y-3">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                Metrics
              </h3>
              <div className="grid gap-2 md:grid-cols-2">
                {Object.entries(data.metrics).map(([key, value]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between rounded-md border bg-muted/30 px-3 py-2"
                  >
                    <span className="text-sm text-muted-foreground">{key}</span>
                    <span className="font-mono text-sm">
                      {formatMetricValue(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {Object.keys(data.hyperparameters).length > 0 && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                  Hyperparameters
                </h3>
                <div className="rounded-md border bg-muted/30 p-3">
                  <pre className="overflow-auto text-xs">
                    {JSON.stringify(data.hyperparameters, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">Metrics unavailable.</p>
        )}
      </CardContent>
    </Card>
  );
}
