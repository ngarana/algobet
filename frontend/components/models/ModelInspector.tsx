"use client";

import { X, BarChart3, Check, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useModelMetrics } from "@/lib/queries/use-models";
import { formatMetricValue } from "./utils";
import type { ModelVersion } from "@/lib/types/api";

interface ModelInspectorProps {
  model: ModelVersion;
  onClose: () => void;
  onActivate: (id: number) => void;
  onDelete: (id: number) => void;
}

export function ModelInspector({
  model,
  onClose,
  onActivate,
  onDelete,
}: ModelInspectorProps) {
  const { data: metrics, isLoading: metricsLoading } = useModelMetrics(model.id);

  const isActive = model.is_active;

  return (
    <Card className="sticky top-4">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-lg">
              <span className="font-mono">{model.version}</span>
              {isActive && <Badge className="bg-green-600">Active</Badge>}
            </CardTitle>
            <div className="mt-1 flex items-center gap-2 text-sm text-muted-foreground">
              <Badge variant="outline">{model.algorithm}</Badge>
              {model.accuracy !== null && (
                <span>{(model.accuracy * 100).toFixed(1)}% accuracy</span>
              )}
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {model.description && (
          <div>
            <h4 className="mb-1 text-xs font-medium text-muted-foreground">
              Description
            </h4>
            <p className="text-sm">{model.description}</p>
          </div>
        )}

        <div>
          <h4 className="mb-1 text-xs font-medium text-muted-foreground">Created</h4>
          <p className="text-sm">{new Date(model.created_at).toLocaleString()}</p>
        </div>

        {model.feature_schema_version && (
          <div>
            <h4 className="mb-1 text-xs font-medium text-muted-foreground">
              Feature Schema
            </h4>
            <p className="font-mono text-sm">{model.feature_schema_version}</p>
          </div>
        )}

        {model.hyperparameters && Object.keys(model.hyperparameters).length > 0 && (
          <div>
            <h4 className="mb-2 text-xs font-medium text-muted-foreground">
              Hyperparameters
            </h4>
            <div className="grid grid-cols-2 gap-1 rounded-md bg-muted p-2">
              {Object.entries(model.hyperparameters).map(([key, value]) => (
                <div key={key} className="col-span-2 flex justify-between text-xs">
                  <span className="text-muted-foreground">{key}:</span>
                  <span className="font-mono">{formatMetricValue(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div>
          <h4 className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
            <BarChart3 className="h-3 w-3" />
            Metrics
          </h4>
          {metricsLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
            </div>
          ) : metrics ? (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(metrics)
                .slice(0, 12)
                .map(([key, value]) => (
                  <div key={key} className="rounded-md bg-muted p-2 text-xs">
                    <div className="truncate text-muted-foreground">{key}</div>
                    <div className="font-mono font-medium">
                      {formatMetricValue(value)}
                    </div>
                  </div>
                ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No metrics available</p>
          )}
        </div>

        {!isActive && (
          <div className="flex gap-2">
            <Button size="sm" className="flex-1" onClick={() => onActivate(model.id)}>
              <Check className="mr-1 h-4 w-4" />
              Activate
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => {
                if (confirm(`Delete model ${model.version}?`)) {
                  onDelete(model.id);
                }
              }}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
