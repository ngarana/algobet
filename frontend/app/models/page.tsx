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
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Box, RefreshCw, Check, Trash2, BarChart3 } from "lucide-react";
import {
  useModels,
  useActiveModel,
  useActivateModel,
  useDeleteModel,
  useModelMetrics,
} from "@/lib/queries/use-models";
import type { ModelVersion } from "@/lib/types/api";

function ModelMetricsPanel({
  model,
  onClose,
}: {
  model: ModelVersion;
  onClose: () => void;
}) {
  const { data: metrics, isLoading } = useModelMetrics(model.id);

  return (
    <Card className="mt-4">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Model Metrics - {model.version}</CardTitle>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {[...Array(6)].map((_, i) => (
              <Skeleton key={i} className="h-8" />
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="text-sm text-muted-foreground">Algorithm</div>
                  <div className="text-lg font-semibold">{model.algorithm}</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="text-sm text-muted-foreground">Accuracy</div>
                  <div className="text-lg font-semibold">
                    {model.accuracy ? `${(model.accuracy * 100).toFixed(1)}%` : "N/A"}
                  </div>
                </CardContent>
              </Card>
            </div>

            {metrics && Object.keys(metrics).length > 0 && (
              <div className="space-y-2">
                <h4 className="font-semibold">Detailed Metrics</h4>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(metrics).map(([key, value]) => (
                    <div
                      key={key}
                      className="flex justify-between rounded bg-muted p-2"
                    >
                      <span className="text-sm">{key}</span>
                      <span className="font-mono text-sm">
                        {typeof value === "number" ? value.toFixed(4) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {model.hyperparameters && Object.keys(model.hyperparameters).length > 0 && (
              <div className="space-y-2">
                <h4 className="font-semibold">Hyperparameters</h4>
                <pre className="max-h-48 overflow-auto rounded bg-muted p-4 text-sm">
                  {JSON.stringify(model.hyperparameters, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ModelRow({
  model,
  isActive,
  onActivate,
  onDelete,
  onShowMetrics,
  isExpanded,
}: {
  model: ModelVersion;
  isActive: boolean;
  onActivate: (id: number) => void;
  onDelete: (id: number) => void;
  onShowMetrics: (model: ModelVersion) => void;
  isExpanded: boolean;
}) {
  return (
    <>
      <TableRow className={isActive ? "bg-green-50 dark:bg-green-950/20" : ""}>
        <TableCell className="font-mono text-sm">{model.version}</TableCell>
        <TableCell>
          <Badge variant="outline">{model.algorithm}</Badge>
        </TableCell>
        <TableCell className="font-mono">
          {model.accuracy ? `${(model.accuracy * 100).toFixed(1)}%` : "-"}
        </TableCell>
        <TableCell>
          {isActive ? (
            <Badge className="bg-green-500">Active</Badge>
          ) : (
            <Badge variant="secondary">Inactive</Badge>
          )}
        </TableCell>
        <TableCell className="text-sm text-muted-foreground">
          {new Date(model.created_at).toLocaleDateString()}
        </TableCell>
        <TableCell className="text-sm text-muted-foreground">
          {model.description || "-"}
        </TableCell>
        <TableCell>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => onShowMetrics(model)}>
              <BarChart3 className="mr-1 h-4 w-4" />
              Metrics
            </Button>
            {!isActive && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onActivate(model.id)}
                >
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
              </>
            )}
          </div>
        </TableCell>
      </TableRow>
      {isExpanded && (
        <TableRow>
          <TableCell colSpan={7} className="p-0">
            <ModelMetricsPanel
              model={model}
              onClose={() => onShowMetrics(null as unknown as ModelVersion)}
            />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

export default function ModelsPage() {
  const { data: modelsData, isLoading, refetch } = useModels();
  const { data: activeModel } = useActiveModel();
  const activateMutation = useActivateModel();
  const deleteMutation = useDeleteModel();
  const [expandedModel, setExpandedModel] = useState<ModelVersion | null>(null);

  const models = modelsData?.items ?? [];
  const activeVersion = activeModel?.version;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-3xl font-bold tracking-tight">
            <Box className="h-8 w-8" />
            Models
          </h1>
          <p className="text-muted-foreground">
            Manage ML prediction models and view performance metrics
          </p>
        </div>
        <div className="flex items-center gap-2">
          {activeModel && (
            <Badge variant="secondary" className="flex items-center gap-1">
              <Check className="h-3 w-3" />
              Active: {activeModel.version}
            </Badge>
          )}
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">{models.length}</div>
            <div className="text-xs text-muted-foreground">Total Models</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600">
              {models.filter((m) => m.is_active).length}
            </div>
            <div className="text-xs text-muted-foreground">Active Models</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">
              {models.length > 0
                ? `${((models.reduce((sum, m) => sum + (m.accuracy || 0), 0) / models.length) * 100).toFixed(1)}%`
                : "-"}
            </div>
            <div className="text-xs text-muted-foreground">Avg Accuracy</div>
          </CardContent>
        </Card>
      </div>

      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-64" />
        </div>
      ) : models.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Model Versions</CardTitle>
            <CardDescription>All registered prediction models</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Version</TableHead>
                  <TableHead>Algorithm</TableHead>
                  <TableHead>Accuracy</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Description</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((model) => (
                  <ModelRow
                    key={model.id}
                    model={model}
                    isActive={model.version === activeVersion}
                    onActivate={(id) => activateMutation.mutate(id)}
                    onDelete={(id) => deleteMutation.mutate(id)}
                    onShowMetrics={setExpandedModel}
                    isExpanded={expandedModel?.id === model.id}
                  />
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Box className="mb-4 h-12 w-12" />
            <p className="text-lg font-medium">No models found</p>
            <p className="text-sm">Train a model using the CLI to get started</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
