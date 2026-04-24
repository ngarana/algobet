"use client";

import { useState } from "react";
import {
  AlertCircle,
  BarChart3,
  Box,
  Check,
  Play,
  RefreshCw,
  Target,
  Trash2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  useActivateModel,
  useActiveModel,
  useDeleteModel,
  useModelMetrics,
  useModels,
} from "@/lib/queries/use-models";
import { useTrainModel } from "@/lib/queries/use-ml-operations";
import type { ModelVersion } from "@/lib/types/api";
import type { TrainModelResult } from "@/lib/types/ml-operations";

function formatMetricValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (Array.isArray(value)) {
    return value.join(", ");
  }

  return String(value);
}

function ModelMetricsPanel({
  model,
  onClose,
}: {
  model: ModelVersion;
  onClose: () => void;
}) {
  const { data, isLoading } = useModelMetrics(model.id);

  return (
    <Card className="m-4 border-dashed">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-lg">{model.version}</CardTitle>
            <CardDescription>
              Detailed metrics and training metadata
            </CardDescription>
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
                    <span className="font-mono text-sm">{formatMetricValue(value)}</span>
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

function TrainModelCard() {
  const trainMutation = useTrainModel();
  const [modelType, setModelType] = useState("xgboost");
  const [description, setDescription] = useState("");
  const [tune, setTune] = useState(false);
  const [activate, setActivate] = useState(true);
  const [result, setResult] = useState<TrainModelResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    try {
      const trained = await trainMutation.mutateAsync({
        model_type: modelType as "xgboost" | "lightgbm" | "random_forest",
        tune_hyperparameters: tune,
        description: description.trim() || undefined,
        activate,
      });
      setResult(trained);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to train model. Please try again."
      );
    }
  };

  const testAccuracy = result?.test_metrics.accuracy;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Train Model
        </CardTitle>
        <CardDescription>
          Start a new training run from the UI using the historical match data in the database.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <Label htmlFor="model-type">Model Type</Label>
            <Select value={modelType} onValueChange={setModelType}>
              <SelectTrigger id="model-type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="xgboost">XGBoost</SelectItem>
                <SelectItem value="lightgbm">LightGBM</SelectItem>
                <SelectItem value="random_forest">Random Forest</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Input
              id="description"
              placeholder="Optional label for this training run"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </div>

          <div className="grid gap-3">
            <label className="flex items-center gap-3 rounded-md border p-3">
              <Checkbox checked={tune} onCheckedChange={(checked) => setTune(Boolean(checked))} />
              <div>
                <div className="text-sm font-medium">Hyperparameter tuning</div>
                <div className="text-xs text-muted-foreground">
                  Runs Optuna tuning when available before final training.
                </div>
              </div>
            </label>

            <label className="flex items-center gap-3 rounded-md border p-3">
              <Checkbox
                checked={activate}
                onCheckedChange={(checked) => setActivate(Boolean(checked))}
              />
              <div>
                <div className="text-sm font-medium">Activate after training</div>
                <div className="text-xs text-muted-foreground">
                  Makes the new model the default for predictions immediately.
                </div>
              </div>
            </label>
          </div>

          <Button className="w-full" disabled={trainMutation.isPending} type="submit">
            {trainMutation.isPending ? "Training..." : "Train Model"}
          </Button>
        </form>

        {result && (
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
                  {result.training_duration_seconds.toFixed(1)}s
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ModelRow({
  model,
  isActive,
  isExpanded,
  onActivate,
  onDelete,
  onToggleMetrics,
}: {
  model: ModelVersion;
  isActive: boolean;
  isExpanded: boolean;
  onActivate: (id: number) => void;
  onDelete: (id: number) => void;
  onToggleMetrics: (model: ModelVersion | null) => void;
}) {
  return (
    <>
      <TableRow className={isActive ? "bg-green-50/70 dark:bg-green-950/20" : ""}>
        <TableCell className="font-mono text-xs">{model.version}</TableCell>
        <TableCell>
          <Badge variant="outline">{model.algorithm}</Badge>
        </TableCell>
        <TableCell className="font-mono">
          {model.accuracy !== null ? `${(model.accuracy * 100).toFixed(1)}%` : "-"}
        </TableCell>
        <TableCell>
          {isActive ? (
            <Badge className="bg-green-600">Active</Badge>
          ) : (
            <Badge variant="secondary">Inactive</Badge>
          )}
        </TableCell>
        <TableCell className="text-sm text-muted-foreground">
          {new Date(model.created_at).toLocaleString()}
        </TableCell>
        <TableCell className="max-w-64 text-sm text-muted-foreground">
          {model.description ?? "-"}
        </TableCell>
        <TableCell>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onToggleMetrics(isExpanded ? null : model)}
            >
              <BarChart3 className="mr-1 h-4 w-4" />
              {isExpanded ? "Hide" : "Metrics"}
            </Button>

            {!isActive && (
              <Button variant="outline" size="sm" onClick={() => onActivate(model.id)}>
                <Check className="mr-1 h-4 w-4" />
                Activate
              </Button>
            )}

            {!isActive && (
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
            )}
          </div>
        </TableCell>
      </TableRow>

      {isExpanded && (
        <TableRow>
          <TableCell className="p-0" colSpan={7}>
            <ModelMetricsPanel model={model} onClose={() => onToggleMetrics(null)} />
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
  const activeVersion = activeModel?.version ?? null;
  const averageAccuracy =
    models.length > 0
      ? models.reduce((sum, model) => sum + (model.accuracy ?? 0), 0) / models.length
      : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-3xl font-bold tracking-tight">
            <Box className="h-8 w-8" />
            Models
          </h1>
          <p className="text-muted-foreground">
            Train, inspect, activate, and retire prediction models from the frontend.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {activeModel && (
            <Badge variant="secondary" className="flex items-center gap-1">
              <Target className="h-3 w-3" />
              Active: {activeModel.version}
            </Badge>
          )}
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">{models.length}</div>
            <div className="text-xs text-muted-foreground">Registered Models</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600">
              {models.filter((model) => model.is_active).length}
            </div>
            <div className="text-xs text-muted-foreground">Active Models</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">
              {averageAccuracy !== null ? `${(averageAccuracy * 100).toFixed(1)}%` : "-"}
            </div>
            <div className="text-xs text-muted-foreground">Average Accuracy</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div>
          <TrainModelCard />
        </div>

        <div>
          {isLoading ? (
            <div className="space-y-3">
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-64 w-full" />
            </div>
          ) : models.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle>Model Registry</CardTitle>
                <CardDescription>
                  Activate the model you want to use for default predictions, or inspect metrics before switching.
                </CardDescription>
              </CardHeader>
              <CardContent className="overflow-x-auto">
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
                        isExpanded={expandedModel?.id === model.id}
                        onActivate={(id) => activateMutation.mutate(id)}
                        onDelete={(id) => deleteMutation.mutate(id)}
                        onToggleMetrics={setExpandedModel}
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
                <p className="text-sm">
                  Train your first model from the panel on the left.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
