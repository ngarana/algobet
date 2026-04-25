"use client";

import { useState } from "react";
import { Box, RefreshCw, Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  useActivateModel,
  useActiveModel,
  useDeleteModel,
  useModels,
} from "@/lib/queries/use-models";
import { TrainModelCard, ModelRow } from "@/components/models";
import type { ModelVersion } from "@/lib/types/api";

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
            Train, inspect, activate, and retire prediction models
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
              {averageAccuracy !== null
                ? `${(averageAccuracy * 100).toFixed(1)}%`
                : "-"}
            </div>
            <div className="text-xs text-muted-foreground">Average Accuracy</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)]">
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
                  Activate the model you want to use for default predictions, or inspect
                  metrics before switching.
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
