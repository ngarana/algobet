"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { Box, RefreshCw, Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  useActivateModel,
  useActiveModel,
  useDeleteModel,
  useModels,
} from "@/lib/queries/use-models";
import { useTrainModel } from "@/lib/queries/use-ml-operations";
import {
  GuidedTrainingWorkspace,
  ModelRegistry,
  ModelInspector,
  TrainingResultDisplay,
} from "@/components/models";
import { defaultConfig } from "@/components/models/utils";
import type { TrainingConfig } from "@/components/models/types";
import type { ModelVersion } from "@/lib/types/api";
import type { TrainModelResult } from "@/lib/types/ml-operations";

export default function ModelsPage() {
  const searchParams = useSearchParams();
  const { data: modelsData, isLoading, refetch } = useModels();
  const { data: activeModel } = useActiveModel();
  const activateMutation = useActivateModel();
  const deleteMutation = useDeleteModel();
  const trainMutation = useTrainModel();

  const [config, setConfig] = useState<TrainingConfig>(defaultConfig);
  const [result, setResult] = useState<TrainModelResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);

  const models = modelsData?.items ?? [];
  const activeVersion = activeModel?.version ?? null;

  const selectedModel = useMemo(() => {
    if (selectedModelId === null) return null;
    return models.find((m) => m.id === selectedModelId) ?? null;
  }, [selectedModelId, models]);

  useEffect(() => {
    const idParam = searchParams.get("id");
    if (idParam) {
      const id = parseInt(idParam, 10);
      const exists = models.some((m) => m.id === id);
      if (exists) {
        setSelectedModelId(id);
      } else if (activeVersion) {
        const active = models.find((m) => m.version === activeVersion);
        setSelectedModelId(active?.id ?? null);
      } else if (models.length > 0) {
        setSelectedModelId(models[0].id);
      }
    } else if (activeVersion) {
      const active = models.find((m) => m.version === activeVersion);
      setSelectedModelId(active?.id ?? null);
    } else if (models.length > 0) {
      setSelectedModelId(models[0].id);
    }
  }, [searchParams, models, activeVersion]);

  const updateConfig = useCallback(
    <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
      setConfig((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleTrainSubmit = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      setError(null);

      const totalRatio = config.trainRatio + config.valRatio + config.testRatio;
      if (Math.abs(totalRatio - 1.0) > 0.001) {
        setError(
          `Split ratios must sum to 1.0, currently ${(totalRatio * 100).toFixed(1)}%`
        );
        return;
      }

      try {
        const trained = await trainMutation.mutateAsync({
          request: {
            model_type: config.modelType,
            tune_hyperparameters: config.tune,
            description: config.description.trim() || undefined,
            activate: config.activate,
            start_date: config.startDate || undefined,
            end_date: config.endDate || undefined,
            min_matches: config.minMatches,
            tournament_ids:
              config.tournamentIds.length > 0 ? config.tournamentIds : undefined,
            team_ids: config.teamIds.length > 0 ? config.teamIds : undefined,
            venue_filter: config.venueFilter,
            min_total_goals: config.minTotalGoals ?? undefined,
            max_total_goals: config.maxTotalGoals ?? undefined,
            train_ratio: config.trainRatio,
            val_ratio: config.valRatio,
            test_ratio: config.testRatio,
            random_seed: config.randomSeed,
            early_stopping_rounds: config.earlyStoppingRounds,
            tuning_trials: config.tuningTrials,
            calibrate_probabilities: config.calibrateProbabilities,
            calibration_method: config.calibrationMethod,
            outcome_balance: config.outcomeBalance,
            outcome_balance_strength: config.outcomeBalanceStrength,
            feature_groups:
              config.featureGroups.length > 0 ? config.featureGroups : undefined,
            feature_selection: config.featureSelection,
            feature_selection_threshold: config.featureSelectionThreshold,
            min_samples_per_feature:
              config.minSamplesPerFeature === null
                ? undefined
                : config.minSamplesPerFeature,
            use_ensemble: config.useEnsemble,
            ensemble_types: config.useEnsemble ? config.ensembleTypes : undefined,
            split_strategy: config.splitStrategy,
            gap_days: config.gapDays,
            min_train_size: config.minTrainSize,
            ew_val_size: config.ewValSize,
            ew_test_size: config.ewTestSize,
            step_size: config.stepSize,
            train_seasons: config.trainSeasons,
            val_seasons: config.valSeasons,
            test_seasons: config.testSeasons,
            tags: {},
            hyperparameters:
              Object.keys(config.customHyperparameters).length > 0
                ? config.customHyperparameters
                : {},
          },
          useGpuWorker: config.useGpuWorker,
        });
        setResult(trained);
        refetch();
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to train model. Please try again."
        );
      }
    },
    [config, trainMutation, refetch]
  );

  const handleSelectModel = useCallback((model: ModelVersion | null) => {
    setSelectedModelId(model?.id ?? null);
  }, []);

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

      <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg">
                Train Model
              </CardTitle>
            </CardHeader>
            <CardContent>
              <GuidedTrainingWorkspace
                config={config}
                onConfigChange={updateConfig}
                onSubmit={handleTrainSubmit}
                isTraining={trainMutation.isPending}
                error={error}
              />
              {result && (
                <div className="mt-4">
                  <TrainingResultDisplay result={result} />
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Model Registry</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-3">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                </div>
              ) : (
                <ModelRegistry
                  models={models}
                  activeVersion={activeVersion}
                  selectedModelId={selectedModelId}
                  onSelectModel={handleSelectModel}
                  onActivate={(id) => activateMutation.mutate(id)}
                  onDelete={(id) => deleteMutation.mutate(id)}
                />
              )}
            </CardContent>
          </Card>

          {selectedModel && (
            <ModelInspector
              model={selectedModel}
              onClose={() => setSelectedModelId(null)}
              onActivate={(id) => activateMutation.mutate(id)}
              onDelete={(id) => deleteMutation.mutate(id)}
            />
          )}
        </div>
      </div>
    </div>
  );
}
