"use client";

import { useState } from "react";
import { AlertCircle, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useTrainModel } from "@/lib/queries/use-ml-operations";
import { BasicSettings } from "./BasicSettings";
import { AdvancedSettings } from "./AdvancedSettings";
import { TrainingResultDisplay } from "./TrainingResultDisplay";
import { defaultConfig } from "./utils";
import type { TrainingConfig } from "./types";
import type { TrainModelResult } from "@/lib/types/ml-operations";

export function TrainModelCard() {
  const trainMutation = useTrainModel();
  const [config, setConfig] = useState<TrainingConfig>(defaultConfig);
  const [result, setResult] = useState<TrainModelResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const updateConfig = <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K]
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    // Validate split ratios
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
          // Data range settings
          start_date: config.startDate || undefined,
          end_date: config.endDate || undefined,
          min_matches: config.minMatches,
          // Tournament and team filtering
          tournament_ids:
            config.tournamentIds.length > 0 ? config.tournamentIds : undefined,
          team_ids: config.teamIds.length > 0 ? config.teamIds : undefined,
          venue_filter: config.venueFilter,
          // Match quality filters
          min_total_goals: config.minTotalGoals ?? undefined,
          max_total_goals: config.maxTotalGoals ?? undefined,
          // Split ratios
          train_ratio: config.trainRatio,
          val_ratio: config.valRatio,
          test_ratio: config.testRatio,
          // Training settings
          random_seed: config.randomSeed,
          early_stopping_rounds: config.earlyStoppingRounds,
          tuning_trials: config.tuningTrials,
          // Calibration settings
          calibrate_probabilities: config.calibrateProbabilities,
          calibration_method: config.calibrationMethod,
          // Outcome balancing
          outcome_balance: config.outcomeBalance,
          outcome_balance_strength: config.outcomeBalanceStrength,
          // Feature groups
          feature_groups:
            config.featureGroups.length > 0 ? config.featureGroups : undefined,
          feature_selection: config.featureSelection,
          feature_selection_threshold: config.featureSelectionThreshold,
          min_samples_per_feature:
            config.minSamplesPerFeature === null
              ? undefined
              : config.minSamplesPerFeature,
          // Ensemble training
          use_ensemble: config.useEnsemble,
          ensemble_types: config.useEnsemble ? config.ensembleTypes : undefined,
          // Split strategy
          split_strategy: config.splitStrategy,
          gap_days: config.gapDays,
          // Expanding window params
          min_train_size: config.minTrainSize,
          ew_val_size: config.ewValSize,
          ew_test_size: config.ewTestSize,
          step_size: config.stepSize,
          // Season-aware params
          train_seasons: config.trainSeasons,
          val_seasons: config.valSeasons,
          test_seasons: config.testSeasons,
          // Model tags
          tags: {},
          // Custom hyperparameters
          hyperparameters:
            Object.keys(config.customHyperparameters).length > 0
              ? config.customHyperparameters
              : {},
        },
        useGpuWorker: config.useGpuWorker,
      });
      setResult(trained);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to train model. Please try again."
      );
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Train Model
        </CardTitle>
        <CardDescription>
          Configure and train a new prediction model with customizable parameters
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        <form className="space-y-6" onSubmit={handleSubmit}>
          <BasicSettings config={config} onConfigChange={updateConfig} />
          <AdvancedSettings config={config} onConfigChange={updateConfig} />

          <Button className="w-full" disabled={trainMutation.isPending} type="submit">
            {trainMutation.isPending ? "Training..." : "Train Model"}
          </Button>
        </form>

        {result && <TrainingResultDisplay result={result} />}
      </CardContent>
    </Card>
  );
}
