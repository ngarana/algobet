"use client";

import { useState, useEffect } from "react";
import { AlertCircle, Brain, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useActivateModel, useActiveModel, useModels } from "@/lib/queries/use-models";
import {
  useGeneratePredictions,
  usePredictionHistory,
  useUpcomingPredictions,
} from "@/lib/queries/use-predictions";
import type { GeneratePredictionsResult } from "@/lib/api/predictions";

import PredictionDashboard from "@/components/predictions/PredictionDashboard";

export default function PredictionsPage() {
  const { data: activeModel } = useActiveModel();
  const { data: modelsData } = useModels();
  const activateMutation = useActivateModel();
  const generateMutation = useGeneratePredictions();

  const [view, setView] = useState<"upcoming" | "history">("upcoming");
  const [daysAhead, setDaysAhead] = useState(7);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [generationResult, setGenerationResult] =
    useState<GeneratePredictionsResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const models = modelsData?.items ?? [];

  useEffect(() => {
    if (models.length === 0) {
      setSelectedModelId("");
      return;
    }

    const selectedStillExists = models.some(
      (model) => String(model.id) === selectedModelId
    );
    if (selectedStillExists) {
      return;
    }

    if (activeModel?.id) {
      setSelectedModelId(String(activeModel.id));
      return;
    }

    setSelectedModelId(String(models[0].id));
  }, [activeModel?.id, models, selectedModelId]);

  const selectedModel =
    models.find((model) => String(model.id) === selectedModelId) ?? null;

  const {
    data: upcomingData,
    isLoading: upcomingLoading,
    refetch: refetchUpcoming,
  } = useUpcomingPredictions(daysAhead, selectedModel?.id);

  const {
    data: historyData,
    isLoading: historyLoading,
    refetch: refetchHistory,
  } = usePredictionHistory({
    model_version_id: selectedModel?.id,
    limit: 100,
  });

  const predictions =
    view === "upcoming" ? (upcomingData?.items ?? []) : (historyData?.items ?? []);
  const isLoading = view === "upcoming" ? upcomingLoading : historyLoading;

  const handleRefresh = () => {
    if (view === "upcoming") {
      void refetchUpcoming();
      return;
    }

    void refetchHistory();
  };

  const handleGenerate = async () => {
    if (!selectedModel) {
      setError("Select a model before generating predictions.");
      return;
    }

    setError(null);

    try {
      const result = await generateMutation.mutateAsync({
        model_version: selectedModel.version,
        days_ahead: daysAhead,
      });
      setGenerationResult(result);
      await refetchUpcoming();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to generate predictions. Please try again."
      );
    }
  };

  const handleActivate = async () => {
    if (!selectedModel) {
      return;
    }

    setError(null);

    try {
      await activateMutation.mutateAsync(selectedModel.id);
      await refetchUpcoming();
      await refetchHistory();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to activate the selected model."
      );
    }
  };

  const title = view === "upcoming" ? "Upcoming Predictions" : "Prediction History";
  const description = selectedModel
    ? `Showing ${predictions.length} predictions for ${selectedModel.version}`
    : `${predictions.length} predictions`;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-3xl font-bold tracking-tight">
            <Brain className="h-8 w-8" />
            Predictions
          </h1>
          <p className="text-muted-foreground">
            Generate and review predictions for the model you select from the UI.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={view}
            onChange={(e) => setView(e.target.value as "upcoming" | "history")}
            className="h-9 w-40 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
          >
            <option value="upcoming">Upcoming</option>
            <option value="history">History</option>
          </select>

          <Button variant="outline" size="sm" onClick={handleRefresh}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-2 p-4 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <p>{error}</p>
          </CardContent>
        </Card>
      )}

      <PredictionDashboard
        activeModel={activeModel ?? null}
        models={models}
        selectedModel={selectedModel}
        selectedModelId={selectedModelId}
        daysAhead={daysAhead}
        generationResult={generationResult}
        isGenerating={generateMutation.isPending}
        isActivating={activateMutation.isPending}
        onChangeDaysAhead={setDaysAhead}
        onChangeSelectedModelId={setSelectedModelId}
        onGenerate={handleGenerate}
        onActivate={handleActivate}
        predictions={predictions}
        isLoading={isLoading}
        title={title}
        description={description}
        showRoi={view === "history"}
        _onRefresh={handleRefresh}
      />
    </div>
  );
}
