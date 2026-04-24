"use client";

import { useEffect, useState } from "react";
import {
  AlertCircle,
  Brain,
  Check,
  Play,
  RefreshCw,
  Target,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useActivateModel, useActiveModel, useModels } from "@/lib/queries/use-models";
import {
  useGeneratePredictions,
  usePredictionHistory,
  useUpcomingPredictions,
} from "@/lib/queries/use-predictions";
import type { GeneratePredictionsResult } from "@/lib/api/predictions";
import type { ModelVersion, Prediction } from "@/lib/types/api";

const outcomeLabels: Record<string, string> = {
  H: "Home Win",
  D: "Draw",
  A: "Away Win",
};

const outcomeBadgeClass: Record<string, string> = {
  H: "bg-blue-600",
  D: "bg-amber-500 text-black",
  A: "bg-red-600",
};

function PredictionsSummary({ predictions }: { predictions: Prediction[] }) {
  const avgConfidence =
    predictions.length > 0
      ? predictions.reduce((sum, prediction) => sum + prediction.confidence, 0) /
        predictions.length
      : 0;

  const counts = predictions.reduce(
    (accumulator, prediction) => {
      accumulator[prediction.predicted_outcome] =
        (accumulator[prediction.predicted_outcome] || 0) + 1;
      return accumulator;
    },
    {} as Record<string, number>
  );

  return (
    <div className="grid gap-4 md:grid-cols-4">
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold">{predictions.length}</div>
          <div className="text-xs text-muted-foreground">Predictions</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold">{(avgConfidence * 100).toFixed(1)}%</div>
          <div className="text-xs text-muted-foreground">Avg Confidence</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">{counts.H ?? 0}</div>
          <div className="text-xs text-muted-foreground">Home Picks</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-red-600">{counts.A ?? 0}</div>
          <div className="text-xs text-muted-foreground">Away Picks</div>
        </CardContent>
      </Card>
    </div>
  );
}

function PredictionRow({
  prediction,
  showRoi,
}: {
  prediction: Prediction;
  showRoi: boolean;
}) {
  const match = prediction.match;
  const matchLabel = match
    ? `${match.home_team_name} vs ${match.away_team_name}`
    : `Match #${prediction.match_id}`;

  return (
    <TableRow>
      <TableCell>
        <div className="font-medium">{matchLabel}</div>
        {match?.tournament_name && (
          <div className="text-xs text-muted-foreground">
            {match.tournament_name}
            {match.season_name ? ` • ${match.season_name}` : ""}
          </div>
        )}
      </TableCell>
      <TableCell>
        <div className="flex flex-col gap-1">
          <Badge className={outcomeBadgeClass[prediction.predicted_outcome]}>
            {outcomeLabels[prediction.predicted_outcome]}
          </Badge>
          {prediction.model_version && (
            <span className="text-xs text-muted-foreground">
              {prediction.model_version.version}
            </span>
          )}
        </div>
      </TableCell>
      <TableCell className="font-mono text-xs">
        {(prediction.prob_home * 100).toFixed(1)} /{" "}
        {(prediction.prob_draw * 100).toFixed(1)} /{" "}
        {(prediction.prob_away * 100).toFixed(1)}
      </TableCell>
      <TableCell className="font-mono">
        {(prediction.confidence * 100).toFixed(1)}%
      </TableCell>
      <TableCell>
        <div>{match ? new Date(match.match_date).toLocaleString() : "-"}</div>
        <div className="text-xs text-muted-foreground">
          Generated {new Date(prediction.predicted_at).toLocaleString()}
        </div>
      </TableCell>
      {showRoi && (
        <TableCell
          className={
            prediction.actual_roi !== null
              ? prediction.actual_roi >= 0
                ? "font-mono text-green-600"
                : "font-mono text-red-600"
              : "font-mono text-muted-foreground"
          }
        >
          {prediction.actual_roi !== null
            ? `${prediction.actual_roi >= 0 ? "+" : ""}${(prediction.actual_roi * 100).toFixed(1)}%`
            : "-"}
        </TableCell>
      )}
    </TableRow>
  );
}

function PredictionControls({
  activeModel,
  models,
  selectedModel,
  selectedModelId,
  daysAhead,
  generationResult,
  isGenerating,
  isActivating,
  onChangeDaysAhead,
  onChangeSelectedModelId,
  onGenerate,
  onActivate,
}: {
  activeModel: ModelVersion | null;
  models: ModelVersion[];
  selectedModel: ModelVersion | null;
  selectedModelId: string;
  daysAhead: number;
  generationResult: GeneratePredictionsResult | null;
  isGenerating: boolean;
  isActivating: boolean;
  onChangeDaysAhead: (value: number) => void;
  onChangeSelectedModelId: (value: string) => void;
  onGenerate: () => void;
  onActivate: () => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Prediction Controls
        </CardTitle>
        <CardDescription>
          Choose the model you want to use, generate upcoming predictions, and optionally make it the active default.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_180px_auto_auto]">
          <div className="space-y-2">
            <Label htmlFor="prediction-model">Model</Label>
            <Select value={selectedModelId} onValueChange={onChangeSelectedModelId}>
              <SelectTrigger id="prediction-model">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.id} value={String(model.id)}>
                    {model.version}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="days-ahead">Days Ahead</Label>
            <Select
              value={String(daysAhead)}
              onValueChange={(value) => onChangeDaysAhead(Number(value))}
            >
              <SelectTrigger id="days-ahead">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="3">3 days</SelectItem>
                <SelectItem value="7">7 days</SelectItem>
                <SelectItem value="14">14 days</SelectItem>
                <SelectItem value="30">30 days</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button
            className="self-end"
            disabled={!selectedModel || isGenerating}
            onClick={onGenerate}
          >
            {isGenerating ? "Generating..." : "Generate"}
          </Button>

          <Button
            className="self-end"
            disabled={!selectedModel || selectedModel.is_active || isActivating}
            onClick={onActivate}
            variant="outline"
          >
            {isActivating ? "Activating..." : "Set Active"}
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          {selectedModel ? (
            <>
              <Badge variant="secondary">{selectedModel.algorithm}</Badge>
              <span>Selected: {selectedModel.version}</span>
              {activeModel && activeModel.id === selectedModel.id && (
                <Badge className="bg-green-600">
                  <Check className="mr-1 h-3 w-3" />
                  Active
                </Badge>
              )}
            </>
          ) : (
            <span>No model selected.</span>
          )}
        </div>

        {generationResult && (
          <div className="rounded-md border bg-muted/30 p-4">
            <div className="text-sm font-medium">Latest generation run</div>
            <div className="mt-2 grid gap-3 md:grid-cols-4">
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  Model
                </div>
                <div className="font-mono text-sm">{generationResult.model_version}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  Generated
                </div>
                <div className="text-lg font-semibold">{generationResult.generated}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  Skipped
                </div>
                <div className="text-lg font-semibold">
                  {generationResult.existing_predictions_skipped}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  Processed
                </div>
                <div className="text-lg font-semibold">
                  {generationResult.matches_processed}
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function PredictionsPage() {
  const { data: activeModel } = useActiveModel();
  const { data: modelsData } = useModels();
  const activateMutation = useActivateModel();
  const generateMutation = useGeneratePredictions();

  const [view, setView] = useState<"upcoming" | "history">("upcoming");
  const [daysAhead, setDaysAhead] = useState(7);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [generationResult, setGenerationResult] = useState<GeneratePredictionsResult | null>(
    null
  );
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
    view === "upcoming" ? upcomingData?.items ?? [] : historyData?.items ?? [];
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
          <Select value={view} onValueChange={(value) => setView(value as "upcoming" | "history")}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="upcoming">Upcoming</SelectItem>
              <SelectItem value="history">History</SelectItem>
            </SelectContent>
          </Select>

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

      <PredictionControls
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
      />

      {isLoading ? (
        <div className="space-y-4">
          <div className="grid gap-4 md:grid-cols-4">
            {[...Array(4)].map((_, index) => (
              <Skeleton key={index} className="h-20" />
            ))}
          </div>
          <Skeleton className="h-72" />
        </div>
      ) : (
        <>
          <PredictionsSummary predictions={predictions} />

          {predictions.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle>
                  {view === "upcoming" ? "Upcoming Predictions" : "Prediction History"}
                </CardTitle>
                <CardDescription>
                  {selectedModel
                    ? `Showing ${predictions.length} predictions for ${selectedModel.version}`
                    : `${predictions.length} predictions`}
                </CardDescription>
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Match</TableHead>
                      <TableHead>Prediction</TableHead>
                      <TableHead>Probabilities</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Timing</TableHead>
                      {view === "history" && <TableHead>ROI</TableHead>}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {predictions.map((prediction) => (
                      <PredictionRow
                        key={prediction.id}
                        prediction={prediction}
                        showRoi={view === "history"}
                      />
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <Brain className="mb-4 h-12 w-12" />
                <p className="text-lg font-medium">No predictions found</p>
                <p className="text-sm">
                  {view === "upcoming"
                    ? "Generate predictions for the selected model to populate this view."
                    : "Historical predictions will appear here once matches settle."}
                </p>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
