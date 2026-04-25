import { useState, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Grid, List } from "lucide-react";

import { PredictionControls } from "@/components/predictions";
import { PredictionStats } from "@/components/predictions";
import { PredictionFilters } from "@/components/predictions";
import { PredictionTable } from "@/components/predictions";
import { PredictionCard } from "@/components/predictions";
import { ModelPerformanceCard } from "@/components/predictions";
import { PredictionDetailModal } from "@/components/predictions";
import { ExportButton } from "@/components/predictions";
import type { Prediction } from "@/lib/types/api";
import type { PredictionFilterState } from "@/components/predictions/PredictionFilters";
import type { ModelVersion, PredictionMatchSummary } from "@/lib/types/api";
import type { GeneratePredictionsResult } from "@/lib/api/predictions";

interface PredictionDashboardProps {
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
  predictions: Prediction[];
  isLoading: boolean;
  title: string;
  description: string;
  showRoi: boolean;
  _onRefresh?: () => void;
}

type ViewMode = "table" | "card";

export default function PredictionDashboard({
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
  predictions,
  isLoading,
  title,
  description,
  showRoi,
  _onRefresh,
}: PredictionDashboardProps) {
  const [filters, setFilters] = useState<PredictionFilterState>({});
  const [sortConfig, setSortConfig] = useState<{
    key: keyof Prediction | "match" | "probabilities";
    direction: "asc" | "desc";
  } | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);

  // Filter predictions based on filter state
  const filteredPredictions = useMemo(() => {
    let result = [...predictions];

    // Search filter
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      result = result.filter((p) => {
        const match = p.match as PredictionMatchSummary | null | undefined;
        return (
          match?.home_team_name?.toLowerCase().includes(searchLower) ||
          match?.away_team_name?.toLowerCase().includes(searchLower)
        );
      });
    }

    // Outcome filter
    if (filters.outcome) {
      result = result.filter((p) => p.predicted_outcome === filters.outcome);
    }

    // Min confidence filter
    if (filters.minConfidence !== undefined) {
      result = result.filter((p) => p.confidence >= (filters.minConfidence ?? 0));
    }

    // Date range filter
    if (filters.fromDate) {
      const fromDate = new Date(filters.fromDate);
      result = result.filter((p) => {
        const match = p.match as PredictionMatchSummary | null | undefined;
        return match && new Date(match.match_date) >= fromDate;
      });
    }

    if (filters.toDate) {
      const toDate = new Date(filters.toDate);
      toDate.setHours(23, 59, 59, 999);
      result = result.filter((p) => {
        const match = p.match as PredictionMatchSummary | null | undefined;
        return match && new Date(match.match_date) <= toDate;
      });
    }

    // Model version filter
    if (filters.modelVersionId !== undefined) {
      result = result.filter((p) => p.model_version_id === filters.modelVersionId);
    }

    // Value bets filter (only applies to historical predictions)
    if (filters.onlyValueBets) {
      result = result.filter(
        (p) => p.actual_roi !== null && p.actual_roi !== undefined && p.actual_roi > 0
      );
    }

    return result;
  }, [predictions, filters]);

  // Sort predictions
  const sortedPredictions = useMemo(() => {
    if (!sortConfig) return filteredPredictions;

    return [...filteredPredictions].sort((a, b) => {
      let aVal: unknown;
      let bVal: unknown;

      switch (sortConfig.key) {
        case "match": {
          const matchA = a.match as PredictionMatchSummary | null;
          const matchB = b.match as PredictionMatchSummary | null;
          aVal = matchA?.home_team_name || "";
          bVal = matchB?.home_team_name || "";
          break;
        }
        case "predicted_outcome":
          aVal = a.predicted_outcome;
          bVal = b.predicted_outcome;
          break;
        case "confidence":
          aVal = a.confidence;
          bVal = b.confidence;
          break;
        case "max_probability":
          aVal = a.max_probability;
          bVal = b.max_probability;
          break;
        case "actual_roi":
          aVal = a.actual_roi ?? (showRoi ? -1 : 0);
          bVal = b.actual_roi ?? (showRoi ? -1 : 0);
          break;
        case "probabilities":
          aVal = a.max_probability;
          bVal = b.max_probability;
          break;
        default: {
          aVal = (a as unknown as Record<string, unknown>)[sortConfig.key as string];
          bVal = (b as unknown as Record<string, unknown>)[sortConfig.key as string];
          break;
        }
      }

      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortConfig.direction === "asc"
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortConfig.direction === "asc" ? aVal - bVal : bVal - aVal;
      }

      return 0;
    });
  }, [filteredPredictions, sortConfig, showRoi]);

  const handleSort = (key: keyof Prediction | "match" | "probabilities") => {
    setSortConfig((current) => {
      if (!current || current.key !== key) {
        return { key, direction: "desc" };
      }
      if (current.direction === "desc") {
        return { key, direction: "asc" };
      }
      return null;
    });
  };

  const activeFilterCount = Object.values(filters).filter(
    (v) => v !== undefined && v !== ""
  ).length;

  const handleViewDetails = (prediction: Prediction) => {
    setSelectedPrediction(prediction);
    setDetailModalOpen(true);
  };

  const handleExportPrediction = (prediction: Prediction) => {
    const data = [prediction];
    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `prediction-${prediction.id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <PredictionControls
        activeModel={activeModel}
        models={models}
        selectedModel={selectedModel}
        selectedModelId={selectedModelId}
        daysAhead={daysAhead}
        generationResult={generationResult}
        isGenerating={isGenerating}
        isActivating={isActivating}
        onChangeDaysAhead={onChangeDaysAhead}
        onChangeSelectedModelId={onChangeSelectedModelId}
        onGenerate={onGenerate}
        onActivate={onActivate}
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <PredictionStats
            predictions={predictions}
            filteredPredictions={filteredPredictions}
          />

          <Card>
            <div className="space-y-4">
              <div className="flex flex-col gap-4 p-6">
                <div className="flex flex-col justify-between gap-4 sm:flex-row">
                  <PredictionFilters
                    filters={filters}
                    onFiltersChange={setFilters}
                    availableModels={models}
                    activeFilterCount={activeFilterCount}
                  />
                  <div className="flex items-center gap-2">
                    <ExportButton predictions={sortedPredictions} />
                    <div className="flex rounded-md border">
                      <Button
                        variant={viewMode === "table" ? "default" : "ghost"}
                        size="sm"
                        onClick={() => setViewMode("table")}
                        className="rounded-r-none"
                      >
                        <List className="h-4 w-4" />
                      </Button>
                      <Button
                        variant={viewMode === "card" ? "default" : "ghost"}
                        size="sm"
                        onClick={() => setViewMode("card")}
                        className="rounded-l-none"
                      >
                        <Grid className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              </div>

              {viewMode === "table" ? (
                <PredictionTable
                  predictions={sortedPredictions}
                  originalPredictions={predictions}
                  filteredPredictions={filteredPredictions}
                  showRoi={showRoi}
                  isLoading={isLoading}
                  title={title}
                  description={description}
                  sortConfig={sortConfig}
                  onSort={handleSort}
                  onViewDetails={handleViewDetails}
                />
              ) : (
                <div className="p-6">
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    {sortedPredictions.map((prediction) => (
                      <PredictionCard
                        key={prediction.id}
                        prediction={prediction}
                        showRoi={showRoi}
                        onViewDetails={handleViewDetails}
                      />
                    ))}
                  </div>
                  {sortedPredictions.length === 0 && !isLoading && (
                    <div className="py-8 text-center text-muted-foreground">
                      No predictions found
                    </div>
                  )}
                </div>
              )}
            </div>
          </Card>
        </div>

        {selectedModel && (
          <div className="space-y-6">
            <ModelPerformanceCard
              model={selectedModel}
              totalPredictions={predictions.length}
            />
          </div>
        )}
      </div>

      <PredictionDetailModal
        prediction={selectedPrediction}
        open={detailModalOpen}
        onOpenChange={setDetailModalOpen}
        onExport={handleExportPrediction}
      />
    </div>
  );
}
