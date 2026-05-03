"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle } from "lucide-react";
import type { TrainingConfig } from "./types";

interface TrainingSummaryProps {
  config: TrainingConfig;
  isTraining: boolean;
}

export function TrainingSummary({ config, isTraining }: TrainingSummaryProps) {
  const totalRatio = config.trainRatio + config.valRatio + config.testRatio;
  const isValid = Math.abs(totalRatio - 1.0) <= 0.001;

  const activeFilters = [
    config.startDate || config.endDate ? "Date" : null,
    config.minMatches > 100 ? "Min matches" : null,
    config.tournamentIds.length > 0 ? "Tournaments" : null,
    config.teamIds.length > 0 ? "Teams" : null,
    config.venueFilter !== "both" ? "Venue" : null,
    config.requireOdds ? "Odds" : null,
    config.minTotalGoals !== null || config.maxTotalGoals !== null ? "Goals" : null,
  ].filter(Boolean).length;

  const strategyNames: Record<string, string> = {
    temporal: "Temporal",
    expanding_window: "Expanding Window",
    season_aware: "Season-Aware",
  };

  return (
    <Card className="bg-muted/30">
      <CardContent className="p-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Type:</span>
            <Badge variant="outline">{config.modelType}</Badge>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Data:</span>
            <Badge variant="outline">
              {activeFilters > 0 ? `${activeFilters} filters` : "All"}
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Split:</span>
            <Badge variant="outline">{strategyNames[config.splitStrategy]}</Badge>
          </div>

          <div className="flex items-center gap-2">
            {config.tune && <Badge>Auto-tuning</Badge>}
            {config.useEnsemble && <Badge>Ensemble</Badge>}
            {config.calibrateProbabilities && (
              <Badge variant="secondary">Calibrated</Badge>
            )}
            {isTraining && <Badge>Training...</Badge>}
          </div>

          {!isValid && (
            <div className="flex items-center gap-1 text-destructive">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-xs">Invalid split ratios</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
