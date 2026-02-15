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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Brain, RefreshCw, TrendingUp, Target, Calendar } from "lucide-react";
import {
  usePredictions,
  useUpcomingPredictions,
  usePredictionHistory,
} from "@/lib/queries/use-predictions";
import { useActiveModel } from "@/lib/queries/use-models";
import type { Prediction } from "@/lib/types/api";

const outcomeLabels: Record<string, string> = {
  H: "Home Win",
  D: "Draw",
  A: "Away Win",
};

const outcomeColors: Record<string, string> = {
  H: "bg-blue-500",
  D: "bg-yellow-500",
  A: "bg-red-500",
};

function PredictionCard({ prediction }: { prediction: Prediction }) {
  return (
    <TableRow>
      <TableCell className="font-medium">Match #{prediction.match_id}</TableCell>
      <TableCell>
        <Badge className={outcomeColors[prediction.predicted_outcome]}>
          {outcomeLabels[prediction.predicted_outcome]}
        </Badge>
      </TableCell>
      <TableCell className="font-mono">
        {(prediction.prob_home * 100).toFixed(1)}% /{" "}
        {(prediction.prob_draw * 100).toFixed(1)}% /{" "}
        {(prediction.prob_away * 100).toFixed(1)}%
      </TableCell>
      <TableCell className="font-mono">
        {(prediction.confidence * 100).toFixed(1)}%
      </TableCell>
      <TableCell className="text-sm text-muted-foreground">
        {new Date(prediction.predicted_at).toLocaleDateString()}
      </TableCell>
      {prediction.actual_roi !== null && (
        <TableCell
          className={`font-mono ${prediction.actual_roi >= 0 ? "text-green-600" : "text-red-600"}`}
        >
          {prediction.actual_roi >= 0 ? "+" : ""}
          {(prediction.actual_roi * 100).toFixed(1)}%
        </TableCell>
      )}
    </TableRow>
  );
}

function PredictionsSummary({ predictions }: { predictions: Prediction[] }) {
  const avgConfidence =
    predictions.length > 0
      ? predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length
      : 0;

  const outcomeCounts = predictions.reduce(
    (acc, p) => {
      acc[p.predicted_outcome] = (acc[p.predicted_outcome] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  return (
    <div className="grid grid-cols-4 gap-4">
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold">{predictions.length}</div>
          <div className="text-xs text-muted-foreground">Total Predictions</div>
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
          <div className="text-2xl font-bold text-blue-600">
            {outcomeCounts["H"] || 0}
          </div>
          <div className="text-xs text-muted-foreground">Home Predictions</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-red-600">
            {outcomeCounts["A"] || 0}
          </div>
          <div className="text-xs text-muted-foreground">Away Predictions</div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function PredictionsPage() {
  const { data: activeModel } = useActiveModel();
  const [view, setView] = useState<"upcoming" | "history">("upcoming");

  const {
    data: upcomingData,
    isLoading: upcomingLoading,
    refetch: refetchUpcoming,
  } = useUpcomingPredictions(7);
  const {
    data: historyData,
    isLoading: historyLoading,
    refetch: refetchHistory,
  } = usePredictionHistory();

  const predictions =
    view === "upcoming" ? (upcomingData?.items ?? []) : (historyData?.items ?? []);

  const isLoading = view === "upcoming" ? upcomingLoading : historyLoading;
  const refetch = view === "upcoming" ? refetchUpcoming : refetchHistory;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-3xl font-bold tracking-tight">
            <Brain className="h-8 w-8" />
            Predictions
          </h1>
          <p className="text-muted-foreground">
            View model predictions and historical accuracy
          </p>
        </div>
        <div className="flex items-center gap-2">
          {activeModel && (
            <Badge variant="secondary" className="flex items-center gap-1">
              <Target className="h-3 w-3" />
              {activeModel.algorithm} v{activeModel.version}
            </Badge>
          )}
          <Select
            value={view}
            onValueChange={(v) => setView(v as "upcoming" | "history")}
          >
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="upcoming">Upcoming</SelectItem>
              <SelectItem value="history">History</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-20" />
            ))}
          </div>
          <Skeleton className="h-64" />
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
                  {predictions.length} predictions from the active model
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Match</TableHead>
                      <TableHead>Prediction</TableHead>
                      <TableHead>Probabilities (H/D/A)</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Date</TableHead>
                      {view === "history" && <TableHead>ROI</TableHead>}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {predictions.map((prediction) => (
                      <PredictionCard key={prediction.id} prediction={prediction} />
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
                    ? "Generate predictions for upcoming matches first"
                    : "No historical predictions available"}
                </p>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
