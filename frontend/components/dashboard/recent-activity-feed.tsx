"use client";

import { useMemo } from "react";
import { usePredictions } from "@/lib/queries/use-predictions";
import { useValueBets } from "@/lib/queries/use-value-bets";
import { useModels } from "@/lib/queries/use-models";
import type { Prediction, ValueBet, ModelVersion } from "@/lib/types/api";

interface ActivityItem {
  id: string;
  type: "prediction" | "value_bet" | "model_update" | "profit";
  title: string;
  description: string;
  timestamp: Date;
  color: string;
}

function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? "s" : ""} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
  if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? "s" : ""} ago`;
  return date.toLocaleDateString();
}

function getOutcomeLabel(outcome: string): string {
  switch (outcome) {
    case "H":
      return "Home Win";
    case "D":
      return "Draw";
    case "A":
      return "Away Win";
    default:
      return outcome;
  }
}

export function RecentActivityFeed() {
  const { data: predictionsData, isLoading: predictionsLoading } = usePredictions();
  const { data: valueBetsData, isLoading: valueBetsLoading } = useValueBets({
    max_matches: 50,
  });
  const { data: modelsData, isLoading: modelsLoading } = useModels();

  const isLoading = predictionsLoading || valueBetsLoading || modelsLoading;

  const activities = useMemo(() => {
    const items: ActivityItem[] = [];
    const predictions = predictionsData?.items || [];
    const valueBets = valueBetsData || [];
    const models = modelsData?.items || [];

    if (valueBets.length > 0) {
      const recentValueBets = valueBets.slice(0, 5);
      for (const vb of recentValueBets) {
        items.push({
          id: `vb-${vb.prediction_id}-${vb.predicted_outcome}`,
          type: "value_bet",
          title: "Value Bet Found",
          description: `${vb.match.home_team_id} vs ${vb.match.away_team_id} - ${getOutcomeLabel(vb.predicted_outcome)} @ ${vb.market_odds.toFixed(2)} (EV: +${(vb.expected_value * 100).toFixed(1)}%)`,
          timestamp: new Date(vb.match.match_date),
          color: "bg-green-500",
        });
      }
    }

    if (predictions.length > 0) {
      const recentPredictions = predictions.slice(0, 5);
      for (const pred of recentPredictions) {
        items.push({
          id: `pred-${pred.id}`,
          type: "prediction",
          title: "Prediction Generated",
          description: `Match #${pred.match_id} - ${getOutcomeLabel(pred.predicted_outcome)} (${(pred.confidence * 100).toFixed(0)}% confidence)`,
          timestamp: new Date(pred.predicted_at),
          color: "bg-purple-500",
        });
      }
    }

    const profitable = predictions
      .filter((p: Prediction) => p.actual_roi !== null && p.actual_roi > 0)
      .slice(0, 3);
    for (const pred of profitable) {
      items.push({
        id: `profit-${pred.id}`,
        type: "profit",
        title: "Profit Recorded",
        description: `+$${pred.actual_roi!.toFixed(2)} from Match #${pred.match_id}`,
        timestamp: new Date(pred.predicted_at),
        color: "bg-yellow-500",
      });
    }

    if (models.length > 0) {
      const recentModels = [...models]
        .sort(
          (a: ModelVersion, b: ModelVersion) =>
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        )
        .slice(0, 2);
      for (const model of recentModels) {
        items.push({
          id: `model-${model.id}`,
          type: "model_update",
          title: model.is_active ? "Model Activated" : "Model Created",
          description: `${model.name} (${model.algorithm})${model.accuracy ? ` - ${(model.accuracy * 100).toFixed(1)}% accuracy` : ""}`,
          timestamp: new Date(model.created_at),
          color: "bg-blue-500",
        });
      }
    }

    return items
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 10);
  }, [predictionsData, valueBetsData, modelsData]);

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="h-2 w-2 animate-pulse rounded-full bg-gray-300"></div>
            <div className="flex-1 space-y-2">
              <div className="h-4 w-3/4 animate-pulse rounded bg-gray-200"></div>
              <div className="h-3 w-1/2 animate-pulse rounded bg-gray-200"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (activities.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No recent activity to display.</p>
    );
  }

  return (
    <div className="space-y-3">
      {activities.map((activity) => (
        <div key={activity.id} className="flex items-start gap-3">
          <div className={`mt-1.5 h-2 w-2 rounded-full ${activity.color}`}></div>
          <div className="min-w-0 flex-1">
            <p className="text-sm">
              <span className="font-medium">{activity.title}:</span>{" "}
              {activity.description}
            </p>
            <p className="text-xs text-muted-foreground">
              {formatRelativeTime(activity.timestamp)}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
