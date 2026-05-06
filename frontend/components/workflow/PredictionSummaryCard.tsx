"use client";

import Link from "next/link";
import { Calendar, Eye, TrendingUp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { Prediction, PredictedOutcome, ValueBet } from "@/lib/types/api";

const outcomeLabels: Record<PredictedOutcome, string> = {
  H: "Home",
  D: "Draw",
  A: "Away",
};

export function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "N/A";
  }
  return `${(value * 100).toFixed(0)}%`;
}

export function PredictionSummaryCard({
  prediction,
  valueBet,
}: {
  prediction: Prediction;
  valueBet?: ValueBet | null;
}) {
  const match = prediction.match;
  const home = match?.home_team_name ?? `Team ${prediction.match_id}`;
  const away = match?.away_team_name ?? "Opponent";
  const tournament = match?.tournament_name;
  const matchDate = match?.match_date ? new Date(match.match_date) : null;

  return (
    <Card className="h-full transition-shadow hover:shadow-md">
      <CardContent className="space-y-4 p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate font-semibold">
              {home} vs {away}
            </p>
            <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
              <Calendar className="h-3.5 w-3.5" />
              <span>{matchDate ? matchDate.toLocaleString() : "Time unavailable"}</span>
            </div>
          </div>
          {tournament && <Badge variant="outline">{tournament}</Badge>}
        </div>

        <div className="grid grid-cols-3 gap-2 rounded-md border p-3 text-center">
          <div>
            <p className="text-xs text-muted-foreground">Home</p>
            <p className="font-semibold text-blue-600">
              {formatPercent(prediction.prob_home)}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Draw</p>
            <p className="font-semibold text-amber-600">
              {formatPercent(prediction.prob_draw)}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Away</p>
            <p className="font-semibold text-red-600">
              {formatPercent(prediction.prob_away)}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge>{outcomeLabels[prediction.predicted_outcome]}</Badge>
          <Badge variant="secondary">
            Confidence {formatPercent(prediction.confidence)}
          </Badge>
          {valueBet && (
            <Badge className="gap-1 bg-emerald-600 text-white">
              <TrendingUp className="h-3 w-3" />
              EV +{(valueBet.expected_value * 100).toFixed(1)}%
            </Badge>
          )}
        </div>

        <Button asChild variant="outline" size="sm" className="w-full">
          <Link href={`/matches/${prediction.match_id}`}>
            <Eye className="mr-2 h-4 w-4" />
            View match
          </Link>
        </Button>
      </CardContent>
    </Card>
  );
}
