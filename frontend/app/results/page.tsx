"use client";

import Link from "next/link";
import { CheckCircle2, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatPercent } from "@/components/workflow";
import { useResultsReview } from "@/lib/queries/use-workflow";
import type { PredictedOutcome } from "@/lib/types/api";

const outcomeLabels: Record<PredictedOutcome, string> = {
  H: "Home",
  D: "Draw",
  A: "Away",
};

function ResultIcon({ value }: { value: boolean | null }) {
  if (value === null) {
    return <span className="text-muted-foreground">Pending</span>;
  }
  return value ? (
    <span className="inline-flex items-center gap-1 text-emerald-600">
      <CheckCircle2 className="h-4 w-4" />
      Correct
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-red-600">
      <XCircle className="h-4 w-4" />
      Miss
    </span>
  );
}

export default function ResultsPage() {
  const { data, isLoading } = useResultsReview();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-72" />
        <div className="grid gap-4 md:grid-cols-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <Skeleton key={index} className="h-28" />
          ))}
        </div>
        <Skeleton className="h-96" />
      </div>
    );
  }

  const summaries = data?.summaries ?? [];
  const items = data?.items ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Results Review</h1>
        <p className="text-muted-foreground">
          Compare final scores against model predictions and your picks.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {summaries.map((summary) => (
          <Card key={summary.label}>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">{summary.label}</CardTitle>
              <CardDescription>
                {summary.model_predictions} model predictions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Model</span>
                <span className="font-semibold">
                  {formatPercent(summary.model_accuracy)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Your picks</span>
                <span className="font-semibold">
                  {formatPercent(summary.user_accuracy)}
                </span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Finished Matches</CardTitle>
          <CardDescription>
            Weekly review of actual outcomes, model calls, and your picks.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {items.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Match</TableHead>
                  <TableHead>Score</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Your Pick</TableHead>
                  <TableHead>Model Result</TableHead>
                  <TableHead>Your Result</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((item) => (
                  <TableRow key={item.match.id}>
                    <TableCell>
                      <Link
                        href={`/matches/${item.match.id}`}
                        className="font-medium hover:underline"
                      >
                        {item.match.home_team_name ?? "Home"} vs{" "}
                        {item.match.away_team_name ?? "Away"}
                      </Link>
                      <p className="text-xs text-muted-foreground">
                        {item.match.tournament_name ?? "Tournament unavailable"}
                      </p>
                    </TableCell>
                    <TableCell className="font-mono">
                      {item.match.home_score} - {item.match.away_score}
                    </TableCell>
                    <TableCell>
                      {item.model_prediction ? (
                        <Badge variant="outline">
                          {outcomeLabels[item.model_prediction.predicted_outcome]}
                        </Badge>
                      ) : (
                        "N/A"
                      )}
                    </TableCell>
                    <TableCell>
                      {item.user_prediction?.pick_1x2 ? (
                        <Badge variant="secondary">
                          {outcomeLabels[item.user_prediction.pick_1x2]}
                        </Badge>
                      ) : (
                        "N/A"
                      )}
                    </TableCell>
                    <TableCell>
                      <ResultIcon value={item.model_correct} />
                    </TableCell>
                    <TableCell>
                      <ResultIcon value={item.user_correct} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="py-10 text-center text-muted-foreground">
              No finished matches are available for review yet.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
