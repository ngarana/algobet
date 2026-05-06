"use client";

import { useParams } from "next/navigation";
import { Calendar } from "lucide-react";

import { UserPredictionPanel, WatchlistToggle, formatPercent } from "@/components/workflow";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useMatchWorkflowDetail } from "@/lib/queries/use-workflow";
import type { PredictedOutcome } from "@/lib/types/api";

const outcomeLabels: Record<PredictedOutcome, string> = {
  H: "Home win",
  D: "Draw",
  A: "Away win",
};

function LoadingState() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-12 w-72" />
      <Skeleton className="h-64" />
      <Skeleton className="h-96" />
    </div>
  );
}

export default function MatchDetailPage() {
  const params = useParams();
  const matchId = Number(params.id);
  const { data, isLoading, error } = useMatchWorkflowDetail(matchId);

  if (isLoading) {
    return <LoadingState />;
  }

  if (error || !data) {
    return (
      <Card className="border-destructive">
        <CardContent className="p-6 text-destructive">
          Failed to load match workflow detail.
        </CardContent>
      </Card>
    );
  }

  const match = data.match;
  const latestPrediction = match.predictions[0] ?? null;

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="space-y-6 p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <Badge>{match.status}</Badge>
                {match.tournament && <Badge variant="outline">{match.tournament.name}</Badge>}
              </div>
              <h1 className="text-3xl font-bold tracking-tight">
                {match.home_team.name} vs {match.away_team.name}
              </h1>
              <p className="flex items-center gap-2 text-muted-foreground">
                <Calendar className="h-4 w-4" />
                {new Date(match.match_date).toLocaleString()}
              </p>
            </div>
            <WatchlistToggle
              entryType="match"
              entryId={match.id}
              watched={data.watched}
              label={data.watched ? "Watching" : "Watch match"}
            />
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-md border p-4 text-center">
              <p className="text-sm text-muted-foreground">Home</p>
              <p className="text-2xl font-semibold">{match.home_team.name}</p>
              {match.odds_home && (
                <p className="text-sm text-muted-foreground">
                  Odds {match.odds_home.toFixed(2)}
                </p>
              )}
            </div>
            <div className="rounded-md border p-4 text-center">
              <p className="text-sm text-muted-foreground">Score</p>
              <p className="text-3xl font-bold">
                {match.status === "FINISHED"
                  ? `${match.home_score} - ${match.away_score}`
                  : "vs"}
              </p>
            </div>
            <div className="rounded-md border p-4 text-center">
              <p className="text-sm text-muted-foreground">Away</p>
              <p className="text-2xl font-semibold">{match.away_team.name}</p>
              {match.odds_away && (
                <p className="text-sm text-muted-foreground">
                  Odds {match.odds_away.toFixed(2)}
                </p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="prediction" className="w-full">
        <TabsList className="grid w-full grid-cols-2 lg:grid-cols-6">
          <TabsTrigger value="prediction">Prediction</TabsTrigger>
          <TabsTrigger value="form">Form</TabsTrigger>
          <TabsTrigger value="stats">Stats</TabsTrigger>
          <TabsTrigger value="odds">Odds</TabsTrigger>
          <TabsTrigger value="explain">Explain</TabsTrigger>
          <TabsTrigger value="h2h">H2H</TabsTrigger>
        </TabsList>

        <TabsContent value="prediction" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Prediction</CardTitle>
              <CardDescription>
                Latest model probabilities and similar-match accuracy.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {latestPrediction ? (
                <>
                  <div className="grid gap-3 md:grid-cols-3">
                    <ProbabilityBox label="Home" value={latestPrediction.prob_home} />
                    <ProbabilityBox label="Draw" value={latestPrediction.prob_draw} />
                    <ProbabilityBox label="Away" value={latestPrediction.prob_away} />
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Badge>
                      Pick {outcomeLabels[latestPrediction.predicted_outcome]}
                    </Badge>
                    <Badge variant="secondary">
                      Confidence {formatPercent(latestPrediction.confidence)}
                    </Badge>
                    <Badge variant="outline">
                      Similar accuracy {formatPercent(data.similar_accuracy.accuracy)}
                    </Badge>
                  </div>
                </>
              ) : (
                <p className="text-muted-foreground">
                  No model prediction is available for this match.
                </p>
              )}
            </CardContent>
          </Card>

          <UserPredictionPanel
            matchId={match.id}
            userPrediction={data.user_prediction}
          />
        </TabsContent>

        <TabsContent value="form" className="mt-4 grid gap-4 lg:grid-cols-2">
          <RecentFormCard title={match.home_team.name} rows={data.recent_form.home} />
          <RecentFormCard title={match.away_team.name} rows={data.recent_form.away} />
        </TabsContent>

        <TabsContent value="stats" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Key Stats Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    <TableHead>{data.stats_comparison.home.team_name}</TableHead>
                    <TableHead>{data.stats_comparison.away.team_name}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <StatsRow
                    label="Avg goals for"
                    home={data.stats_comparison.home.avg_goals_for}
                    away={data.stats_comparison.away.avg_goals_for}
                  />
                  <StatsRow
                    label="Avg goals against"
                    home={data.stats_comparison.home.avg_goals_against}
                    away={data.stats_comparison.away.avg_goals_against}
                  />
                  <StatsRow
                    label="Avg shots"
                    home={data.stats_comparison.home.avg_shots}
                    away={data.stats_comparison.away.avg_shots}
                  />
                  <StatsRow
                    label="Avg shots on target"
                    home={data.stats_comparison.home.avg_shots_on_target}
                    away={data.stats_comparison.away.avg_shots_on_target}
                  />
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="odds" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Odds Comparison</CardTitle>
              <CardDescription>
                Available bookmaker rows, or the market aggregate when only match odds
                exist.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.odds_comparison.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Bookmaker</TableHead>
                      <TableHead>Home</TableHead>
                      <TableHead>Draw</TableHead>
                      <TableHead>Away</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.odds_comparison.map((row) => (
                      <TableRow key={`${row.bookmaker}-${row.scraped_at ?? "market"}`}>
                        <TableCell>{row.bookmaker}</TableCell>
                        <TableCell>{row.odds_home.toFixed(2)}</TableCell>
                        <TableCell>{row.odds_draw.toFixed(2)}</TableCell>
                        <TableCell>{row.odds_away.toFixed(2)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <p className="text-muted-foreground">No odds are available.</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="explain" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Explanation</CardTitle>
              <CardDescription>
                Top available cached features for this match.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {data.model_explanation.length > 0 ? (
                data.model_explanation.map((item) => (
                  <div
                    key={item.feature}
                    className="flex items-center justify-between rounded-md border p-3"
                  >
                    <div>
                      <p className="font-medium">{item.label}</p>
                      <p className="text-sm text-muted-foreground">{item.direction}</p>
                    </div>
                    <Badge variant="outline">{item.value.toFixed(2)}</Badge>
                  </div>
                ))
              ) : (
                <p className="text-muted-foreground">
                  No cached model features are available for explanation.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="h2h" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Head To Head</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {match.h2h_matches.length > 0 ? (
                match.h2h_matches.map((row) => (
                  <div
                    key={row.id}
                    className="flex items-center justify-between rounded-md border p-3"
                  >
                    <span>{new Date(row.match_date).toLocaleDateString()}</span>
                    <span className="font-mono">
                      {row.home_score} - {row.away_score}
                    </span>
                    {row.result && <Badge variant="outline">{row.result}</Badge>}
                  </div>
                ))
              ) : (
                <p className="text-muted-foreground">No previous meetings found.</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function ProbabilityBox({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border p-4 text-center">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="text-2xl font-bold">{formatPercent(value)}</p>
    </div>
  );
}

function RecentFormCard({
  title,
  rows,
}: {
  title: string;
  rows: { match_id: number; match_date: string; opponent_name: string; result: string }[];
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title} Recent Form</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {rows.length > 0 ? (
          rows.map((row) => (
            <div
              key={row.match_id}
              className="flex items-center justify-between rounded-md border p-3"
            >
              <div>
                <p className="font-medium">vs {row.opponent_name}</p>
                <p className="text-sm text-muted-foreground">
                  {new Date(row.match_date).toLocaleDateString()}
                </p>
              </div>
              <Badge variant={row.result === "W" ? "success" : "secondary"}>
                {row.result}
              </Badge>
            </div>
          ))
        ) : (
          <p className="text-muted-foreground">No recent form available.</p>
        )}
      </CardContent>
    </Card>
  );
}

function StatsRow({
  label,
  home,
  away,
}: {
  label: string;
  home: number | null;
  away: number | null;
}) {
  return (
    <TableRow>
      <TableCell className="font-medium">{label}</TableCell>
      <TableCell>{home !== null ? home.toFixed(2) : "N/A"}</TableCell>
      <TableCell>{away !== null ? away.toFixed(2) : "N/A"}</TableCell>
    </TableRow>
  );
}
