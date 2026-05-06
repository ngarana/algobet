"use client";

import Link from "next/link";
import {
  BarChart3,
  CalendarDays,
  Eye,
  RefreshCw,
  Star,
  Target,
  TrendingUp,
} from "lucide-react";

import { PredictionSummaryCard, formatPercent } from "@/components/workflow";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useDailyWorkflow } from "@/lib/queries/use-workflow";
import type { Prediction, ValueBet } from "@/lib/types/api";

function MetricCard({
  title,
  value,
  detail,
  icon: Icon,
}: {
  title: string;
  value: string;
  detail: string;
  icon: typeof CalendarDays;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <p className="text-xs text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  );
}

function matchValueBet(
  prediction: Prediction,
  valueBets: ValueBet[]
): ValueBet | null {
  return (
    valueBets.find((valueBet) => valueBet.prediction_id === prediction.id) ?? null
  );
}

export default function DashboardPage() {
  const { data, isLoading, isRefetching, refetch, error } = useDailyWorkflow();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-80" />
        <div className="grid gap-4 md:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-28" />
          ))}
        </div>
        <Skeleton className="h-80" />
      </div>
    );
  }

  const todayMatches = data?.today_matches ?? [];
  const highConfidence = data?.high_confidence ?? [];
  const valueBets = data?.value_bets ?? [];
  const watchedFixtures = data?.watched_fixtures ?? [];
  const summary = data?.results_summary;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Daily Dashboard</h1>
          <p className="text-muted-foreground">
            Fresh predictions, watched fixtures, and model review for today.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => void refetch()}
          disabled={isRefetching}
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${isRefetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="p-4 text-sm text-destructive">
            Failed to load daily workflow.
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-4">
        <MetricCard
          title="Today's Matches"
          value={String(todayMatches.length)}
          detail="Predictions available"
          icon={CalendarDays}
        />
        <MetricCard
          title="High Confidence"
          value={String(highConfidence.length)}
          detail="Above your threshold"
          icon={Target}
        />
        <MetricCard
          title="Value Bets"
          value={String(valueBets.length)}
          detail="Positive expected value"
          icon={TrendingUp}
        />
        <MetricCard
          title="Today Accuracy"
          value={formatPercent(summary?.model_accuracy)}
          detail={`${summary?.model_correct ?? 0}/${summary?.model_predictions ?? 0} model picks`}
          icon={BarChart3}
        />
      </div>

      <section className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold">Today's Matches</h2>
            <p className="text-sm text-muted-foreground">
              Sorted by kickoff with model probabilities and value signals.
            </p>
          </div>
          <Button asChild variant="outline" size="sm">
            <Link href="/matches">Browse all</Link>
          </Button>
        </div>

        {todayMatches.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {todayMatches.slice(0, 9).map((prediction) => (
              <PredictionSummaryCard
                key={prediction.id}
                prediction={prediction}
                valueBet={matchValueBet(prediction, valueBets)}
              />
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="py-10 text-center text-muted-foreground">
              No predictions are available for today.
            </CardContent>
          </Card>
        )}
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <TrendingUp className="h-5 w-5" />
              Best Value Bets Today
            </CardTitle>
            <CardDescription>
              Kelly suggestions are informational, not betting advice.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {valueBets.length > 0 ? (
              valueBets.slice(0, 6).map((valueBet) => (
                <Link
                  key={valueBet.prediction_id}
                  href={`/matches/${valueBet.match.id}`}
                  className="flex items-center justify-between rounded-md border p-3 transition-colors hover:bg-muted/50"
                >
                  <div>
                    <p className="font-medium">
                      {valueBet.match.home_team_name ?? "Home"} vs{" "}
                      {valueBet.match.away_team_name ?? "Away"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Odds {valueBet.market_odds.toFixed(2)} | Kelly{" "}
                      {(valueBet.kelly_fraction * 100).toFixed(1)}%
                    </p>
                  </div>
                  <Badge className="bg-emerald-600 text-white">
                    +{(valueBet.expected_value * 100).toFixed(1)}%
                  </Badge>
                </Link>
              ))
            ) : (
              <p className="py-6 text-center text-sm text-muted-foreground">
                No value bets meet your threshold today.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Star className="h-5 w-5" />
              Watchlist
            </CardTitle>
            <CardDescription>Upcoming fixtures for teams and leagues you follow.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {watchedFixtures.length > 0 ? (
              watchedFixtures.slice(0, 6).map((match) => (
                <Link
                  key={match.id}
                  href={`/matches/${match.id}`}
                  className="flex items-center justify-between rounded-md border p-3 transition-colors hover:bg-muted/50"
                >
                  <div>
                    <p className="font-medium">
                      {match.home_team.name} vs {match.away_team.name}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {new Date(match.match_date).toLocaleString()}
                    </p>
                  </div>
                  <Eye className="h-4 w-4 text-muted-foreground" />
                </Link>
              ))
            ) : (
              <div className="space-y-3 py-6 text-center">
                <p className="text-sm text-muted-foreground">
                  Add teams, leagues, or matches to build your daily watchlist.
                </p>
                <Button asChild variant="outline" size="sm">
                  <Link href="/watchlist">Manage watchlist</Link>
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
