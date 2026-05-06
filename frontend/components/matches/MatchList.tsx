"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { MatchCard } from "./MatchCard";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useMatches } from "@/lib/queries/use-matches";
import { useUpcomingPredictions } from "@/lib/queries/use-predictions";
import { useWatchlist } from "@/lib/queries/use-workflow";
import type { Match, MatchFilters, Prediction } from "@/lib/types/api";

interface MatchListProps {
  initialFilters?: MatchFilters;
}

export function MatchList({ initialFilters }: MatchListProps) {
  const searchParams = useSearchParams();
  const [offset, setOffset] = useState(0);
  const limit = 20;

  // Build filters from URL params
  const filters: MatchFilters = {
    ...initialFilters,
    status: (searchParams.get("status") as MatchFilters["status"]) || undefined,
    tournament_id: searchParams.get("tournament_id")
      ? parseInt(searchParams.get("tournament_id") ?? "0")
      : undefined,
    team_id: searchParams.get("team_id")
      ? parseInt(searchParams.get("team_id") ?? "0")
      : undefined,
    days_ahead: searchParams.get("days_ahead")
      ? parseInt(searchParams.get("days_ahead") ?? "0")
      : undefined,
    from_date: searchParams.get("from_date") || undefined,
    to_date: searchParams.get("to_date") || undefined,
    limit,
    offset,
  };

  const { data, isLoading, isFetching, error } = useMatches(filters);
  const sort = searchParams.get("sort") || "kickoff";
  const daysAhead = searchParams.get("days_ahead")
    ? parseInt(searchParams.get("days_ahead") ?? "7")
    : 30;
  const { data: predictionsData } = useUpcomingPredictions(daysAhead);
  const { data: watchlist } = useWatchlist();

  // Reset offset when filters change
  useEffect(() => {
    setOffset(0);
  }, [searchParams]);

  const loadMore = () => {
    setOffset((prev) => prev + limit);
  };

  if (isLoading) {
    return <MatchListSkeleton />;
  }

  if (error) {
    return (
      <div className="py-8 text-center">
        <p className="text-destructive">Failed to load matches</p>
        <p className="mt-2 text-sm text-muted-foreground">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
      </div>
    );
  }

  const matches = sortMatches(
    data?.items || [],
    predictionsData?.items ?? [],
    watchlist?.tournaments.map((entry) => entry.entry_id) ?? [],
    sort
  );
  const total = data?.total || 0;
  const hasMore = matches.length < total;

  if (matches.length === 0) {
    return (
      <div className="py-12 text-center">
        <p className="text-lg font-medium">No matches found</p>
        <p className="mt-2 text-muted-foreground">
          Try adjusting your filters to see more results
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Showing {matches.length} of {total} matches
        </p>
      </div>

      <div className="grid gap-4">
        {matches.map((match) => (
          <MatchCard key={match.id} match={match} />
        ))}
      </div>

      {hasMore && (
        <div className="flex justify-center pt-4">
          <Button onClick={loadMore} disabled={isFetching} variant="outline" size="lg">
            {isFetching ? "Loading..." : "Load More"}
          </Button>
        </div>
      )}
    </div>
  );
}

function sortMatches(
  matches: Match[],
  predictions: Prediction[],
  favoriteTournamentIds: number[],
  sort: string
) {
  const predictionByMatchId = new Map(
    predictions.map((prediction) => [prediction.match_id, prediction])
  );

  return [...matches].sort((a, b) => {
    if (sort === "confidence") {
      return (
        (predictionByMatchId.get(b.id)?.confidence ?? 0) -
        (predictionByMatchId.get(a.id)?.confidence ?? 0)
      );
    }

    if (sort === "value") {
      return (
        valueEdge(b, predictionByMatchId.get(b.id)) -
        valueEdge(a, predictionByMatchId.get(a.id))
      );
    }

    if (sort === "favorites") {
      const aFav =
        a.tournament_id !== null && favoriteTournamentIds.includes(a.tournament_id)
          ? 1
          : 0;
      const bFav =
        b.tournament_id !== null && favoriteTournamentIds.includes(b.tournament_id)
          ? 1
          : 0;
      if (aFav !== bFav) {
        return bFav - aFav;
      }
    }

    return new Date(a.match_date).getTime() - new Date(b.match_date).getTime();
  });
}

function valueEdge(match: Match, prediction?: Prediction) {
  if (!prediction) {
    return Number.NEGATIVE_INFINITY;
  }

  if (prediction.predicted_outcome === "H" && match.odds_home) {
    return prediction.prob_home * match.odds_home - 1;
  }
  if (prediction.predicted_outcome === "D" && match.odds_draw) {
    return prediction.prob_draw * match.odds_draw - 1;
  }
  if (prediction.predicted_outcome === "A" && match.odds_away) {
    return prediction.prob_away * match.odds_away - 1;
  }
  return Number.NEGATIVE_INFINITY;
}

export function MatchListSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-32" />
      </div>

      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="space-y-3 rounded-lg border p-4">
          <div className="flex items-center justify-between">
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-4 w-24" />
          </div>
          <div className="flex items-center justify-between gap-4">
            <div className="flex-1 space-y-1 text-right">
              <Skeleton className="ml-auto h-5 w-24" />
              <Skeleton className="ml-auto h-4 w-16" />
            </div>
            <Skeleton className="h-8 w-16" />
            <div className="flex-1 space-y-1 text-left">
              <Skeleton className="h-5 w-24" />
              <Skeleton className="h-4 w-16" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
