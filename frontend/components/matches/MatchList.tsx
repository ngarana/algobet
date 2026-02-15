"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { MatchCard } from "./MatchCard";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useMatches } from "@/lib/queries/use-matches";
import type { MatchFilters } from "@/lib/types/api";

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
      ? parseInt(searchParams.get("tournament_id")!)
      : undefined,
    team_id: searchParams.get("team_id")
      ? parseInt(searchParams.get("team_id")!)
      : undefined,
    days_ahead: searchParams.get("days_ahead")
      ? parseInt(searchParams.get("days_ahead")!)
      : undefined,
    limit,
    offset,
  };

  const { data, isLoading, isFetching, error } = useMatches(filters);

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

  const matches = data?.items || [];
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
