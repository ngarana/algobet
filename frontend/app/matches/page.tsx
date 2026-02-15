"use client";

import { Suspense } from "react";
import { MatchFilters } from "@/components/matches/MatchFilters";
import { MatchList, MatchListSkeleton } from "@/components/matches/MatchList";

export default function MatchesPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Matches</h1>
        <p className="text-muted-foreground">
          Browse and filter football matches from all tournaments
        </p>
      </div>

      <MatchFilters />

      <Suspense fallback={<MatchListSkeleton />}>
        <MatchList />
      </Suspense>
    </div>
  );
}
