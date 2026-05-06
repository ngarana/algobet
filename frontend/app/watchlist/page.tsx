"use client";

import { useDeferredValue, useState } from "react";
import { Star, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { useTeams } from "@/lib/queries/use-teams";
import { useTournaments } from "@/lib/queries/use-tournaments";
import {
  useAddWatchlistEntry,
  useRemoveWatchlistEntry,
  useWatchlist,
} from "@/lib/queries/use-workflow";
import type { WatchlistEntry } from "@/lib/types/api";

function WatchlistSection({
  title,
  description,
  entries,
}: {
  title: string;
  description: string;
  entries: WatchlistEntry[];
}) {
  const removeMutation = useRemoveWatchlistEntry();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {entries.length > 0 ? (
          entries.map((entry) => (
            <div
              key={entry.id}
              className="flex items-center justify-between rounded-md border p-3"
            >
              <div>
                <p className="font-medium">{entry.label}</p>
                {entry.meta && (
                  <p className="text-sm text-muted-foreground">{entry.meta}</p>
                )}
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() =>
                  removeMutation.mutate({
                    entryType: entry.entry_type,
                    entryId: entry.entry_id,
                  })
                }
              >
                <Trash2 className="h-4 w-4" />
                <span className="sr-only">Remove</span>
              </Button>
            </div>
          ))
        ) : (
          <p className="py-6 text-center text-sm text-muted-foreground">
            Nothing watched yet.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function AddWatchlistControls() {
  const [teamSearch, setTeamSearch] = useState("");
  const [leagueSearch, setLeagueSearch] = useState("");
  const deferredTeamSearch = useDeferredValue(teamSearch.trim());
  const deferredLeagueSearch = useDeferredValue(leagueSearch.trim());
  const { data: teams = [] } = useTeams({
    search: deferredTeamSearch || undefined,
    limit: 8,
  });
  const { data: tournaments = [] } = useTournaments({
    search: deferredLeagueSearch || undefined,
    limit: 8,
  });
  const addMutation = useAddWatchlistEntry();

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Add Team</CardTitle>
          <CardDescription>Search clubs and add them to watched fixtures.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Input
            placeholder="Search teams..."
            value={teamSearch}
            onChange={(event) => setTeamSearch(event.target.value)}
          />
          <div className="space-y-2">
            {teams.map((team) => (
              <div
                key={team.id}
                className="flex items-center justify-between rounded-md border p-3"
              >
                <span className="font-medium">{team.name}</span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    addMutation.mutate({ entry_type: "team", entry_id: team.id })
                  }
                >
                  Add
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Follow League</CardTitle>
          <CardDescription>Follow leagues to shape the daily dashboard.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Input
            placeholder="Search leagues..."
            value={leagueSearch}
            onChange={(event) => setLeagueSearch(event.target.value)}
          />
          <div className="space-y-2">
            {tournaments.map((tournament) => (
              <div
                key={tournament.id}
                className="flex items-center justify-between rounded-md border p-3"
              >
                <div>
                  <p className="font-medium">{tournament.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {tournament.country}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    addMutation.mutate({
                      entry_type: "tournament",
                      entry_id: tournament.id,
                    })
                  }
                >
                  Follow
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function WatchlistPage() {
  const { data, isLoading } = useWatchlist();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-72" />
        <div className="grid gap-6 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <Skeleton key={index} className="h-80" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="flex items-center gap-2 text-3xl font-bold tracking-tight">
          <Star className="h-8 w-8" />
          Watchlist
        </h1>
        <p className="text-muted-foreground">
          Follow leagues, teams, and matches that should shape your daily view.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <WatchlistSection
          title="Teams"
          description="Watched teams for upcoming fixtures."
          entries={data?.teams ?? []}
        />
        <WatchlistSection
          title="Leagues"
          description="Followed leagues for daily filtering."
          entries={data?.tournaments ?? []}
        />
        <WatchlistSection
          title="Matches"
          description="Individual matches you want to track."
          entries={data?.matches ?? []}
        />
      </div>

      <AddWatchlistControls />
    </div>
  );
}
