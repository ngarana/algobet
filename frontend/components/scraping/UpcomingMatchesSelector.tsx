"use client";

import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useUpcomingMatchesWithTeams } from "@/lib/queries/use-matches";
import type { MatchDetail } from "@/lib/types/api";
import { CalendarIcon, CheckCircle2Icon, GlobeIcon, RefreshCwIcon } from "lucide-react";

const MAX_VISIBLE_SOURCES = 6;
const MAX_PREVIEW_MATCHES = 3;

export interface UpcomingSourceSelection {
  key: string;
  tournamentId: number | null;
  tournamentName: string;
  tournamentUrl: string;
  totalMatches: number;
  previewMatches: MatchDetail[];
  firstKickoff: string | null;
}

interface UpcomingMatchesSelectorProps {
  onSelectionChange: (selection: UpcomingSourceSelection | null) => void;
}

function buildTournamentUrl(match: MatchDetail) {
  const slug = match.tournament?.url_slug;
  return slug ? `https://www.oddsportal.com/football/${slug}/` : null;
}

function formatKickoff(dateStr: string) {
  return new Date(dateStr).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function UpcomingMatchesSelector({
  onSelectionChange,
}: UpcomingMatchesSelectorProps) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const { data, isLoading, isRefetching, refetch } = useUpcomingMatchesWithTeams();
  const matches = data?.items ?? [];

  const tournamentSources = useMemo(() => {
    const grouped = new Map<string, UpcomingSourceSelection>();

    for (const match of matches) {
      const tournamentUrl = buildTournamentUrl(match);
      if (!tournamentUrl) {
        continue;
      }

      const key = `${match.tournament_id ?? "unknown"}:${tournamentUrl}`;
      const existing = grouped.get(key);

      if (existing) {
        existing.totalMatches += 1;
        existing.previewMatches = [...existing.previewMatches, match]
          .sort(
            (left, right) =>
              new Date(left.match_date).getTime() - new Date(right.match_date).getTime()
          )
          .slice(0, MAX_PREVIEW_MATCHES);
        if (
          existing.firstKickoff === null ||
          new Date(match.match_date).getTime() <
            new Date(existing.firstKickoff).getTime()
        ) {
          existing.firstKickoff = match.match_date;
        }
        continue;
      }

      grouped.set(key, {
        key,
        tournamentId: match.tournament_id ?? null,
        tournamentName: match.tournament?.name ?? "Unknown tournament",
        tournamentUrl,
        totalMatches: 1,
        previewMatches: [match],
        firstKickoff: match.match_date,
      });
    }

    return Array.from(grouped.values())
      .sort((left, right) => {
        if (right.totalMatches !== left.totalMatches) {
          return right.totalMatches - left.totalMatches;
        }

        if (!left.firstKickoff || !right.firstKickoff) {
          return 0;
        }

        return (
          new Date(left.firstKickoff).getTime() - new Date(right.firstKickoff).getTime()
        );
      })
      .slice(0, MAX_VISIBLE_SOURCES);
  }, [matches]);

  const selectedSource = useMemo(
    () => tournamentSources.find((source) => source.key === selectedKey) ?? null,
    [selectedKey, tournamentSources]
  );

  useEffect(() => {
    if (!selectedKey) {
      onSelectionChange(null);
      return;
    }

    onSelectionChange(selectedSource);
  }, [onSelectionChange, selectedKey, selectedSource]);

  useEffect(() => {
    if (!selectedKey) {
      return;
    }

    const stillExists = tournamentSources.some((source) => source.key === selectedKey);
    if (!stillExists) {
      setSelectedKey(null);
    }
  }, [selectedKey, tournamentSources]);

  if (isLoading) {
    return (
      <div className="grid gap-3 md:grid-cols-2">
        {[...Array(4)].map((_, index) => (
          <Card key={index} className="border-border/60 bg-background/60">
            <CardContent className="space-y-4 p-4">
              <Skeleton className="h-5 w-28" />
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-16 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/70 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <p className="text-sm font-semibold text-foreground">
            Guided Tournament Sources
          </p>
          <p className="text-sm text-muted-foreground">
            Pick one live tournament source from the local upcoming matches feed.
          </p>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isRefetching}
          className="self-start sm:self-auto"
        >
          <RefreshCwIcon
            className={`mr-2 h-4 w-4 ${isRefetching ? "animate-spin" : ""}`}
          />
          Refresh feed
        </Button>
      </div>

      {tournamentSources.length === 0 ? (
        <Card className="border-dashed border-border/70 bg-background/60">
          <CardContent className="flex flex-col items-center justify-center gap-3 py-10 text-center text-muted-foreground">
            <CalendarIcon className="h-10 w-10" />
            <div className="space-y-1">
              <p className="font-medium text-foreground">
                No tournament sources available yet
              </p>
              <p className="text-sm">
                We could not derive a valid OddsPortal tournament URL from the current
                upcoming matches feed.
              </p>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-3 xl:grid-cols-2">
          {tournamentSources.map((source) => {
            const isSelected = source.key === selectedKey;

            return (
              <button
                key={source.key}
                type="button"
                onClick={() => setSelectedKey(source.key)}
                className={`rounded-2xl border p-4 text-left transition-all ${
                  isSelected
                    ? "border-primary bg-primary/10 shadow-[0_0_0_1px_hsl(var(--primary)/0.35)]"
                    : "border-border/60 bg-background/60 hover:border-primary/40 hover:bg-background"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant={isSelected ? "default" : "secondary"}>
                        {source.totalMatches} matches
                      </Badge>
                      <Badge variant="outline" className="border-border/60 text-xs">
                        Tournament source
                      </Badge>
                    </div>
                    <div>
                      <p className="text-base font-semibold text-foreground">
                        {source.tournamentName}
                      </p>
                      <p className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                        <GlobeIcon className="h-3.5 w-3.5" />
                        <span className="truncate">{source.tournamentUrl}</span>
                      </p>
                    </div>
                  </div>
                  {isSelected && <CheckCircle2Icon className="h-5 w-5 text-primary" />}
                </div>

                <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
                  <CalendarIcon className="h-3.5 w-3.5" />
                  <span>
                    First kickoff{" "}
                    {source.firstKickoff
                      ? formatKickoff(source.firstKickoff)
                      : "unknown"}
                  </span>
                </div>

                <div className="mt-4 space-y-2 rounded-xl border border-border/50 bg-muted/30 p-3">
                  <p className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
                    Match preview
                  </p>
                  {source.previewMatches.map((match) => (
                    <div
                      key={match.id}
                      className="flex items-center justify-between gap-3 text-sm"
                    >
                      <span className="min-w-0 truncate font-medium text-foreground">
                        {match.home_team?.name ?? "Home"} vs{" "}
                        {match.away_team?.name ?? "Away"}
                      </span>
                      <span className="whitespace-nowrap text-xs text-muted-foreground">
                        {formatKickoff(match.match_date)}
                      </span>
                    </div>
                  ))}
                </div>
              </button>
            );
          })}
        </div>
      )}

      {selectedSource && (
        <div className="rounded-2xl border border-primary/20 bg-primary/5 p-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-semibold text-foreground">
                {selectedSource.tournamentName} is ready to scrape
              </p>
              <p className="text-sm text-muted-foreground">
                The run will target {selectedSource.totalMatches} locally tracked
                upcoming matches through one tournament URL.
              </p>
            </div>
            <Badge variant="outline" className="border-primary/30 bg-background/60">
              Guided source selected
            </Badge>
          </div>
        </div>
      )}
    </div>
  );
}
