"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { Check, Search, Trophy, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useTournaments } from "@/lib/queries/use-tournaments";
import { useTeams } from "@/lib/queries/use-teams";
import { cn } from "@/lib/utils";
import type { DataRangeSectionProps } from "./types";

export function DataSelectionSection({
  config,
  onConfigChange,
}: DataRangeSectionProps) {
  const [tournamentSearch, setTournamentSearch] = useState("");
  const [teamSearch, setTeamSearch] = useState("");
  const [tournamentLabels, setTournamentLabels] = useState<Record<number, string>>({});
  const [teamLabels, setTeamLabels] = useState<Record<number, string>>({});

  const deferredTournamentSearch = useDeferredValue(tournamentSearch.trim());
  const deferredTeamSearch = useDeferredValue(teamSearch.trim());
  const { data: tournaments = [], isLoading: tournamentsLoading } = useTournaments({
    search: deferredTournamentSearch || undefined,
    limit: 24,
  });
  const scopedTournamentId =
    config.tournamentIds.length === 1 ? config.tournamentIds[0] : undefined;
  const { data: teams = [], isLoading: teamsLoading } = useTeams({
    search: deferredTeamSearch || undefined,
    tournament_id: scopedTournamentId,
    limit: 50,
  });

  useEffect(() => {
    if (tournaments.length === 0) {
      return;
    }

    setTournamentLabels((current) => {
      const next = { ...current };
      let changed = false;

      tournaments.forEach((tournament) => {
        if (next[tournament.id] !== tournament.name) {
          next[tournament.id] = tournament.name;
          changed = true;
        }
      });

      return changed ? next : current;
    });
  }, [tournaments]);

  useEffect(() => {
    if (teams.length === 0) {
      return;
    }

    setTeamLabels((current) => {
      const next = { ...current };
      let changed = false;

      teams.forEach((team) => {
        if (next[team.id] !== team.name) {
          next[team.id] = team.name;
          changed = true;
        }
      });

      return changed ? next : current;
    });
  }, [teams]);

  const toggleTournament = (tournamentId: number) => {
    const current = config.tournamentIds;
    const updated = current.includes(tournamentId)
      ? current.filter((id) => id !== tournamentId)
      : [...current, tournamentId];
    onConfigChange("tournamentIds", updated);
  };

  const toggleTeam = (teamId: number) => {
    const current = config.teamIds;
    const updated = current.includes(teamId)
      ? current.filter((id) => id !== teamId)
      : [...current, teamId];
    onConfigChange("teamIds", updated);
  };

  const visibleTeams = useMemo(
    () => teams.slice(0, teamSearch.trim() ? 24 : 12),
    [teamSearch, teams]
  );

  return (
    <div className="space-y-5">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Trophy className="h-4 w-4" />
        Tournament & Team Selection
      </h4>

      <div className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div className="space-y-1">
            <Label htmlFor="tournament-search">Tournaments</Label>
            <p className="text-xs text-muted-foreground">
              Search and pick one or more competitions to narrow the training set.
            </p>
          </div>
          {config.tournamentIds.length > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => onConfigChange("tournamentIds", [])}
            >
              Clear
            </Button>
          )}
        </div>

        <div className="space-y-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              id="tournament-search"
              placeholder="Search tournaments..."
              value={tournamentSearch}
              onChange={(event) => setTournamentSearch(event.target.value)}
              className="pl-9"
            />
          </div>

          <div className="max-h-60 overflow-y-auto rounded-md border bg-background">
            {tournamentsLoading ? (
              <p className="px-3 py-2 text-sm text-muted-foreground">
                Loading tournaments...
              </p>
            ) : tournaments.length > 0 ? (
              tournaments.map((tournament) => (
                <OptionRow
                  key={tournament.id}
                  isSelected={config.tournamentIds.includes(tournament.id)}
                  label={tournament.name}
                  meta={tournament.country}
                  onSelect={() => toggleTournament(tournament.id)}
                />
              ))
            ) : (
              <p className="px-3 py-2 text-sm text-muted-foreground">
                No tournaments match your search.
              </p>
            )}
          </div>
        </div>

        {config.tournamentIds.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {config.tournamentIds.map((tournamentId) => (
              <SelectedChip
                key={tournamentId}
                label={tournamentLabels[tournamentId] ?? `Tournament #${tournamentId}`}
                onRemove={() => toggleTournament(tournamentId)}
              />
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Leave empty to include all tournaments.
          </p>
        )}
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div className="space-y-1">
            <Label htmlFor="team-search">Teams</Label>
            <p className="text-xs text-muted-foreground">
              Search for clubs to include. When one tournament is selected, team search
              is scoped to that competition.
            </p>
          </div>
          {config.teamIds.length > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => onConfigChange("teamIds", [])}
            >
              Clear
            </Button>
          )}
        </div>

        <div className="space-y-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              id="team-search"
              placeholder="Search teams..."
              value={teamSearch}
              onChange={(event) => setTeamSearch(event.target.value)}
              className="pl-9"
            />
          </div>

          <div className="max-h-60 overflow-y-auto rounded-md border bg-background">
            {teamsLoading ? (
              <p className="px-3 py-2 text-sm text-muted-foreground">
                Loading teams...
              </p>
            ) : visibleTeams.length > 0 ? (
              visibleTeams.map((team) => (
                <OptionRow
                  key={team.id}
                  isSelected={config.teamIds.includes(team.id)}
                  label={team.name}
                  meta={
                    config.tournamentIds.length > 1
                      ? "Search runs across all tournaments"
                      : scopedTournamentId
                        ? "Scoped to selected tournament"
                        : "Available team"
                  }
                  onSelect={() => toggleTeam(team.id)}
                />
              ))
            ) : (
              <p className="px-3 py-2 text-sm text-muted-foreground">
                No teams match your search.
              </p>
            )}
          </div>
        </div>

        {config.teamIds.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {config.teamIds.map((teamId) => (
              <SelectedChip
                key={teamId}
                label={teamLabels[teamId] ?? `Team #${teamId}`}
                onRemove={() => toggleTeam(teamId)}
              />
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Leave empty to include all teams.
          </p>
        )}
      </div>

      <div className="space-y-2">
        <Label>Match Venue</Label>
        <div className="flex gap-2">
          {(["both", "home", "away"] as const).map((venue) => (
            <button
              type="button"
              key={venue}
              className={cn(
                "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                config.venueFilter === venue
                  ? "border-primary bg-primary text-primary-foreground"
                  : "bg-background text-foreground hover:bg-muted"
              )}
              onClick={() => onConfigChange("venueFilter", venue)}
            >
              {venue === "both"
                ? "Home & Away"
                : venue === "home"
                  ? "Home Only"
                  : "Away Only"}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

interface OptionRowProps {
  isSelected: boolean;
  label: string;
  meta: string;
  onSelect: () => void;
}

function OptionRow({ isSelected, label, meta, onSelect }: OptionRowProps) {
  return (
    <button
      type="button"
      className={cn(
        "flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm transition-colors hover:bg-muted/60",
        isSelected && "bg-primary/5"
      )}
      onClick={onSelect}
    >
      <div className="min-w-0">
        <div className="truncate font-medium">{label}</div>
        <div className="truncate text-xs text-muted-foreground">{meta}</div>
      </div>
      <div
        className={cn(
          "flex h-5 w-5 items-center justify-center rounded-full border",
          isSelected
            ? "border-primary bg-primary text-primary-foreground"
            : "border-muted-foreground/30 text-transparent"
        )}
      >
        <Check className="h-3 w-3" />
      </div>
    </button>
  );
}

interface SelectedChipProps {
  label: string;
  onRemove: () => void;
}

function SelectedChip({ label, onRemove }: SelectedChipProps) {
  return (
    <button
      type="button"
      className="inline-flex items-center gap-1 rounded-full border bg-muted px-3 py-1 text-xs font-medium transition-colors hover:bg-muted/70"
      onClick={onRemove}
    >
      <span>{label}</span>
      <X className="h-3 w-3" />
    </button>
  );
}
