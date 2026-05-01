"use client";

import { Trophy } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { useTournaments } from "@/lib/queries/use-tournaments";
import { useTeams } from "@/lib/queries/use-teams";
import type { DataRangeSectionProps } from "./types";

export function DataSelectionSection({
  config,
  onConfigChange,
}: DataRangeSectionProps) {
  const { data: tournaments = [], isLoading: tournamentsLoading } = useTournaments();
  const { data: teams = [], isLoading: teamsLoading } = useTeams({ limit: 100 });

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

  return (
    <div className="space-y-4">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Trophy className="h-4 w-4" />
        Tournament & Team Selection
      </h4>

      {/* Tournament Selection */}
      <div className="space-y-2">
        <Label>Tournaments</Label>
        <div className="flex flex-wrap gap-2">
          {tournamentsLoading ? (
            <span className="text-xs text-muted-foreground">Loading...</span>
          ) : tournaments.length > 0 ? (
            tournaments.slice(0, 8).map((t) => (
              <Badge
                key={t.id}
                variant={config.tournamentIds.includes(t.id) ? "default" : "outline"}
                className="cursor-pointer"
                onClick={() => toggleTournament(t.id)}
              >
                {t.name}
              </Badge>
            ))
          ) : (
            <span className="text-xs text-muted-foreground">
              No tournaments available
            </span>
          )}
          {config.tournamentIds.length > 0 && (
            <Badge
              variant="secondary"
              className="cursor-pointer"
              onClick={() => onConfigChange("tournamentIds", [])}
            >
              Clear ({config.tournamentIds.length})
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {config.tournamentIds.length > 0
            ? `Selected ${config.tournamentIds.length} tournament(s). Leave empty for all.`
            : "Leave empty to include all tournaments."}
        </p>
      </div>

      {/* Team Selection */}
      <div className="space-y-2">
        <Label>Teams</Label>
        <div className="flex flex-wrap gap-2">
          {teamsLoading ? (
            <span className="text-xs text-muted-foreground">Loading...</span>
          ) : teams.length > 0 ? (
            teams.slice(0, 12).map((t) => (
              <Badge
                key={t.id}
                variant={config.teamIds.includes(t.id) ? "default" : "outline"}
                className="cursor-pointer"
                onClick={() => toggleTeam(t.id)}
              >
                {t.name}
              </Badge>
            ))
          ) : (
            <span className="text-xs text-muted-foreground">No teams available</span>
          )}
          {config.teamIds.length > 0 && (
            <Badge
              variant="secondary"
              className="cursor-pointer"
              onClick={() => onConfigChange("teamIds", [])}
            >
              Clear ({config.teamIds.length})
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {config.teamIds.length > 0
            ? `Selected ${config.teamIds.length} team(s). Matches involving these teams will be included.`
            : "Leave empty to include all teams."}
        </p>
      </div>

      {/* Venue Filter */}
      <div className="space-y-2">
        <Label>Match Venue</Label>
        <div className="flex gap-2">
          {(["both", "home", "away"] as const).map((venue) => (
            <Badge
              key={venue}
              variant={config.venueFilter === venue ? "default" : "outline"}
              className="cursor-pointer"
              onClick={() => onConfigChange("venueFilter", venue)}
            >
              {venue === "both"
                ? "Home & Away"
                : venue === "home"
                  ? "Home Only"
                  : "Away Only"}
            </Badge>
          ))}
        </div>
      </div>
    </div>
  );
}
