"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { MatchStatus } from "@/lib/types/api";

interface MatchFiltersProps {
  onFilterChange?: (filters: Record<string, string | null>) => void;
}

const statusOptions: { value: MatchStatus | "all"; label: string }[] = [
  { value: "all", label: "All Statuses" },
  { value: "SCHEDULED", label: "Scheduled" },
  { value: "LIVE", label: "Live" },
  { value: "FINISHED", label: "Finished" },
];

export function MatchFilters({ onFilterChange }: MatchFiltersProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const createQueryString = useCallback(
    (name: string, value: string | null) => {
      const params = new URLSearchParams(searchParams.toString());
      if (value === null || value === "all") {
        params.delete(name);
      } else {
        params.set(name, value);
      }
      return params.toString();
    },
    [searchParams]
  );

  const updateFilter = (name: string, value: string | null) => {
    const queryString = createQueryString(name, value);
    router.push(`/matches${queryString ? `?${queryString}` : ""}`);
    onFilterChange?.({ [name]: value });
  };

  const clearFilters = () => {
    router.push("/matches");
    onFilterChange?.({});
  };

  const hasFilters = searchParams.toString().length > 0;

  const currentStatus = (searchParams.get("status") as MatchStatus) || "all";
  const currentTournament = searchParams.get("tournament_id") || "";
  const currentTeam = searchParams.get("team_id") || "";
  const currentDaysAhead = searchParams.get("days_ahead") || "";

  return (
    <div className="space-y-4 rounded-lg border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Filters</h3>
        {hasFilters && (
          <Button variant="ghost" size="sm" onClick={clearFilters}>
            Clear All
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* Status Filter */}
        <div className="space-y-2">
          <Label htmlFor="status">Status</Label>
          <Select
            value={currentStatus}
            onValueChange={(value) =>
              updateFilter("status", value === "all" ? null : value)
            }
          >
            <SelectTrigger id="status">
              <SelectValue placeholder="Select status" />
            </SelectTrigger>
            <SelectContent>
              {statusOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Tournament Filter */}
        <div className="space-y-2">
          <Label htmlFor="tournament">Tournament ID</Label>
          <Input
            id="tournament"
            type="number"
            placeholder="Enter tournament ID"
            value={currentTournament}
            onChange={(e) => updateFilter("tournament_id", e.target.value || null)}
          />
        </div>

        {/* Team Filter */}
        <div className="space-y-2">
          <Label htmlFor="team">Team ID</Label>
          <Input
            id="team"
            type="number"
            placeholder="Enter team ID"
            value={currentTeam}
            onChange={(e) => updateFilter("team_id", e.target.value || null)}
          />
        </div>

        {/* Days Ahead Filter */}
        <div className="space-y-2">
          <Label htmlFor="days">Days Ahead</Label>
          <Input
            id="days"
            type="number"
            placeholder="e.g., 7"
            value={currentDaysAhead}
            onChange={(e) => updateFilter("days_ahead", e.target.value || null)}
          />
        </div>
      </div>

      {/* Active Filters */}
      {hasFilters && (
        <div className="flex flex-wrap gap-2 pt-2">
          {Array.from(searchParams.entries()).map(([key, value]) => (
            <Badge key={key} variant="secondary" className="gap-1">
              {key}: {value}
              <button
                onClick={() => updateFilter(key, null)}
                className="ml-1 hover:text-destructive"
              >
                ×
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
