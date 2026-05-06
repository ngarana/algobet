"use client";

import { Layers } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { DataRangeSectionProps } from "./types";

const AVAILABLE_FEATURE_GROUPS = [
  {
    id: "team_form",
    name: "Team Form",
    description: "Recent form, win rates, goals stats",
  },
  {
    id: "head_to_head",
    name: "Head-to-Head",
    description: "Historical matches between teams",
  },
  {
    id: "odds",
    name: "Market Odds",
    description: "Bookmaker odds, implied probabilities",
  },
  {
    id: "temporal",
    name: "Temporal",
    description: "Day of week, month, rest days, season period",
  },
  {
    id: "standings",
    name: "Standings",
    description: "League position, points, relegation, Euro spots",
  },
];

export function FeatureGroupsSection({
  config,
  onConfigChange,
}: DataRangeSectionProps) {
  const toggleFeatureGroup = (groupId: string) => {
    const current = config.featureGroups;
    const updated = current.includes(groupId)
      ? current.filter((id) => id !== groupId)
      : [...current, groupId];
    onConfigChange("featureGroups", updated);
  };

  return (
    <div className="space-y-4">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Layers className="h-4 w-4" />
        Feature Groups
      </h4>

      <div className="flex flex-wrap gap-2">
        {AVAILABLE_FEATURE_GROUPS.map((group) => {
          const isSelected =
            config.featureGroups.length === 0 ||
            config.featureGroups.includes(group.id);
          return (
            <Badge
              key={group.id}
              variant={isSelected ? "default" : "outline"}
              className="cursor-pointer px-3 py-2"
              onClick={() => toggleFeatureGroup(group.id)}
            >
              <div className="flex flex-col items-start">
                <span className="font-medium">{group.name}</span>
                <span className="text-xs opacity-70">{group.description}</span>
              </div>
            </Badge>
          );
        })}
      </div>
      <p className="text-xs text-muted-foreground">
        {config.featureGroups.length === 0
          ? "All feature groups enabled (default). Click to toggle specific groups."
          : `Selected ${config.featureGroups.length} feature group(s). Unselected groups will be excluded.`}
      </p>
    </div>
  );
}
