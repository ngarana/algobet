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
    id: "temporal",
    name: "Temporal",
    description: "Day of week, month, rest days, season period",
  },
  {
    id: "standings",
    name: "Standings",
    description: "League position, points, relegation, Euro spots",
  },
  {
    id: "enriched_stats",
    name: "Enriched Stats",
    description: "Rolling xG, shot, corners, PPDA, and player stats",
  },
  {
    id: "draw_signals",
    name: "Draw Signals",
    description: "Low-scoring, parity, and xG draw indicators",
  },
  {
    id: "matchup_interaction",
    name: "Matchup Interaction",
    description: "Home-away style and strength interaction terms",
  },
  {
    id: "elo_rating",
    name: "Elo Ratings",
    description: "Rolling team strength ratings and rating gaps",
  },
  {
    id: "expected_points",
    name: "Expected Points",
    description: "Recent points expectation and performance trend",
  },
  {
    id: "player_quality",
    name: "Player Quality",
    description: "Player availability and rolling contribution signals",
  },
  {
    id: "odds",
    name: "1X2 Odds",
    description: "Stored home, draw, and away market probabilities",
  },
  {
    id: "odds_residual",
    name: "Odds Residuals",
    description: "Market residuals against football-only model signals",
  },
  {
    id: "detailed_odds",
    name: "Detailed Odds",
    description: "Bookmaker consensus, AH, and over-under market signals",
  },
];

const DEFAULT_FEATURE_GROUP_IDS = [
  "team_form",
  "head_to_head",
  "temporal",
  "standings",
  "enriched_stats",
  "draw_signals",
  "matchup_interaction",
];

export function FeatureGroupsSection({
  config,
  onConfigChange,
}: DataRangeSectionProps) {
  const toggleFeatureGroup = (groupId: string) => {
    const current =
      config.featureGroups.length > 0
        ? config.featureGroups
        : DEFAULT_FEATURE_GROUP_IDS;
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
            config.featureGroups.length === 0
              ? DEFAULT_FEATURE_GROUP_IDS.includes(group.id)
              : config.featureGroups.includes(group.id);
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
          ? "Backend default feature set enabled. Select groups to override it."
          : `Selected ${config.featureGroups.length} feature group(s). Unselected groups will be excluded.`}
      </p>
    </div>
  );
}
