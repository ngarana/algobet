"use client";

import { Target } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import type { DataRangeSectionProps } from "./types";

export function MatchQualitySection({ config, onConfigChange }: DataRangeSectionProps) {
  return (
    <div className="space-y-4">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Target className="h-4 w-4" />
        Match Quality Filters
      </h4>

      {/* Require Odds */}
      <div className="flex items-center gap-2">
        <Checkbox
          id="require-odds"
          checked={config.requireOdds}
          onCheckedChange={(checked) => onConfigChange("requireOdds", Boolean(checked))}
        />
        <Label htmlFor="require-odds" className="text-sm font-normal">
          Require betting odds available
        </Label>
      </div>

      {/* Goals Range */}
      <div className="space-y-2">
        <Label>Total Goals Range</Label>
        <div className="flex items-center gap-2">
          <Input
            type="number"
            placeholder="Min (e.g. 1)"
            value={config.minTotalGoals ?? ""}
            onChange={(e) =>
              onConfigChange(
                "minTotalGoals",
                e.target.value ? Number(e.target.value) : null
              )
            }
            className="w-24"
            min={0}
          />
          <span className="text-muted-foreground">-</span>
          <Input
            type="number"
            placeholder="Max (e.g. 6)"
            value={config.maxTotalGoals ?? ""}
            onChange={(e) =>
              onConfigChange(
                "maxTotalGoals",
                e.target.value ? Number(e.target.value) : null
              )
            }
            className="w-24"
            min={0}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Filter matches by total goals (home_score + away_score). Leave empty for no
          filter.
        </p>
      </div>
    </div>
  );
}
