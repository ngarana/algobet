"use client";

import { Database } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import type { DataRangeSectionProps } from "./types";

export function DataRangeSection({ config, onConfigChange }: DataRangeSectionProps) {
  return (
    <div className="space-y-3">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Database className="h-4 w-4" />
        Data Range
      </h4>
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="start-date">Start Date</Label>
          <Input
            id="start-date"
            type="date"
            value={config.startDate}
            onChange={(e) => onConfigChange("startDate", e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="end-date">End Date</Label>
          <Input
            id="end-date"
            type="date"
            value={config.endDate}
            onChange={(e) => onConfigChange("endDate", e.target.value)}
          />
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="min-matches">Minimum Matches</Label>
          <span className="text-xs text-muted-foreground">{config.minMatches}</span>
        </div>
        <Slider
          id="min-matches"
          min={100}
          max={10000}
          step={100}
          value={[config.minMatches]}
          onValueChange={([value]) => onConfigChange("minMatches", value)}
        />
      </div>
    </div>
  );
}
