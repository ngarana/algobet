"use client";

import { Split } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import type { DataRangeSectionProps } from "./types";

const SPLIT_STRATEGIES = [
  {
    id: "temporal" as const,
    name: "Temporal",
    description: "Single chronological split by ratio",
  },
  {
    id: "expanding_window" as const,
    name: "Expanding Window",
    description: "Rolling window cross-validation",
  },
  {
    id: "season_aware" as const,
    name: "Season-Aware",
    description: "Split by complete football seasons",
  },
];

export function SplitStrategySection({
  config,
  onConfigChange,
}: DataRangeSectionProps) {
  return (
    <div className="space-y-4">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Split className="h-4 w-4" />
        Split Strategy
      </h4>

      {/* Strategy Selection */}
      <div className="flex flex-wrap gap-2">
        {SPLIT_STRATEGIES.map((strategy) => (
          <Badge
            key={strategy.id}
            variant={config.splitStrategy === strategy.id ? "default" : "outline"}
            className="cursor-pointer px-3 py-2"
            onClick={() => onConfigChange("splitStrategy", strategy.id)}
          >
            <div className="flex flex-col items-start">
              <span className="font-medium">{strategy.name}</span>
              <span className="text-xs opacity-70">{strategy.description}</span>
            </div>
          </Badge>
        ))}
      </div>

      {/* Temporal-specific: gap_days */}
      {config.splitStrategy === "temporal" && (
        <div className="space-y-2">
          <Label htmlFor="gap-days" className="text-sm">
            Gap Days
          </Label>
          <Input
            id="gap-days"
            type="number"
            value={config.gapDays}
            onChange={(e) => onConfigChange("gapDays", Number(e.target.value))}
            className="w-24"
            min={0}
            max={30}
          />
          <p className="text-xs text-muted-foreground">
            Days of gap between train/val and val/test sets to prevent data leakage at
            boundaries. Default: 0.
          </p>
        </div>
      )}

      {/* Expanding Window params */}
      {config.splitStrategy === "expanding_window" && (
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label htmlFor="min-train-size" className="text-xs">
              Min Train Size
            </Label>
            <Input
              id="min-train-size"
              type="number"
              value={config.minTrainSize}
              onChange={(e) => onConfigChange("minTrainSize", Number(e.target.value))}
              min={50}
              max={5000}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="step-size" className="text-xs">
              Step Size
            </Label>
            <Input
              id="step-size"
              type="number"
              value={config.stepSize}
              onChange={(e) => onConfigChange("stepSize", Number(e.target.value))}
              min={10}
              max={500}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="ew-val-size" className="text-xs">
              Validation Size
            </Label>
            <Input
              id="ew-val-size"
              type="number"
              value={config.ewValSize}
              onChange={(e) => onConfigChange("ewValSize", Number(e.target.value))}
              min={10}
              max={1000}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="ew-test-size" className="text-xs">
              Test Size
            </Label>
            <Input
              id="ew-test-size"
              type="number"
              value={config.ewTestSize}
              onChange={(e) => onConfigChange("ewTestSize", Number(e.target.value))}
              min={10}
              max={1000}
            />
          </div>
          <p className="col-span-2 text-xs text-muted-foreground">
            Expanding window uses a growing training set with fixed validation and test
            windows. Step size controls how much the window expands each iteration.
          </p>
        </div>
      )}

      {/* Season-Aware params */}
      {config.splitStrategy === "season_aware" && (
        <div className="grid grid-cols-3 gap-3">
          <div className="space-y-1">
            <Label htmlFor="train-seasons" className="text-xs">
              Train Seasons
            </Label>
            <Input
              id="train-seasons"
              type="number"
              value={config.trainSeasons}
              onChange={(e) => onConfigChange("trainSeasons", Number(e.target.value))}
              min={1}
              max={10}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="val-seasons" className="text-xs">
              Val Seasons
            </Label>
            <Input
              id="val-seasons"
              type="number"
              value={config.valSeasons}
              onChange={(e) => onConfigChange("valSeasons", Number(e.target.value))}
              min={1}
              max={5}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="test-seasons" className="text-xs">
              Test Seasons
            </Label>
            <Input
              id="test-seasons"
              type="number"
              value={config.testSeasons}
              onChange={(e) => onConfigChange("testSeasons", Number(e.target.value))}
              min={1}
              max={5}
            />
          </div>
          <p className="col-span-3 text-xs text-muted-foreground">
            Splits data by complete football seasons. Ensures no partial seasons in any
            set. Requires data spanning at least{" "}
            {config.trainSeasons + config.valSeasons + config.testSeasons} seasons.
          </p>
        </div>
      )}
    </div>
  );
}
