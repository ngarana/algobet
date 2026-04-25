"use client";

import { SlidersHorizontal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import type { DataSplitSectionProps } from "./types";

export function DataSplitSection({ config, onConfigChange }: DataSplitSectionProps) {
  const totalRatio = config.trainRatio + config.valRatio + config.testRatio;
  const isValid = Math.abs(totalRatio - 1.0) <= 0.001;

  return (
    <div className="space-y-3">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <SlidersHorizontal className="h-4 w-4" />
        Data Split
        {!isValid && (
          <Badge variant="destructive" className="text-xs">
            Must sum to 100%
          </Badge>
        )}
      </h4>
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Training Set</Label>
            <span className="text-xs text-muted-foreground">
              {(config.trainRatio * 100).toFixed(0)}%
            </span>
          </div>
          <Slider
            min={0.1}
            max={0.9}
            step={0.05}
            value={[config.trainRatio]}
            onValueChange={([value]) => {
              const remaining = 1 - value;
              onConfigChange("trainRatio", value);
              onConfigChange("valRatio", remaining * 0.5);
              onConfigChange("testRatio", remaining * 0.5);
            }}
          />
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Validation Set</Label>
            <span className="text-xs text-muted-foreground">
              {(config.valRatio * 100).toFixed(0)}%
            </span>
          </div>
          <Slider
            min={0.05}
            max={0.45}
            step={0.05}
            value={[config.valRatio]}
            onValueChange={([value]) => {
              const maxTest = 1 - config.trainRatio - 0.05;
              const clampedValue = Math.min(value, maxTest);
              onConfigChange("valRatio", clampedValue);
              onConfigChange("testRatio", 1 - config.trainRatio - clampedValue);
            }}
          />
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Test Set</Label>
            <span className="text-xs text-muted-foreground">
              {(config.testRatio * 100).toFixed(0)}%
            </span>
          </div>
          <Slider
            min={0.05}
            max={0.45}
            step={0.05}
            value={[config.testRatio]}
            onValueChange={([value]) => {
              const maxVal = 1 - config.trainRatio - 0.05;
              const clampedValue = Math.min(value, maxVal);
              onConfigChange("testRatio", clampedValue);
              onConfigChange("valRatio", 1 - config.trainRatio - clampedValue);
            }}
          />
        </div>
      </div>
    </div>
  );
}
