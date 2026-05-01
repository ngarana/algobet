"use client";

import { Scale } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import type { DataRangeSectionProps } from "./types";

export function OutcomeBalancingSection({
  config,
  onConfigChange,
}: DataRangeSectionProps) {
  return (
    <div className="space-y-3">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Scale className="h-4 w-4" />
        Outcome Balancing
      </h4>

      <div className="flex items-center gap-2">
        <Checkbox
          id="outcome-balance"
          checked={config.outcomeBalance}
          onCheckedChange={(checked) =>
            onConfigChange("outcomeBalance", Boolean(checked))
          }
        />
        <Label htmlFor="outcome-balance" className="text-sm font-normal">
          Enable inverse-frequency class weighting
        </Label>
      </div>

      <p className="ml-6 text-xs text-muted-foreground">
        Applies higher weights to minority outcomes (e.g., draws, away wins) during
        training to balance the class distribution. Disable to use raw frequencies.
      </p>
    </div>
  );
}
