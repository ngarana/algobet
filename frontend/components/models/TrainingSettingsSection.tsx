"use client";

import { Clock } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { TrainingSettingsSectionProps } from "./types";

export function TrainingSettingsSection({
  config,
  onConfigChange,
}: TrainingSettingsSectionProps) {
  return (
    <div className="space-y-3">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Clock className="h-4 w-4" />
        Training Settings
      </h4>
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="random-seed">Random Seed</Label>
          <Input
            id="random-seed"
            type="number"
            min={0}
            max={999999}
            value={config.randomSeed}
            onChange={(e) =>
              onConfigChange("randomSeed", parseInt(e.target.value) || 42)
            }
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="early-stopping">Early Stopping</Label>
          <Input
            id="early-stopping"
            type="number"
            min={10}
            max={500}
            value={config.earlyStoppingRounds}
            onChange={(e) =>
              onConfigChange("earlyStoppingRounds", parseInt(e.target.value) || 50)
            }
          />
        </div>
      </div>
      {config.tune && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="tuning-trials">Tuning Trials</Label>
            <span className="text-xs text-muted-foreground">{config.tuningTrials}</span>
          </div>
          <Slider
            id="tuning-trials"
            min={10}
            max={200}
            step={10}
            value={[config.tuningTrials]}
            onValueChange={([value]) => onConfigChange("tuningTrials", value)}
          />
        </div>
      )}
      <div className="space-y-2">
        <Label htmlFor="calibration-method">Calibration Method</Label>
        <Select
          value={config.calibrationMethod}
          onValueChange={(value) =>
            onConfigChange(
              "calibrationMethod",
              value as typeof config.calibrationMethod
            )
          }
        >
          <SelectTrigger id="calibration-method">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="temperature">Temperature</SelectItem>
            <SelectItem value="isotonic">Isotonic</SelectItem>
            <SelectItem value="sigmoid">Sigmoid (Platt)</SelectItem>
            <SelectItem value="venn_abers">Venn-Abers</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-3 rounded-md border p-3">
        <div className="flex items-center gap-2">
          <Checkbox
            id="feature-selection"
            checked={config.featureSelection}
            onCheckedChange={(checked) =>
              onConfigChange("featureSelection", Boolean(checked))
            }
          />
          <Label htmlFor="feature-selection" className="text-sm font-normal">
            Enable feature selection
          </Label>
        </div>
        {config.featureSelection && (
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="feature-selection-threshold">Importance Threshold</Label>
              <Input
                id="feature-selection-threshold"
                type="number"
                min={0}
                max={1}
                step={0.001}
                value={config.featureSelectionThreshold}
                onChange={(e) =>
                  onConfigChange(
                    "featureSelectionThreshold",
                    parseFloat(e.target.value) || 0.005
                  )
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="min-samples-per-feature">Min Samples per Feature</Label>
              <Input
                id="min-samples-per-feature"
                type="number"
                min={1}
                max={5000}
                value={config.minSamplesPerFeature ?? ""}
                onChange={(e) =>
                  onConfigChange(
                    "minSamplesPerFeature",
                    e.target.value === "" ? null : parseInt(e.target.value, 10)
                  )
                }
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
