"use client";

import { Combine, Info } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import type { DataRangeSectionProps } from "./types";

const ENSEMBLE_MODEL_TYPES = [
  { id: "xgboost", name: "XGBoost" },
  { id: "lightgbm", name: "LightGBM" },
  { id: "random_forest", name: "Random Forest" },
];

export function EnsembleSection({ config, onConfigChange }: DataRangeSectionProps) {
  const toggleEnsembleType = (modelType: string) => {
    const current = config.ensembleTypes;
    const updated = current.includes(modelType)
      ? current.filter((t) => t !== modelType)
      : [...current, modelType];
    // Ensure at least 2 types when ensemble is enabled
    if (config.useEnsemble && updated.length < 2) return;
    onConfigChange("ensembleTypes", updated);
  };

  return (
    <div className="space-y-3">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Combine className="h-4 w-4" />
        Ensemble Training
      </h4>

      <div className="flex items-center gap-2">
        <Checkbox
          id="use-ensemble"
          checked={config.useEnsemble}
          onCheckedChange={(checked) => {
            const enabled = Boolean(checked);
            onConfigChange("useEnsemble", enabled);
            // Default to xgboost + lightgbm if turning on with fewer than 2
            if (enabled && config.ensembleTypes.length < 2) {
              onConfigChange("ensembleTypes", ["xgboost", "lightgbm"]);
            }
          }}
        />
        <Label htmlFor="use-ensemble" className="text-sm font-normal">
          Train ensemble of multiple model types
        </Label>
      </div>

      {config.useEnsemble && (
        <div className="ml-6 space-y-2">
          <Label className="text-xs text-muted-foreground">
            Select models to combine (minimum 2)
          </Label>
          <div className="flex flex-wrap gap-2">
            {ENSEMBLE_MODEL_TYPES.map((model) => (
              <Badge
                key={model.id}
                variant={
                  config.ensembleTypes.includes(model.id) ? "default" : "outline"
                }
                className="cursor-pointer"
                onClick={() => toggleEnsembleType(model.id)}
              >
                {model.name}
              </Badge>
            ))}
          </div>
          <div className="flex items-start gap-1.5 rounded-md bg-muted/50 p-2">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <p className="text-xs text-muted-foreground">
              Ensemble uses weighted soft-vote averaging with optimized weights (grid
              search over validation set). The individual model type setting above is
              ignored when ensemble is enabled.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
