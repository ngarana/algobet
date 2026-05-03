"use client";

import { SlidersHorizontal } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import type { DataRangeSectionProps } from "./types";

interface HyperparamDef {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
}

const HYPERPARAMS_BY_MODEL: Record<string, HyperparamDef[]> = {
  xgboost: [
    { key: "max_depth", label: "Max Depth", min: 3, max: 15, step: 1, defaultValue: 6 },
    {
      key: "learning_rate",
      label: "Learning Rate",
      min: 0.01,
      max: 0.3,
      step: 0.01,
      defaultValue: 0.1,
    },
    {
      key: "n_estimators",
      label: "N Estimators",
      min: 50,
      max: 1000,
      step: 50,
      defaultValue: 100,
    },
    {
      key: "subsample",
      label: "Subsample",
      min: 0.5,
      max: 1.0,
      step: 0.05,
      defaultValue: 0.8,
    },
    {
      key: "colsample_bytree",
      label: "Col Sample",
      min: 0.5,
      max: 1.0,
      step: 0.05,
      defaultValue: 0.8,
    },
    {
      key: "min_child_weight",
      label: "Min Child Weight",
      min: 1,
      max: 10,
      step: 1,
      defaultValue: 1,
    },
  ],
  lightgbm: [
    {
      key: "num_leaves",
      label: "Num Leaves",
      min: 15,
      max: 127,
      step: 1,
      defaultValue: 31,
    },
    {
      key: "learning_rate",
      label: "Learning Rate",
      min: 0.01,
      max: 0.3,
      step: 0.01,
      defaultValue: 0.1,
    },
    {
      key: "n_estimators",
      label: "N Estimators",
      min: 50,
      max: 1000,
      step: 50,
      defaultValue: 100,
    },
    {
      key: "subsample",
      label: "Subsample",
      min: 0.5,
      max: 1.0,
      step: 0.05,
      defaultValue: 0.8,
    },
    {
      key: "colsample_bytree",
      label: "Col Sample",
      min: 0.5,
      max: 1.0,
      step: 0.05,
      defaultValue: 0.8,
    },
  ],
  random_forest: [
    {
      key: "n_estimators",
      label: "N Estimators",
      min: 50,
      max: 1000,
      step: 50,
      defaultValue: 100,
    },
    {
      key: "max_depth",
      label: "Max Depth",
      min: 3,
      max: 20,
      step: 1,
      defaultValue: 10,
    },
    {
      key: "min_samples_split",
      label: "Min Samples Split",
      min: 2,
      max: 20,
      step: 1,
      defaultValue: 2,
    },
    {
      key: "min_samples_leaf",
      label: "Min Samples Leaf",
      min: 1,
      max: 10,
      step: 1,
      defaultValue: 1,
    },
  ],
};

export function HyperparametersSection({
  config,
  onConfigChange,
}: DataRangeSectionProps) {
  const params = HYPERPARAMS_BY_MODEL[config.modelType] ?? [];

  const updateHyperparam = (key: string, value: number) => {
    onConfigChange("customHyperparameters", {
      ...config.customHyperparameters,
      [key]: value,
    });
  };

  const clearHyperparams = () => {
    onConfigChange("customHyperparameters", {});
  };

  // Don't show when tuning is enabled (tuner will find optimal values)
  if (config.tune) {
    return (
      <div className="space-y-3">
        <h4 className="flex items-center gap-2 text-sm font-semibold">
          <SlidersHorizontal className="h-4 w-4" />
          Custom Hyperparameters
        </h4>
        <p className="text-xs text-muted-foreground">
          Hyperparameter tuning is enabled — Optuna will search for optimal values
          automatically. Disable tuning to set custom values manually.
        </p>
      </div>
    );
  }

  // Don't show when ensemble is enabled
  if (config.useEnsemble) {
    return (
      <div className="space-y-3">
        <h4 className="flex items-center gap-2 text-sm font-semibold">
          <SlidersHorizontal className="h-4 w-4" />
          Custom Hyperparameters
        </h4>
        <p className="text-xs text-muted-foreground">
          Ensemble training is enabled — hyperparameters are applied uniformly to all
          ensemble members. Per-model customization is not yet supported.
        </p>
      </div>
    );
  }

  const hasCustomValues = Object.keys(config.customHyperparameters).length > 0;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="flex items-center gap-2 text-sm font-semibold">
          <SlidersHorizontal className="h-4 w-4" />
          Custom Hyperparameters
        </h4>
        {hasCustomValues && (
          <button
            type="button"
            className="text-xs text-muted-foreground underline hover:text-foreground"
            onClick={clearHyperparams}
          >
            Reset to defaults
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {params.map((param) => {
          const currentValue =
            config.customHyperparameters[param.key] ?? param.defaultValue;
          return (
            <div key={param.key} className="space-y-1">
              <Label htmlFor={`hp-${param.key}`} className="text-xs">
                {param.label}
              </Label>
              <Input
                id={`hp-${param.key}`}
                type="number"
                value={currentValue}
                onChange={(e) => updateHyperparam(param.key, Number(e.target.value))}
                min={param.min}
                max={param.max}
                step={param.step}
              />
            </div>
          );
        })}
      </div>

      <p className="text-xs text-muted-foreground">
        {hasCustomValues
          ? `${Object.keys(config.customHyperparameters).length} custom parameter(s) set. Empty fields use model defaults.`
          : "Leave unchanged to use model defaults. Or enable hyperparameter tuning to auto-search."}
      </p>
    </div>
  );
}
