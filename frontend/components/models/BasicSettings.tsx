"use client";

import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import type { BasicSettingsProps } from "./types";

export function BasicSettings({ config, onConfigChange }: BasicSettingsProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="model-type">Model Type</Label>
        <Select
          value={config.modelType}
          onValueChange={(value) =>
            onConfigChange("modelType", value as typeof config.modelType)
          }
        >
          <SelectTrigger id="model-type">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="xgboost">XGBoost</SelectItem>
            <SelectItem value="lightgbm">LightGBM</SelectItem>
            <SelectItem value="random_forest">Random Forest</SelectItem>
            <SelectItem value="dixon_coles">Dixon-Coles</SelectItem>
            <SelectItem value="hybrid_poisson">Hybrid Poisson</SelectItem>
            <SelectItem value="market_mediation">Market Mediation</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">Description</Label>
        <Input
          id="description"
          placeholder="Optional label for this training run (e.g., 'XGBoost with tuned params')"
          value={config.description}
          onChange={(e) => onConfigChange("description", e.target.value)}
        />
      </div>

      <div className="grid gap-3">
        <label className="flex cursor-pointer items-center gap-3 rounded-md border p-3 hover:bg-muted/50">
          <Checkbox
            checked={config.tune}
            onCheckedChange={(checked) => onConfigChange("tune", Boolean(checked))}
          />
          <div>
            <div className="text-sm font-medium">Hyperparameter tuning</div>
            <div className="text-xs text-muted-foreground">
              Runs Optuna tuning ({config.tuningTrials} trials) before final training
            </div>
          </div>
        </label>

        <label className="flex cursor-pointer items-center gap-3 rounded-md border p-3 hover:bg-muted/50">
          <Checkbox
            checked={config.activate}
            onCheckedChange={(checked) => onConfigChange("activate", Boolean(checked))}
          />
          <div>
            <div className="text-sm font-medium">Activate after training</div>
            <div className="text-xs text-muted-foreground">
              Makes the new model the default for predictions immediately
            </div>
          </div>
        </label>

        <label className="flex cursor-pointer items-center gap-3 rounded-md border p-3 hover:bg-muted/50">
          <Checkbox
            checked={config.calibrateProbabilities}
            onCheckedChange={(checked) =>
              onConfigChange("calibrateProbabilities", Boolean(checked))
            }
          />
          <div>
            <div className="text-sm font-medium">Calibrate probabilities</div>
            <div className="text-xs text-muted-foreground">
              Apply {config.calibrationMethod} calibration for better probability
              estimates
            </div>
          </div>
        </label>

        <label
          className={`flex cursor-pointer items-center gap-3 rounded-md border p-3 hover:bg-muted/50 ${
            config.useGpuWorker ? "border-blue-500/50 bg-blue-500/5" : ""
          }`}
        >
          <Checkbox
            checked={config.useGpuWorker}
            onCheckedChange={(checked) =>
              onConfigChange("useGpuWorker", Boolean(checked))
            }
          />
          <div className="flex-1">
            <div className="flex items-center gap-2 text-sm font-medium">
              Use GPU worker
              <Badge variant="secondary" className="text-xs">
                Intel iGPU · :8011
              </Badge>
            </div>
            <div className="text-xs text-muted-foreground">
              Routes training to the Intel iGPU-accelerated worker (XGBoost SYCL /
              LightGBM OpenCL)
            </div>
          </div>
        </label>
      </div>
    </div>
  );
}
